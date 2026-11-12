"""Rendering: frames -> ffmpeg (parallel segments) -> audio mix -> loudness -> mux. Stills & contact sheets."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import audio as AU
from . import media, project as PJ
from .compositor import ProjectRuntime
from .util import fmt_tc

PRESETS = {
    "final":   {"ext": ".mp4", "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-pix_fmt", "yuv420p"]},
    "youtube": {"ext": ".mp4", "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "14", "-profile:v", "high", "-pix_fmt", "yuv420p",
                                     "-bf", "2", "-g", "60"]},
    "draft":   {"ext": ".mp4", "v": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"]},
    "hevc":    {"ext": ".mp4", "v": ["-c:v", "libx265", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", "log-level=error"]},
    "prores":  {"ext": ".mov", "v": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]},
    "prores4444": {"ext": ".mov", "alpha": True, "v": ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]},
    "webm_alpha": {"ext": ".webm", "alpha": True, "v": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24"]},
    "hdr10": {"ext": ".mp4", "hdr": True, "v": ["-c:v", "libx265", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p10le",
                                                 "-tag:v", "hvc1", "-x265-params",
                                                 "log-level=error:hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                                                 "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):"
                                                 "max-cll=1000,400"]},
}
HDR_TAGS = ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-color_range", "tv"]
COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def to_uint8(frame: np.ndarray, alpha: bool, dither: bool = True, seed: int = 0) -> np.ndarray:
    if alpha:
        a = frame[..., 3:4]
        rgb = np.where(a > 1e-6, frame[..., :3] / np.maximum(a, 1e-6), 0)
        out = np.concatenate([rgb, a], -1)
    else:
        out = frame[..., :3]
    out = np.clip(out, 0, 1) * 255.0
    if dither:  # kills banding in dark gradients / vignettes
        rng = np.random.default_rng(seed)
        out = out + rng.random(out.shape[:2] + (1,), dtype=np.float32) - 0.5
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- worker
_RT = {}


def _init_worker(norm, scale, output="sdr", proxies=False):
    _RT["rt"] = ProjectRuntime(norm, scale, proxies=proxies)
    _RT["rt"].color.output = output
    cv2.setNumThreads(1)


def _render_segment(args):
    f0, f1, path, vargs, alpha, fps, size, quiet = args[:8]
    hdr = len(args) > 8 and args[8]
    rt = _RT["rt"]
    comp = rt.main
    W, H = size
    pf = "rgb48le" if hdr else ("rgba" if alpha else "rgb24")
    cmd = [media.FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pf, "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    if hdr:
        cmd += ["-vf", "scale=out_color_matrix=bt2020nc:out_range=tv"] + vargs + HDR_TAGS + ["-an", path]
    else:
        if not alpha:
            cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv"]
        cmd += vargs + (COLOR_TAGS if not alpha else []) + ["-an", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    for i, f in enumerate(range(f0, f1)):
        fr = comp.render(f / fps)
        if hdr:
            p.stdin.write((np.clip(fr[..., :3], 0, 1) * 65535 + 0.5).astype("<u2").tobytes())
        else:
            p.stdin.write(to_uint8(fr, alpha, seed=f).tobytes())
        if not quiet and (i % 15 == 0 or f == f1 - 1):
            el = time.time() - t_start
            spd = (i + 1) / max(el, 1e-6)
            eta = (f1 - f - 1) / max(spd, 1e-6)
            sys.stderr.write(f"\r  seg {f0}-{f1}: frame {f + 1}/{f1}  {spd:.1f} fps  eta {eta:.0f}s   ")
            sys.stderr.flush()
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    open(path + ".done", "w").write("ok")
    return path


# ---------------------------------------------------------------- public API
def render(project, out: str, start: float | None = None, end: float | None = None, scale: float = 1.0,
           workers: int | None = None, preset: str = "final", audio: bool = True, loudness: float | None = None,
           quiet: bool = False, resume: bool = True, proxies: bool = False, segment_seconds: float = 10.0) -> dict:
    norm = project if isinstance(project, dict) and "comps" in project else PJ.load(project)
    comp = norm["comps"]["main"]
    fps = float(comp["fps"])
    t0 = float(start or 0.0)
    t1 = float(end if end is not None else comp["duration"])
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    if f1 <= f0:
        raise ValueError("Empty render range")
    W = max(2, int(round(comp["width"] * scale)))
    H = max(2, int(round(comp["height"] * scale)))
    if preset not in PRESETS and preset not in ("png", "gif"):
        raise ValueError(f"Unknown preset {preset}. Known: {', '.join(list(PRESETS) + ['png', 'gif'])}")
    alpha = PRESETS.get(preset, {}).get("alpha", False) or preset == "png"
    if W % 2 or H % 2:
        if not alpha and preset != "png":
            raise ValueError(f"Output size {W}x{H} must be even for {preset}")
    workers = workers or max(1, min(os.cpu_count() or 1, 8))
    tmp = tempfile.mkdtemp(prefix="cinegraph_")
    t_begin = time.time()
    try:
        if preset == "png":
            os.makedirs(out, exist_ok=True)
            rt = ProjectRuntime(norm, scale)
            for f in range(f0, f1):
                fr = to_uint8(rt.main.render(f / fps), True, seed=f)
                cv2.imwrite(os.path.join(out, f"frame_{f:06d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGBA2BGRA))
            return {"output": out, "frames": f1 - f0}
        vpreset = PRESETS["final" if preset == "gif" else preset]
        n = f1 - f0
        # short segments => crash-safe resume + parallelism; finished segments are reused on re-run
        nseg = max(1, min(max(workers, int(math.ceil(n / max(1.0, segment_seconds * fps)))), n // 15 or 1))
        import hashlib, json as _json
        sig = hashlib.md5(_json.dumps([norm, scale, preset, f0, f1, nseg], sort_keys=True, default=str).encode()).hexdigest()[:12]
        parts = os.path.abspath(out) + f".parts_{sig}"
        if resume:
            os.makedirs(parts, exist_ok=True)
        else:
            parts = tmp
        bounds = [f0 + round(n * k / nseg) for k in range(nseg + 1)]
        hdr = bool(vpreset.get("hdr"))
        if hdr and not (norm["comps"]["main"].get("color", {}).get("working", "display") != "display"):
            raise ValueError('hdr10 needs scene-linear compositing: set comp "color": {"working": "acescg"} (or "linear")')
        jobs = [(bounds[k], bounds[k + 1], os.path.join(parts, f"seg{k:04d}{vpreset['ext']}"), vpreset["v"], alpha, fps, (W, H), quiet, hdr)
                for k in range(nseg)]
        todo = [j for j in jobs if not (os.path.exists(j[2] + ".done") and os.path.exists(j[2]))]
        reused = len(jobs) - len(todo)
        if reused and not quiet:
            sys.stderr.write(f"  resuming: {reused}/{len(jobs)} segments already rendered\n")
        if len(todo) == 1 or workers == 1:
            _init_worker(norm, scale, "hdr10" if hdr else "sdr", proxies)
            for j in todo:
                _render_segment(j)
        elif todo:
            import multiprocessing as mp
            ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=ctx, initializer=_init_worker,
                                     initargs=(norm, scale, "hdr10" if hdr else "sdr", proxies)) as ex:
                list(ex.map(_render_segment, todo))
        segs = [j[2] for j in jobs]
        if not quiet:
            sys.stderr.write("\n")
        video = os.path.join(tmp, "video" + vpreset["ext"])
        if len(segs) == 1:
            shutil.copy2(segs[0], video)
        else:
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as f:
                for s in segs:
                    f.write(f"file '{s}'\n")
            r = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("concat failed: " + r.stderr)
        stats = {}
        final_audio = None
        if audio and preset != "gif":
            mixed, st = AU.mix(norm, t0, t1)
            if st["has_audio"]:
                raw = os.path.join(tmp, "mix.wav")
                AU.write_wav(raw, mixed)
                mc = AU.master_chain(norm.get("audio_settings", {}))
                if mc:
                    raw = AU.apply_chain(raw, os.path.join(tmp, "mix_master.wav"), mc)
                target = loudness if loudness is not None else norm.get("audio_settings", {}).get("loudness", -14.0)
                if loudness is False:
                    target = None
                if target is not None and target is not False:
                    final_audio = os.path.join(tmp, "mix_norm.wav")
                    stats["loudness_in"] = AU.loudnorm(raw, final_audio, float(target))["input_i"]
                    stats["loudness_target"] = target
                else:
                    final_audio = raw
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        if preset == "gif":
            r = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", video, "-vf",
                                "fps=15,scale='min(720,iw)':-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
                                out], capture_output=True, text=True)
        elif final_audio:
            acodec = ["-c:a", "pcm_s24le"] if out.endswith(".mov") else (["-c:a", "libopus", "-b:a", "192k"] if out.endswith(".webm") else ["-c:a", "aac", "-b:a", "320k"])
            r = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", video, "-i", final_audio, "-map", "0:v", "-map", "1:a",
                                "-c:v", "copy", *acodec, "-shortest", "-movflags", "+faststart", out], capture_output=True, text=True)
        else:
            r = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", video, "-c", "copy", "-movflags", "+faststart", out],
                               capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mux failed: " + r.stderr)
        el = time.time() - t_begin
        stats.update({"output": out, "frames": n, "seconds": round(el, 1), "render_fps": round(n / max(el, 1e-6), 2),
                      "size": f"{W}x{H}", "segments": nseg, "segments_reused": reused})
        if resume:
            shutil.rmtree(parts, ignore_errors=True)
        return stats
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def still(project, t: float, out: str, scale: float = 1.0, alpha: bool = False):
    norm = project if isinstance(project, dict) and "comps" in project else PJ.load(project)
    rt = ProjectRuntime(norm, scale)
    fr = rt.main.render(float(t))
    img = to_uint8(fr, alpha, dither=False)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA if alpha else cv2.COLOR_RGB2BGR))
    return out


def contact_sheet(project, out: str, times=None, every: float | None = None, cols: int = 4, width: int = 480):
    """Grid of frames with timecodes — fast visual review (great for Claude to inspect)."""
    norm = project if isinstance(project, dict) and "comps" in project else PJ.load(project)
    comp = norm["comps"]["main"]
    if times is None:
        step = every or max(0.5, comp["duration"] / 16)
        times = list(np.arange(0, comp["duration"] - 1e-6, step))
    scale = width / comp["width"]
    rt = ProjectRuntime(norm, scale)
    tiles = []
    for t in times:
        img = to_uint8(rt.main.render(float(t)), False, dither=False)
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (0, 0), (110, 26), (0, 0, 0), -1)
        cv2.putText(img, fmt_tc(t), (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * (h + 6) + 6, cols * (w + 6) + 6, 3), 30, np.uint8)
    for i, tl in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[6 + r * (h + 6):6 + r * (h + 6) + h, 6 + c * (w + 6):6 + c * (w + 6) + w] = tl
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out
