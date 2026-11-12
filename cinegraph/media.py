"""Media IO: probing, frame-accurate video decoding, images, audio decoding."""
from __future__ import annotations

import functools
import json
import math
import os
import subprocess
from collections import OrderedDict

import cv2
import numpy as np

FFMPEG = os.environ.get("CINEGRAPH_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("CINEGRAPH_FFPROBE", "ffprobe")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"}
AUDIO_EXT = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}


class MediaError(RuntimeError):
    pass


@functools.lru_cache(maxsize=512)
def probe(path: str) -> dict:
    if not os.path.exists(path):
        raise MediaError(f"File not found: {path}")
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    info = json.loads(r.stdout)
    out = {"path": path, "has_video": False, "has_audio": False, "duration": None}
    fmt = info.get("format", {})
    if fmt.get("duration"):
        out["duration"] = float(fmt["duration"])
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not out["has_video"]:
            if s.get("disposition", {}).get("attached_pic"):
                continue
            out["has_video"] = True
            out["width"] = int(s["width"])
            out["height"] = int(s["height"])
            num, den = (s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1").split("/")
            fps = float(num) / float(den) if float(den) else 30.0
            if fps <= 0 or fps > 1000:
                num, den = s.get("r_frame_rate", "30/1").split("/")
                fps = float(num) / float(den)
            out["fps"] = fps
            out["codec"] = s.get("codec_name")
            out["pix_fmt"] = s.get("pix_fmt")
            rot = 0
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(sd["rotation"])
            rot = int(s.get("tags", {}).get("rotate", rot))
            out["rotation"] = rot
            if s.get("duration") and not out["duration"]:
                out["duration"] = float(s["duration"])
            if s.get("nb_frames") and s["nb_frames"].isdigit():
                out["frames"] = int(s["nb_frames"])
        elif s.get("codec_type") == "audio" and not out["has_audio"]:
            out["has_audio"] = True
            out["sample_rate"] = int(s.get("sample_rate", 48000))
            out["channels"] = int(s.get("channels", 2))
    if out["has_video"] and abs(out.get("rotation", 0)) in (90, 270):
        out["width"], out["height"] = out["height"], out["width"]
    return out


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


def is_audio_only(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXT


# ---------------------------------------------------------------- images
_IMG_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def load_image_hi(path: str) -> np.ndarray:
    """Straight-alpha RGBA float32 0..1 keeping 16-bit/float precision (for log/HDR stills)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise MediaError(f"Cannot read image: {path}")
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(img.dtype, 1.0)
    img = img.astype(np.float32) / scale
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    if img.shape[2] == 3:
        img = np.concatenate([img[..., ::-1], np.ones(img.shape[:2] + (1,), np.float32)], -1)
    else:
        img = img[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(img)


def load_image(path: str, max_side: int | None = None) -> np.ndarray:
    """Return straight-alpha RGBA uint8."""
    key = (path, max_side, os.path.getmtime(path) if os.path.exists(path) else 0)
    if key in _IMG_CACHE:
        _IMG_CACHE.move_to_end(key)
        return _IMG_CACHE[key]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MediaError(f"Cannot read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    _IMG_CACHE[key] = img
    while len(_IMG_CACHE) > 64:
        _IMG_CACHE.popitem(last=False)
    return img


# ---------------------------------------------------------------- video
class VideoReader:
    """Frame-accurate decoder over an ffmpeg pipe with forward streaming + small LRU cache.

    Decodes at (out_w, out_h) to keep memory low (proxy scale supported)."""

    def __init__(self, path: str, scale: float = 1.0, depth: int = 8):
        self.path = path
        self.depth = depth
        self.info = probe(path)
        if not self.info["has_video"]:
            raise MediaError(f"No video stream: {path}")
        self.fps = self.info["fps"]
        self.w = max(2, int(round(self.info["width"] * scale)))
        self.h = max(2, int(round(self.info["height"] * scale)))
        self.bpp = 8 if depth > 8 else 4
        self.frame_bytes = self.w * self.h * self.bpp
        self.proc = None
        self.next_idx = 0
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        dur = self.info.get("duration") or 0
        self.n_frames = self.info.get("frames") or max(1, int(math.floor(dur * self.fps + 1e-6)))
        self.last_good = None

    def _open(self, idx: int):
        self.close()
        t = idx / self.fps
        cmd = [FFMPEG, "-v", "error", "-nostdin"]
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-an", "-sn",
                "-vf", f"scale={self.w}:{self.h}:flags=bicubic",
                "-f", "rawvideo", "-pix_fmt", "rgba64le" if self.depth > 8 else "rgba", "-vsync", "passthrough", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.frame_bytes * 2)
        self.next_idx = idx

    def _read_one(self):
        buf = self.proc.stdout.read(self.frame_bytes) if self.proc else b""
        if len(buf) < self.frame_bytes:
            return None
        if self.depth > 8:
            return np.frombuffer(buf, "<u2").reshape(self.h, self.w, 4)
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)

    def frame(self, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), self.n_frames - 1))
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if self.proc is None or idx < self.next_idx or idx - self.next_idx > max(48, int(self.fps * 2)):
            self._open(idx)
        img = None
        while self.next_idx <= idx:
            img = self._read_one()
            if img is None:
                break
            self.cache[self.next_idx] = img
            self.next_idx += 1
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
        if img is None:
            # past the real end: hold the last decodable frame
            if self.last_good is None:
                self.close()
                if idx > 0:
                    self.n_frames = max(1, idx)  # shrink and retry once
                    return self.frame(idx - 1)
                return np.zeros((self.h, self.w, 4), np.uint16 if self.depth > 8 else np.uint8)
            return self.last_good
        self.last_good = img
        return img

    def at(self, t: float) -> np.ndarray:
        return self.frame(int(math.floor(t * self.fps + 1e-4)))

    def close(self):
        if self.proc:
            try:
                self.proc.stdout.close()
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:
                pass
            self.proc = None

    def __del__(self):
        self.close()


# ---------------------------------------------------------------- audio
SR = 48000


def decode_audio(path: str, sr: int = SR, start: float = 0.0, duration: float | None = None, af: str | None = None) -> np.ndarray:
    """Return float32 array (n, 2) at sr. `af` = optional ffmpeg audio filter chain applied after decode."""
    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-i", path, "-vn"]
    if af:
        cmd += ["-af", af]
    cmd += ["-ac", "2", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise MediaError(f"Audio decode failed {path}: {r.stderr.decode(errors='ignore')[-400:]}")
    a = np.frombuffer(r.stdout, np.float32)
    return a.reshape(-1, 2).copy() if a.size else np.zeros((0, 2), np.float32)
