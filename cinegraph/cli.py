"""Command line: cinegraph <command> ...

  render   project.json -o out.mp4 [--preset final|youtube|draft|hevc|prores|prores4444|webm_alpha|png|gif]
           [--from S --to S] [--scale 0.5] [--workers N] [--no-audio] [--loudness -14]
  still    project.json -t 3.5 -o frame.png [--scale 1] [--alpha]
  contact  project.json -o sheet.png [--every 2 | --times 0,1.5,3] [--cols 4] [--width 480]
  validate project.json [--deep]
  probe    media files...
  srt      project.json -o captions.srt      (export caption layers)
  chapters project.json                      (YouTube chapter list from markers)
  serve    project.json [--port 8765]        (live preview UI: timeline, scrubber, hot reload)
  track    clip.mp4 --point x,y | --quad x1,y1,...,x4,y4 [--from S --to S] [-o track.json]
  stabilize clip.mp4 -o out.mp4 [--smoothness 1.0] [--lock]
  slowmo   clip.mp4 -o out.mp4 --speed 0.25 [--mode flow|blend]
  denoise  clip.mp4 -o out.mp4 [--strength 0.5] [--mode temporal|nlm|bilateral] [--audio 0.5]
  transcribe audio_or_video -o subs.srt [--words words.json] [--language en]
  cutout   image.png -o cut.png [--model general|human|fast]
  scopes   project.json -t 3.5 -o scopes.png   (waveform, RGB parade, vectorscope, histogram)
  beats    music.wav [--json]                 (tempo + beat times for cutting on the beat)
  loudness project.json | file.wav           (integrated LUFS, LRA, true peak of the final mix)
  presets                                     (looks, text styles, transition presets)
  export   project.json -o timeline.xml|.edl|.otio   (to Premiere / Resolve / Avid)
  import   timeline.xml|.edl|.otio -o project.json [--fps 25]
  proxy    project.json                       (540p all-intra proxies; used by serve + --proxy drafts)
  info                                        (effects, easings, blend modes, transitions)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cinegraph", description="Code-driven compositor & editor.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("project")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--preset", default="final")
    r.add_argument("--from", dest="t0", type=float)
    r.add_argument("--to", dest="t1", type=float)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--workers", type=int)
    r.add_argument("--no-audio", action="store_true")
    r.add_argument("--loudness", type=float)
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--proxy", action="store_true")
    r.add_argument("--segment-seconds", type=float, default=10.0)
    s = sub.add_parser("still")
    s.add_argument("project")
    s.add_argument("-t", "--time", type=float, default=0.0)
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--alpha", action="store_true")
    c = sub.add_parser("contact")
    c.add_argument("project")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--every", type=float)
    c.add_argument("--times")
    c.add_argument("--cols", type=int, default=4)
    c.add_argument("--width", type=int, default=480)
    v = sub.add_parser("validate")
    v.add_argument("project")
    v.add_argument("--deep", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("files", nargs="+")
    sr = sub.add_parser("srt")
    sr.add_argument("project")
    sr.add_argument("-o", "--out", required=True)
    ch = sub.add_parser("chapters")
    ch.add_argument("project")
    sv = sub.add_parser("serve")
    sv.add_argument("project")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--host", default="127.0.0.1")
    tk = sub.add_parser("track")
    tk.add_argument("clip")
    tk.add_argument("--point")
    tk.add_argument("--quad")
    tk.add_argument("--from", dest="t0", type=float, default=0.0)
    tk.add_argument("--to", dest="t1", type=float)
    tk.add_argument("-o", "--out")
    stb = sub.add_parser("stabilize")
    stb.add_argument("clip")
    stb.add_argument("-o", "--out", required=True)
    stb.add_argument("--smoothness", type=float, default=1.0)
    stb.add_argument("--lock", action="store_true")
    sm = sub.add_parser("slowmo")
    sm.add_argument("clip")
    sm.add_argument("-o", "--out", required=True)
    sm.add_argument("--speed", type=float, default=0.25)
    sm.add_argument("--mode", default="flow")
    dn = sub.add_parser("denoise")
    dn.add_argument("clip")
    dn.add_argument("-o", "--out", required=True)
    dn.add_argument("--strength", type=float, default=0.5)
    dn.add_argument("--mode", default="temporal")
    dn.add_argument("--audio", type=float)
    tr = sub.add_parser("transcribe")
    tr.add_argument("media")
    tr.add_argument("-o", "--out", required=True)
    tr.add_argument("--words")
    tr.add_argument("--language", default="en")
    tr.add_argument("--max-chars", type=int, default=32)
    cu = sub.add_parser("cutout")
    cu.add_argument("image")
    cu.add_argument("-o", "--out", required=True)
    cu.add_argument("--model", default="general")
    sc = sub.add_parser("scopes")
    sc.add_argument("project")
    sc.add_argument("-t", "--time", type=float, default=0.0)
    sc.add_argument("-o", "--out", required=True)
    bt = sub.add_parser("beats")
    bt.add_argument("audio")
    bt.add_argument("--json", action="store_true")
    ld = sub.add_parser("loudness")
    ld.add_argument("target")
    sub.add_parser("presets")
    ex = sub.add_parser("export")
    ex.add_argument("project")
    ex.add_argument("-o", "--out", required=True)
    im = sub.add_parser("import")
    im.add_argument("timeline")
    im.add_argument("-o", "--out", required=True)
    im.add_argument("--fps", type=float)
    px = sub.add_parser("proxy")
    px.add_argument("project")
    px.add_argument("--height", type=int, default=540)
    sub.add_parser("info")
    a = ap.parse_args(argv)

    from . import project as PJ
    if a.cmd == "render":
        from .render import render
        st = render(a.project, a.out, a.t0, a.t1, a.scale, a.workers, a.preset, not a.no_audio, a.loudness, a.quiet,
                    resume=not a.no_resume, proxies=a.proxy, segment_seconds=a.segment_seconds)
        print(json.dumps(st, indent=2))
    elif a.cmd == "still":
        from .render import still
        print(still(a.project, a.time, a.out, a.scale, a.alpha))
    elif a.cmd == "contact":
        from .render import contact_sheet
        times = [float(x) for x in a.times.split(",")] if a.times else None
        print(contact_sheet(a.project, a.out, times, a.every, a.cols, a.width))
    elif a.cmd == "validate":
        from .validate import validate
        res = validate(PJ.load(a.project), deep=a.deep)
        for e in res["errors"]:
            print("ERROR  ", e)
        for w in res["warnings"]:
            print("WARN   ", w)
        print("OK" if res["ok"] else "FAILED")
        sys.exit(0 if res["ok"] else 1)
    elif a.cmd == "probe":
        from .media import probe
        for f in a.files:
            print(json.dumps(probe(f), indent=2))
    elif a.cmd == "srt":
        from .captions import export_srt
        print(export_srt(PJ.load(a.project), a.out), "cues written")
    elif a.cmd == "chapters":
        from .captions import chapters
        print(chapters(PJ.load(a.project)))
    elif a.cmd == "serve":
        from .server import serve
        serve(a.project, a.port, a.host)
    elif a.cmd == "track":
        from . import motion as MO
        if a.quad:
            v = [float(x) for x in a.quad.split(",")]
            res = MO.track_planar(a.clip, [v[i:i + 2] for i in range(0, 8, 2)], a.t0, a.t1, os.path.dirname(os.path.abspath(a.clip)))
        else:
            res = MO.track_points(a.clip, [[float(x) for x in a.point.split(",")]], a.t0, a.t1, os.path.dirname(os.path.abspath(a.clip)))
        txt = json.dumps(res)
        if a.out:
            open(a.out, "w").write(txt)
            print(f"{len(res['frames'])} frames tracked -> {a.out}")
        else:
            print(txt)
    elif a.cmd in ("stabilize", "slowmo", "denoise"):
        from .render import render
        L = {"type": "video", "src": os.path.abspath(a.clip), "fit": "cover"}
        info = __import__("cinegraph.media", fromlist=["probe"]).probe(L["src"])
        dur = info["duration"]
        if a.cmd == "stabilize":
            L["stabilize"] = {"smoothness": a.smoothness, "mode": "lock" if a.lock else "smooth"}
        elif a.cmd == "slowmo":
            L["speed"] = a.speed
            L["frame_blend"] = a.mode
            dur = dur / a.speed
        else:
            L["denoise"] = {"mode": a.mode, "strength": a.strength}
            if a.audio is not None:
                L["audio_denoise"] = a.audio
        W, H = info["width"] // 2 * 2, info["height"] // 2 * 2
        proj = {"comp": {"width": W, "height": H, "fps": info["fps"], "duration": dur}, "layers": [L],
                "audio_settings": {"loudness": None, "duck": False}}
        print(json.dumps(render(PJ.load(proj, os.getcwd()), a.out, loudness=False), indent=2))
    elif a.cmd == "transcribe":
        from . import ai
        from .captions import write_srt
        words = ai.transcribe(a.media, a.language)
        if a.words:
            json.dump(words, open(a.words, "w"), indent=1)
        n = write_srt(ai.build_captions(words, a.max_chars), a.out)
        print(f"{len(words)} words, {n} cues -> {a.out}")
    elif a.cmd == "cutout":
        import cv2
        from . import ai, media
        img = media.load_image(a.image)
        m = ai.segment(img, a.model)
        out = img.copy()
        out[..., 3] = (m * 255 + 0.5).astype("uint8")
        cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
        print(a.out)
    elif a.cmd == "scopes":
        import cv2
        from .color import scopes
        from .compositor import ProjectRuntime
        from .render import to_uint8
        rt = ProjectRuntime(PJ.load(a.project), 0.5)
        img = scopes(to_uint8(rt.main.render(a.time), False, dither=False))
        cv2.imwrite(a.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(a.out)
    elif a.cmd == "beats":
        from .audio import beats
        res = beats(a.audio)
        if a.json:
            print(json.dumps(res))
        else:
            print(f"{res['bpm']} BPM, {len(res['beats'])} beats")
            print(", ".join(f"{b:.3f}" for b in res["beats"]))
    elif a.cmd == "loudness":
        import tempfile
        from . import audio as AU
        if a.target.endswith(".json"):
            norm = PJ.load(a.target)
            mixed, _ = AU.mix(norm, 0, norm["comps"]["main"]["duration"])
            tmp = tempfile.mktemp(suffix=".wav")
            AU.write_wav(tmp, mixed)
            path = tmp
        else:
            path = a.target
        print(json.dumps(AU.loudness_report(path), indent=2))
    elif a.cmd == "presets":
        from . import presets as PR
        print("LOOKS:", ", ".join(sorted(PR.LOOKS)))
        print("TEXT STYLES:", ", ".join(sorted(PR.TEXT_STYLES)))
        print("TRANSITIONS:", ", ".join(sorted(PR.TRANSITIONS)))
        from .audio import VOICE_PRESETS
        print("VOICE PRESETS:", ", ".join(sorted(VOICE_PRESETS)))
        from .captions import CAPTION_STYLES
        print("CAPTION STYLES:", ", ".join(sorted(CAPTION_STYLES)))
    elif a.cmd == "export":
        from .interchange import export
        print(export(PJ.load(a.project), a.out))
    elif a.cmd == "import":
        from .interchange import import_timeline
        d = import_timeline(a.timeline, a.fps)
        json.dump(d, open(a.out, "w"), indent=2)
        print(f"{len(d['layers'])} clips, {len(d['audio'])} audio -> {a.out}")
    elif a.cmd == "proxy":
        from .proxy import make_all
        for src, p, new in make_all(PJ.load(a.project), a.height):
            print(("made   " if new else "exists ") + os.path.basename(src) + " -> " + p)
    elif a.cmd == "info":
        from . import blend, easing, effects
        from .project import LAYER_TYPES, OVERLAP_TRANSITIONS
        print("LAYER TYPES:", ", ".join(sorted(LAYER_TYPES)))
        print("\nEFFECTS:")
        for k in sorted(effects.REGISTRY):
            d = effects.REGISTRY[k]["defaults"]
            print(f"  {k}: " + ", ".join(f"{p}={v}" for p, v in d.items()))
        print("\nEFFECT ALIASES:", ", ".join(f"{a}->{b}" for a, b in sorted(effects.ALIASES.items())))
        print("\nBLEND MODES:", ", ".join(blend.MODES))
        print("\nEASINGS:", ", ".join(easing.names()), "+ bezier(x1,y1,x2,y2), steps(n)")
        print("\nTRANSITIONS:", ", ".join(sorted(OVERLAP_TRANSITIONS | {"dip_black", "dip_white"})))


if __name__ == "__main__":
    main()
