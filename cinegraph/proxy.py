"""Proxy media: 540p all-intra H.264 copies for instant scrubbing / fast drafts. Originals are always used
for final renders unless explicitly told otherwise."""
from __future__ import annotations

import hashlib
import os
import subprocess

from . import media


def proxy_path(src, base_dir, height=540):
    st = os.stat(src)
    h = hashlib.md5(f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}|{height}".encode()).hexdigest()[:16]
    return os.path.join(base_dir, ".cinegraph_cache", "proxy", f"{h}.mp4")


def make_proxy(src, base_dir, height=540):
    out = proxy_path(src, base_dir, height)
    if os.path.exists(out):
        return out, False
    os.makedirs(os.path.dirname(out), exist_ok=True)
    info = media.probe(src)
    h = min(height, info["height"]) // 2 * 2
    tmp = out + ".part.mp4"
    r = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-an", "-vf", f"scale=-2:{h}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-g", "1", "-pix_fmt", "yuv420p", tmp],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"proxy failed for {src}: {r.stderr[-300:]}")
    os.replace(tmp, out)
    return out, True


def make_all(norm, height=540):
    made = []
    seen = set()
    for comp in norm["comps"].values():
        for L in comp["layers"]:
            if L.get("type") == "video" and L.get("src") and L["src"] not in seen and not media.is_image(L["src"]):
                seen.add(L["src"])
                p, new = make_proxy(L["src"], norm["base_dir"], height)
                made.append((L["src"], p, new))
    return made
