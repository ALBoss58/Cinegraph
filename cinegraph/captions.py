"""SRT import (burned-in caption layers) and export (sidecar for YouTube)."""
from __future__ import annotations

import re

from .util import fmt_tc

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(path: str):
    cues = []
    with open(path, encoding="utf-8-sig") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    for b in blocks:
        lines = [l for l in b.strip().splitlines() if l.strip()]
        for i, l in enumerate(lines):
            m = _TS.search(l)
            if m:
                g = [int(x) for x in m.groups()]
                t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append((t0, t1, "\n".join(lines[i + 1:])))
                break
    return cues


def srt_to_layers(path: str, style: dict, comp: dict):
    W, H = comp["width"], comp["height"]
    out = []
    for i, (t0, t1, txt) in enumerate(parse_srt(path)):
        L = {"type": "text", "id": f"cap{i}", "text": txt, "start": t0, "duration": max(0.05, t1 - t0),
             "size": style.get("size", 44), "font": style.get("font", "Inter"), "bold": style.get("bold", True),
             "color": style.get("color", "#ffffff"), "max_width": style.get("max_width", W * 0.8),
             "stroke_width": style.get("stroke_width", 0), "stroke_color": style.get("stroke_color", "#000000"),
             "position": [W / 2, style.get("y", H * 0.88)], "caption": True,
             "transition_in": {"type": "fade", "duration": 0.08}, "transition_out": {"type": "fade", "duration": 0.08}}
        if style.get("box", True):
            L["box"] = style.get("box") if isinstance(style.get("box"), dict) else {"color": "#000000a8", "padding": [18, 8], "radius": 6}
        out.append(L)
    return out


def _srt_ts(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(norm: dict, path: str):
    caps = [L for L in norm["comps"]["main"]["layers"] if L.get("type") == "text" and L.get("caption")]
    caps.sort(key=lambda L: L["start"])
    with open(path, "w", encoding="utf-8") as f:
        for i, L in enumerate(caps, 1):
            t0 = L["start"]
            t1 = t0 + L.get("duration", 2)
            f.write(f"{i}\n{_srt_ts(t0)} --> {_srt_ts(t1)}\n{L['text']}\n\n")
    return len(caps)


def chapters(norm: dict) -> str:
    """YouTube chapter list from markers with chapter: true (first must be 0:00)."""
    ms = [m for m in norm.get("markers", []) if m.get("chapter", True)]
    ms.sort(key=lambda m: float(m.get("t", 0)))
    lines = []
    for m in ms:
        t = float(m.get("t", 0))
        mm, ss = divmod(int(t), 60)
        hh, mm = divmod(mm, 60)
        stamp = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
        lines.append(f"{stamp} {m.get('name', '')}")
    if ms and float(ms[0].get("t", 0)) != 0:
        lines.insert(0, "0:00 Intro")
    return "\n".join(lines)


CAPTION_STYLES = {
    "clean":   {"size": 46, "bold": True, "color": "#ffffff", "box": {"color": "#000000a6", "padding": [18, 8], "radius": 8}},
    "outline": {"size": 54, "bold": True, "color": "#ffffff", "stroke_width": 6, "stroke_color": "#000000"},
    "karaoke": {"size": 56, "bold": True, "color": "#ffffff", "stroke_width": 6, "stroke_color": "#000000",
                "highlight": "#ffd400"},
    "pop":     {"size": 60, "bold": True, "color": "#ffffff", "stroke_width": 7, "stroke_color": "#000000", "pop": True},
    "documentary": {"size": 42, "color": "#f2efe6", "font": "Inter", "box": {"color": "#00000080", "padding": [16, 6], "radius": 4}},
}


def cues_to_layers(cues, style, comp, offset=0.0, id_prefix="acap"):
    """Caption cues (from ai.build_captions) -> styled text layers. Style: name or dict (overrides)."""
    if isinstance(style, str):
        st = dict(CAPTION_STYLES.get(style, CAPTION_STYLES["clean"]))
    else:
        st = dict(CAPTION_STYLES.get(style.get("preset", "clean"), CAPTION_STYLES["clean"]))
        st.update({k: v for k, v in style.items() if k != "preset"})
    W, H = comp["width"], comp["height"]
    out = []
    for i, c in enumerate(cues):
        t0, t1 = c["start"] + offset, c["end"] + offset
        L = {"type": "text", "id": f"{id_prefix}{i}", "text": c["text"], "start": t0, "duration": max(0.1, t1 - t0),
             "caption": True, "font": st.get("font", "Inter"), "bold": st.get("bold", True), "size": st["size"],
             "color": st["color"], "max_width": st.get("max_width", W * 0.84), "align": "center",
             "position": [W / 2, st.get("y", H * 0.86)], "line_height": 1.15}
        for k in ("stroke_width", "stroke_color", "box"):
            if k in st:
                L[k] = st[k]
        spans = [[w["start"] - c["start"], w["end"] - c["start"]] for w in c["words"]]
        if st.get("highlight"):
            L["word_highlight"] = {"times": spans, "color": st["highlight"]}
        if st.get("pop"):
            L["animator"] = {"type": "rise", "unit": "word", "times": [s[0] for s in spans], "duration": 0.12,
                             "offset": 14, "ease": "out_back"}
        else:
            L["transition_in"] = {"type": "fade", "duration": 0.06}
            L["transition_out"] = {"type": "fade", "duration": 0.06}
        out.append(L)
    return out


def write_srt(cues, path, offset=0.0):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{_srt_ts(c['start'] + offset)} --> {_srt_ts(c['end'] + offset)}\n{c['text']}\n\n")
    return len(cues)
