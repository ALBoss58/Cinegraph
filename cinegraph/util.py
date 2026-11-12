"""Shared helpers: colors, fonts, paths, time parsing."""
from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess

import numpy as np

NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00ff00", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff", "gray": "#808080", "grey": "#808080",
    "orange": "#ff8800", "gold": "#d4af37", "crimson": "#b3001b", "navy": "#0b1d3a", "transparent": "#00000000",
    "parchment": "#e8d8b0", "sepia": "#704214", "ivory": "#fffff0",
}


def parse_color(c) -> np.ndarray:
    """Return RGBA float32 in 0..1. Accepts '#rgb', '#rrggbb', '#rrggbbaa', names, [r,g,b(,a)] 0..1 or 0..255."""
    if isinstance(c, np.ndarray):
        a = c.astype(np.float32)
        return a if a.size == 4 else np.append(a, 1.0).astype(np.float32)
    if isinstance(c, str):
        s = NAMED_COLORS.get(c.lower(), c).lstrip("#")
        if len(s) in (3, 4):
            s = "".join(ch * 2 for ch in s)
        if len(s) not in (6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", s):
            raise ValueError(f"Bad color: {c!r}")
        vals = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
        if len(vals) == 3:
            vals.append(1.0)
        return np.array(vals, np.float32)
    if isinstance(c, (list, tuple)):
        v = [float(x) for x in c]
        if any(x > 1.0 for x in v[:3]):
            v = [x / 255.0 for x in v[:3]] + v[3:]
        if len(v) == 3:
            v.append(1.0)
        return np.array(v, np.float32)
    raise ValueError(f"Bad color: {c!r}")


def is_color_string(v) -> bool:
    return isinstance(v, str) and (v.startswith("#") or v.lower() in NAMED_COLORS)


@functools.lru_cache(maxsize=256)
def resolve_font(font: str | None, bold: bool = False, italic: bool = False) -> str:
    """Resolve a font file path from a path or a family name (uses fc-match)."""
    if font and os.path.isfile(font):
        return font
    family = font or "Inter"
    style = []
    if bold:
        style.append("Bold")
    if italic:
        style.append("Italic")
    pattern = family + (":style=" + " ".join(style) if style else "")
    if shutil.which("fc-match"):
        try:
            out = subprocess.run(["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout and os.path.isfile(out.stdout):
                return out.stdout
        except Exception:
            pass
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"]:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Font not found: {font}")


def parse_time(v, fps: float = 30.0) -> float:
    """Seconds from number, 'HH:MM:SS.ms', 'MM:SS', '12f' (frames) or '1.5s'."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.endswith("f"):
        return float(s[:-1]) / fps
    if s.endswith("s"):
        return float(s[:-1])
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(s)


def stable_seed(*parts) -> int:
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def fmt_tc(t: float, fps: float = 30.0) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"
