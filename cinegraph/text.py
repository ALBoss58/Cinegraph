"""Text rendering with kerning-preserving layout, tracking, wrapping, stroke, boxes and
per-unit (char / word / line) animators (typewriter, fade, rise, drop, scale-free blur-free)."""
from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import easing
from .util import parse_color, resolve_font


@functools.lru_cache(maxsize=128)
def get_font(path: str, px: int):
    return ImageFont.truetype(path, max(1, int(px)))


def _wrap(text: str, font, max_w: float, tracking: float):
    lines = []
    for para in text.split("\n"):
        if not max_w or max_w <= 0:
            lines.append(para)
            continue
        words = para.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w) if cur else w
            if font.getlength(trial) + tracking * len(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def layout(spec: dict, rs: float):
    """Return layout dict with glyph positions in raster pixels (scale rs)."""
    font_path = resolve_font(spec.get("font"), bool(spec.get("bold")), bool(spec.get("italic")))
    size = float(spec.get("size", 72))
    font = get_font(font_path, round(size * rs))
    text = str(spec.get("text", ""))
    if spec.get("all_caps"):
        text = text.upper()
    tracking = float(spec.get("tracking", 0.0)) * size / 1000.0 * rs  # AE-style tracking in 1/1000 em
    max_w = float(spec.get("max_width") or 0) * rs
    lines = _wrap(text, font, max_w, tracking)
    ascent, descent = font.getmetrics()
    lh = float(spec.get("line_height", 1.2)) * size * rs
    widths = [font.getlength(l) + tracking * max(0, len(l) - 1) for l in lines]
    block_w = max(widths + [max_w if spec.get("box_fixed_width") else 0, 1.0])
    block_h = (len(lines) - 1) * lh + ascent + descent
    align = spec.get("align", "center")
    glyphs = []  # (char, x, y_top, line_idx, word_idx)
    word_idx = 0
    for li, (line, lw) in enumerate(zip(lines, widths)):
        if align == "left":
            x0 = 0.0
        elif align == "right":
            x0 = block_w - lw
        else:
            x0 = (block_w - lw) / 2
        y = li * lh
        prev_space = True
        for ci, ch in enumerate(line):
            if ch == " ":
                if not prev_space:
                    word_idx += 1
                prev_space = True
                continue
            prev_space = False
            x = x0 + font.getlength(line[:ci]) + tracking * ci
            glyphs.append((ch, x, y, li, word_idx))
        if not prev_space:
            word_idx += 1
    return {"font": font, "glyphs": glyphs, "w": block_w, "h": block_h, "lines": len(lines),
            "words": word_idx, "ascent": ascent}


def _unit_progress(anim: dict, t: float, n_units: int):
    start = float(anim.get("start", 0.0))
    stagger = float(anim.get("stagger", 0.05))
    dur = float(anim.get("duration", 0.4))
    ease = easing.get(anim.get("ease", "out_cubic"))
    mode = anim.get("mode", "in")
    order = anim.get("order", "forward")
    out = []
    for i in range(n_units):
        k = i if order == "forward" else (n_units - 1 - i if order == "reverse" else
                                         abs(i - (n_units - 1) / 2.0))
        times = anim.get("times")
        t0 = float(times[i]) if times is not None and i < len(times) else start + k * stagger
        if dur <= 0:
            u = 1.0 if t >= t0 else 0.0
        else:
            u = min(1.0, max(0.0, (t - t0) / dur))
        u = ease(u)
        out.append(1 - u if mode == "out" else u)
    return out


def render(spec: dict, t: float, rs: float):
    """Render text. Returns (rgba premultiplied float32, ox, oy, natural_w, natural_h) where pixel (0,0)
    corresponds to layer coordinate (ox, oy) and natural size is in layer units."""
    L = layout(spec, rs)
    font = L["font"]
    size = float(spec.get("size", 72))
    sw = float(spec.get("stroke_width", 0) or 0) * rs
    box = spec.get("box")
    pad = 0.0
    if box:
        pv = box.get("padding", [24, 12])
        pv = pv if isinstance(pv, (list, tuple)) else [pv, pv]
        pad_x, pad_y = float(pv[0]) * rs, float(pv[1]) * rs
    else:
        pad_x = pad_y = 0.0
    anim = spec.get("animator")
    rise = float(anim.get("offset", 0)) * rs if anim else 0.0
    margin = math.ceil(sw + 4 + abs(rise))
    mx, my = int(math.ceil(margin + pad_x)), int(math.ceil(margin + pad_y))
    W = int(math.ceil(L["w"])) + 2 * mx
    H = int(math.ceil(L["h"])) + 2 * my
    glyphs = L["glyphs"]

    # per-glyph alpha / dy from animator
    alphas = np.ones(len(glyphs), np.float32)
    dys = np.zeros(len(glyphs), np.float32)
    if anim:
        unit = anim.get("unit", "char")
        idx = [g[4] if unit == "word" else g[3] if unit == "line" else i for i, g in enumerate(glyphs)]
        n = (max(idx) + 1) if idx else 0
        prog = _unit_progress(anim, t, n)
        typ = anim.get("type", "fade")
        for gi, ui in enumerate(idx):
            u = prog[ui]
            if typ == "typewriter":
                alphas[gi] = 1.0 if u > 0 else 0.0
            else:
                alphas[gi] = u
                if typ in ("rise", "slide_up"):
                    dys[gi] = (1 - u) * (rise or size * rs * 0.4)
                elif typ == "drop":
                    dys[gi] = -(1 - u) * (rise or size * rs * 0.4)

    # karaoke-style word highlight: glyphs of the word being spoken switch colour
    hl = spec.get("word_highlight")
    hl_on = np.zeros(len(glyphs), bool)
    if hl:
        spans = hl.get("times", [])
        for gi, g in enumerate(glyphs):
            wi = g[4]
            if wi < len(spans) and spans[wi][0] <= t < spans[wi][1] + float(hl.get("hold", 0.0)):
                hl_on[gi] = True
            elif hl.get("mode") == "fill" and wi < len(spans) and t >= spans[wi][0]:
                hl_on[gi] = True
    fill_m = np.zeros((H, W), np.float32)
    hl_m = np.zeros((H, W), np.float32)
    stroke_m = np.zeros((H, W), np.float32)
    # group glyphs by identical (alpha, dy, highlight) to minimize draws
    groups: dict = {}
    for gi, g in enumerate(glyphs):
        if alphas[gi] <= 0.001:
            continue
        groups.setdefault((round(float(alphas[gi]), 3), round(float(dys[gi]), 2), bool(hl_on[gi])), []).append(g)
    for (a, dy, is_hl), gl in groups.items():
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        ims = Image.new("L", (W, H), 0) if sw > 0 else None
        drs = ImageDraw.Draw(ims) if ims is not None else None
        for ch, x, y, _, _ in gl:
            pos = (mx + x, my + y + dy)
            dr.text(pos, ch, font=font, fill=255)
            if drs is not None:
                drs.text(pos, ch, font=font, fill=255, stroke_width=int(round(sw)), stroke_fill=255)
        gm = np.asarray(im, np.float32) / 255.0 * a
        if is_hl:
            hl_m = np.maximum(hl_m, gm)
        else:
            fill_m = np.maximum(fill_m, gm)
        if ims is not None:
            stroke_m = np.maximum(stroke_m, np.asarray(ims, np.float32) / 255.0 * a)

    out = np.zeros((H, W, 4), np.float32)
    if box:
        bc = parse_color(box.get("color", "#000000b0"))
        bo = float(box.get("opacity", 1.0))
        rad = int(float(box.get("radius", 8)) * rs)
        bim = Image.new("L", (W * 2, H * 2), 0)
        ImageDraw.Draw(bim).rounded_rectangle(
            [2 * (mx - pad_x), 2 * (my - pad_y), 2 * (mx + L["w"] + pad_x), 2 * (my + L["h"] + pad_y)],
            radius=2 * rad, fill=255)
        bm = np.asarray(bim.resize((W, H), Image.LANCZOS), np.float32) / 255.0 * bc[3] * bo
        if anim and spec.get("box_follows_text", True) and len(alphas):
            bm = bm * float(alphas.max())
        out[..., :3] = bc[:3] * bm[..., None]
        out[..., 3] = bm
    if sw > 0:
        sc = parse_color(spec.get("stroke_color", "#000000"))
        sa = stroke_m[..., None] * sc[3]
        out = out * (1 - sa) + np.concatenate([sc[:3] * sa, sa], -1)
    fc = parse_color(spec.get("color", "#ffffff"))
    grad = spec.get("gradient")
    fa = fill_m[..., None] * fc[3]
    if grad:
        c0, c1 = parse_color(grad[0])[:3], parse_color(grad[1])[:3]
        v = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        col = c0 + (c1 - c0) * v
        out = out * (1 - fa) + np.concatenate([col * fa, fa], -1)
    else:
        out = out * (1 - fa) + np.concatenate([fc[:3] * fa, fa], -1)
    if hl is not None and hl_m.any():
        hc = parse_color(hl.get("color", "#ffd400"))
        ha = hl_m[..., None] * hc[3]
        out = out * (1 - ha) + np.concatenate([hc[:3] * ha, ha], -1)
    return out.astype(np.float32), -mx / rs, -my / rs, L["w"] / rs, L["h"] / rs


def render_path(spec: dict, t: float, rs: float):
    """Text along a path. spec["path"] = {"points": [[x,y]...] (layer coords), "smooth": bool, "offset": % along,
    "align": "start"|"center"|"end", "baseline": px (+ = away from path, toward the left normal), "flip": bool}."""
    import cv2
    from .shapes import _catmull_sample
    P = spec["path"]
    pts = np.asarray(P["points"], np.float32).reshape(-1, 2)
    if P.get("smooth", True) and len(pts) >= 3:
        pts = _catmull_sample(pts, bool(P.get("closed")), 24)
    if P.get("flip"):
        pts = pts[::-1].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = float(cum[-1]) or 1.0
    font_path = resolve_font(spec.get("font"), bool(spec.get("bold")), bool(spec.get("italic")))
    size = float(spec.get("size", 72))
    font = get_font(font_path, round(size * rs))
    text = str(spec.get("text", ""))
    if spec.get("all_caps"):
        text = text.upper()
    tracking = float(spec.get("tracking", 0.0)) * size / 1000.0
    adv = [font.getlength(text[:i]) / rs + tracking * i for i in range(len(text) + 1)]
    width = adv[-1]
    align = P.get("align", "start")
    base = float(P.get("offset", 0)) / 100.0 * total
    if align == "center":
        base += (total - width) / 2
    elif align == "end":
        base += total - width
    ascent, descent = font.getmetrics()
    sw = float(spec.get("stroke_width", 0) or 0)
    m = size * 1.3 + sw + 4
    x0, y0 = pts.min(0) - m
    x1, y1 = pts.max(0) + m
    W, H = int(np.ceil((x1 - x0) * rs)), int(np.ceil((y1 - y0) * rs))
    fill = np.zeros((H, W), np.float32)
    strk = np.zeros((H, W), np.float32)
    anim = spec.get("animator")
    alphas = _unit_progress(anim, t, len(text)) if anim else [1.0] * len(text)
    bl = float(P.get("baseline", 0))
    pad = int(size * rs * 0.6 + sw * rs + 4)
    for i, ch in enumerate(text):
        if ch == " " or alphas[i] <= 0.001:
            continue
        cw = adv[i + 1] - adv[i]
        s = base + adv[i] + cw / 2
        if s < 0 or s > total:
            continue
        px = np.interp(s, cum, pts[:, 0])
        py = np.interp(s, cum, pts[:, 1])
        k = min(len(pts) - 2, max(0, int(np.searchsorted(cum, s) - 1)))
        d = pts[k + 1] - pts[k]
        ang = np.degrees(np.arctan2(d[1], d[0]))
        nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)  # left normal (screen-up for left-to-right)
        gw = int(font.getlength(ch)) + 2 * pad
        gh = ascent + descent + 2 * pad
        im = Image.new("L", (gw, gh), 0)
        ImageDraw.Draw(im).text((pad, pad), ch, font=font, fill=255)
        glyph = np.asarray(im, np.float32) / 255.0
        gs = None
        if sw > 0:
            ims = Image.new("L", (gw, gh), 0)
            ImageDraw.Draw(ims).text((pad, pad), ch, font=font, fill=255, stroke_width=int(round(sw * rs)), stroke_fill=255)
            gs = np.asarray(ims, np.float32) / 255.0
        # glyph pivot: horizontal centre on the baseline
        pivot = (pad + font.getlength(ch) / 2, pad + ascent)
        tx = (px - x0 + nrm[0] * bl) * rs
        ty = (py - y0 + nrm[1] * bl) * rs
        M = cv2.getRotationMatrix2D(pivot, -ang, 1.0)
        M[0, 2] += tx - pivot[0]
        M[1, 2] += ty - pivot[1]
        a = float(alphas[i])
        fill = np.maximum(fill, cv2.warpAffine(glyph, M, (W, H), flags=cv2.INTER_LINEAR) * a)
        if gs is not None:
            strk = np.maximum(strk, cv2.warpAffine(gs, M, (W, H), flags=cv2.INTER_LINEAR) * a)
    out = np.zeros((H, W, 4), np.float32)
    if sw > 0:
        sc = parse_color(spec.get("stroke_color", "#000000"))
        sa = strk[..., None] * sc[3]
        out = np.concatenate([sc[:3] * sa, sa], -1)
    fc = parse_color(spec.get("color", "#ffffff"))
    fa = fill[..., None] * fc[3]
    out = out * (1 - fa) + np.concatenate([fc[:3] * fa, fa], -1)
    return out.astype(np.float32), float(x0), float(y0), float(x1 - x0), float(y1 - y0)
