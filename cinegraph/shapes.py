"""Vector shapes: rect, ellipse, polygon, star, path/line/arrow with fill, stroke, dashes and
trim paths (animated line drawing for maps, routes and diagrams)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .util import parse_color


def _has(s, k):
    v = s.get(k)
    if v is None or v is False:
        return False
    if isinstance(v, str):
        return v.lower() not in ("", "none", "transparent")
    return True


def _catmull_sample(pts: np.ndarray, closed: bool, per_seg: int = 16) -> np.ndarray:
    n = len(pts)
    if n < 3:
        return pts
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (closed or i > 0) else pts[0] - (pts[1] - pts[0])
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (closed or i + 2 < n) else pts[-1] + (pts[-1] - pts[-2])
        for k in range(per_seg):
            u = k / per_seg
            u2, u3 = u * u, u * u * u
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    if not closed:
        out.append(pts[-1])
    return np.array(out, np.float32)


def geometry(s: dict):
    """Return (points Nx2 float32, closed) in layer coordinates for an evaluated shape dict."""
    kind = s.get("kind", "path")
    if kind == "rect":
        cx, cy = s.get("center", [0, 0])
        w, h = s.get("size", [100, 100])
        r = min(float(s.get("radius", 0)), w / 2, h / 2)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if r <= 0:
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), True
        pts = []
        for (ccx, ccy, a0) in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
            for k in range(9):
                a = math.radians(a0 + k * 90 / 8)
                pts.append([ccx + r * math.cos(a), ccy + r * math.sin(a)])
        return np.array(pts, np.float32), True
    if kind == "ellipse":
        cx, cy = s.get("center", [0, 0])
        w, h = s.get("size", [100, 100])
        n = 160
        a = np.linspace(0, 2 * math.pi, n, endpoint=False) - math.pi / 2
        return np.stack([cx + w / 2 * np.cos(a), cy + h / 2 * np.sin(a)], -1).astype(np.float32), True
    if kind == "star":
        cx, cy = s.get("center", [0, 0])
        n = int(s.get("points", 5))
        ro, ri = float(s.get("outer", 100)), float(s.get("inner", 45))
        rot = math.radians(float(s.get("rotation", 0))) - math.pi / 2
        pts = []
        for i in range(2 * n):
            r = ro if i % 2 == 0 else ri
            a = rot + i * math.pi / n
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        return np.array(pts, np.float32), True
    if kind == "polygon" and "sides" in s:
        cx, cy = s.get("center", [0, 0])
        n = int(s["sides"])
        r = float(s.get("radius", 100))
        a = np.linspace(0, 2 * math.pi, n, endpoint=False) - math.pi / 2 + math.radians(float(s.get("rotation", 0)))
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], -1).astype(np.float32), True
    pts = np.asarray(s.get("points", [[0, 0], [100, 0]]), np.float32).reshape(-1, 2)
    closed = bool(s.get("closed", kind == "polygon"))
    if s.get("smooth") and len(pts) >= 3:
        pts = _catmull_sample(pts, closed)
    return pts, closed


def _trim(pts: np.ndarray, closed: bool, start: float, end: float, offset: float):
    """start/end in percent [0..100], offset in percent. Returns list of polylines."""
    if closed:
        pts = np.vstack([pts, pts[:1]])
    if start <= 0.0001 and end >= 99.9999 and abs(offset) < 1e-6:
        return [pts]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return []
    a, b = (min(start, end) + offset) / 100.0, (max(start, end) + offset) / 100.0
    if b - a <= 1e-6:
        return []

    def sub(a, b):
        da, db = a * total, b * total
        xs = []
        xs.append(np.array([np.interp(da, cum, pts[:, 0]), np.interp(da, cum, pts[:, 1])]))
        inner = (cum > da) & (cum < db)
        xs.extend(pts[inner])
        xs.append(np.array([np.interp(db, cum, pts[:, 0]), np.interp(db, cum, pts[:, 1])]))
        return np.array(xs, np.float32)

    if closed or abs(offset) > 1e-6:
        a0 = a % 1.0
        b0 = a0 + (b - a)
        if b0 <= 1.0:
            return [sub(a0, b0)]
        return [sub(a0, 1.0), sub(0.0, b0 - 1.0)]
    return [sub(max(0, a), min(1, b))]


def _dash(poly: np.ndarray, on: float, off: float):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    out = []
    d = 0.0
    while d < total:
        e = min(total, d + on)
        xs = [[np.interp(d, cum, poly[:, 0]), np.interp(d, cum, poly[:, 1])]]
        mask = (cum > d) & (cum < e)
        xs.extend(poly[mask].tolist())
        xs.append([np.interp(e, cum, poly[:, 0]), np.interp(e, cum, poly[:, 1])])
        out.append(np.array(xs, np.float32))
        d = e + off
    return out


def _arrow_head(poly, size):
    if len(poly) < 2:
        return None
    p1 = poly[-1]
    # walk back to find a point at a usable distance for direction
    for k in range(len(poly) - 2, -1, -1):
        if np.linalg.norm(p1 - poly[k]) > 1e-3:
            p0 = poly[k]
            break
    else:
        return None
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-d[1], d[0]])
    tip = p1 + d * size * 0.35
    base = p1 - d * size * 0.65
    return np.array([tip, base + nrm * size * 0.55, base - nrm * size * 0.55], np.float32)


SHIFT = 4
SC = 1 << SHIFT


def _to_fixed(p, ox, oy, rs):
    return np.round(((p - [ox, oy]) * rs) * SC).astype(np.int32)


def render(shapes: list, rs: float):
    """shapes: evaluated shape dicts. Returns (rgba premul, ox, oy) with pixel(0,0) = layer (ox, oy)."""
    items = []
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    for s in shapes:
        pts, closed = geometry(s)
        sw = float(s.get("stroke_width", 0) or 0) if _has(s, "stroke") else 0.0
        trim = (float(s.get("trim_start", 0)), float(s.get("trim_end", 100)), float(s.get("trim_offset", 0)))
        polys = _trim(pts, closed, *trim)
        head = None
        if s.get("arrow") and polys:
            head = _arrow_head(polys[-1], float(s.get("arrow_size", max(14.0, sw * 3.5))))
        items.append((s, pts, closed, polys, sw, head))
        allp = [pts] + ([head] if head is not None else [])
        allp = np.vstack(allp)
        m = sw / 2 + 3 + (float(s.get("arrow_size", sw * 3.5)) if s.get("arrow") else 0)
        xs_min.append(allp[:, 0].min() - m)
        ys_min.append(allp[:, 1].min() - m)
        xs_max.append(allp[:, 0].max() + m)
        ys_max.append(allp[:, 1].max() + m)
    if not items:
        return np.zeros((2, 2, 4), np.float32), 0.0, 0.0
    ox, oy = math.floor(min(xs_min)), math.floor(min(ys_min))
    W = max(2, int(math.ceil((max(xs_max) - ox) * rs)))
    H = max(2, int(math.ceil((max(ys_max) - oy) * rs)))
    out = np.zeros((H, W, 4), np.float32)
    for s, pts, closed, polys, sw, head in items:
        op = float(s.get("opacity", 1.0))
        trimmed = not (len(polys) == 1 and len(polys[0]) == len(pts) + (1 if closed else 0))
        if _has(s, "fill") and len(pts) >= 3:
            fc = parse_color(s["fill"])
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [_to_fixed(pts, ox, oy, rs)], 255, cv2.LINE_AA, SHIFT)
            fa = m.astype(np.float32)[..., None] / 255 * fc[3] * op * float(s.get("fill_opacity", 1.0))
            out = out * (1 - fa) + np.concatenate([fc[:3] * fa, fa], -1)
        if _has(s, "stroke") and sw > 0:
            sc = parse_color(s["stroke"])
            m = np.zeros((H, W), np.uint8)
            th = max(1, int(round(sw * rs)))
            dash = s.get("dash")
            for poly in polys:
                if len(poly) < 2:
                    continue
                segs = _dash(poly, float(dash[0]), float(dash[1])) if dash else [poly]
                for sg in segs:
                    is_closed = closed and not trimmed and not dash
                    cv2.polylines(m, [_to_fixed(sg, ox, oy, rs)], is_closed, 255, th, cv2.LINE_AA, SHIFT)
            if head is not None:
                cv2.fillPoly(m, [_to_fixed(head, ox, oy, rs)], 255, cv2.LINE_AA, SHIFT)
            sa = m.astype(np.float32)[..., None] / 255 * sc[3] * op
            out = out * (1 - sa) + np.concatenate([sc[:3] * sa, sa], -1)
    return out, float(ox), float(oy)


def bounds(shapes: list):
    xs, ys = [], []
    for s in shapes:
        pts, _ = geometry(s)
        xs += [pts[:, 0].min(), pts[:, 0].max()]
        ys += [pts[:, 1].min(), pts[:, 1].max()]
    if not xs:
        return 0, 0, 0, 0
    return min(xs), min(ys), max(xs), max(ys)
