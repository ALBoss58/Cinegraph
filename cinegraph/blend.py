"""Blend modes (W3C compositing model) on premultiplied float32 RGBA."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _soft_light(cs, cd):
    d = np.where(cd <= 0.25, ((16 * cd - 12) * cd + 4) * cd, np.sqrt(np.maximum(cd, 0)))
    return np.where(cs <= 0.5, cd - (1 - 2 * cs) * cd * (1 - cd), cd + (2 * cs - 1) * (d - cd))


def _hard_light(cs, cd):
    return np.where(cs <= 0.5, cd * 2 * cs, 1 - (1 - cd) * (1 - (2 * cs - 1)))


SEPARABLE = {
    "multiply": lambda cs, cd: cs * cd,
    "screen": lambda cs, cd: cs + cd - cs * cd,
    "overlay": lambda cs, cd: _hard_light(cd, cs),
    "darken": np.minimum,
    "lighten": np.maximum,
    "color_dodge": lambda cs, cd: np.where(cd <= 0, 0, np.where(cs >= 1, 1, np.minimum(1, cd / np.maximum(1 - cs, EPS)))),
    "color_burn": lambda cs, cd: np.where(cd >= 1, 1, np.where(cs <= 0, 0, 1 - np.minimum(1, (1 - cd) / np.maximum(cs, EPS)))),
    "hard_light": _hard_light,
    "soft_light": _soft_light,
    "difference": lambda cs, cd: np.abs(cs - cd),
    "exclusion": lambda cs, cd: cs + cd - 2 * cs * cd,
    "linear_light": lambda cs, cd: np.clip(cd + 2 * cs - 1, 0, 1),
}

MODES = ["normal", "add", "subtract"] + list(SEPARABLE) + ["luminosity", "color"]


def _lum(c):
    return c[..., 0:1] * 0.2126 + c[..., 1:2] * 0.7152 + c[..., 2:3] * 0.0722


def _set_lum(c, l):
    d = l - _lum(c)
    c = c + d
    L = _lum(c)
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    c = np.where(n < 0, L + (c - L) * L / np.maximum(L - n, EPS), c)
    c = np.where(x > 1, L + (c - L) * (1 - L) / np.maximum(x - L, EPS), c)
    return c


def composite(dst: np.ndarray, src: np.ndarray, x0: int, y0: int, mode: str = "normal", opacity: float = 1.0):
    """Blend premultiplied `src` patch onto `dst` in place at (x0, y0). Handles clipping."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return dst
    s = src[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    if opacity < 1.0:
        s = s * opacity
    d = dst[ay0:ay1, ax0:ax1]
    sa = s[..., 3:4]
    if mode in ("normal", None):
        d *= (1.0 - sa)
        d += s
        return dst
    if mode == "add":
        d[..., :3] = np.minimum(1.0, d[..., :3] + s[..., :3])
        d[..., 3:4] = np.minimum(1.0, d[..., 3:4] + sa)
        return dst
    if mode == "subtract":
        d[..., :3] = np.maximum(0.0, d[..., :3] - s[..., :3])
        return dst
    da = d[..., 3:4]
    cs = np.where(sa > EPS, s[..., :3] / np.maximum(sa, EPS), 0)
    cd = np.where(da > EPS, d[..., :3] / np.maximum(da, EPS), 0)
    if mode in SEPARABLE:
        B = SEPARABLE[mode](cs, cd)
    elif mode == "luminosity":
        B = _set_lum(cd, _lum(cs))
    elif mode == "color":
        B = _set_lum(cs, _lum(cd))
    else:
        raise ValueError(f"Unknown blend mode '{mode}'. Known: {', '.join(MODES)}")
    rgb = (1 - da) * s[..., :3] + (1 - sa) * d[..., :3] + sa * da * np.clip(B, 0, 1)
    d[..., :3] = rgb
    d[..., 3:4] = sa + da * (1 - sa)
    return dst
