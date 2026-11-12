"""Effects library. All effects take/return premultiplied float32 RGBA (H, W, 4).

Each effect is registered with defaults; any parameter may be animated (Prop syntax).
Pixel-size parameters are in layer units and are scaled by ctx.unit (raster/proxy scale).
"""
from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .util import parse_color, stable_seed


@dataclass
class FxCtx:
    t: float = 0.0          # layer-local time
    comp_t: float = 0.0
    fps: float = 30.0
    unit: float = 1.0       # pixels per layer unit
    seed: int = 0
    comp_w: int = 1920
    comp_h: int = 1080
    base_dir: str = "."


REGISTRY: dict[str, dict] = {}


def effect(name, pad=None, color_params=(), **defaults):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "defaults": defaults, "pad": pad, "color_params": set(color_params)}
        return fn
    return deco


ALIASES = {"blur": "gaussian_blur", "bloom": "glow", "radial_blur": "zoom_blur", "grade": "color",
           "color_correct": "color", "lumetri": "color", "outline": "stroke", "shadow": "drop_shadow",
           "film": "old_film", "anime": "cartoon", "keylight": "chroma_key", "greenscreen": "chroma_key",
           "brightness_contrast": "color", "hue_saturation": "color", "exposure": "color"}


def resolve_name(name: str) -> str:
    n = name.lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(n, n)


# ---------------------------------------------------------------- helpers
def unpremul(img):
    a = img[..., 3:4]
    if float(a.min()) > 0.9995:  # opaque fast path (full frames, adjustment layers)
        return img[..., :3].copy(), a
    inv = np.where(a > 1e-6, 1.0 / np.maximum(a, 1e-6), 0.0).astype(np.float32)
    return img[..., :3] * inv, a


def premul(rgb, a):
    out = np.empty(rgb.shape[:2] + (4,), np.float32)
    np.clip(rgb, 0, None, out=out[..., :3])
    if float(a.min()) <= 0.9995:
        out[..., :3] *= a
    out[..., 3:4] = a
    return out


def luminance(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def gblur(img, sigma):
    """Fast gaussian blur (downsample for large sigma)."""
    if sigma <= 0.05:
        return img
    h, w = img.shape[:2]
    if sigma > 12 and min(h, w) > 32:
        f = min(8, sigma / 6.0)
        sw, sh = max(4, int(w / f)), max(4, int(h / f))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_CONSTANT)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)


def _vec3(v):
    a = np.atleast_1d(np.asarray(v, np.float32))
    return np.repeat(a, 3) if a.size == 1 else a[:3]


def _rng(ctx, salt=0):
    return np.random.default_rng(stable_seed(ctx.seed, salt, int(round(ctx.comp_t * ctx.fps))))


def _smoothstep(a, b, x):
    u = np.clip((x - a) / max(1e-6, (b - a)), 0, 1)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- blur family
@effect("gaussian_blur", pad=lambda p: p["radius"] * 1.6, radius=10.0, dims="both")
def fx_gaussian_blur(img, p, ctx):
    s = p["radius"] * ctx.unit / 2.0
    if p["dims"] == "h":
        return cv2.GaussianBlur(img, (0, 1), sigmaX=max(0.01, s), sigmaY=0.01) if s > 0.05 else img
    if p["dims"] == "v":
        return cv2.GaussianBlur(img, (1, 0), sigmaX=0.01, sigmaY=max(0.01, s)) if s > 0.05 else img
    return gblur(img, s)


@effect("directional_blur", pad=lambda p: p["length"], angle=0.0, length=20.0)
def fx_directional_blur(img, p, ctx):
    L = int(round(p["length"] * ctx.unit))
    if L < 2:
        return img
    k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
    a = math.radians(p["angle"])
    for i in range(-L, L + 1):
        x = int(round(L + i * math.cos(a)))
        y = int(round(L + i * math.sin(a)))
        k[y, x] = 1.0
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT)


@effect("zoom_blur", amount=0.15, center=[0.5, 0.5], samples=12)
def fx_zoom_blur(img, p, ctx):
    amt = float(p["amount"])
    if abs(amt) < 1e-4:
        return img
    h, w = img.shape[:2]
    cx, cy = float(p["center"][0]) * w, float(p["center"][1]) * h
    n = int(p["samples"])
    acc = np.zeros_like(img)
    for i in range(n):
        s = 1.0 + amt * i / max(1, n - 1)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return acc / n


@effect("glow", pad=lambda p: p["radius"] * 2, color_params=("color",), threshold=0.6, radius=30.0, intensity=1.0, color=None)
def fx_glow(img, p, ctx):
    rgb, a = unpremul(img)
    lum = luminance(rgb)[..., None]
    knee = np.clip((lum - p["threshold"]) / max(1e-3, 1 - p["threshold"]), 0, 1)
    bright = img[..., :3] * knee
    if p.get("color") is not None:
        bright = luminance(bright)[..., None] * parse_color(p["color"])[:3]
    sig = p["radius"] * ctx.unit / 2
    g = gblur(bright, sig) * 0.6 + gblur(bright, sig * 0.35) * 0.4
    out = img.copy()
    add = g * p["intensity"]
    out[..., :3] = out[..., :3] + add - out[..., :3] * add  # screen
    ga = np.clip(luminance(add), 0, 1)[..., None]
    out[..., 3:4] = np.maximum(out[..., 3:4], ga)
    return out


@effect("god_rays", threshold=0.7, length=0.35, intensity=1.0, center=[0.5, 0.3], color="#fff2d0")
def fx_god_rays(img, p, ctx):
    rgb, a = unpremul(img)
    lum = luminance(rgb)
    bright = (np.clip((lum - p["threshold"]) / max(1e-3, 1 - p["threshold"]), 0, 1) * a[..., 0])[..., None]
    col = parse_color(p["color"])[:3]
    src = np.concatenate([bright * col, bright], axis=-1).astype(np.float32)
    h, w = img.shape[:2]
    small = cv2.resize(src, (max(4, w // 4), max(4, h // 4)), interpolation=cv2.INTER_AREA)
    rays = fx_zoom_blur(small, {"amount": p["length"], "center": p["center"], "samples": 24}, ctx)
    rays = cv2.resize(rays, (w, h), interpolation=cv2.INTER_LINEAR) * p["intensity"]
    out = img.copy()
    out[..., :3] = out[..., :3] + rays[..., :3] - out[..., :3] * np.clip(rays[..., :3], 0, 1)
    return out


@effect("sharpen", amount=0.5, radius=1.5)
def fx_sharpen(img, p, ctx):
    b = gblur(img, p["radius"] * ctx.unit)
    out = img + (img - b) * p["amount"]
    out[..., :3] = np.clip(out[..., :3], 0, out[..., 3:4])
    out[..., 3] = img[..., 3]
    return out


# ---------------------------------------------------------------- color
@effect("color", exposure=0.0, contrast=0.0, pivot=0.435, saturation=1.0, vibrance=0.0, temperature=0.0, tint=0.0,
        lift=0.0, gamma=1.0, gain=1.0, offset=0.0, hue=0.0, brightness=0.0, shadows=0.0, highlights=0.0, mix=1.0)
def fx_color(img, p, ctx):
    rgb, a = unpremul(img)
    src = rgb
    rgb = rgb * (2.0 ** p["exposure"])
    if p["brightness"]:
        rgb = rgb + p["brightness"]
    t, ti = p["temperature"] / 100.0, p["tint"] / 100.0
    if t or ti:
        rgb = rgb * np.array([1 + 0.18 * t, 1 - 0.18 * ti, 1 - 0.18 * t], np.float32)
    lift, gam, gain = _vec3(p["lift"]), _vec3(p["gamma"]), _vec3(p["gain"])
    rgb = gain * (rgb + lift * (1.0 - rgb)) + _vec3(p["offset"])
    if np.any(gam != 1.0):
        rgb = np.power(np.clip(rgb, 0, None), 1.0 / np.maximum(gam, 1e-3))
    if p["shadows"] or p["highlights"]:
        L = np.clip(luminance(rgb), 0, 1)[..., None]
        for amt, w in ((p["shadows"], 1 - _smoothstep(0.0, 0.55, L)), (p["highlights"], _smoothstep(0.45, 1.0, L))):
            if amt:
                k = w * amt * 0.5
                rgb = rgb + (k * (1 - rgb) if amt > 0 else k * rgb)
    if p["contrast"]:
        c = 1.0 + p["contrast"] / 100.0 if abs(p["contrast"]) > 2 else 1.0 + p["contrast"]
        rgb = (rgb - p["pivot"]) * c + p["pivot"]
    if p["hue"]:
        h = math.radians(p["hue"])
        cosA, sinA = math.cos(h), math.sin(h)
        k = 1 / 3.0
        sq = math.sqrt(k)
        m = np.array([
            [cosA + (1 - cosA) * k, k * (1 - cosA) - sq * sinA, k * (1 - cosA) + sq * sinA],
            [k * (1 - cosA) + sq * sinA, cosA + k * (1 - cosA), k * (1 - cosA) - sq * sinA],
            [k * (1 - cosA) - sq * sinA, k * (1 - cosA) + sq * sinA, cosA + k * (1 - cosA)]], np.float32)
        rgb = rgb @ m.T
    if p["saturation"] != 1.0 or p["vibrance"]:
        L = luminance(rgb)[..., None]
        s = p["saturation"]
        if p["vibrance"]:
            sat_now = rgb.max(-1, keepdims=True) - rgb.min(-1, keepdims=True)
            s = s * (1 + p["vibrance"] * (1 - np.clip(sat_now, 0, 1)))
        rgb = L + (rgb - L) * s
    rgb = np.clip(rgb, 0, 1)
    if p["mix"] < 1:
        rgb = src + (rgb - src) * p["mix"]
    return premul(rgb, a)


def _monotone_curve(points, n=1024):
    pts = sorted([(float(x), float(y)) for x, y in points])
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1] if pts[0][0] < 1e-3 else 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    xs = np.array([q[0] for q in pts])
    ys = np.array([q[1] for q in pts])
    d = np.diff(ys) / np.maximum(np.diff(xs), 1e-9)
    m = np.zeros_like(ys)
    m[1:-1] = (d[:-1] + d[1:]) / 2
    m[0], m[-1] = d[0], d[-1]
    for i in range(len(d)):  # Fritsch-Carlson
        if d[i] == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / d[i], m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9:
                tau = 3 / math.sqrt(s)
                m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]
    x = np.linspace(0, 1, n)
    idx = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    h = xs[idx + 1] - xs[idx]
    t = (x - xs[idx]) / np.maximum(h, 1e-9)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    y = h00 * ys[idx] + h10 * h * m[idx] + h01 * ys[idx + 1] + h11 * h * m[idx + 1]
    return np.clip(y, 0, 1).astype(np.float32)


@effect("curves", master=None, r=None, g=None, b=None)
def fx_curves(img, p, ctx):
    rgb, a = unpremul(img)
    rgb = np.clip(rgb, 0, 1)
    n = 1024
    if p["master"]:
        lut = _monotone_curve(p["master"], n)
        rgb = lut[(rgb * (n - 1)).astype(np.int32)]
    for i, ch in enumerate("rgb"):
        if p[ch]:
            lut = _monotone_curve(p[ch], n)
            rgb[..., i] = lut[(rgb[..., i] * (n - 1)).astype(np.int32)]
    return premul(rgb, a)


@effect("levels", in_black=0.0, in_white=1.0, gamma=1.0, out_black=0.0, out_white=1.0)
def fx_levels(img, p, ctx):
    rgb, a = unpremul(img)
    x = np.clip((rgb - p["in_black"]) / max(1e-4, p["in_white"] - p["in_black"]), 0, 1)
    x = np.power(x, 1.0 / max(1e-3, p["gamma"]))
    return premul(p["out_black"] + x * (p["out_white"] - p["out_black"]), a)


@functools.lru_cache(maxsize=16)
def load_cube(path):
    size, data, dmin, dmax = None, [], [0, 0, 0], [1, 1, 1]
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(s.split()[1])
            elif up.startswith("DOMAIN_MIN"):
                dmin = [float(v) for v in s.split()[1:4]]
            elif up.startswith("DOMAIN_MAX"):
                dmax = [float(v) for v in s.split()[1:4]]
            elif up.startswith(("TITLE", "LUT_1D")):
                if up.startswith("LUT_1D"):
                    raise ValueError("1D .cube LUTs not supported; use curves")
                continue
            else:
                vals = s.split()
                if len(vals) == 3:
                    data.append([float(v) for v in vals])
    if not size:
        raise ValueError(f"Not a 3D cube LUT: {path}")
    lut = np.array(data, np.float32).reshape(size, size, size, 3)  # [b][g][r]
    return lut, np.array(dmin, np.float32), np.array(dmax, np.float32)


def apply_cube(rgb, lut, dmin, dmax):
    n = lut.shape[0]
    x = np.clip((rgb - dmin) / (dmax - dmin), 0, 1) * (n - 1)
    i0 = np.floor(x).astype(np.int32)
    i0 = np.minimum(i0, n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c = lambda db, dg, dr: lut[b0 + db, g0 + dg, r0 + dr]
    c00 = c(0, 0, 0) * (1 - fr) + c(0, 0, 1) * fr
    c01 = c(0, 1, 0) * (1 - fr) + c(0, 1, 1) * fr
    c10 = c(1, 0, 0) * (1 - fr) + c(1, 0, 1) * fr
    c11 = c(1, 1, 0) * (1 - fr) + c(1, 1, 1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb


@effect("lut", file="", intensity=1.0)
def fx_lut(img, p, ctx):
    path = p["file"]
    if not os.path.isabs(path):
        path = os.path.join(ctx.base_dir, path)
    lut, dmin, dmax = load_cube(path)
    rgb, a = unpremul(img)
    out = apply_cube(np.clip(rgb, 0, 1), lut, dmin, dmax)
    if p["intensity"] < 1:
        out = rgb + (out - rgb) * p["intensity"]
    return premul(out, a)


@effect("tint", color_params=("black", "white"), black="#1a120b", white="#f3e3c3", amount=1.0)
def fx_tint(img, p, ctx):
    rgb, a = unpremul(img)
    L = np.clip(luminance(rgb), 0, 1)[..., None]
    b, w = parse_color(p["black"])[:3], parse_color(p["white"])[:3]
    out = b + (w - b) * L
    return premul(rgb + (out - rgb) * p["amount"], a)


@effect("black_white", amount=1.0)
def fx_bw(img, p, ctx):
    rgb, a = unpremul(img)
    L = luminance(rgb)[..., None]
    return premul(rgb + (L - rgb) * p["amount"], a)


@effect("sepia", amount=1.0)
def fx_sepia(img, p, ctx):
    return fx_tint(img, {"black": "#140c06", "white": "#f0dcb4", "amount": p["amount"]}, ctx)


@effect("invert", amount=1.0)
def fx_invert(img, p, ctx):
    rgb, a = unpremul(img)
    return premul(rgb + ((1 - rgb) - rgb) * p["amount"], a)


@effect("posterize", levels=6)
def fx_posterize(img, p, ctx):
    rgb, a = unpremul(img)
    n = max(2, int(p["levels"]))
    return premul(np.round(np.clip(rgb, 0, 1) * (n - 1)) / (n - 1), a)


@effect("fill", color_params=("color",), color="#ffffff", amount=1.0)
def fx_fill(img, p, ctx):
    rgb, a = unpremul(img)
    c = parse_color(p["color"])[:3]
    return premul(rgb + (c - rgb) * p["amount"], a)


@effect("gradient_overlay", colors=["#000000", "#ffffff"], angle=90.0, opacity=1.0, mode="multiply")
def fx_gradient_overlay(img, p, ctx):
    from .blend import SEPARABLE
    h, w = img.shape[:2]
    rgb, a = unpremul(img)
    ang = math.radians(p["angle"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = ((xx - w / 2) * math.cos(ang) + (yy - h / 2) * math.sin(ang))
    span = abs(w * math.cos(ang)) + abs(h * math.sin(ang))
    u = np.clip(proj / max(1, span) + 0.5, 0, 1)
    cols = [parse_color(c)[:3] for c in p["colors"]]
    stops = np.linspace(0, 1, len(cols))
    g = np.stack([np.interp(u, stops, [c[i] for c in cols]) for i in range(3)], -1).astype(np.float32)
    mode = p["mode"]
    B = SEPARABLE[mode](g, rgb) if mode in SEPARABLE else g
    return premul(rgb + (B - rgb) * p["opacity"], a)


# ---------------------------------------------------------------- stylize / film
@effect("vignette", color_params=("color",), amount=0.5, size=0.75, softness=0.6, roundness=0.5, color="#000000")
def fx_vignette(img, p, ctx):
    h, w = img.shape[:2]
    m = _vignette_mask(h, w, round(float(p["size"]), 4), round(float(p["softness"]), 4),
                       round(float(p["roundness"]), 4)) * float(p["amount"])
    c = parse_color(p["color"])[:3]
    out = img.copy()
    out[..., :3] = img[..., :3] * (1 - m) + c * img[..., 3:4] * m
    return out


@functools.lru_cache(maxsize=8)
def _vignette_mask(h, w, size, softness, roundness):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    asp = w / h
    nx = (xx / w - 0.5) * 2 * (1 - roundness + roundness * asp)
    ny = (yy / h - 0.5) * 2
    d = np.sqrt(nx * nx + ny * ny)
    return _smoothstep(size - softness / 2, size + softness / 2, d)[..., None].astype(np.float32)


@effect("grain", amount=0.06, size=1.2, mono=True, animated=True)
def fx_grain(img, p, ctx):
    h, w = img.shape[:2]
    rng = _rng(ctx, 11) if p["animated"] else np.random.default_rng(ctx.seed)
    sz = max(1.0, p["size"] * ctx.unit)
    gh, gw = max(2, int(h / sz)), max(2, int(w / sz))
    n = rng.standard_normal((gh, gw, 1 if p["mono"] else 3)).astype(np.float32)
    if (gh, gw) != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
        if n.ndim == 2:
            n = n[..., None]
    rgb, a = unpremul(img)
    L = np.clip(luminance(rgb), 0, 1)[..., None]
    weight = 0.35 + 0.65 * 4 * L * (1 - L)  # strongest in mid-tones like film
    return premul(rgb + n * p["amount"] * weight, a)


@effect("chromatic_aberration", amount=4.0)
def fx_ca(img, p, ctx):
    amt = p["amount"] * ctx.unit
    if abs(amt) < 0.1:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for ch, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * amt / max(w, h)
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


@effect("flicker", amount=0.08, speed=12.0)
def fx_flicker(img, p, ctx):
    from .props import noise1d
    f = 1 + p["amount"] * noise1d(ctx.comp_t * p["speed"], ctx.seed + 5)
    out = img.copy()
    out[..., :3] *= f
    return out


@effect("light_leak", color_params=("color",), color="#ff7a2a", amount=0.5, speed=0.2, size=0.6)
def fx_light_leak(img, p, ctx):
    from .props import noise1d
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = np.zeros((h, w), np.float32)
    for k in range(3):
        cx = (0.5 + 0.7 * noise1d(ctx.comp_t * p["speed"] + k * 10, ctx.seed + k)) * w
        cy = (0.5 + 0.7 * noise1d(ctx.comp_t * p["speed"] + k * 20 + 5, ctx.seed + k + 3)) * h
        r = p["size"] * max(w, h) * (0.6 + 0.3 * k)
        acc += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))
    leak = np.clip(acc * p["amount"], 0, 1)[..., None] * parse_color(p["color"])[:3]
    out = img.copy()
    out[..., :3] = out[..., :3] + leak * img[..., 3:4] - out[..., :3] * leak
    return out


@effect("old_film", amount=1.0, sepia=0.6, scratches=0.5, dust=0.5, flicker=0.08, grain=0.08, vignette=0.55)
def fx_old_film(img, p, ctx):
    k = p["amount"]
    out = fx_sepia(img, {"amount": p["sepia"] * k}, ctx)
    out = fx_flicker(out, {"amount": p["flicker"] * k, "speed": 14}, ctx)
    out = fx_grain(out, {"amount": p["grain"] * k, "size": 1.5, "mono": True, "animated": True}, ctx)
    h, w = out.shape[:2]
    rng = _rng(ctx, 77)
    lines = np.zeros((h, w), np.float32)
    for _ in range(int(rng.poisson(3 * p["scratches"] * k))):
        x = int(rng.uniform(0, w))
        cv2.line(lines, (x, 0), (x + int(rng.normal(0, 6)), h), float(rng.uniform(0.2, 0.6)), 1, cv2.LINE_AA)
    for _ in range(int(rng.poisson(12 * p["dust"] * k))):
        cv2.circle(lines, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(1, 3) * ctx.unit),
                   float(rng.uniform(0.3, 0.8)), -1, cv2.LINE_AA)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    out[..., :3] = np.clip(out[..., :3] + (sgn * lines * 0.6)[..., None] * out[..., 3:4], 0, None)
    return fx_vignette(out, {"amount": p["vignette"] * k, "size": 0.8, "softness": 0.7, "roundness": 0.5, "color": "#000"}, ctx)


@effect("cartoon", levels=8, edge=0.6, smooth=7, edge_width=1)
def fx_cartoon(img, p, ctx):
    """Anime/cel look: edge-preserving smoothing + quantized color + ink lines."""
    rgb, a = unpremul(img)
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    d = int(max(3, p["smooth"]))
    sm = u8
    for _ in range(2):
        sm = cv2.bilateralFilter(sm, d, 40, 7)
    n = max(2, int(p["levels"]))
    q = (np.round(sm.astype(np.float32) / 255 * (n - 1)) / (n - 1))
    gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
    if p["edge_width"] > 1:
        edges = cv2.erode(edges, np.ones((int(p["edge_width"]),) * 2, np.uint8))
    e = (edges.astype(np.float32) / 255)[..., None]
    q = q * (1 - p["edge"] * (1 - e))
    return premul(q, a)


@effect("pixelate", size=12.0)
def fx_pixelate(img, p, ctx):
    h, w = img.shape[:2]
    s = max(1.0, p["size"] * ctx.unit)
    small = cv2.resize(img, (max(1, int(w / s)), max(1, int(h / s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


@effect("wave_warp", amplitude=10.0, wavelength=120.0, speed=1.0, direction="horizontal")
def fx_wave(img, p, ctx):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    A, L = p["amplitude"] * ctx.unit, max(1, p["wavelength"] * ctx.unit)
    ph = ctx.t * p["speed"] * 2 * math.pi
    if p["direction"] == "vertical":
        mx, my = xx + A * np.sin(yy / L * 2 * math.pi + ph), yy
    else:
        mx, my = xx, yy + A * np.sin(xx / L * 2 * math.pi + ph)
    return cv2.remap(img, np.float32(mx), np.float32(my), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


@effect("turbulent_displace", amount=12.0, size=80.0, speed=0.5)
def fx_turb(img, p, ctx):
    h, w = img.shape[:2]
    s = max(4.0, p["size"] * ctx.unit)
    gh, gw = max(2, int(h / s) + 2), max(2, int(w / s) + 2)
    t = ctx.t * p["speed"]
    i0 = math.floor(t)
    f = t - i0
    f = f * f * (3 - 2 * f)

    def field(i, salt):
        r = np.random.default_rng(stable_seed(ctx.seed, salt, i))
        return r.standard_normal((gh, gw)).astype(np.float32)

    fx_ = field(i0, 1) * (1 - f) + field(i0 + 1, 1) * f
    fy_ = field(i0, 2) * (1 - f) + field(i0 + 1, 2) * f
    fx_ = cv2.resize(fx_, (w, h), interpolation=cv2.INTER_CUBIC)
    fy_ = cv2.resize(fy_, (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    A = p["amount"] * ctx.unit
    return cv2.remap(img, np.float32(xx + fx_ * A), np.float32(yy + fy_ * A), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@effect("lens_distortion", amount=0.1)
def fx_lens(img, p, ctx):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
    r2 = nx * nx + ny * ny
    k = p["amount"]
    f = 1 + k * r2
    return cv2.remap(img, np.float32(nx * f * (w / 2) + w / 2), np.float32(ny * f * (w / 2) + h / 2), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------- matte / keying / layer styles
@effect("drop_shadow", pad=lambda p: p["distance"] + p["softness"] * 2, color_params=("color",),
        color="#000000", opacity=0.6, distance=10.0, angle=135.0, softness=12.0)
def fx_drop_shadow(img, p, ctx):
    h, w = img.shape[:2]
    a = img[..., 3]
    d = p["distance"] * ctx.unit
    ang = math.radians(p["angle"])
    dx, dy = -math.cos(ang) * d, math.sin(ang) * d
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sa = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    sa = gblur(sa[..., None], p["softness"] * ctx.unit / 2)
    if sa.ndim == 2:
        sa = sa[..., None]
    c = parse_color(p["color"])
    sa = sa * p["opacity"] * c[3]
    shadow = np.concatenate([c[:3] * sa, sa], -1)
    return img + shadow * (1 - img[..., 3:4])


@effect("stroke", pad=lambda p: p["width"] + 2, color_params=("color",), color="#000000", width=4.0, opacity=1.0)
def fx_stroke(img, p, ctx):
    r = max(1, int(round(p["width"] * ctx.unit)))
    a = img[..., 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    da = cv2.dilate(a, k)[..., None] * p["opacity"]
    c = parse_color(p["color"])
    st = np.concatenate([c[:3] * da, da], -1)
    return img + st * (1 - img[..., 3:4])


@effect("chroma_key", color_params=("color",), color="#00ff00", tolerance=0.18, softness=0.12, spill=0.6)
def fx_chroma(img, p, ctx):
    rgb, a = unpremul(img)
    k = parse_color(p["color"])[:3]

    def cbcr(c):
        y = luminance(c)
        return np.stack([(c[..., 2] - y) * 0.5389, (c[..., 0] - y) * 0.6350], -1)

    d = np.linalg.norm(cbcr(rgb) - cbcr(k[None, None, :])[0, 0], axis=-1)
    m = _smoothstep(p["tolerance"], p["tolerance"] + max(1e-3, p["softness"]), d)[..., None]
    if p["spill"] > 0:
        dom = int(np.argmax(k))
        others = [i for i in range(3) if i != dom]
        lim = np.maximum(rgb[..., others[0]], rgb[..., others[1]])
        rgb = rgb.copy()
        rgb[..., dom] = rgb[..., dom] - p["spill"] * np.maximum(0, rgb[..., dom] - lim)
    return premul(rgb, a * m)


@effect("luma_key", threshold=0.1, softness=0.1, invert=False)
def fx_luma_key(img, p, ctx):
    rgb, a = unpremul(img)
    L = luminance(rgb)[..., None]
    m = _smoothstep(p["threshold"], p["threshold"] + max(1e-3, p["softness"]), L)
    if p["invert"]:
        m = 1 - m
    return premul(rgb, a * m)


@effect("letterbox", color_params=("color",), ratio=2.39, color="#000000", opacity=1.0)
def fx_letterbox(img, p, ctx):
    h, w = img.shape[:2]
    target_h = w / max(0.1, p["ratio"])
    bar = max(0, int(round((h - target_h) / 2)))
    if bar <= 0:
        return img
    out = img.copy()
    c = parse_color(p["color"])
    o = p["opacity"] * c[3]
    for sl in (slice(0, bar), slice(h - bar, h)):
        out[sl] = out[sl] * (1 - o) + np.array([*(c[:3] * o), o], np.float32)
    return out


@effect("opacity", amount=1.0)
def fx_opacity(img, p, ctx):
    return img * p["amount"]


# ---------------------------------------------------------------- dispatcher
def apply_effects(img: np.ndarray, effects: list, eval_param, ctx: FxCtx):
    """effects: list of dicts with 'type'; eval_param(effect_index, name, raw, is_color) -> value."""
    for i, e in enumerate(effects):
        if e.get("enabled", True) is False:
            continue
        name = resolve_name(e["type"])
        spec = REGISTRY.get(name)
        if spec is None:
            raise ValueError(f"Unknown effect '{e['type']}'. Known: {', '.join(sorted(REGISTRY))}")
        params = {}
        for k, dv in spec["defaults"].items():
            raw = e.get(k, dv)
            params[k] = eval_param(i, k, raw, k in spec["color_params"]) if raw is not None else None
        out = spec["fn"](img, params, ctx)
        mix = e.get("mix_layer")
        img = out.astype(np.float32, copy=False) if mix is None else (img + (out - img) * float(mix)).astype(np.float32)
    return img


def effects_pad(effects: list, eval_param) -> float:
    pad = 0.0
    for i, e in enumerate(effects):
        spec = REGISTRY.get(resolve_name(e["type"]))
        if spec and spec["pad"] and e.get("enabled", True) is not False:
            params = {k: eval_param(i, k, e.get(k, dv), k in spec["color_params"]) if e.get(k, dv) is not None else None
                      for k, dv in spec["defaults"].items()}
            try:
                pad += float(spec["pad"](params))
            except Exception:
                pass
    return pad


# ---------------------------------------------------------------- secondary grading / matching
def _hsl_key(rgb, p):
    import cv2
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)  # H 0..360, S 0..1, V 0..1
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    soft = max(1e-3, float(p["softness"]))
    d = np.abs(((h - float(p["hue"])) + 180) % 360 - 180)
    kh = 1 - _smoothstep(float(p["hue_width"]) / 2, float(p["hue_width"]) / 2 + soft * 60, d)
    ks = _smoothstep(float(p["sat_min"]) - soft * 0.2, float(p["sat_min"]) + soft * 0.05, s) * \
        (1 - _smoothstep(float(p["sat_max"]) - soft * 0.05, float(p["sat_max"]) + soft * 0.2, s))
    kl = _smoothstep(float(p["lum_min"]) - soft * 0.2, float(p["lum_min"]) + soft * 0.05, v) * \
        (1 - _smoothstep(float(p["lum_max"]) - soft * 0.05, float(p["lum_max"]) + soft * 0.2, v))
    k = kh * ks * kl
    if p["invert"]:
        k = 1 - k
    return k.astype(np.float32)


@effect("secondary", hue=30.0, hue_width=40.0, sat_min=0.15, sat_max=1.0, lum_min=0.0, lum_max=1.0, softness=0.3,
        invert=False, show_mask=False, exposure=0.0, saturation=1.0, hue_shift=0.0, temperature=0.0, tint=0.0,
        contrast=0.0, blur_key=2.0)
def fx_secondary(img, p, ctx):
    """HSL qualifier (Resolve-style secondary): isolate a colour range, grade only that."""
    rgb, a = unpremul(img)
    k = _hsl_key(rgb, p)
    if p["blur_key"] > 0:
        k = gblur(k[..., None], p["blur_key"] * ctx.unit)[..., 0] if gblur(k[..., None], 0.1).ndim == 3 else gblur(k, p["blur_key"] * ctx.unit)
    if p["show_mask"]:
        return premul(np.repeat(k[..., None], 3, -1), a)
    graded, _ = unpremul(fx_color(img, {**REGISTRY["color"]["defaults"], "exposure": p["exposure"], "saturation": p["saturation"],
                                         "hue": p["hue_shift"], "temperature": p["temperature"], "tint": p["tint"],
                                         "contrast": p["contrast"]}, ctx))
    out = rgb + (graded - rgb) * k[..., None]
    return premul(out, a)


_REF_CACHE = {}


@effect("match_color", reference="", strength=1.0, preserve_luma=False)
def fx_match_color(img, p, ctx):
    """Shot match: transfer the colour statistics (Lab mean/std) of a reference frame/image."""
    from .color import lab_stats, match_to
    from .media import load_image, is_image, VideoReader
    ref = p["reference"]
    if not ref:
        return img
    path = ref if os.path.isabs(ref) else os.path.join(ctx.base_dir, ref)
    if path not in _REF_CACHE:
        if is_image(path):
            r = load_image(path)
        else:
            rd = VideoReader(path, scale=0.25)
            r = rd.at(float(p.get("ref_time", 0) or 0))
        _REF_CACHE[path] = lab_stats(r[..., :3].astype(np.float32) / 255.0)
    m, s = _REF_CACHE[path]
    rgb, a = unpremul(img)
    return premul(match_to(rgb, m, s, float(p["strength"]), bool(p["preserve_luma"])), a)


@effect("auto_balance", strength=1.0)
def fx_auto_balance(img, p, ctx):
    """Grey-world white balance (removes colour casts)."""
    rgb, a = unpremul(img)
    w = a[..., 0] > 0.5
    if not w.any():
        return img
    mean = rgb[w].mean(0) + 1e-6
    gain = mean.mean() / mean
    gain = 1 + (gain - 1) * float(p["strength"])
    return premul(rgb * gain, a)


# ---------------------------------------------------------------- stylize / distort pack
def _polar_remap(img, fn):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = fn(xx, yy, w, h)
    return cv2.remap(img, np.float32(mx), np.float32(my), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


@effect("twirl", angle=90.0, radius=0.5, center=[0.5, 0.5])
def fx_twirl(img, p, ctx):
    def fn(xx, yy, w, h):
        cx, cy = p["center"][0] * w, p["center"][1] * h
        R = max(1.0, p["radius"] * min(w, h))
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        k = np.clip(1 - r / R, 0, 1) ** 2
        a = np.radians(p["angle"]) * k
        c, s = np.cos(a), np.sin(a)
        return cx + dx * c - dy * s, cy + dx * s + dy * c
    return _polar_remap(img, fn)


@effect("bulge", amount=0.5, radius=0.35, center=[0.5, 0.5])
def fx_bulge(img, p, ctx):
    """amount > 0 magnifies the centre (bulge), < 0 pinches. Smooth falloff to zero distortion at the radius."""
    def fn(xx, yy, w, h):
        cx, cy = p["center"][0] * w, p["center"][1] * h
        R = max(1.0, p["radius"] * min(w, h))
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy) / R
        amt = float(np.clip(p["amount"], -0.95, 0.95))
        f = np.where(r < 1, 1 - amt * (1 - np.clip(r, 0, 1)) ** 2, 1.0)  # sample closer to centre -> magnify
        return cx + dx * f, cy + dy * f
    return _polar_remap(img, fn)


@effect("mirror", axis="vertical", position=0.5, flip="first")
def fx_mirror(img, p, ctx):
    h, w = img.shape[:2]
    out = img.copy()
    if p["axis"] == "vertical":
        c = int(p["position"] * w)
        if p["flip"] == "first":
            src = img[:, :c][:, ::-1]
            n = min(src.shape[1], w - c)
            out[:, c:c + n] = src[:, :n]
        else:
            src = img[:, c:][:, ::-1]
            n = min(src.shape[1], c)
            out[:, c - n:c] = src[:, src.shape[1] - n:]
    else:
        c = int(p["position"] * h)
        if p["flip"] == "first":
            src = img[:c][::-1]
            n = min(src.shape[0], h - c)
            out[c:c + n] = src[:n]
        else:
            src = img[c:][::-1]
            n = min(src.shape[0], c)
            out[c - n:c] = src[src.shape[0] - n:]
    return out


@effect("kaleidoscope", segments=6, rotation=0.0, center=[0.5, 0.5])
def fx_kaleido(img, p, ctx):
    def fn(xx, yy, w, h):
        cx, cy = p["center"][0] * w, p["center"][1] * h
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        a = np.arctan2(dy, dx) - np.radians(p["rotation"])
        seg = 2 * np.pi / max(2, int(p["segments"]))
        a = np.abs(((a % seg) + seg) % seg - seg / 2)
        return cx + r * np.cos(a), cy + r * np.sin(a)
    return _polar_remap(img, fn)


@effect("edge_glow", color_params=("color",), color="#7fd4ff", threshold=0.15, radius=6.0, intensity=1.5, keep=0.6)
def fx_edge_glow(img, p, ctx):
    rgb, a = unpremul(img)
    L = luminance(rgb)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    e = np.clip((np.sqrt(gx * gx + gy * gy) - p["threshold"]) * 3, 0, 1)
    g = gblur(e[..., None], p["radius"] * ctx.unit / 2)
    g = g if g.ndim == 3 else g[..., None]
    col = parse_color(p["color"])[:3]
    out = rgb * p["keep"] + (e[..., None] + g) * col * p["intensity"]
    return premul(np.clip(out, 0, None), a)


@effect("halftone", size=8.0, angle=45.0, mono=False)
def fx_halftone(img, p, ctx):
    rgb, a = unpremul(img)
    h, w = img.shape[:2]
    s = max(2.0, p["size"] * ctx.unit)
    ang = np.radians(p["angle"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx * np.cos(ang) + yy * np.sin(ang)) / s
    v = (-xx * np.sin(ang) + yy * np.cos(ang)) / s
    d = np.sqrt((u - np.floor(u) - 0.5) ** 2 + (v - np.floor(v) - 0.5) ** 2)
    if p["mono"]:
        L = luminance(rgb)
        dot = (d < np.sqrt(np.clip(1 - L, 0, 1) / np.pi) * 1.0).astype(np.float32)
        out = np.repeat((1 - dot)[..., None], 3, -1)
    else:
        out = np.stack([(d >= np.sqrt(np.clip(1 - rgb[..., i], 0, 1) / np.pi)).astype(np.float32) for i in range(3)], -1)
    return premul(out, a)


@effect("emboss", amount=1.0, angle=135.0, mix=1.0)
def fx_emboss(img, p, ctx):
    rgb, a = unpremul(img)
    L = luminance(rgb)
    ang = np.radians(p["angle"])
    k = np.array([[-np.cos(ang) - np.sin(ang), -np.sin(ang), np.cos(ang) - np.sin(ang)],
                  [-np.cos(ang), 0, np.cos(ang)],
                  [-np.cos(ang) + np.sin(ang), np.sin(ang), np.cos(ang) + np.sin(ang)]], np.float32) * p["amount"]
    e = np.clip(cv2.filter2D(L, -1, k) + 0.5, 0, 1)
    out = rgb + (np.repeat(e[..., None], 3, -1) - rgb) * p["mix"]
    return premul(out, a)


@effect("lens_flare", color_params=("color",), position=[0.75, 0.25], intensity=1.0, size=1.0, color="#ffd9a0",
        ghosts=5, streak=0.6)
def fx_lens_flare(img, p, ctx):
    """Procedural lens flare: core glow + anamorphic streak + ghosts along the axis through frame centre."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["position"][0] * w, p["position"][1] * h
    cx, cy = w / 2, h / 2
    D = max(w, h)
    col = parse_color(p["color"])[:3]
    sz = float(p["size"])
    r2 = (xx - px) ** 2 + (yy - py) ** 2
    light = np.exp(-r2 / (2 * (0.03 * D * sz) ** 2))[..., None] * 1.2 + np.exp(-r2 / (2 * (0.12 * D * sz) ** 2))[..., None] * 0.35
    light = light * col
    st = float(p["streak"])
    if st > 0:
        streak = np.exp(-((yy - py) ** 2) / (2 * (0.004 * D * sz) ** 2)) * np.exp(-np.abs(xx - px) / (0.45 * D * st))
        light = light + streak[..., None] * np.array([0.55, 0.75, 1.0], np.float32) * 0.8
    rng = np.random.default_rng(int(ctx.seed))
    tints = [(1, .6, .3), (.4, .8, 1), (.6, 1, .6), (1, .4, .8), (1, 1, .6), (.5, .6, 1)]
    for k in range(int(p["ghosts"])):
        f = -0.4 - 1.6 * (k + 1) / max(1, int(p["ghosts"]))
        gx, gy = cx + (px - cx) * f, cy + (py - cy) * f
        rad = D * sz * rng.uniform(0.02, 0.07)
        d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        ring = np.clip(1 - np.abs(d - rad * 0.85) / (rad * 0.25), 0, 1) * 0.35 + np.clip(1 - d / rad, 0, 1) * 0.12
        light = light + ring[..., None] * np.array(tints[k % len(tints)], np.float32)
    light = light * float(p["intensity"])
    out = img.copy()
    out[..., :3] = out[..., :3] + light - out[..., :3] * np.clip(light, 0, 1)
    out[..., 3:4] = np.maximum(out[..., 3:4], np.clip(luminance(light)[..., None], 0, 1))
    return out


@effect("glitch", amount=0.5, block=0.06, rgb_split=8.0, scanlines=0.25, frequency=0.35)
def fx_glitch(img, p, ctx):
    """Digital glitch: displaced slices, RGB split, scanlines; only fires on some frames (frequency)."""
    rng = _rng(ctx, 31)
    if rng.random() > float(p["frequency"]) or p["amount"] <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    amt = float(p["amount"])
    for _ in range(int(3 + 12 * amt)):
        y = int(rng.uniform(0, h))
        bh = max(2, int(rng.uniform(0.2, 1.0) * p["block"] * h))
        dx = int(rng.normal(0, 0.08 * w * amt))
        out[y:y + bh] = np.roll(out[y:y + bh], dx, axis=1)
    s = int(p["rgb_split"] * ctx.unit * amt)
    if s:
        out[..., 0] = np.roll(out[..., 0], s, axis=1)
        out[..., 2] = np.roll(out[..., 2], -s, axis=1)
    if p["scanlines"] > 0:
        out[::2, :, :3] *= 1 - float(p["scanlines"]) * amt
    return out
