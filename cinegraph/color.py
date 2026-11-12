"""Colour management (OpenColorIO, ACES 2.0 studio config) + scopes + shot matching.

comp "color": {"working": "display" (default: classic sRGB compositing) | "linear" (Linear Rec.709) | "acescg",
               "view": "standard" (Un-tone-mapped) | "aces" (ACES 2.0 SDR tone map) | "video"}
layer "colorspace": any OCIO colorspace name, e.g. "ARRI LogC4", "S-Log3 S-Gamut3.Cine", "V-Log V-Gamut",
                    "Apple Log", "ACEScg", "Linear Rec.709 (sRGB)". Default for media/graphics: sRGB.
"""
from __future__ import annotations

import functools

import numpy as np

SRGB = "sRGB Encoded Rec.709 (sRGB)"
WORKING = {"linear": "Linear Rec.709 (sRGB)", "acescg": "ACEScg"}
VIEWS = {"standard": "Un-tone-mapped", "aces": "ACES 2.0 - SDR 100 nits (Rec.709)", "video": "Video (colorimetric)"}
SDR_DISPLAY = "sRGB - Display"
HDR_DISPLAY = "Rec.2100-PQ - Display"
HDR_VIEW = "ACES 2.0 - HDR 1000 nits (Rec.2020)"
ALIASES = {"logc": "ARRI LogC4", "logc4": "ARRI LogC4", "logc3": "ARRI LogC3 (EI800)", "slog3": "S-Log3 S-Gamut3.Cine",
           "vlog": "V-Log V-Gamut", "applelog": "Apple Log", "clog3": "CanonLog3 CinemaGamut D55",
           "bmd": "BMDFilm WideGamut Gen5", "redlog": "Log3G10 REDWideGamutRGB", "srgb": SRGB, "rec709": "Camera Rec.709",
           "linear": "Linear Rec.709 (sRGB)", "acescg": "ACEScg", "acescct": "ACEScct", "dlog": "D-Log D-Gamut"}


@functools.lru_cache(maxsize=1)
def config():
    import PyOpenColorIO as O
    return O.Config.CreateFromBuiltinConfig("studio-config-latest")


def resolve(name):
    if name is None:
        return SRGB
    key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in ALIASES:
        return ALIASES[key]
    names = list(config().getColorSpaceNames())
    if name in names:
        return name
    raise ValueError(f"Unknown colorspace '{name}'. Aliases: {sorted(ALIASES)}; OCIO names: {names}")


@functools.lru_cache(maxsize=64)
def _proc(src, dst):
    return config().getProcessor(src, dst).getDefaultCPUProcessor()


@functools.lru_cache(maxsize=64)
def _dproc(src, display, view):
    import PyOpenColorIO as O
    t = O.DisplayViewTransform(src=src, display=display, view=view)
    return config().getProcessor(t).getDefaultCPUProcessor()


def _apply(proc, rgb):
    a = np.ascontiguousarray(rgb, np.float32).copy()
    proc.applyRGB(a)
    return a


def convert(rgb, src, dst):
    if src == dst:
        return rgb
    return _apply(_proc(src, dst), rgb)


def to_display(rgb, src, view="standard", display=SDR_DISPLAY):
    v = VIEWS.get(view, view)
    return _apply(_dproc(src, display, v), rgb)


def convert_premul(img, fn):
    """Apply an RGB conversion to a premultiplied RGBA image."""
    a = img[..., 3:4]
    opaque = float(a.min()) > 0.9995
    rgb = img[..., :3] if opaque else np.where(a > 1e-6, img[..., :3] / np.maximum(a, 1e-6), 0)
    out = np.empty_like(img)
    c = fn(rgb)
    out[..., :3] = c if opaque else c * a
    out[..., 3:4] = a
    return out


class ColorSettings:
    def __init__(self, spec: dict | None):
        spec = spec or {}
        w = spec.get("working", "display")
        self.working = WORKING.get(w) if w != "display" else None
        if w not in ("display",) and self.working is None:
            self.working = resolve(w)
        self.view = spec.get("view", "standard")
        self.output = "sdr"

    @property
    def linear(self):
        return self.working is not None

    def input(self, img, cs):
        """Convert a display-referred / camera source into the working space (premultiplied in, out)."""
        src = resolve(cs)
        if self.working:
            return convert_premul(img, lambda rgb: convert(rgb, src, self.working))
        if src != SRGB:  # display mode: camera log -> viewable display image
            return convert_premul(img, lambda rgb: np.clip(to_display(rgb, src, cs_view(self.view, src)), 0, 1))
        return img

    def color(self, rgba):
        """Linearize a UI colour value (solid/background/material) when compositing in linear."""
        if not self.working:
            return rgba
        out = np.array(rgba, np.float32).copy()
        out[:3] = convert(out[None, None, :3], SRGB, self.working)[0, 0]
        return out

    def output_frame(self, img):
        if not self.working:
            return img
        if self.output == "hdr10":
            return convert_premul(img, lambda rgb: to_display(rgb, self.working, HDR_VIEW, HDR_DISPLAY))
        return convert_premul(img, lambda rgb: np.clip(to_display(rgb, self.working, self.view), 0, 1))


def cs_view(view, src):
    return view if view != "standard" or src == SRGB else "aces"


# ---------------------------------------------------------------- shot matching / auto balance
def lab_stats(rgb):
    import cv2
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6


def match_to(rgb, ref_mean, ref_std, strength=1.0, preserve_luma=False):
    """Reinhard colour transfer in Lab (mean/std per channel)."""
    import cv2
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - m) * (ref_std / s) + ref_mean
    if preserve_luma:
        out[..., 0] = lab[..., 0]
    out = lab + (out - lab) * strength
    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


# ---------------------------------------------------------------- scopes
def scopes(rgb_u8: np.ndarray, size=(512, 256)) -> np.ndarray:
    """Return an RGB uint8 image: [luma waveform | RGB parade] over [vectorscope | histogram]."""
    import cv2
    W, H = size
    img = cv2.resize(rgb_u8[..., :3], (W, max(16, int(rgb_u8.shape[0] * W / rgb_u8.shape[1]))), interpolation=cv2.INTER_AREA)
    f = img.astype(np.float32) / 255.0
    luma = f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722

    def waveform(ch_vals, w, color):
        acc = np.zeros((H, w), np.float32)
        hh, ww = ch_vals.shape
        xs = np.repeat(np.arange(ww)[None], hh, 0) * w // ww
        ys = (H - 1 - np.clip(ch_vals * (H - 1), 0, H - 1)).astype(np.int32)
        np.add.at(acc, (ys.ravel(), xs.ravel()), 1)
        acc = np.clip(np.log1p(acc) / np.log1p(hh * 0.25), 0, 1)
        return (acc[..., None] * np.array(color, np.float32))

    def grid(a):
        for k in range(0, 11):
            y = int((H - 1) * (1 - k / 10))
            a[y, :] = np.maximum(a[y, :], 0.18 if k % 5 else 0.35)
        return a

    wf = grid(waveform(luma, W, (0.9, 0.95, 0.9)))
    pw = W // 3
    parade = np.concatenate([waveform(f[..., i], pw, c) for i, c in enumerate([(1, .25, .25), (.25, 1, .25), (.35, .55, 1)])], 1)
    parade = grid(cv2.resize(parade, (W, H), interpolation=cv2.INTER_NEAREST))
    # vectorscope (BT.709 Cb/Cr)
    cb = (f[..., 2] - luma) / 1.8556
    cr = (f[..., 0] - luma) / 1.5748
    V = H
    vs = np.zeros((V, V), np.float32)
    xs = np.clip(((cb * 0.9 + 0.5) * (V - 1)).astype(np.int32), 0, V - 1)
    ys = np.clip(((0.5 - cr * 0.9) * (V - 1)).astype(np.int32), 0, V - 1)
    np.add.at(vs, (ys.ravel(), xs.ravel()), 1)
    vs = np.clip(np.log1p(vs) / np.log1p(vs.size * 0.002), 0, 1)
    vsc = np.repeat(vs[..., None], 3, -1) * 0.9
    cv2.circle(vsc, (V // 2, V // 2), V // 2 - 2, (0.3, 0.3, 0.3), 1, cv2.LINE_AA)
    # skin-tone line (~123 degrees)
    ang = np.radians(123)
    cv2.line(vsc, (V // 2, V // 2), (int(V / 2 + np.cos(ang) * V / 2), int(V / 2 - np.sin(ang) * V / 2)), (0.6, 0.45, 0.3), 1, cv2.LINE_AA)
    for name, c in {"R": (1, 0, 0), "G": (0, 1, 0), "B": (0, 0, 1), "Y": (1, 1, 0), "C": (0, 1, 1), "M": (1, 0, 1)}.items():
        c = np.array(c, np.float32)
        l_ = c @ np.array([0.2126, 0.7152, 0.0722])
        x = int(((c[2] - l_) / 1.8556 * 0.9 * 0.75 + 0.5) * (V - 1))  # 75% bars
        y = int((0.5 - (c[0] - l_) / 1.5748 * 0.9 * 0.75) * (V - 1))
        cv2.rectangle(vsc, (x - 5, y - 5), (x + 5, y + 5), tuple(float(v) for v in c * 0.9), 1)
    vsc_full = np.zeros((H, W, 3), np.float32)
    vsc_full[:, (W - V) // 2:(W - V) // 2 + V] = vsc
    # histogram
    hist = np.zeros((H, W, 3), np.float32)
    for i in range(3):
        h_, _ = np.histogram(f[..., i], 256, (0, 1))
        h_ = np.clip(h_ / max(1, np.percentile(h_[1:-1], 99)), 0, 1)  # ignore clipped-black/white spikes
        for x in range(W):
            v = h_[int(x / W * 255)]
            hist[H - int(v * (H - 1)) - 1:, x, i] = 0.75
    top = np.concatenate([wf, parade], 1)
    bot = np.concatenate([vsc_full, hist], 1)
    out = (np.clip(np.concatenate([top, bot], 0), 0, 1) * 255).astype(np.uint8)
    for (x, y, t) in [(6, 16, "LUMA WAVEFORM"), (W + 6, 16, "RGB PARADE"), (6, H + 16, "VECTORSCOPE"), (W + 6, H + 16, "HISTOGRAM")]:
        cv2.putText(out, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return out
