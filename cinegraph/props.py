"""Animatable properties: constants, keyframes, spatial paths, loops and AE-style expressions.

A property in a project file can be written as:

    5                                   constant
    [960, 540]                          constant vector
    "#ff8800"                           constant color (for color props)
    {"keys": [[0, 100], [2, 120, "ease"]]}          keyframes  [time, value, easing-of-segment-starting-here]
    {"keys": [{"t":0,"v":[0,0],"ease":"out_cubic"}, {"t":1.5,"v":[100,0]}],
     "path": "smooth",                  Catmull-Rom spatial interpolation (motion paths)
     "loop": "cycle"|"pingpong"|"continue",
     "expr": "value + wiggle(2, 15)"}   expression applied on top of keys
    "=wiggle(1.5, 10) + 960"            pure expression (leading '=')

Times inside keys are LAYER-LOCAL seconds (0 = layer start) unless the property belongs to the
camera or the composition, which use comp time.
Expression namespace: t (local), time (comp), value, wiggle(freq, amp, octaves=1, seed=0),
noise(x, seed=0), linear(x, x0, x1, y0, y1), ease(x, x0, x1, y0, y1), clamp(x, a, b),
sin cos tan abs min max pow sqrt pi floor ceil round, random(seed), smoothstep(a,b,x), np.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import easing
from .util import parse_color, is_color_string


# ---------------------------------------------------------------- noise / wiggle
def _hash01(i: np.ndarray, seed: int) -> np.ndarray:
    x = (np.asarray(i, np.int64) * 374761393 + seed * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF) / float(0xFFFFFF)


def noise1d(x: float, seed: int = 0) -> float:
    """Smooth value noise in [-1, 1]."""
    i = math.floor(x)
    f = x - i
    a = _hash01(np.array(i), seed) * 2 - 1
    b = _hash01(np.array(i + 1), seed) * 2 - 1
    u = f * f * f * (f * (f * 6 - 15) + 10)  # quintic fade
    return float(a + (b - a) * u)


def wiggle_value(t: float, freq: float, amp: float, octaves: int, seed: int, dims: int):
    out = []
    for d in range(dims):
        v, a, f = 0.0, 1.0, freq
        norm = 0.0
        for o in range(max(1, int(octaves))):
            v += a * noise1d(t * f, seed * 131 + d * 17 + o * 7919)
            norm += a
            a *= 0.5
            f *= 2.0
        out.append(amp * v / norm)
    return out[0] if dims == 1 else np.array(out, np.float32)


# ---------------------------------------------------------------- helpers
def _to_val(v, is_color: bool):
    if is_color or is_color_string(v):
        return parse_color(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, np.float32)
    if isinstance(v, bool):
        return float(v)
    return float(v)


def _catmull(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def _remap(x, x0, x1, y0, y1, fn):
    if x1 == x0:
        u = 1.0 if x >= x1 else 0.0
    else:
        u = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
    u = fn(u)
    if isinstance(y0, (list, tuple)):
        y0 = np.asarray(y0, np.float32)
    if isinstance(y1, (list, tuple)):
        y1 = np.asarray(y1, np.float32)
    return y0 + (y1 - y0) * u


class Prop:
    """An animatable value. Call .at(t_local, t_comp)."""

    __slots__ = ("raw", "is_color", "const", "keys", "path", "loop", "expr", "_code", "default")

    def __init__(self, raw: Any, is_color: bool = False, default: Any = 0.0):
        self.raw = raw
        self.is_color = is_color
        self.default = default
        self.const = None
        self.keys = None
        self.path = None
        self.loop = None
        self.expr = None
        self._code = None
        if raw is None:
            raw = default
        if isinstance(raw, Prop):
            raise TypeError("Prop of Prop")
        if isinstance(raw, str) and raw.startswith("="):
            self.expr = raw[1:]
        elif isinstance(raw, dict) and ("keys" in raw or "expr" in raw):
            ks = raw.get("keys") or []
            parsed = []
            for k in ks:
                if isinstance(k, dict):
                    t, v, e = k["t"], k["v"], k.get("ease", k.get("easing"))
                    if k.get("hold"):
                        e = "hold"
                else:
                    t, v = k[0], k[1]
                    e = k[2] if len(k) > 2 else None
                parsed.append((float(t), _to_val(v, is_color), easing.get(e or raw.get("ease") or "linear")))
            parsed.sort(key=lambda x: x[0])
            self.keys = parsed or None
            self.path = raw.get("path")
            self.loop = raw.get("loop")
            self.expr = raw.get("expr")
            if self.keys is None and self.expr is None:
                self.const = _to_val(default, is_color)
        else:
            self.const = _to_val(raw, is_color)
        if self.expr:
            self._code = compile(self.expr, "<expr>", "eval")

    @property
    def animated(self) -> bool:
        return self.const is None

    # ---------------------------------------------------------------- evaluation
    def _keyed(self, t: float):
        ks = self.keys
        if len(ks) == 1:
            return ks[0][1]
        t0, tN = ks[0][0], ks[-1][0]
        span = tN - t0
        if self.loop and span > 0 and t > tN:
            if self.loop == "cycle":
                t = t0 + (t - t0) % span
            elif self.loop == "pingpong":
                n = (t - t0) / span
                f = n % 1.0
                t = t0 + (f if int(n) % 2 == 0 else 1 - f) * span
            elif self.loop == "continue":
                # extrapolate using last segment velocity
                (ta, va, _), (tb, vb, _) = ks[-2], ks[-1]
                vel = (vb - va) / max(1e-9, tb - ta)
                return vb + vel * (t - tb)
        if t <= t0:
            return ks[0][1]
        if t >= tN:
            return ks[-1][1]
        # binary search segment
        lo, hi = 0, len(ks) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ks[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        ta, va, fn = ks[lo]
        tb, vb, _ = ks[hi]
        u = fn((t - ta) / (tb - ta)) if tb > ta else 1.0
        if self.path == "smooth" and isinstance(va, np.ndarray) and len(ks) > 2:
            p0 = ks[lo - 1][1] if lo > 0 else va - (vb - va)
            p3 = ks[hi + 1][1] if hi + 1 < len(ks) else vb + (vb - va)
            return _catmull(p0, va, vb, p3, u).astype(np.float32)
        return va + (vb - va) * u

    def at(self, t: float, comp_t: float | None = None):
        if self.const is not None:
            return self.const
        value = self._keyed(t) if self.keys else _to_val(self.default, self.is_color)
        if self._code is None:
            return value
        ns = _expr_ns(t, comp_t if comp_t is not None else t, value)
        try:
            out = eval(self._code, {"__builtins__": {}}, ns)
        except Exception as e:
            raise ValueError(f"Expression error in '{self.expr}' at t={t:.3f}: {e}") from e
        return _to_val(out, self.is_color) if not isinstance(out, np.ndarray) else out.astype(np.float32)


def _expr_ns(t, comp_t, value):
    dims = 1 if np.isscalar(value) or np.ndim(value) == 0 else len(value)

    def wiggle(freq, amp, octaves=1, seed=0):
        return wiggle_value(t, freq, amp, octaves, seed, dims)

    def noise(x, seed=0):
        return noise1d(x, seed)

    def random(seed=0):
        return float(_hash01(np.array(int(seed * 1000003) & 0x7FFFFFFF), 99))

    return {
        "t": t, "time": comp_t, "value": value, "np": np,
        "wiggle": wiggle, "noise": noise, "random": random,
        "linear": lambda x, x0, x1, y0, y1: _remap(x, x0, x1, y0, y1, lambda u: u),
        "ease": lambda x, x0, x1, y0, y1, name="ease": _remap(x, x0, x1, y0, y1, easing.get(name)),
        "clamp": lambda x, a, b: min(b, max(a, x)),
        "smoothstep": lambda a, b, x: (lambda u: u * u * (3 - 2 * u))(min(1, max(0, (x - a) / (b - a)))),
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "floor": math.floor, "ceil": math.ceil,
        "round": round, "exp": math.exp, "log": math.log, "array": lambda *a: np.array(a, np.float32),
        "vec": lambda *a: np.array(a, np.float32),
    }


def P(raw, default=0.0, color=False) -> Prop:
    return raw if isinstance(raw, Prop) else Prop(raw, is_color=color, default=default)


def as_vec(v, n: int, fill: float = 0.0, broadcast: bool = False) -> np.ndarray:
    a = np.atleast_1d(np.asarray(v, np.float32)).ravel()
    if a.size == 1 and n > 1 and broadcast:
        return np.repeat(a, n).astype(np.float32)
    if a.size < n:
        a = np.concatenate([a, np.full(n - a.size, fill, np.float32)])
    return a[:n].astype(np.float32)
