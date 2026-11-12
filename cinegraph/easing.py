"""Easing curves.

Every easing maps progress u in [0,1] -> eased value (usually in [0,1], back/elastic overshoot).
Names follow the easings.net convention plus After Effects style aliases:

    linear, hold,
    ease (AE "Easy Ease" = bezier(.333,0,.667,1)), ease_in, ease_out,
    in_sine/out_sine/in_out_sine, *_quad, *_cubic, *_quart, *_quint, *_expo, *_circ,
    *_back, *_elastic, *_bounce,
    bezier(x1,y1,x2,y2)  -- CSS / AE speed-graph cubic bezier,
    steps(n)             -- stepped (stop-motion / typewriter feel).

camelCase (easeInOutCubic) and kebab-case (ease-in-out-cubic) spellings are accepted.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable

Easing = Callable[[float], float]


def _bezier(x1: float, y1: float, x2: float, y2: float) -> Easing:
    """Cubic bezier easing identical to CSS cubic-bezier()."""
    cx = 3.0 * x1
    bx = 3.0 * (x2 - x1) - cx
    ax = 1.0 - cx - bx
    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by

    def sx(t):
        return ((ax * t + bx) * t + cx) * t

    def sy(t):
        return ((ay * t + by) * t + cy) * t

    def dsx(t):
        return (3.0 * ax * t + 2.0 * bx) * t + cx

    def solve(x):
        t = x
        for _ in range(8):  # Newton
            err = sx(t) - x
            if abs(err) < 1e-7:
                return t
            d = dsx(t)
            if abs(d) < 1e-6:
                break
            t -= err / d
        lo, hi = 0.0, 1.0  # bisection fallback
        t = x
        for _ in range(40):
            v = sx(t)
            if abs(v - x) < 1e-7:
                return t
            if v < x:
                lo = t
            else:
                hi = t
            t = (lo + hi) / 2
        return t

    def f(u):
        if u <= 0:
            return 0.0
        if u >= 1:
            return 1.0
        return sy(solve(u))

    return f


def _out_bounce(u):
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1
_C4 = (2 * math.pi) / 3
_C5 = (2 * math.pi) / 4.5

_BASE: dict[str, Easing] = {
    "linear": lambda u: u,
    "hold": lambda u: 0.0 if u < 1 else 1.0,
    "in_sine": lambda u: 1 - math.cos(u * math.pi / 2),
    "out_sine": lambda u: math.sin(u * math.pi / 2),
    "in_out_sine": lambda u: -(math.cos(math.pi * u) - 1) / 2,
    "in_quad": lambda u: u * u,
    "out_quad": lambda u: 1 - (1 - u) ** 2,
    "in_out_quad": lambda u: 2 * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 2 / 2,
    "in_cubic": lambda u: u ** 3,
    "out_cubic": lambda u: 1 - (1 - u) ** 3,
    "in_out_cubic": lambda u: 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2,
    "in_quart": lambda u: u ** 4,
    "out_quart": lambda u: 1 - (1 - u) ** 4,
    "in_out_quart": lambda u: 8 * u ** 4 if u < 0.5 else 1 - (-2 * u + 2) ** 4 / 2,
    "in_quint": lambda u: u ** 5,
    "out_quint": lambda u: 1 - (1 - u) ** 5,
    "in_out_quint": lambda u: 16 * u ** 5 if u < 0.5 else 1 - (-2 * u + 2) ** 5 / 2,
    "in_expo": lambda u: 0.0 if u == 0 else 2 ** (10 * u - 10),
    "out_expo": lambda u: 1.0 if u == 1 else 1 - 2 ** (-10 * u),
    "in_out_expo": lambda u: 0.0 if u == 0 else 1.0 if u == 1 else (
        2 ** (20 * u - 10) / 2 if u < 0.5 else (2 - 2 ** (-20 * u + 10)) / 2),
    "in_circ": lambda u: 1 - math.sqrt(max(0.0, 1 - u * u)),
    "out_circ": lambda u: math.sqrt(max(0.0, 1 - (u - 1) ** 2)),
    "in_out_circ": lambda u: (1 - math.sqrt(max(0.0, 1 - (2 * u) ** 2))) / 2 if u < 0.5 else (
        math.sqrt(max(0.0, 1 - (-2 * u + 2) ** 2)) + 1) / 2,
    "in_back": lambda u: _C3 * u ** 3 - _C1 * u * u,
    "out_back": lambda u: 1 + _C3 * (u - 1) ** 3 + _C1 * (u - 1) ** 2,
    "in_out_back": lambda u: ((2 * u) ** 2 * ((_C2 + 1) * 2 * u - _C2)) / 2 if u < 0.5 else (
        (2 * u - 2) ** 2 * ((_C2 + 1) * (u * 2 - 2) + _C2) + 2) / 2,
    "in_elastic": lambda u: 0.0 if u == 0 else 1.0 if u == 1 else
    -(2 ** (10 * u - 10)) * math.sin((u * 10 - 10.75) * _C4),
    "out_elastic": lambda u: 0.0 if u == 0 else 1.0 if u == 1 else
    2 ** (-10 * u) * math.sin((u * 10 - 0.75) * _C4) + 1,
    "in_out_elastic": lambda u: 0.0 if u == 0 else 1.0 if u == 1 else (
        -(2 ** (20 * u - 10) * math.sin((20 * u - 11.125) * _C5)) / 2 if u < 0.5 else
        (2 ** (-20 * u + 10) * math.sin((20 * u - 11.125) * _C5)) / 2 + 1),
    "out_bounce": _out_bounce,
    "in_bounce": lambda u: 1 - _out_bounce(1 - u),
    "in_out_bounce": lambda u: (1 - _out_bounce(1 - 2 * u)) / 2 if u < 0.5 else (1 + _out_bounce(2 * u - 1)) / 2,
}

# After Effects style presets
_BASE["ease"] = _bezier(0.333, 0.0, 0.667, 1.0)          # Easy Ease (F9)
_BASE["ease_in"] = _bezier(0.333, 0.0, 1.0, 1.0)         # Easy Ease In (incoming)
_BASE["ease_out"] = _bezier(0.0, 0.0, 0.667, 1.0)        # Easy Ease Out
_BASE["smooth"] = _bezier(0.65, 0.0, 0.35, 1.0)          # strong cinematic S-curve
_BASE["whip"] = _bezier(0.85, 0.0, 0.15, 1.0)            # very fast middle (whip pans)


def _normalize(name: str) -> str:
    n = name.strip()
    n = re.sub(r"(?<!^)(?=[A-Z])", "_", n).lower()  # camelCase -> snake
    n = n.replace("-", "_").replace(" ", "_")
    if n.startswith("ease_") and n[5:] in _BASE:
        n = n[5:]
    return n


@lru_cache(maxsize=512)
def get(name: str | None) -> Easing:
    if name is None:
        return _BASE["linear"]
    m = re.match(r"^\s*(?:cubic[-_]?)?bezier\(([^)]*)\)\s*$", name)
    if m:
        vals = [float(v) for v in m.group(1).split(",")]
        if len(vals) != 4:
            raise ValueError(f"bezier() needs 4 numbers: {name}")
        return _bezier(*vals)
    m = re.match(r"^\s*steps\((\d+)\)\s*$", name)
    if m:
        n = int(m.group(1))
        return lambda u: min(1.0, math.floor(u * n) / n) if u < 1 else 1.0
    key = _normalize(name)
    if key not in _BASE:
        raise ValueError(f"Unknown easing '{name}'. Known: {', '.join(sorted(_BASE))}, bezier(a,b,c,d), steps(n)")
    return _BASE[key]


def names() -> list[str]:
    return sorted(_BASE)
