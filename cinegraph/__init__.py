"""Cinegraph — a free, code-driven compositor and video editor built to be driven by Claude.

Timeline-as-code (JSON or Python) -> frame-accurate renders with keyframes, expressions, 2.5D/3D camera,
effects, color grading, masks, mattes, transitions, text animators, shapes with trim paths,
particles, multitrack audio with ducking and loudness normalization.
"""
__version__ = "0.3.0"

from .builder import Project, keys, kenburns, lower_third, route_line, title_card  # noqa: F401
