import math
import os

import numpy as np
import pytest

from cinegraph import easing, project as PJ
from cinegraph.blend import composite
from cinegraph.compositor import ProjectRuntime
from cinegraph.props import P
from cinegraph.render import to_uint8

HERE = os.path.dirname(__file__)
MEDIA = os.path.join(HERE, "..", "examples", "media")


def test_easing_endpoints():
    for n in easing.names():
        f = easing.get(n)
        assert abs(f(0.0)) < 1e-6, n
        assert abs(f(1.0) - 1.0) < 1e-6, n
    b = easing.get("bezier(0.25,0.1,0.25,1)")
    assert 0.0 < b(0.5) < 1.0
    assert easing.get("easeInOutCubic") is easing.get("in_out_cubic")
    assert easing.get("steps(4)")(0.3) == 0.25


def test_keyframes_and_loop():
    p = P({"keys": [[0, 0], [1, 10]]})
    assert p.at(0.5) == pytest.approx(5)
    assert p.at(2) == pytest.approx(10)
    p = P({"keys": [[0, 0], [1, 10]], "loop": "pingpong"})
    assert p.at(1.5) == pytest.approx(5)
    p = P({"keys": [[0, [0, 0]], [1, [10, 20]]]})
    assert np.allclose(p.at(0.5), [5, 10])
    c = P({"keys": [[0, "#000000"], [1, "#ffffff"]]}, color=True)
    assert np.allclose(c.at(0.5)[:3], 0.5, atol=1e-3)


def test_expressions():
    p = P("=100 + 10*sin(t)")
    assert p.at(math.pi / 2) == pytest.approx(110)
    w = P({"keys": [[0, [960, 540]]], "expr": "value + wiggle(2, 30)"})
    v1, v2 = w.at(0.3), w.at(0.3)
    assert np.allclose(v1, v2)  # deterministic
    assert not np.allclose(w.at(0.3), w.at(1.7))


def test_blend_normal_over():
    dst = np.zeros((4, 4, 4), np.float32)
    dst[..., 3] = 1
    src = np.zeros((2, 2, 4), np.float32)
    src[...] = [0.5, 0, 0, 0.5]  # 50% red premultiplied
    composite(dst, src, 1, 1)
    assert np.allclose(dst[1, 1], [0.5, 0, 0, 1])
    assert np.allclose(dst[0, 0], [0, 0, 0, 1])


def _rt(layers, **comp):
    data = {"comp": {"width": 320, "height": 180, "fps": 30, "duration": 2, **comp}, "layers": layers}
    return ProjectRuntime(PJ.load(data, MEDIA))


def test_3d_at_z0_matches_2d():
    """AE invariant: a 3D layer at z=0 with the default camera must match the 2D render."""
    base = {"type": "image", "src": "map.jpg", "fit": "contain", "scale": 60, "rotation": 12}
    a = _rt([dict(base)]).main.render(0.5)
    b = _rt([dict(base, three_d=True)]).main.render(0.5)
    diff = np.abs(a - b).max()
    assert diff < 0.02, diff


def test_solid_and_position():
    rt = _rt([{"type": "solid", "color": "#ff0000", "size": [20, 20], "position": [50, 60]}])
    fr = rt.main.render(0)
    assert fr[60, 50, 0] > 0.99 and fr[60, 50, 1] < 0.01
    assert fr[60, 80, 0] < 0.01


def test_layer_timing_and_fade():
    rt = _rt([{"type": "solid", "color": "#ffffff", "start": 0.5, "duration": 1.0,
               "transition_in": {"type": "fade", "duration": 0.5, "ease": "linear"}}])
    assert rt.main.render(0.25)[90, 160, 0] < 0.01
    assert rt.main.render(0.75)[90, 160, 0] == pytest.approx(0.5, abs=0.02)
    assert rt.main.render(1.25)[90, 160, 0] > 0.99
    assert rt.main.render(1.6)[90, 160, 0] < 0.01


def test_sequence_expansion():
    n = PJ.load({"comp": {"duration": 10}, "layers": [{"type": "sequence", "transition": {"type": "crossfade", "duration": 1},
                                                         "clips": [{"type": "solid", "duration": 3}, {"type": "solid", "duration": 3},
                                                                   {"type": "solid", "duration": 3}]}]}, MEDIA)
    starts = [L["start"] for L in n["comps"]["main"]["layers"]]
    assert starts == [0, 2, 4]


def test_video_decode_frame_accuracy():
    from cinegraph.media import VideoReader
    r = VideoReader(os.path.join(MEDIA, "clip.mp4"))
    f45 = r.frame(45).copy()
    r.frame(10)  # seek back
    assert np.array_equal(r.frame(45), f45)


def test_dither_output_range():
    fr = np.zeros((4, 4, 4), np.float32)
    fr[..., 3] = 1
    u = to_uint8(fr, False)
    assert u.max() <= 1
