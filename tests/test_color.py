"""Colour management, secondaries, shot match, scopes, HDR."""
import os

import numpy as np
import pytest

from cinegraph import project as PJ
from cinegraph.compositor import ProjectRuntime
from cinegraph.media import load_image

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")
C = pytest.importorskip("cinegraph.color")
pytest.importorskip("PyOpenColorIO")


def rt(layers, **comp):
    return ProjectRuntime(PJ.load({"comp": {"width": 480, "height": 270, "duration": 1, **comp}, "layers": layers}, MEDIA)).main


def test_linear_workflow_roundtrips_opaque_media():
    a = rt([{"type": "image", "src": "map.jpg"}]).render(0)
    for w in ("linear", "acescg"):
        b = rt([{"type": "image", "src": "map.jpg"}], color={"working": w}).render(0)
        assert np.abs(a - b)[..., :3].mean() < 2e-3


def test_log_footage_roundtrip(tmp_path):
    import cv2
    src = load_image(os.path.join(MEDIA, "map.jpg"))[..., :3].astype(np.float32) / 255
    lin = C.convert(src, C.SRGB, "Linear Rec.709 (sRGB)")
    log = C.convert(lin, "Linear Rec.709 (sRGB)", "ARRI LogC4")
    p = str(tmp_path / "log.png")
    cv2.imwrite(p, cv2.cvtColor((np.clip(log, 0, 1) * 65535 + .5).astype(np.uint16), cv2.COLOR_RGB2BGR))
    out = ProjectRuntime(PJ.load({"comp": {"width": 1920, "height": 1080, "duration": 1, "color": {"working": "acescg"}},
                                  "layers": [{"type": "image", "src": p, "colorspace": "logc4"}]}, MEDIA)).main.render(0)[..., :3]
    assert np.abs(out - src).mean() < 1e-3


def test_secondary_isolates_hue():
    # red square + blue square; secondary desaturates only red
    L = [{"type": "solid", "color": "#d02020", "size": [100, 100], "position": [120, 135]},
         {"type": "solid", "color": "#2040d0", "size": [100, 100], "position": [360, 135]},
         {"type": "adjustment", "effects": [{"type": "secondary", "hue": 0, "hue_width": 30, "saturation": 0.0, "blur_key": 0}]}]
    fr = rt(L).render(0)
    red, blue = fr[135, 120, :3], fr[135, 360, :3]
    assert np.ptp(red) < 0.05          # red became grey
    assert blue[2] - blue[0] > 0.5      # blue untouched


def test_match_color_moves_stats():
    ref = os.path.join(MEDIA, "painting.png")
    a = rt([{"type": "image", "src": "map.jpg"}]).render(0)[..., :3]
    b = rt([{"type": "image", "src": "map.jpg", "effects": [{"type": "match_color", "reference": ref}]}]).render(0)[..., :3]
    r = load_image(ref)[..., :3].astype(np.float32).mean((0, 1)) / 255
    assert np.abs(b.mean((0, 1)) - r).sum() < np.abs(a.mean((0, 1)) - r).sum()


def test_scopes_image():
    img = (np.random.default_rng(0).random((90, 160, 3)) * 255).astype(np.uint8)
    s = C.scopes(img, (256, 128))
    assert s.shape == (256, 512, 3) and s.max() > 100
