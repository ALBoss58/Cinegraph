"""AI tools: cut-out vs rendered ground truth, clean-plate object removal, speech-to-text + captions."""
import json
import os

import numpy as np
import pytest

from cinegraph import project as PJ
from cinegraph.compositor import ProjectRuntime
from cinegraph.render import to_uint8

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")
ai = pytest.importorskip("cinegraph.ai")
pytest.importorskip("onnxruntime")
pytest.importorskip("moderngl")


def test_cutout_matches_ground_truth():
    objs = [{"src": "pawn.glb", "size": 380, "position": [560, 380, 0], "rotation_y": 30},
            {"primitive": "torus", "radius": 110, "tube": 40, "position": [820, 400, -50], "rotation_x": 60, "color": "#d4af37"},
            {"type": "light", "light_type": "directional", "position": [200, 0, -800], "poi": [640, 360, 0], "intensity": 110},
            {"type": "light", "light_type": "ambient", "intensity": 30}]
    c = {"width": 1280, "height": 720, "duration": 1}
    gt = ProjectRuntime(PJ.load({"comp": dict(c, background="transparent"), "layers": objs}, MEDIA)).main.render(0)[..., 3] > 0.5
    bg = [{"type": "image", "src": "painting.png", "effects": [{"type": "gaussian_blur", "radius": 24},
                                                                 {"type": "color", "saturation": 0.35, "exposure": -0.8}]}]
    img = to_uint8(ProjectRuntime(PJ.load({"comp": c, "layers": bg + objs}, MEDIA)).main.render(0), False, dither=False)
    pr = ai.segment(img, "fast") > 0.5
    assert (pr & gt).sum() / (pr | gt).sum() > 0.9


def test_cutout_layer_makes_alpha():
    r = ProjectRuntime(PJ.load({"comp": {"width": 320, "height": 180, "duration": 1, "background": "transparent"},
                                "layers": [{"type": "image", "src": "map.jpg", "cutout": {"model": "fast"}}]}, MEDIA))
    a = r.main.render(0)[..., 3]
    assert a.min() < 0.5  # something was cut away


def test_clean_plate_object_removal():
    path = os.path.join(MEDIA, "pan_person.mp4")
    if not os.path.exists(path):
        pytest.skip("pan_person.mp4 missing")
    rem = [{"shape": "ellipse", "center": {"keys": [[0, [200, 300]], [3, [560, 300]]]}, "size": [95, 235], "expand": 3}]
    r = ProjectRuntime(PJ.load({"comp": {"width": 960, "height": 540, "fps": 30, "duration": 3},
                                "layers": [{"type": "video", "src": "pan_person.mp4", "remove": rem}]}, MEDIA)).main
    out = to_uint8(r.render(1.5), False, dither=False)
    reg = out[200:400, 340:420]
    assert reg.mean(-1).min() > 40 or (reg.mean(-1) < 30).mean() < 0.02  # the dark figure is gone


def test_transcribe_and_captions():
    words = ai.transcribe(os.path.join(MEDIA, "speech.wav"))
    text = " ".join(w["word"] for w in words).lower()
    assert "direct consequence of the sin" in text and "blessed soul in heaven" in text
    assert all(w["end"] > w["start"] for w in words)
    assert words[0]["start"] < words[5]["start"] < words[-1]["start"]
    cues = ai.build_captions(words)
    assert all(len(line) <= 32 for c in cues for line in c["text"].split("\n"))
    assert all(c["end"] - c["start"] >= 0.7 for c in cues)
    assert len(cues[-1]["words"]) > 1  # no orphan word cue
