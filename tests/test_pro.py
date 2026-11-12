"""Batch 3: audio processing, beats, text on path, new effects, presets, posterize/echo."""
import os

import numpy as np
import pytest

from cinegraph import audio as AU, effects as FX, presets as PR, project as PJ
from cinegraph.compositor import ProjectRuntime

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")


def rt(layers, **c):
    return ProjectRuntime(PJ.load({"comp": {"width": 480, "height": 270, "duration": 2, **c}, "layers": layers}, MEDIA)).main


def test_beat_tracking(tmp_path):
    sr, bpm, T = 22050, 117, 16
    x = np.zeros(sr * T, np.float32)
    truth = np.arange(0.3, T, 60 / bpm)
    for k, t in enumerate(truth):
        i, n = int(t * sr), int(0.04 * sr)
        x[i:i + n] += 0.6 * np.exp(-np.arange(n) / (0.008 * sr)) * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)
    p = str(tmp_path / "c.wav")
    AU.write_wav(p, np.repeat(x[:, None], 2, 1), sr)
    res = AU.beats(p)
    assert abs(res["bpm"] - bpm) < 1.5
    b = np.array(res["beats"])
    assert np.median([np.min(np.abs(b - t)) for t in truth[2:-2]]) < 0.02


@pytest.mark.parametrize("preset", sorted(AU.VOICE_PRESETS))
def test_voice_presets_run(preset):
    n = PJ.load({"comp": {"duration": 2}, "audio": [{"src": "speech.wav", "voice": preset, "duration": 2}]}, MEDIA)
    o, st = AU.mix(n, 0, 2)
    assert st["has_audio"] and np.isfinite(o).all()


def test_eq_highpass_removes_lows():
    n = PJ.load({"comp": {"duration": 3}, "audio": [{"src": "music.wav", "duration": 3,
                                                       "eq": [{"type": "highpass", "f": 400}]}]}, MEDIA)
    o, _ = AU.mix(n, 0, 3)
    F = np.abs(np.fft.rfft(o[:, 0]))
    fr = np.fft.rfftfreq(len(o), 1 / 48000)
    n2 = PJ.load({"comp": {"duration": 3}, "audio": [{"src": "music.wav", "duration": 3}]}, MEDIA)
    F2 = np.abs(np.fft.rfft(AU.mix(n2, 0, 3)[0][:, 0]))
    assert F[(fr > 120) & (fr < 140)].sum() < 0.1 * F2[(fr > 120) & (fr < 140)].sum()


def test_text_on_path_renders_along_curve():
    pts = [[40, 200], [240, 60], [440, 200]]
    fr = rt([{"type": "text", "text": "ALONG THE ARC", "size": 30, "path": {"points": pts}}]).render(0)
    a = fr[..., 3]
    ys, xs = np.nonzero(a > 0.5)
    assert xs.min() < 150 and xs.max() > 300 and ys.min() < 140  # spans the arc, rises toward the apex


@pytest.mark.parametrize("look", sorted(PR.LOOKS))
def test_looks_expand_and_render(look):
    fr = rt([{"type": "image", "src": "map.jpg", "effects": [{"type": "look", "name": look}]}]).render(0.5)
    assert np.isfinite(fr).all()


def test_text_styles_and_transition_presets():
    for st in PR.TEXT_STYLES:
        assert np.isfinite(rt([{"type": "text", "text": "Hi", "style": st, "transition_in": "soft_fade"}]).render(0.5)).all()


def test_posterize_time_and_echo():
    L = {"type": "solid", "color": "#fff", "size": [20, 20], "position": {"keys": [[0, [40, 135]], [2, [440, 135]]]}}
    c = rt([dict(L, posterize_time=4)])
    assert np.array_equal(c.render(0.30), c.render(0.45))  # same pose within a 1/4 s step
    e = rt([dict(L, echo={"count": 3, "delay": 0.1, "decay": 0.5})]).render(1.0)
    assert (e[..., 0] > 0.05).sum() > (rt([L]).render(1.0)[..., 0] > 0.05).sum()
