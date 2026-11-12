"""Smoke + behaviour tests across every effect, transition, blend mode and layer type."""
import os

import numpy as np
import pytest

from cinegraph import effects as FX, project as PJ
from cinegraph.audio import mix
from cinegraph.blend import MODES
from cinegraph.compositor import ProjectRuntime
from cinegraph.project import OVERLAP_TRANSITIONS

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")


def rt(layers, dur=2, **kw):
    d = {"comp": {"width": 320, "height": 180, "fps": 30, "duration": dur}, "layers": layers, **kw}
    return ProjectRuntime(PJ.load(d, MEDIA))


@pytest.mark.parametrize("name", sorted(n for n in FX.REGISTRY if n != "lut"))
def test_every_effect_runs(name):
    fr = rt([{"type": "image", "src": "map.jpg"}, {"type": "text", "text": "Hi", "size": 60, "effects": [{"type": name}]}]).main.render(0.5)
    assert fr.shape == (180, 320, 4) and np.isfinite(fr).all()


def test_lut_identity(tmp_path):
    n = 5
    p = tmp_path / "id.cube"
    lines = [f"LUT_3D_SIZE {n}"]
    for b in range(n):
        for g in range(n):
            for r in range(n):
                lines.append(f"{r/(n-1)} {g/(n-1)} {b/(n-1)}")
    p.write_text("\n".join(lines))
    a = rt([{"type": "image", "src": "map.jpg"}]).main.render(0)
    b = rt([{"type": "image", "src": "map.jpg", "effects": [{"type": "lut", "file": str(p)}]}]).main.render(0)
    assert np.abs(a - b).max() < 1e-3


@pytest.mark.parametrize("tr", sorted(OVERLAP_TRANSITIONS | {"dip_black", "dip_white"}))
def test_every_transition_runs(tr):
    r = rt([{"type": "image", "src": "map.jpg", "duration": 2, "transition_in": {"type": tr, "duration": 1},
             "transition_out": {"type": tr, "duration": 1}}])
    for t in (0.1, 0.5, 1.5, 1.9):
        assert np.isfinite(r.main.render(t)).all()


@pytest.mark.parametrize("mode", MODES)
def test_blend_modes(mode):
    fr = rt([{"type": "image", "src": "map.jpg"}, {"type": "solid", "color": "#4080c0", "blend": mode, "opacity": 70}]).main.render(0)
    assert np.isfinite(fr).all() and fr[..., :3].max() <= 1.0001


def test_mask_and_matte_and_precomp():
    r = rt([{"type": "solid", "color": "#ffffff", "masks": [{"shape": "ellipse", "center": [160, 90], "size": [100, 100]}]}])
    fr = r.main.render(0)
    assert fr[90, 160, 0] > 0.99 and fr[5, 5, 0] < 0.01
    r = rt([{"type": "solid", "color": "#ff0000", "matte": {"layer": "m", "mode": "alpha"}},
            {"type": "solid", "id": "m", "color": "#fff", "size": [40, 40]}])
    fr = r.main.render(0)
    assert fr[90, 160, 0] > 0.99 and fr[10, 10, 0] < 0.01
    d = {"comp": {"width": 320, "height": 180, "fps": 30, "duration": 2},
         "comps": {"inner": {"layers": [{"type": "solid", "color": "#00ff00", "size": [50, 50]}]}},
         "layers": [{"type": "comp", "comp": "inner", "position": [100, 90]}]}
    fr = ProjectRuntime(PJ.load(d, MEDIA)).main.render(0)
    assert fr[90, 100, 1] > 0.99 and fr[90, 160, 1] < 0.01


def test_parenting_and_3d_camera_and_extrude():
    r = rt([{"type": "null", "id": "rig", "position": [100, 90], "rotation": 90},
            {"type": "solid", "color": "#fff", "size": [10, 10], "parent": "rig", "position": [40, 0]}])
    fr = r.main.render(0)
    assert fr[130, 100, 0] > 0.9  # rotated 90deg: +x becomes +y
    r = rt([{"type": "text", "text": "3D", "size": 60, "three_d": True, "rotation_y": 30, "extrude": {"depth": 20}}],
           camera={"position": [160, 90, -400], "poi": [160, 90, 0], "dof": {"aperture": 4}})
    fr = r.main.render(0.5)
    assert fr[..., 3].max() > 0.9


def test_layer_types_render():
    r = rt([{"type": "gradient", "colors": ["#000", "#fff"], "kind": "radial"},
            {"type": "particles", "kind": "embers", "count": 50},
            {"type": "shape", "shapes": [{"kind": "star", "center": [160, 90], "fill": "#ff0", "stroke": "#000", "stroke_width": 3},
                                          {"kind": "path", "points": [[10, 10], [200, 150]], "stroke": "#f00", "stroke_width": 4,
                                           "arrow": True, "dash": [10, 5], "trim_end": 60}]},
            {"type": "video", "src": "clip.mp4", "scale": 30, "position": [60, 40], "speed": 2},
            {"type": "text", "text": "", "counter": {"from": 0, "to": 100}}])
    assert np.isfinite(r.main.render(1.0)).all()


def test_audio_mix_ducking():
    n = PJ.load({"comp": {"duration": 6}, "audio": [{"src": "music.wav", "role": "music"},
                                                      {"src": "voice.wav", "role": "voice", "start": 0}]}, MEDIA)
    out, st = mix(n, 0, 6)
    assert st["has_audio"] and out.shape == (6 * 48000, 2)
    n2 = PJ.load({"comp": {"duration": 6}, "audio": [{"src": "music.wav", "role": "music"}]}, MEDIA)
    solo, _ = mix(n2, 0, 6)
    # voice is on during 0..1s; music there must be quieter than solo music
    v_only = out[:48000] - mix(PJ.load({"comp": {"duration": 6}, "audio": [{"src": "voice.wav", "role": "voice"}]}, MEDIA), 0, 6)[0][:48000]
    assert np.sqrt((v_only[12000:40000] ** 2).mean()) < 0.7 * np.sqrt((solo[12000:40000] ** 2).mean())


def test_validate_flags_short_text():
    from cinegraph.validate import validate
    n = PJ.load({"comp": {"duration": 5}, "layers": [{"type": "text", "text": "Too fast to read this", "duration": 0.3}]}, MEDIA)
    res = validate(n)
    assert res["ok"] and any("too short" in w or "words" in w for w in res["warnings"])
