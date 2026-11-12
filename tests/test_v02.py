"""v0.2: GL 3D (models, lights, shadows), tracking, stabilization, flow slow-mo, denoise, audio gaps."""
import json
import os

import numpy as np
import pytest

from cinegraph import motion as MO, media, project as PJ
from cinegraph.audio import mix
from cinegraph.compositor import ProjectRuntime

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")
gl = pytest.importorskip("moderngl")


def rt(d):
    return ProjectRuntime(PJ.load(d, MEDIA))


def test_gl_model_light_shadow():
    base = {"comp": {"width": 320, "height": 180, "fps": 30, "duration": 1,
                     "camera": {"position": [160, -150, -300], "poi": [160, 140, 200], "fov": 60}},
            "layers": [{"id": "floor", "primitive": "plane", "size": [800, 800], "position": [160, 160, 200], "color": "#ffffff"},
                       {"id": "box", "primitive": "cube", "size": 60, "position": [160, 120, 200], "color": "#ff0000"},
                       {"type": "light", "light_type": "directional", "position": [160, -500, 200], "poi": [160, 160, 200],
                        "intensity": 100, "casts_shadows": True},
                       {"type": "light", "light_type": "ambient", "intensity": 20}]}
    fr = rt(base).main.render(0)
    assert fr[..., 0].max() > 0.5  # red cube visible & lit
    # shadow: straight down under the cube is darker than open floor
    no_box = dict(base, layers=[L for L in base["layers"] if L.get("id") != "box"])
    a = rt(no_box).main.render(0)
    diff = a[..., :3].mean(-1) - fr[..., :3].mean(-1)
    assert diff.max() > 0.3  # cube + its shadow darkened some floor


def test_gl_formats_and_primitives():
    lay = [{"src": "pawn.glb", "size": 100, "position": [80, 90, 0]},
           {"primitive": "terrain", "heightmap": "map.jpg", "texture": "map.jpg", "size": [200, 200], "height": 30,
            "position": [220, 120, 0], "rotation_x": -30},
           {"primitive": "torus", "position": [160, 40, 0], "material": {"toon": 3, "outline": 2}}]
    for p in ("sphere", "cylinder", "cone", "capsule"):
        lay.append({"primitive": p, "position": [40, 150, 0], "radius": 20, "height": 40})
    fr = rt({"comp": {"width": 320, "height": 180, "duration": 1}, "layers": lay}).main.render(0)
    assert np.isfinite(fr).all() and fr[..., 3].max() > 0.9


def test_gl_cards_intersect_depth():
    """Two crossing 3D cards must intersect (true z-buffer), not paint over each other."""
    d = {"comp": {"width": 200, "height": 200, "duration": 1, "renderer": "gl"},
         "layers": [{"type": "solid", "color": "#ff0000", "size": [160, 160], "three_d": True, "rotation_y": 40},
                    {"type": "solid", "color": "#0000ff", "size": [160, 160], "three_d": True, "rotation_y": -40}]}
    fr = rt(d).main.render(0)
    left, right = fr[100, 40, :3], fr[100, 160, :3]
    assert (left[0] > left[2]) != (right[0] > right[2])  # different card wins on each side


def test_point_track_accuracy(tmp_path):
    truth = np.array(json.load(open(os.path.join(MEDIA, "track_truth.json"))))
    tr = MO.track_points(os.path.join(MEDIA, "track.mp4"), [truth[0].tolist()], 0, None, str(tmp_path))
    pts = np.array([p[0] for p in tr["points"]])
    assert np.linalg.norm(pts - truth[:len(pts)], axis=1).mean() < 1.0


def test_tracked_position_and_corner_pin():
    truth = np.array(json.load(open(os.path.join(MEDIA, "track_truth.json"))))
    d = {"comp": {"width": 960, "height": 540, "fps": 30, "duration": 3},
         "layers": [{"id": "plate", "type": "video", "src": "track.mp4"},
                    {"type": "solid", "color": "#ff00ff", "size": [100, 100],
                     "corner_pin": {"track_planar": {"layer": "plate", "quad": [(truth[0] + o).tolist() for o in
                                                                                  [(-40, -40), (40, -40), (40, 40), (-40, 40)]]}}},
                    {"type": "solid", "color": "#00ff00", "size": [6, 6],
                     "position": {"track": {"layer": "plate", "point": truth[0].tolist()}}}]}
    r = rt(d)
    fr = r.main.render(2.0)
    x, y = truth[60]
    assert fr[int(round(y)), int(round(x)), 1] > 0.9          # tracked dot sits on the target
    px = fr[int(round(y)) + 25, int(round(x)) + 25]
    assert px[0] > 0.9 and px[2] > 0.9 and px[1] < 0.1        # corner-pinned square covers the tracked plane


def test_stabilizer_reduces_jitter(tmp_path):
    import cv2
    path = os.path.join(MEDIA, "shaky.mp4")
    st = MO.Stabilizer(path, {"smoothness": 1.0}, str(tmp_path))
    r = media.VideoReader(path)

    def jitter(apply):
        prev, ds = None, []
        for i in range(40):
            f = r.frame(i).copy()
            if apply:
                f = st.apply(f, i)
            g = cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)[60:-60, 60:-60]
            if prev is not None:
                ds.append(cv2.phaseCorrelate(prev, g)[0])
            prev = g
        d = np.array(ds)
        return np.std(d - d.mean(0), 0).mean()
    assert jitter(True) < 0.3 * jitter(False)


def test_flow_interp_and_denoise():
    r = media.VideoReader(os.path.join(MEDIA, "motion30.mp4"))
    f0, f1 = r.frame(15).copy(), r.frame(16).copy()
    out = MO.interpolate(f0, f1, 0.5)
    assert out.shape == f0.shape
    rn = media.VideoReader(os.path.join(MEDIA, "noisy.mp4"))
    c = rn.frame(10).copy()
    d = MO.temporal_denoise(c, [rn.frame(j).copy() for j in (8, 9, 11, 12)], 0.8)
    assert MO.estimate_noise(d) < 0.6 * MO.estimate_noise(c)


def test_slowmo_layer_renders():
    d = {"comp": {"width": 480, "height": 270, "fps": 30, "duration": 1},
         "layers": [{"type": "video", "src": "motion30.mp4", "speed": 0.25, "frame_blend": "flow",
                     "denoise": {"mode": "temporal", "radius": 1}, "stabilize": True}]}
    r = rt(d)
    assert np.isfinite(r.main.render(0.5)).all()


def _dom(o):
    x = o[:, 0]
    f = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / 48000)[f.argmax()]


def test_audio_retime_reverse_precomp():
    n = PJ.load({"comp": {"duration": 4}, "layers": [{"type": "video", "src": "clip.mp4", "speed": 2}]}, MEDIA)
    o, _ = mix(n, 0, 4)
    assert abs(_dom(o[:48000]) - 440) < 5  # pitch preserved
    n = PJ.load({"comp": {"duration": 4}, "layers": [{"type": "video", "src": "clip.mp4", "speed": 2, "pitch": "tape"}]}, MEDIA)
    assert abs(_dom(mix(n, 0, 4)[0][:48000]) - 880) < 8
    a = mix(PJ.load({"comp": {"duration": 2}, "audio": [{"src": "voice.wav", "duration": 1.5}]}, MEDIA), 0, 1.5)[0]
    b = mix(PJ.load({"comp": {"duration": 2}, "audio": [{"src": "voice.wav", "duration": 1.5, "reverse": True}]}, MEDIA), 0, 1.5)[0]
    assert np.allclose(a[2000:-2000], b[::-1][2000:-2000], atol=0.02)
    d = {"comp": {"duration": 5}, "comps": {"inner": {"duration": 3, "layers": [{"type": "video", "src": "clip.mp4"}]}},
         "layers": [{"type": "comp", "comp": "inner", "start": 2, "duration": 3}]}
    o, _ = mix(PJ.load(d, MEDIA), 0, 5)
    assert np.abs(o[:2 * 48000]).max() < 1e-4 and np.abs(o[3 * 48000:4 * 48000]).max() > 0.05
