"""3D v0.3: skeletal glTF animation, PBR + environment, point-light shadows, GL motion blur."""
import os

import numpy as np
import pytest

from cinegraph import project as PJ
from cinegraph.compositor import ProjectRuntime

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")
pytest.importorskip("moderngl")
CESIUM = os.path.join(MEDIA, "gltf", "CesiumMan.glb")


def rt(layers, **comp):
    c = {"width": 320, "height": 240, "fps": 30, "duration": 2, **comp}
    return ProjectRuntime(PJ.load({"comp": c, "layers": layers}, MEDIA)).main


@pytest.mark.skipif(not os.path.exists(CESIUM), reason="CesiumMan sample not downloaded")
def test_skinned_animation_moves_and_stays_grounded():
    from cinegraph import gltf_anim
    g = gltf_anim.load(CESIUM)
    a = np.vstack([p[0] for p in g.geometry(0.0)])
    b = np.vstack([p[0] for p in g.geometry(0.5)])
    assert np.abs(a - b).max() > 0.05 * np.ptp(a, axis=0).max()  # limbs actually move
    c = rt([{"src": CESIUM, "size": 150, "position": [160, 200, 0]}])
    f0, f1 = c.render(0.0), c.render(0.5)
    assert f0[..., 3].sum() > 100 and np.abs(f0 - f1).sum() > 50


def test_point_light_shadow_darkens_floor():
    base = [{"primitive": "plane", "size": [2000, 2000], "position": [160, 200, 300], "color": "#ffffff"},
            {"type": "light", "light_type": "point", "position": [160, 0, 300], "intensity": 120, "casts_shadows": True}]
    cam = {"position": [160, -200, -300], "poi": [160, 200, 300], "fov": 60}
    a = rt(base, camera=cam).render(0)
    b = rt(base + [{"primitive": "sphere", "radius": 40, "position": [160, 120, 300]}], camera=cam).render(0)
    d = a[..., :3].mean(-1) - b[..., :3].mean(-1)
    assert d.max() > 0.3


def test_environment_reflection_changes_metal():
    sph = [{"primitive": "sphere", "radius": 80, "position": [160, 120, 0], "color": "#ffffff",
            "material": {"metallic": 1.0, "roughness": 0.1}},
           {"type": "light", "light_type": "ambient", "intensity": 5}]
    no_env = rt(sph).render(0)
    env = rt(sph, environment={"src": "sky.png", "intensity": 1.0}).render(0)
    assert env[120, 160, :3].mean() > no_env[120, 160, :3].mean() + 0.1


def test_gl_motion_blur_smears_fast_object():
    L = [{"primitive": "cube", "size": 40, "position": {"keys": [[0, [40, 120, 0]], [1, [280, 120, 0]]]}}]
    sharp = rt(L, background="transparent").render(0.5)
    blur = rt(L, background="transparent", motion_blur={"samples": 8, "shutter": 180}).render(0.5)
    cols = lambda f: (f[..., 3].max(0) > 0.02).sum()
    assert cols(blur) > cols(sharp) + 3
