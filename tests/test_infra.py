"""Interchange round trips, editor edit ops (+undo), proxies, resumable renders."""
import json
import os
import shutil

import numpy as np
import pytest

from cinegraph import project as PJ

MEDIA = os.path.join(os.path.dirname(__file__), "..", "examples", "media")
pytest.importorskip("opentimelineio")


def _proj():
    return {"comp": {"width": 640, "height": 360, "fps": 25, "duration": 12},
            "layers": [{"type": "video", "src": "clip.mp4", "id": "A", "start": 0, "duration": 3, "in": 1},
                       {"type": "video", "src": "track.mp4", "id": "B", "start": 3, "duration": 2.4, "in": 0.4},
                       {"type": "image", "src": "map.jpg", "id": "C", "start": 5.4, "duration": 4},
                       {"type": "video", "src": "shaky.mp4", "id": "D", "start": 2, "duration": 2, "in": 0.2}],
            "audio": [{"src": "music.wav", "start": 0, "duration": 10}], "markers": [{"t": 5.4, "name": "Map"}]}


@pytest.mark.parametrize("ext", ["otio", "xml"])
def test_interchange_roundtrip(tmp_path, ext):
    from cinegraph.interchange import export, import_timeline
    p = str(tmp_path / f"t.{ext}")
    export(PJ.load(_proj(), MEDIA), p)
    back = import_timeline(p, 25)
    got = sorted((L["start"], L["duration"], L["in"], os.path.basename(L["src"])) for L in back["layers"])
    want = sorted((L["start"], L["duration"], L.get("in", 0.0), L["src"]) for L in _proj()["layers"])
    for g, w in zip(got, want):
        assert abs(g[0] - w[0]) < 0.021 and abs(g[1] - w[1]) < 0.021 and abs(g[2] - w[2]) < 0.021 and g[3] == w[3]
    assert len(back["audio"]) == 1 and back["markers"][0]["name"] == "Map"


def test_edl_single_track(tmp_path):
    from cinegraph.interchange import export, import_timeline
    p = str(tmp_path / "t.edl")
    export(PJ.load(_proj(), MEDIA), p)
    assert len(import_timeline(p, 25)["layers"]) == 3  # V1 only


def test_editor_ops_and_undo(tmp_path):
    from cinegraph.server import State
    d = _proj()
    d["layers"] = [dict(L, src=os.path.join(MEDIA, L["src"])) for L in d["layers"]]
    d["audio"] = []
    p = tmp_path / "p.json"
    p.write_text(json.dumps(d))
    s = State(str(p))
    assert s.apply({"op": "move", "id": "C", "start": 6.013})["ok"]
    assert json.loads(p.read_text())["layers"][2]["start"] == pytest.approx(6.0, abs=1e-6)  # frame-snapped
    assert s.apply({"op": "trim_start", "id": "A", "start": 0.6})["ok"]  # 15 frames @25fps
    A = json.loads(p.read_text())["layers"][0]
    assert A["in"] == pytest.approx(1.6) and A["duration"] == pytest.approx(2.4)
    assert s.apply({"op": "split", "id": "B", "at": 4.0})["ok"]
    ids = [L["id"] for L in json.loads(p.read_text())["layers"]]
    assert "B_2" in ids
    assert not s.apply({"op": "replace", "id": "C", "layer": {"type": "nope"}})["ok"]
    for _ in range(3):
        assert s.apply({"op": "undo"})["ok"]
    assert json.loads(p.read_text())["layers"] == [dict(L) for L in d["layers"]]


def test_proxy_and_resume(tmp_path):
    from cinegraph.proxy import make_proxy
    from cinegraph.render import render
    src = os.path.join(MEDIA, "clip.mp4")
    pp, _ = make_proxy(src, str(tmp_path))
    from cinegraph.media import probe
    assert probe(pp)["height"] <= 540
    proj = {"comp": {"width": 320, "height": 180, "fps": 30, "duration": 2},
            "layers": [{"type": "video", "src": src}], "audio_settings": {"loudness": None}}
    out = str(tmp_path / "o.mp4")
    st = render(PJ.load(proj, str(tmp_path)), out, preset="draft", workers=1, quiet=True, segment_seconds=0.5)
    assert st["frames"] == 60 and st["segments"] >= 2
    assert not any(n.startswith("o.mp4.parts_") for n in os.listdir(tmp_path))
