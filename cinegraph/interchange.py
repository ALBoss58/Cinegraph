"""Timeline interchange with Premiere Pro / DaVinci Resolve / Final Cut / Avid via OpenTimelineIO.

export: .otio (lossless timeline), .xml (FCP7 XML: Premiere + Resolve import), .edl (CMX3600, one video track)
import: same formats -> Cinegraph project dict (clips, gaps, in-points, dissolves, audio, markers).
Only editorial data travels (clips, timing, tracks, dissolves, markers); effects/3D/text stay in Cinegraph."""
from __future__ import annotations

import math
import os

from . import media


def _otio():
    import opentimelineio as otio
    return otio


def _rt(otio, sec, rate):
    return otio.opentime.RationalTime(math.floor(sec * rate + 0.5 + 1e-9), rate)


def _range(otio, start, dur, rate):
    return otio.opentime.TimeRange(_rt(otio, start, rate), _rt(otio, dur, rate))


def _assign_tracks(items):
    """Greedy: put non-overlapping items on the same track (lowest free)."""
    tracks = []
    for it in sorted(items, key=lambda x: x["start"]):
        for tr in tracks:
            if tr[-1]["start"] + tr[-1]["duration"] <= it["start"] + 1e-6:
                tr.append(it)
                break
        else:
            tracks.append([it])
    return tracks


def to_timeline(norm: dict, name="Cinegraph"):
    otio = _otio()
    comp = norm["comps"]["main"]
    rate = float(comp["fps"])
    tl = otio.schema.Timeline(name=name)
    tl.global_start_time = _rt(otio, 0, rate)
    vids = []
    for L in comp["layers"]:
        if L["type"] not in ("video", "image") or L.get("enabled") is False:
            continue
        dur = L.get("duration") or (comp["duration"] - L["start"])
        vids.append({"src": L["src"], "start": float(L["start"]), "duration": float(dur), "in": float(L.get("in", 0)),
                     "name": L.get("id", os.path.basename(L["src"])), "image": L["type"] == "image",
                     "speed": float(L.get("speed", 1.0)), "tin": L.get("transition_in")})
    auds = [{"src": a["src"], "start": float(a["start"]), "duration": float(a.get("duration") or
                                                                          max(0.0, (media.probe(a["src"]).get("duration") or 0) - a.get("in", 0))),
             "in": float(a.get("in", 0)), "name": a.get("id", os.path.basename(a["src"]))} for a in norm["audio"]]

    def fill(track, items, kind):
        """Absolute frame quantization (no cumulative rounding drift)."""
        F = lambda sec: int(math.floor(sec * rate + 0.5 + 1e-9))  # half-up, no banker's rounding
        cursor_f = 0
        for it in items:
            s_f = F(it["start"])
            e_f = max(s_f + 1, F(it["start"] + it["duration"]))
            if s_f > cursor_f:
                track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(s_f - cursor_f, rate))))
            ref = otio.schema.ExternalReference(target_url=os.path.abspath(it["src"]))
            try:
                info = media.probe(it["src"])
                if info.get("duration"):
                    ref.available_range = _range(otio, 0, info["duration"], rate)
            except Exception:
                pass
            src_frames = max(1, int(math.floor((e_f - s_f) * it.get("speed", 1.0) + 0.5)))
            clip = otio.schema.Clip(name=str(it["name"]), media_reference=ref,
                                    source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(F(it["in"]), rate),
                                                                         otio.opentime.RationalTime(src_frames, rate)))
            if abs(it.get("speed", 1.0) - 1) > 1e-6:
                clip.effects.append(otio.schema.LinearTimeWarp(time_scalar=it["speed"]))
            track.append(clip)
            cursor_f = e_f

    vtracks = _assign_tracks(vids)
    for i, items in enumerate(vtracks):
        tr = otio.schema.Track(name=f"V{i + 1}", kind=otio.schema.TrackKind.Video)
        fill(tr, items, "video")
        tl.tracks.append(tr)
    for i, items in enumerate(_assign_tracks(auds)):
        tr = otio.schema.Track(name=f"A{i + 1}", kind=otio.schema.TrackKind.Audio)
        fill(tr, items, "audio")
        tl.tracks.append(tr)
    for m in norm.get("markers", []):
        tl.tracks.markers.append(otio.schema.Marker(name=str(m.get("name", "")),
                                                    marked_range=_range(otio, float(m.get("t", 0)), 0, rate)))
    return tl


def export(norm: dict, path: str):
    otio = _otio()
    tl = to_timeline(norm, os.path.splitext(os.path.basename(path))[0])
    ext = os.path.splitext(path)[1].lower()
    if ext == ".edl":
        v = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        if len(v) > 1:
            print(f"[cinegraph] EDL holds one video track: exporting V1 of {len(v)} (use .xml or .otio for all)")
        single = otio.schema.Timeline(name=tl.name)
        single.global_start_time = tl.global_start_time
        if v:
            single.tracks.append(v[0].clone())
        otio.adapters.write_to_file(single, path, adapter_name="cmx_3600")
    elif ext == ".xml":
        otio.adapters.write_to_file(tl, path, adapter_name="fcp_xml")
    else:
        otio.adapters.write_to_file(tl, path)
    return path


def import_timeline(path: str, fps: float | None = None, width=1920, height=1080) -> dict:
    """Read .otio/.xml/.edl -> Cinegraph project dict."""
    otio = _otio()
    ext = os.path.splitext(path)[1].lower()
    kw = {}
    if ext == ".edl":
        kw["rate"] = fps or 30
    tl = otio.adapters.read_from_file(path, **kw)
    layers, audio, markers = [], [], []
    rate = fps
    end_all = 0.0
    vidx = 0
    for track in tl.tracks:
        is_audio = track.kind == otio.schema.TrackKind.Audio
        prev_layer = None
        for item in track:
            if isinstance(item, otio.schema.Transition):
                continue
            rng = track.range_of_child(item)
            start = rng.start_time.to_seconds()
            dur = rng.duration.to_seconds()
            rate = rate or rng.start_time.rate
            end_all = max(end_all, start + dur)
            if not isinstance(item, otio.schema.Clip):
                prev_layer = None
                continue
            ref = item.media_reference
            src = getattr(ref, "target_url", None) or item.name
            if src and src.startswith("file://"):
                from urllib.parse import unquote, urlparse
                src = unquote(urlparse(src).path)
            sin = item.source_range.start_time.to_seconds() if item.source_range else 0.0
            if ref is not None and getattr(ref, "available_range", None) is not None and item.source_range:
                sin -= ref.available_range.start_time.to_seconds()
            speed = 1.0
            for fx in item.effects:
                if isinstance(fx, otio.schema.LinearTimeWarp):
                    speed = float(fx.time_scalar)
            if is_audio:
                audio.append({"src": src, "start": round(start, 6), "in": round(sin, 6), "duration": round(dur, 6),
                              "id": item.name or None})
                continue
            vidx += 1
            L = {"type": "image" if media.is_image(src) else "video", "src": src, "id": f"v{vidx}_{(item.name or 'clip')[:24]}",
                 "start": round(start, 6), "duration": round(dur, 6), "in": round(max(0.0, sin), 6)}
            if abs(speed - 1) > 1e-6:
                L["speed"] = speed
            # in-track dissolve before this clip -> transition_in (clip overlaps the previous by the dissolve)
            i = list(track).index(item)
            if i > 0 and isinstance(track[i - 1], otio.schema.Transition):
                trn = track[i - 1]
                d = (trn.in_offset + trn.out_offset).to_seconds()
                L["start"] = round(start - trn.in_offset.to_seconds(), 6)
                L["duration"] = round(dur + trn.in_offset.to_seconds(), 6)
                L["in"] = round(max(0.0, sin - trn.in_offset.to_seconds()), 6)
                L["transition_in"] = {"type": "crossfade", "duration": round(d, 6)}
            layers.append(L)
            prev_layer = L
    # V1 is the bottom track in NLEs -> draw first (our list is bottom -> top): already in order
    for m in tl.tracks.markers:
        markers.append({"t": round(m.marked_range.start_time.to_seconds(), 6), "name": m.name, "chapter": True})
    return {"comp": {"width": width, "height": height, "fps": float(rate or 30), "duration": round(end_all, 6)},
            "layers": layers, "audio": audio, "markers": markers}
