"""Project loading: JSON/YAML/Python dict -> normalized comps with expanded sequences."""
from __future__ import annotations

import copy
import json
import os

from .util import parse_time

LAYER_TYPES = {"video", "image", "text", "solid", "shape", "gradient", "comp", "adjustment", "null", "particles", "model", "light"}
OVERLAP_TRANSITIONS = {"crossfade", "fade", "dissolve", "slide_left", "slide_right", "slide_up", "slide_down",
                       "push_left", "push_right", "push_up", "push_down", "wipe", "wipe_left", "wipe_right",
                       "wipe_up", "wipe_down", "iris", "zoom", "zoom_in", "zoom_out", "blur", "whip", "whip_left",
                       "whip_right", "flash", "glitch"}


def load(path_or_dict, base_dir: str | None = None) -> dict:
    if isinstance(path_or_dict, dict):
        data = copy.deepcopy(path_or_dict)
        base = base_dir or os.getcwd()
    else:
        path = os.path.abspath(path_or_dict)
        base = base_dir or os.path.dirname(path)
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml  # optional
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    data["_base_dir"] = base
    return normalize(data)


def _resolve_src(src, assets, base):
    if not src:
        return src
    if isinstance(src, str) and src.startswith("@"):
        key = src[1:]
        if key not in assets:
            raise ValueError(f"Unknown asset '@{key}'")
        src = assets[key]
    if not os.path.isabs(src):
        src = os.path.normpath(os.path.join(base, src))
    return src


def _expand_layers(layers, fps, assets, base):
    out = []
    from .presets import apply_layer_presets
    for i, L in enumerate(layers):
        L = apply_layer_presets(dict(L))
        typ = L.get("type")
        if typ == "sequence":
            out.extend(_expand_sequence(L, fps, assets, base))
            continue
        if typ == "group":  # flatten: children inherit start offset + parent
            off = parse_time(L.get("start", 0), fps)
            if L.get("id"):
                nl = {k: v for k, v in L.items() if k not in ("layers",)}
                nl["type"] = "null"
                nl["start"] = off
                nl.pop("duration", None)
                out.append(nl)
            for c in _expand_layers(L.get("layers", []), fps, assets, base):
                c["start"] = parse_time(c.get("start", 0), fps) + off
                if L.get("id") and not c.get("parent"):
                    c["parent"] = L["id"]
                out.append(c)
            continue
        if typ is None:
            src = L.get("src", "")
            if src:
                ext = os.path.splitext(str(src))[1].lower()
                L["type"] = "image" if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else \
                    "model" if ext in {".obj", ".glb", ".gltf", ".stl", ".ply", ".off", ".dae", ".3mf"} else "video"
            elif "primitive" in L:
                L["type"] = "model"
            elif "light_type" in L:
                L["type"] = "light"
            elif "text" in L:
                L["type"] = "text"
            elif "shapes" in L:
                L["type"] = "shape"
            else:
                raise ValueError(f"Layer {L.get('id', i)} has no type")
        if L["type"] not in LAYER_TYPES:
            raise ValueError(f"Unknown layer type '{L['type']}' (layer {L.get('id', i)}). Known: {sorted(LAYER_TYPES)}")
        for key in ("src", "texture", "heightmap"):
            if key in L and isinstance(L[key], str):
                L[key] = _resolve_src(L[key], assets, base)
        L["start"] = parse_time(L.get("start", 0), fps)
        if "duration" in L:
            L["duration"] = parse_time(L["duration"], fps)
        if "end" in L:
            L["duration"] = parse_time(L.pop("end"), fps) - L["start"]
        out.append(L)
    return out


def _expand_sequence(S, fps, assets, base):
    cursor = parse_time(S.get("start", 0), fps)
    default_tr = S.get("transition")
    clips = S.get("clips", [])
    out = []
    prev = None
    for i, c in enumerate(clips):
        c = dict(c)
        for k, v in (S.get("defaults") or {}).items():
            c.setdefault(k, copy.deepcopy(v))
        tr = c.pop("transition", default_tr) if i > 0 else None
        if "duration" not in c:
            raise ValueError(f"Sequence clip {i} needs a duration")
        dur = parse_time(c["duration"], fps)
        if isinstance(tr, str):
            from .presets import TRANSITIONS
            tr = dict(TRANSITIONS[tr])
        if tr and prev is not None:
            ttype = tr.get("type", "crossfade")
            td = parse_time(tr.get("duration", 0.5), fps)
            if ttype in ("cut", "none"):
                pass
            elif ttype.startswith("dip"):
                half = dict(tr, duration=td / 2)
                prev.setdefault("transition_out", half)
                c.setdefault("transition_in", half)
            else:
                cursor -= td
                c.setdefault("transition_in", dict(tr, duration=td))
                if ttype.startswith(("push", "whip")):
                    prev.setdefault("transition_out", dict(tr, duration=td))
        c["start"] = cursor
        c["duration"] = dur
        cursor += dur
        if S.get("id") and "id" not in c:
            c["id"] = f"{S['id']}_{i}"
        out.append(c)
        prev = c
    return _expand_layers(out, fps, assets, base)


def transcript(path, base, language="en"):
    """Cached word-level transcript of an audio/video file."""
    import hashlib
    st = os.stat(path)
    key = hashlib.md5(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{language}".encode()).hexdigest()[:16]
    d = os.path.join(base, ".cinegraph_cache")
    os.makedirs(d, exist_ok=True)
    cp = os.path.join(d, f"words_{key}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    from .ai import transcribe
    words = transcribe(path, language)
    with open(cp, "w") as f:
        json.dump(words, f)
    return words


def auto_caption_layers(cap, assets, base, comp):
    from .ai import build_captions
    from .captions import cues_to_layers
    src = _resolve_src(cap["auto"], assets, base)
    words = transcript(src, base, cap.get("language", "en"))
    for fix in cap.get("replace", []):  # [["Napolean", "Napoleon"], ...] spelling fixes for names
        for w in words:
            if w["word"].strip(".,!?").lower() == fix[0].lower():
                w["word"] = w["word"].lower().replace(fix[0].lower(), fix[1])
    cues = build_captions(words, cap.get("max_chars", 32), cap.get("max_lines", 2), cap.get("max_dur", 3.2))
    return cues_to_layers(cues, cap.get("style", "clean"), comp, float(cap.get("start", 0.0)), cap.get("id", "acap"))


def _norm_audio(a, assets, base, fps):
    a = dict(a)
    a["src"] = _resolve_src(a.get("src") or a.get("file"), assets, base)
    a["start"] = parse_time(a.get("start", 0), fps)
    a["in"] = parse_time(a.get("in", 0), fps)
    if "duration" in a:
        a["duration"] = parse_time(a["duration"], fps)
    return a


def normalize(data: dict) -> dict:
    base = data.get("_base_dir", os.getcwd())
    assets = data.get("assets", {})
    comp = dict(data.get("comp") or {})
    comp.setdefault("width", 1920)
    comp.setdefault("height", 1080)
    comp.setdefault("fps", 30)
    fps = float(comp["fps"])
    comp["duration"] = parse_time(comp.get("duration", 10), fps)
    comp["name"] = comp.get("name", "main")
    comp["layers"] = _expand_layers(data.get("layers", []), fps, assets, base)
    if "camera" in data and "camera" not in comp:
        comp["camera"] = data["camera"]
    comps = {"main": comp}
    for name, c in (data.get("comps") or {}).items():
        c = dict(c)
        c.setdefault("width", comp["width"])
        c.setdefault("height", comp["height"])
        c.setdefault("fps", fps)
        c["duration"] = parse_time(c.get("duration", comp["duration"]), fps)
        c["name"] = name
        c.setdefault("background", "transparent")
        c["layers"] = _expand_layers(c.get("layers", []), fps, assets, base)
        c["audio"] = [_norm_audio(a, assets, base, fps) for a in c.get("audio", [])]
        comps[name] = c
    audio = [_norm_audio(a, assets, base, fps) for a in data.get("audio", [])]
    # caption import
    for cap in data.get("captions", []) if isinstance(data.get("captions"), list) else ([data["captions"]] if data.get("captions") else []):
        if cap.get("srt"):
            from .captions import srt_to_layers
            comp["layers"].extend(srt_to_layers(_resolve_src(cap["srt"], assets, base), cap.get("style", {}), comp))
        elif cap.get("auto"):
            comp["layers"].extend(auto_caption_layers(cap, assets, base, comp))
    markers = list(data.get("markers", []))
    mb = data.get("markers_from_beats")
    if mb:
        from .audio import beats as _beats
        res = _beats(_resolve_src(mb["src"], assets, base))
        every = int(mb.get("every", 4))
        off = float(mb.get("start", 0))
        for i, b in enumerate(res["beats"][::every]):
            markers.append({"t": round(b + off, 4), "name": f"beat {i * every + 1}", "chapter": False, "beat": True})
    return {"comps": comps, "audio": audio, "audio_settings": data.get("audio_settings", {}),
            "output": data.get("output", {}), "markers": markers, "base_dir": base,
            "assets": assets, "meta": data.get("meta", {})}
