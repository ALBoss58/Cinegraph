"""Project linting: catches errors before a long render and flags retention/readability problems."""
from __future__ import annotations

import os

import numpy as np

from . import easing, effects as FX, media
from .props import P


def validate(norm: dict, deep: bool = False, min_text_hold: float = 0.5, min_text_px: float = 28.0):
    errors, warnings, info = [], [], []
    for cname, comp in norm["comps"].items():
        H = comp["height"]
        dur = comp["duration"]
        ids = set()
        for i, L in enumerate(comp["layers"]):
            lid = L.get("id", f"#{i}")
            where = f"[{cname}:{lid}]"
            if lid in ids:
                errors.append(f"{where} duplicate layer id")
            ids.add(lid)
            src = L.get("src")
            if L["type"] in ("video", "image"):
                if not src or not os.path.exists(src):
                    errors.append(f"{where} missing media: {src}")
                elif L["type"] == "video" and not media.is_image(src):
                    try:
                        inf = media.probe(src)
                        if "duration" in L and inf.get("duration") and not L.get("loop") and "time_remap" not in L \
                                and "freeze_at" not in L:
                            need = float(L.get("in", 0)) + L["duration"] * abs(float(L.get("speed", 1)))
                            if need > inf["duration"] + 0.05:
                                warnings.append(f"{where} clip needs {need:.2f}s but source is {inf['duration']:.2f}s "
                                                f"(last frame will freeze for {need - inf['duration']:.2f}s)")
                    except media.MediaError as e:
                        errors.append(f"{where} {e}")
            if L["type"] == "model" and not L.get("primitive") and (not src or not os.path.exists(src)):
                errors.append(f"{where} missing model file: {src}")
            if L["type"] == "model" and L.get("primitive") == "terrain" and L.get("heightmap") and not os.path.exists(L["heightmap"]):
                errors.append(f"{where} missing heightmap: {L['heightmap']}")
            if L.get("texture") and not os.path.exists(L["texture"]):
                errors.append(f"{where} missing texture: {L['texture']}")
            if L["type"] == "light" and L.get("light_type", "spot") not in ("ambient", "parallel", "directional", "sun", "point", "spot"):
                errors.append(f"{where} unknown light_type '{L.get('light_type')}'")
            if L["type"] == "light" and L.get("casts_shadows") and L.get("light_type", "spot") in ("point", "ambient"):
                warnings.append(f"{where} {L.get('light_type')} lights don't cast shadows yet (use spot or directional)")
            if "time_remap" in L and L["type"] == "video":
                warnings.append(f"{where} time_remap: clip audio is muted (use speed/reverse for retimed audio)")
            for fx in L.get("effects", []):
                if FX.resolve_name(fx.get("type", "")) not in FX.REGISTRY:
                    errors.append(f"{where} unknown effect '{fx.get('type')}'")
                if FX.resolve_name(fx.get("type", "")) == "lut":
                    p = fx.get("file", "")
                    p = p if os.path.isabs(p) else os.path.join(norm["base_dir"], p)
                    if not os.path.exists(p):
                        errors.append(f"{where} LUT not found: {p}")
            for key in ("transition_in", "transition_out"):
                tr = L.get(key)
                if tr:
                    try:
                        easing.get(tr.get("ease", "in_out_cubic"))
                    except ValueError as e:
                        errors.append(f"{where} {e}")
            # props compile (easing names, expressions)
            for k, v in L.items():
                if isinstance(v, dict) and ("keys" in v or "expr" in v):
                    try:
                        P(v, color=k.endswith("color"))
                    except Exception as e:
                        errors.append(f"{where} property '{k}': {e}")
                elif isinstance(v, str) and v.startswith("="):
                    try:
                        P(v)
                    except Exception as e:
                        errors.append(f"{where} expression '{k}': {e}")
            if L.get("parent") and L["parent"] not in {x.get("id") for x in comp["layers"]}:
                errors.append(f"{where} parent '{L['parent']}' not found")
            if L.get("matte") and L["matte"].get("layer") not in {x.get("id") for x in comp["layers"]}:
                errors.append(f"{where} matte layer '{L['matte'].get('layer')}' not found")
            st = L.get("start", 0)
            d = L.get("duration")
            if st >= dur:
                warnings.append(f"{where} starts at {st:.2f}s after comp end ({dur:.2f}s) — never visible")
            if L["type"] == "text" and not L.get("caption"):
                if d is not None and d < min_text_hold:
                    warnings.append(f"{where} text on screen only {d:.2f}s (< {min_text_hold}s) — too short to read")
                size = L.get("size", 72)
                if isinstance(size, (int, float)) and size * 1080 / H < min_text_px:
                    warnings.append(f"{where} text size {size}px is small for mobile viewers at {H}p")
                words = len(str(L.get("text", "")).split())
                if d is not None and words > 0 and d < 0.3 + words / 4.0:
                    warnings.append(f"{where} {words} words in {d:.2f}s — viewers read ~4 words/s; hold longer")
        if comp.get("motion_blur") and any(L.get("type") in ("model", "light") for L in comp["layers"]):
            warnings.append(f"[{cname}] motion blur is not applied inside the GL 3D renderer (models/lights) yet")
        cam = comp.get("camera")
        if cam and not any(L.get("three_d") or L.get("3d") for L in comp["layers"]):
            warnings.append(f"[{cname}] camera defined but no layer has three_d: true (camera only affects 3D layers)")
    for a in norm["audio"]:
        if not a.get("src") or not os.path.exists(a["src"]):
            errors.append(f"[audio:{a.get('id', '?')}] missing file: {a.get('src')}")
    if not errors:
        try:
            from .compositor import ProjectRuntime
            rt = ProjectRuntime(norm, 0.25)
            if deep:
                comp = rt.main
                dark = []
                for t in np.arange(0, comp.duration, 1.0):
                    fr = comp.render(float(t))
                    if float(fr[..., :3].mean()) < 0.01:
                        dark.append(round(float(t), 2))
                if dark:
                    warnings.append(f"near-black frames at t={dark[:12]}{'...' if len(dark) > 12 else ''}s")
                comp.render(0.0)
        except Exception as e:
            errors.append(f"runtime: {type(e).__name__}: {e}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}
