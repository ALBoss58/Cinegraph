"""Preset library: looks (effect stacks), text styles, transitions and layer templates.

Use in projects:
  effects: [{"type": "look", "name": "archive_warm", "mix": 0.8}]     -> expands to the look's effects
  layer:   {"type": "text", "style": "title_gold", "text": "..."}     -> style keys are defaults, layer wins
  transition_in: "soft_fade" | "whip_left_fast" | ...                  -> named transition
"""
from __future__ import annotations

import copy

LOOKS = {
    "doc_neutral":   [{"type": "color", "contrast": 0.08, "saturation": 0.95}, {"type": "vignette", "amount": 0.25}],
    "doc_cold":      [{"type": "color", "temperature": -18, "contrast": 0.12, "saturation": 0.85, "lift": [0.0, 0.01, 0.03]},
                      {"type": "vignette", "amount": 0.35}, {"type": "grain", "amount": 0.025}],
    "archive_warm":  [{"type": "color", "temperature": 22, "contrast": 0.1, "saturation": 0.75, "gain": [1.04, 1.0, 0.92]},
                      {"type": "grain", "amount": 0.04}, {"type": "vignette", "amount": 0.45}, {"type": "flicker", "amount": 0.03}],
    "film_1970s":    [{"type": "color", "temperature": 15, "saturation": 0.8, "lift": [0.03, 0.02, 0.0], "contrast": -0.05},
                      {"type": "curves", "master": [[0, 0.05], [0.5, 0.52], [1, 0.95]]},
                      {"type": "grain", "amount": 0.05, "size": 1.6}, {"type": "glow", "threshold": 0.75, "radius": 25, "intensity": 0.3}],
    "old_newsreel":  [{"type": "black_white"}, {"type": "old_film", "amount": 0.9, "sepia": 0.15},
                      {"type": "curves", "master": [[0, 0.04], [0.5, 0.5], [1, 0.92]]}],
    "noir":          [{"type": "black_white"}, {"type": "color", "contrast": 0.35}, {"type": "vignette", "amount": 0.6},
                      {"type": "grain", "amount": 0.04}],
    "anime_cel":     [{"type": "cartoon", "levels": 7, "edge": 0.65}, {"type": "color", "saturation": 1.2, "contrast": 0.06},
                      {"type": "glow", "threshold": 0.7, "radius": 24, "intensity": 0.35}],
    "anime_sunset":  [{"type": "color", "temperature": 35, "saturation": 1.25}, {"type": "gradient_overlay",
                      "colors": ["#ff9a5a", "#3a1c5c"], "angle": 90, "mode": "soft_light", "opacity": 0.6},
                      {"type": "glow", "threshold": 0.6, "radius": 40, "intensity": 0.6}, {"type": "light_leak", "amount": 0.25}],
    "epic_battle":   [{"type": "color", "contrast": 0.2, "saturation": 0.7, "temperature": 8, "shadows": -0.2},
                      {"type": "secondary", "hue": 20, "hue_width": 50, "saturation": 1.3},
                      {"type": "vignette", "amount": 0.5}, {"type": "grain", "amount": 0.03}],
    "dream":         [{"type": "glow", "threshold": 0.4, "radius": 60, "intensity": 0.8}, {"type": "color", "saturation": 0.8, "contrast": -0.1},
                      {"type": "chromatic_aberration", "amount": 3}],
}

TEXT_STYLES = {
    "title_gold":     {"font": "DejaVu Serif", "bold": True, "size": 110, "color": "#f1dfb4", "tracking": 80,
                       "effects": [{"type": "drop_shadow", "distance": 6, "softness": 14, "opacity": 0.6},
                                   {"type": "glow", "threshold": 0.6, "radius": 26, "intensity": 0.35}]},
    "chapter_card":   {"font": "Inter", "bold": True, "size": 84, "color": "#ffffff", "tracking": 300, "all_caps": True,
                       "animator": {"type": "fade", "unit": "char", "stagger": 0.03, "duration": 0.5, "order": "center"}},
    "date_stamp":     {"font": "DejaVu Sans Mono", "size": 38, "color": "#e9e2cf", "tracking": 120,
                       "animator": {"type": "typewriter", "unit": "char", "stagger": 0.05}},
    "quote":          {"font": "DejaVu Serif", "italic": True, "size": 52, "color": "#f2efe6", "max_width": 1300,
                       "line_height": 1.35, "animator": {"type": "rise", "unit": "word", "stagger": 0.09, "duration": 0.5}},
    "map_label":      {"font": "Inter", "bold": True, "size": 30, "color": "#1a1a1a", "tracking": 160, "all_caps": True},
    "stat_big":       {"font": "Inter", "bold": True, "size": 200, "color": "#ffffff",
                       "effects": [{"type": "drop_shadow", "distance": 8, "softness": 20, "opacity": 0.5}]},
}

TRANSITIONS = {
    "soft_fade": {"type": "crossfade", "duration": 0.6, "ease": "in_out_sine"},
    "quick_cut_fade": {"type": "crossfade", "duration": 0.2},
    "dip_black_slow": {"type": "dip_black", "duration": 1.2},
    "flash_hit": {"type": "flash", "duration": 0.25},
    "whip_left_fast": {"type": "whip_left", "duration": 0.35, "ease": "whip"},
    "whip_right_fast": {"type": "whip_right", "duration": 0.35, "ease": "whip"},
    "page_wipe": {"type": "wipe_right", "duration": 0.7, "softness": 160},
    "iris_open": {"type": "iris", "duration": 0.9, "softness": 80},
    "zoom_punch": {"type": "zoom", "duration": 0.3, "amount": 0.35},
    "glitch_cut": {"type": "glitch", "duration": 0.3},
}


def expand_effects(effects):
    out = []
    for e in effects or []:
        if isinstance(e, dict) and e.get("type") == "look":
            name = e.get("name")
            if name not in LOOKS:
                raise ValueError(f"Unknown look '{name}'. Looks: {sorted(LOOKS)}")
            fx = copy.deepcopy(LOOKS[name])
            if "mix" in e:
                for f in fx:
                    f["mix_layer"] = float(e["mix"])
            out.extend(fx)
        else:
            out.append(e)
    return out


def apply_layer_presets(L: dict) -> dict:
    st = L.get("style")
    if st:
        if st not in TEXT_STYLES:
            raise ValueError(f"Unknown style '{st}'. Styles: {sorted(TEXT_STYLES)}")
        base = copy.deepcopy(TEXT_STYLES[st])
        base.update({k: v for k, v in L.items() if k != "style"})
        L = base
    for k in ("transition_in", "transition_out"):
        if isinstance(L.get(k), str):
            if L[k] not in TRANSITIONS:
                raise ValueError(f"Unknown transition preset '{L[k]}'. Presets: {sorted(TRANSITIONS)}")
            L[k] = dict(TRANSITIONS[L[k]])
    if L.get("effects"):
        L["effects"] = expand_effects(L["effects"])
    return L
