# Cinegraph

Free (MIT), code-driven compositor and video editor. You describe the edit as **JSON or Python**
(timeline-as-code); Cinegraph renders it frame-accurately. Built so Claude can write, review and fix edits.

```
scripts/install.sh  |  scripts/install.ps1    # one-command install (ffmpeg + venv + all features)
pip install -e .[ai,full]   # manual: Python 3.9+, ffmpeg + ffprobe on PATH
cinegraph validate edit.json
cinegraph serve edit.json   # editor UI at http://127.0.0.1:8765: drag/trim/split clips, properties, undo, scopes
cinegraph contact edit.json -o sheet.png --every 2     # frame grid for fast review
cinegraph still edit.json -t 12.5 -o frame.png
cinegraph render edit.json -o out.mp4 --preset youtube  # final|youtube|draft|hevc|prores|prores4444|webm_alpha|png|gif
cinegraph info              # every effect + params, blend mode, easing, transition
cinegraph track clip.mp4 --point 640,360 -o track.json      # or --quad x1,y1,...,x4,y4
cinegraph stabilize shaky.mp4 -o steady.mp4 [--lock]
cinegraph slowmo clip.mp4 -o slow.mp4 --speed 0.25          # optical-flow in-betweens
cinegraph denoise grainy.mp4 -o clean.mp4 --strength 0.6 --audio 0.5
cinegraph transcribe vo.wav -o vo.srt          # offline speech-to-text, word timings
cinegraph cutout photo.png -o subject.png      # AI background removal
cinegraph scopes edit.json -t 12 -o scopes.png # waveform, parade, vectorscope, histogram
cinegraph beats music.wav                      # BPM + beat times
cinegraph loudness edit.json                   # LUFS / LRA / true peak of the mix
cinegraph export edit.json -o cut.xml          # to Premiere / Resolve (.xml), .edl, .otio
cinegraph import cut.xml -o edit.json          # from Premiere / Resolve / Avid
cinegraph proxy edit.json                      # 540p proxies for instant scrubbing
cinegraph presets                              # looks, text styles, transitions, voice + caption presets
```

## What's in 0.3

| Area | Features |
|---|---|
| Timeline (Premiere) | layers with start/duration/in, `sequence` tracks with auto-overlapping transitions, speed, reverse, freeze, loop, time-remap, groups, markers → YouTube chapters, SRT import/export |
| Motion (After Effects) | keyframes on any property, 30+ easings + `bezier()` speed curves + `steps()`, Catmull-Rom motion paths, loops (cycle/pingpong/continue), **expressions** (`wiggle`, `noise`, `linear`, `ease`, …), parenting/nulls, precomps, masks (add/subtract/intersect, feather, expansion), track mattes (alpha/luma/inverted), adjustment layers, 13 blend modes, motion blur (adaptive) |
| 3D (AE camera / Blender-lite) | 2.5D layers in true perspective, two-node and one-node cameras, fov/zoom/focal length, depth of field, depth fog, camera shake, camera rig parenting, **extruded 3D text/shapes**, near-plane clipping, depth sorting |
| 3D v0.3 | **glTF skeletal + node animation** (`"animation": "Run"`, `anim_speed`, `anim_offset`), **PBR** `material: {"metallic", "roughness"}`, **environment** reflections/IBL + sky background (`"environment": {"src": "sky.hdr", "background": true}`), **point-light shadows**, motion blur in 3D |
| 3D models & lighting (OpenGL) | `model` layers: OBJ / GLB / glTF / STL / PLY (textures, vertex colours) or primitives (cube, sphere, cylinder, cone, torus, capsule, plane, **heightmap terrain** for 3D maps); `light` layers: ambient, directional, point, spot (cone + feather, falloff); **shadow maps with soft PCF**; Blinn-Phong or **toon/cel shading + ink outlines**; rim light; cards, text and models share one z-buffer (true intersections); depth-based DOF + fog; MSAA. GPU when available, headless CPU (llvmpipe) otherwise |
| Tracking (AE/Mocha-lite) | point tracker (LK + template re-anchor, ~0.4 px error on tests), planar tracker → **corner pin**, any position can follow a track: `"position": {"track": {"layer": "plate", "point": [x, y]}}` |
| Stabilization (Warp Stabilizer-lite) | `"stabilize": {"smoothness": 1.0}` or `"mode": "lock"`, auto crop-zoom; jitter 7.8 → 0.75 px on tests |
| Retiming | `"frame_blend": "flow"` optical-flow interpolation (smooth slow motion) or `"blend"` frame mix |
| AI | **cut-out** `"cutout": {"model": "general"|"human"|"fast"}` (IoU 0.97 on tests, temporally smoothed); **object removal** `"remove": [mask shapes]` rebuilds real background from other frames (error 3.8 vs 23.1 for inpainting); **auto-captions** `"captions": {"auto": "vo.wav", "style": "karaoke"|"pop"|"clean"|"outline"|"documentary", "replace": [["Napolean","Napoleon"]]}` (offline sherpa-onnx; faster-whisper used when installed) |
| Colour management | OpenColorIO + ACES 2.0: camera log in (`"colorspace": "logc4"|"slog3"|"vlog"|"applelog"|...`, 16-bit decode), `"color": {"working": "acescg"|"linear", "view": "aces"|"standard"}`, **HDR10 export** (`--preset hdr10`), scopes, `secondary` HSL qualifier, `match_color` shot match, `auto_balance` |
| Noise reduction | video `"denoise": {"mode": "temporal"|"nlm"|"bilateral", "strength": 0.5}` (motion-compensated temporal, +5 dB on tests); audio `"denoise": 0.5` (FFT) |
| Presets & extras | looks (`{"type": "look", "name": "archive_warm"}`: doc_cold, archive_warm, film_1970s, old_newsreel, noir, anime_cel, anime_sunset, epic_battle, dream…), text `style`s, named transitions; **text on path**; effects lens_flare, glitch, twirl, bulge, mirror, kaleidoscope, edge_glow, halftone, emboss; `posterize_time` (on-twos anime motion), `echo` trails |
| Text | any font (path or family), wrap, tracking, line height, stroke, gradient fill, boxes, per-char/word/line animators (typewriter, fade, rise, drop, centre-out order), number counters |
| Shapes | rect, rounded rect, ellipse, polygon, star, smooth paths, arrows, dashes, **trim paths** (animated map routes) |
| Colour (Resolve) | exposure, contrast/pivot, lift/gamma/gain/offset, temperature/tint, hue, saturation/vibrance, shadows/highlights, monotone curves, levels, **3D .cube LUTs**, tint/duotone, B&W, sepia |
| Effects | gaussian/directional/zoom blur, glow, god rays, sharpen, vignette, film grain, chromatic aberration, flicker, light leaks, old film, **cartoon/cel (anime)**, pixelate, wave warp, turbulent displace, lens distortion, drop shadow, stroke, chroma key, luma key, letterbox, gradient overlay, particles (dust, embers, snow, rain, stars) |
| Pro audio | per clip/master: `eq` bands (peak/shelves/HPF/LPF/notch, slopes), `compressor`, `limiter`, `gate`, `deess`, `voice` presets (clean, broadcast, warm, phone, radio_1940s), `audio_settings.master` chain, beat detection (±3 ms on tests), `markers_from_beats` |
| Audio (Fairlight-lite) | multitrack mix, keyframed gain in dB, equal-power fades, pan, loops, **auto-ducking** of music under voice, **two-pass EBU R128 loudness** (−14 LUFS default), audio from precomps (nested), pitch-preserving speed changes (`"pitch": "tape"` for varispeed), reverse, `highpass`, FFT denoise, raw ffmpeg `filters` |
| Editing & pipeline | editor UI writes edits back to the JSON (frame-snapped move/trim/split, properties, undo/redo, validation before save); **OTIO / FCP7 XML / EDL** import-export; **resumable renders** (crash → re-run reuses finished segments); **proxies** |
| Output | parallel segment rendering, BT.709-correct RGB→YUV, anti-banding dither, H.264/HEVC/ProRes/ProRes 4444 alpha/VP9 alpha/PNG sequence/GIF |
| QA | `validate` catches missing media, bad effect/easing names, broken expressions, clips shorter than their slot, unreadable text (too short / too small / too many words), black frames (`--deep`) |

## Project format (essentials)

```jsonc
{
  "comp": {"width": 1920, "height": 1080, "fps": 30, "duration": 60, "background": "#000",
           "motion_blur": {"samples": 8, "shutter": 180},
           "camera": {"position": {"keys": [[0, [960,540,-2600]], [6, [960,540,-1800], "ease"]]},
                      "poi": [960,540,0], "fov": 40, "dof": {"focus_distance": 1800, "aperture": 12}}},
  "assets": {"map": "media/map.jpg"},
  "layers": [                                   // bottom -> top
    {"type": "image", "src": "@map", "start": 0, "duration": 6,
     "scale": {"keys": [[0, 100], [6, 115, "in_out_sine"]]}},          // fit: cover by default
    {"type": "text", "text": "PARIS, 1799", "start": 1, "duration": 4, "size": 96, "font": "Cinzel",
     "position": [960, 900], "animator": {"type": "rise", "unit": "char", "stagger": 0.04},
     "transition_out": {"type": "fade", "duration": 0.4}},
    {"type": "sequence", "start": 6, "transition": {"type": "crossfade", "duration": 0.5},
     "clips": [{"src": "a.mp4", "in": 3, "duration": 4}, {"src": "b.mp4", "duration": 5}]},
    {"type": "adjustment", "effects": [{"type": "color", "contrast": 0.1, "saturation": 0.9},
                                       {"type": "vignette", "amount": 0.4}, {"type": "grain", "amount": 0.03}]}
  ],
  "audio": [{"src": "vo.wav", "role": "voice"}, {"src": "score.wav", "role": "music", "volume": -8, "fade_out": 3}],
  "audio_settings": {"duck": {"amount_db": -10}, "loudness": -14},
  "markers": [{"t": 0, "name": "Intro"}]
}
```

Rules worth knowing:
* Keyframe times are **layer-local** (0 = layer start). Camera keys use comp time.
* Coordinates are comp pixels, origin top-left, y down, z into the screen (AE conventions). A 3D layer at z=0 with the default camera is pixel-identical to its 2D version (tested).
* Transform keys (`position`, `anchor`, `scale`, `rotation`, `rotation_x/y/z`, `orientation`, `opacity`) sit on the layer.
* Default `fit` is `cover` for 2D opaque images/video; `none` for 3D layers and images with transparency.
* Any value can be animated: `{"keys": [...]}`, `"=expression"`, or `{"keys": [...], "expr": "value + wiggle(2, 10)"}`.

Python API: `from cinegraph import Project, keys, kenburns, lower_third, route_line, title_card` (see `examples/demo.py`).

## 3D example

```jsonc
{"comp": {"camera": {"position": [700, 200, -1500], "poi": [960, 650, 300], "fov": 45}},
 "layers": [
   {"primitive": "plane", "size": [4000, 4000], "position": [960, 860, 400], "texture": "map.jpg"},
   {"src": "cannon.glb", "size": 300, "position": [900, 760, 300], "rotation_y": {"keys": [[0, 0], [6, 40]]}},
   {"primitive": "cube", "size": 200, "color": "#d64933", "material": {"toon": 3, "outline": 4}},
   {"type": "text", "text": "AUSTERLITZ", "three_d": true, "extrude": {"depth": 24}, "casts_shadows": true},
   {"type": "light", "light_type": "spot", "position": [500, 0, -200], "poi": [960, 800, 400],
    "intensity": 140, "cone_angle": 80, "casts_shadows": true},
   {"type": "light", "light_type": "ambient", "intensity": 25}]}
```
Models are auto-centred and scaled so their largest side = `size` px (`"normalize": false` keeps file units). Any model or light switches that 3D run to the OpenGL renderer; `"renderer": "gl"` forces it for card-only scenes (true intersections).

## Honest limits (v0.3)
* CPU compositing ≈0.4–1 s per heavy 1080p frame per core; OpenGL 3D ≈1.5 s/frame without a GPU (fast on a real GPU). A GPU compositor is the next big item.
* 3D: no morph targets, no reflections between objects/GI, one shadowed point light; glTF skinning is CPU-side (fine for a few characters).
* AI: offline captions are English (faster-whisper adds 90+ languages, needs its model download); cut-out works best on clear subjects; object removal needs the background to be visible in other frames, otherwise falls back to inpainting.
* Interchange carries editorial only (clips, timing, tracks, dissolves, markers) — effects/3D stay in Cinegraph; EDL is one video track.
* time_remap clips are silent; retimed precomps are silent.
