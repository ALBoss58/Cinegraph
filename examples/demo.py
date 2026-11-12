"""Feature showcase: 3D camera parallax, extruded 3D title, DOF, fog, particles, grading, sequence with
transitions, Ken Burns, trim-path route, lower third, counter, precomp, track matte, cel/anime look,
audio ducking + loudness. Run:  python examples/demo.py  (writes demo.json, then render with the CLI)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cinegraph import Project, keys, kenburns, lower_third, route_line  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
p = Project(1920, 1080, 30, duration=15, background="#05070a", motion_blur={"samples": 6, "shutter": 180})

# ---------------- Scene 1: 3D parallax push with extruded title (0-5s)
p.camera(position=keys((0, [960, 540, -2400]), (5, [1020, 500, -2050]), ease="in_out_sine"),
         poi=keys((0, [960, 540, 0]), (5, [1000, 520, 0]), ease="in_out_sine"),
         fov=40, dof={"focus_distance": 1500, "aperture": 14, "max_blur": 18},
         fog={"color": "#0a0d14", "near": 1800, "far": 5000, "amount": 0.6},
         shake={"amount": 3, "frequency": 0.8, "rotation": 0.15})
p.image("media/painting.png", start=0, duration=5.2, three_d=True, position=[960, 540, 1600], scale=260,
        effects=[{"type": "color", "saturation": 0.6, "exposure": -0.6}])
p.image("media/map.jpg", start=0, duration=5.2, three_d=True, position=[960, 560, 300], scale=70, rotation_y=-8,
        masks=[{"shape": "ellipse", "center": [960, 540], "size": [1700, 950], "feather": 180}])
p.image("media/emblem.png", start=0, duration=5.2, three_d=True, position=[430, 330, -500], scale=60,
        effects=[{"type": "glow", "threshold": 0.3, "radius": 40, "intensity": 0.6}])
p.text("RISE OF AN EAGLE", start=0.4, duration=4.8, three_d=True, font="DejaVu Serif", bold=True, size=118,
       color="#f1dfb4", tracking=40, position=[960, 560, -200], rotation_y=keys((0, -18), (4.8, 6), ease="in_out_sine"),
       extrude={"depth": 26, "color": "#6b4a1c"},
       animator={"type": "rise", "unit": "char", "stagger": 0.05, "duration": 0.7, "offset": 40},
       effects=[{"type": "drop_shadow", "distance": 12, "softness": 18, "opacity": 0.6}],
       transition_out={"type": "fade", "duration": 0.5})
p.particles("dust", start=0, duration=5.2, count=160, opacity=70)

# ---------------- Scene 2: edited sequence with transitions (5-10.5s)
seq_clips = [
    {"type": "video", "src": "media/clip.mp4", "in": 1.0, "duration": 2.5, "volume": -8,
     **kenburns(100, 110, duration=2.5), "effects": [{"type": "cartoon", "levels": 7, "edge": 0.7}]},
    {"type": "image", "src": "media/map.jpg", "duration": 3.5, **kenburns(105, 118, [960, 540], [1010, 520], duration=3.5),
     "effects": [{"type": "color", "temperature": 25, "contrast": 0.15, "saturation": 0.8},
                 {"type": "old_film", "amount": 0.6}]},
]
p.sequence(seq_clips, start=5.0, transition={"type": "wipe_right", "duration": 0.6, "softness": 120}, id="seq")
rl = route_line([[300, 300], [700, 450], [1100, 380], [1500, 700]], start=7.6, draw_time=2.0, duration=2.9,
                color="#c8102e", width=7, dash=[26, 14])
p.add(**rl)
for L in lower_third("NAPOLEON BONAPARTE", "First Consul of France, 1799", start=7.8, duration=2.7):
    p.add(**L)
p.text("", start=5.3, duration=2.2, counter={"from": 1769, "to": 1799, "duration": 1.6, "format": "{:.0f}"},
       font="DejaVu Serif", bold=True, size=120, color="#ffffff", position=[1650, 160],
       box={"color": "#000000a0", "padding": [28, 10], "radius": 10},
       transition_in={"type": "zoom", "duration": 0.4}, transition_out={"type": "fade", "duration": 0.3})

# ---------------- Scene 3: precomp + text track matte reveal (10.5-15s)
inner = Project(1920, 1080, 30, duration=4.5, background="transparent")
inner.image("media/painting.png", start=0, duration=4.5, **kenburns(100, 125, duration=4.5),
            effects=[{"type": "color", "hue": 40, "saturation": 1.3}])
p.precomp("painting_comp", inner, start=10.5, duration=4.5,
          matte={"layer": "matte_text", "mode": "alpha"})
p.text("EMPIRE", id="matte_text", start=10.5, duration=4.5, font="DejaVu Sans", bold=True, size=420, tracking=40,
       scale=keys((0, 80), (4.5, 105), ease="out_cubic"))
p.text("1804 — 1815", start=11.3, duration=3.7, font="DejaVu Serif", size=64, color="#e9e2cf", position=[960, 820],
       tracking=300, animator={"type": "typewriter", "unit": "char", "stagger": 0.06},
       transition_out={"type": "dip_white", "duration": 0.6})

# ---------------- global finishing: grade + letterbox + grain on an adjustment layer
p.adjustment([{"type": "color", "contrast": 0.08, "lift": [0.01, 0.015, 0.03], "gain": [1.03, 1.0, 0.95]},
              {"type": "vignette", "amount": 0.45},
              {"type": "grain", "amount": 0.035},
              {"type": "letterbox", "ratio": 2.2}], start=0, id="finish")

# ---------------- audio
p.audio("media/music.wav", start=0, role="music", volume=-6, fade_in=1.5, fade_out=2.0, duration=15, loop=True)
p.audio("media/voice.wav", start=1.0, role="voice", volume=-3)
p.data["audio_settings"] = {"duck": {"amount_db": -12, "attack": 0.1, "release": 0.4}, "loudness": -14}
p.marker(0, "Cold open")
p.marker(5, "The Campaign")
p.marker(10.5, "Empire")

out = p.save(os.path.join(HERE, "demo.json"))
print("wrote", out)
