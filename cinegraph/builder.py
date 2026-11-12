"""Python API for building projects programmatically (ideal for Claude-generated edits).

    from cinegraph.builder import Project, keys, kenburns
    p = Project(1920, 1080, 30, duration=12)
    p.image("map.jpg", start=0, duration=6, **kenburns(100, 115, ease="smooth", duration=6))
    p.text("PARIS, 1799", start=0.5, duration=4, size=96, font="Cinzel", tracking=200,
           animator={"type": "rise", "unit": "char", "stagger": 0.04})
    p.audio("narration.wav", start=0, role="voice")
    p.save("edit.json"); p.render("out.mp4")
"""
from __future__ import annotations

import copy
import json
import os


def keys(*pairs, ease=None, path=None, loop=None, expr=None):
    """keys((0, 100), (2, 120, "ease")) -> {"keys": [...]}; `ease` sets the default segment easing."""
    d = {"keys": [list(p) for p in pairs]}
    if ease:
        d["ease"] = ease
    if path:
        d["path"] = path
    if loop:
        d["loop"] = loop
    if expr:
        d["expr"] = expr
    return d


def kenburns(scale_from=100, scale_to=112, pos_from=None, pos_to=None, duration=5.0, ease="in_out_sine", W=1920, H=1080):
    """Slow push / pan for stills. Positions default to frame center."""
    pf = pos_from or [W / 2, H / 2]
    pt = pos_to or pf
    return {"scale": keys((0, scale_from), (duration, scale_to), ease=ease),
            "position": keys((0, list(pf)), (duration, list(pt)), ease=ease)}


class Project:
    def __init__(self, width=1920, height=1080, fps=30, duration=10.0, background="#000000", **comp):
        self.data = {"comp": {"width": width, "height": height, "fps": fps, "duration": duration,
                              "background": background, **comp},
                     "layers": [], "audio": [], "comps": {}, "markers": [], "assets": {}}
        self._n = 0

    # ------------------------------------------------------------ core
    @property
    def W(self):
        return self.data["comp"]["width"]

    @property
    def H(self):
        return self.data["comp"]["height"]

    def add(self, **layer):
        self._n += 1
        layer.setdefault("id", f"{layer.get('type', 'layer')}{self._n}")
        self.data["layers"].append(layer)
        return layer

    def video(self, src, start=0, **kw):
        return self.add(type="video", src=src, start=start, **kw)

    def image(self, src, start=0, **kw):
        return self.add(type="image", src=src, start=start, **kw)

    def text(self, text, start=0, **kw):
        return self.add(type="text", text=text, start=start, **kw)

    def solid(self, color="#000000", start=0, **kw):
        return self.add(type="solid", color=color, start=start, **kw)

    def gradient(self, colors, start=0, **kw):
        return self.add(type="gradient", colors=colors, start=start, **kw)

    def shape(self, shapes, start=0, **kw):
        return self.add(type="shape", shapes=shapes if isinstance(shapes, list) else [shapes], start=start, **kw)

    def adjustment(self, effects, start=0, **kw):
        return self.add(type="adjustment", effects=effects, start=start, **kw)

    def null(self, start=0, **kw):
        return self.add(type="null", start=start, **kw)

    def particles(self, kind="dust", start=0, **kw):
        return self.add(type="particles", kind=kind, start=start, **kw)

    def model(self, src=None, start=0, primitive=None, **kw):
        """3D mesh: src=.obj/.glb/.gltf/.stl/.ply or primitive=cube|sphere|cylinder|cone|torus|capsule|plane|terrain."""
        if primitive:
            kw["primitive"] = primitive
        if src:
            kw["src"] = src
        return self.add(type="model", start=start, **kw)

    def light(self, light_type="spot", start=0, **kw):
        """light_type: ambient | directional | point | spot. intensity %, color, poi, cone_angle, casts_shadows."""
        return self.add(type="light", light_type=light_type, start=start, **kw)

    def precomp(self, name, sub: "Project", start=0, **kw):
        c = copy.deepcopy(sub.data["comp"])
        c["layers"] = copy.deepcopy(sub.data["layers"])
        c.setdefault("background", "transparent")
        self.data["comps"][name] = c
        self.data["comps"].update(copy.deepcopy(sub.data["comps"]))
        return self.add(type="comp", comp=name, start=start, **kw)

    def sequence(self, clips, start=0, transition=None, **kw):
        s = {"type": "sequence", "clips": clips, "start": start, **kw}
        if transition:
            s["transition"] = transition
        self.data["layers"].append(s)
        return s

    def audio(self, src, start=0, role="music", **kw):
        a = {"src": src, "start": start, "role": role, **kw}
        self.data["audio"].append(a)
        return a

    def camera(self, **kw):
        self.data["comp"]["camera"] = kw
        return kw

    def marker(self, t, name, chapter=True):
        self.data["markers"].append({"t": t, "name": name, "chapter": chapter})

    # ------------------------------------------------------------ io
    def to_dict(self):
        return copy.deepcopy(self.data)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        return path

    def _norm(self, base_dir=None):
        from . import project as PJ
        return PJ.load(self.to_dict(), base_dir or os.getcwd())

    def render(self, out, base_dir=None, **kw):
        from .render import render
        return render(self._norm(base_dir), out, **kw)

    def still(self, t, out, base_dir=None, **kw):
        from .render import still
        return still(self._norm(base_dir), t, out, **kw)

    def contact(self, out, base_dir=None, **kw):
        from .render import contact_sheet
        return contact_sheet(self._norm(base_dir), out, **kw)

    def validate(self, base_dir=None, **kw):
        from .validate import validate
        return validate(self._norm(base_dir), **kw)


# ---------------------------------------------------------------- documentary presets
def lower_third(title, subtitle="", start=0, duration=5, x=120, y=860, accent="#d4af37", font="Inter", W=1920):
    """Animated lower third: accent bar draws on, title rises in, subtitle fades. Returns list of layers."""
    return [
        {"type": "shape", "id": f"lt_bar_{start}", "start": start, "duration": duration,
         "shapes": [{"kind": "path", "points": [[x, y - 50], [x, y + 50]], "stroke": accent, "stroke_width": 6,
                     "trim_end": keys((0, 0), (0.45, 100), ease="out_expo")}],
         "transition_out": {"type": "fade", "duration": 0.35}},
        {"type": "text", "id": f"lt_title_{start}", "text": title, "start": start + 0.15, "duration": duration - 0.15,
         "font": font, "bold": True, "size": 54, "align": "left", "position": [x + 28, y - 16], "anchor": [0, 32],
         "animator": {"type": "rise", "unit": "char", "stagger": 0.018, "duration": 0.45, "offset": 24},
         "effects": [{"type": "drop_shadow", "opacity": 0.5, "distance": 3, "softness": 8}],
         "transition_out": {"type": "fade", "duration": 0.35}},
        {"type": "text", "id": f"lt_sub_{start}", "text": subtitle, "start": start + 0.45, "duration": duration - 0.45,
         "font": font, "size": 32, "color": "#d8d8d8", "align": "left", "position": [x + 28, y + 38], "anchor": [0, 20],
         "tracking": 60, "transition_in": {"type": "fade", "duration": 0.5},
         "transition_out": {"type": "fade", "duration": 0.35}},
    ] if subtitle else []


def route_line(points, start=0, draw_time=2.5, duration=6, color="#c8102e", width=6, dash=None, arrow=True, smooth=True):
    """Animated map route (trim path) with arrow head."""
    s = {"kind": "path", "points": points, "smooth": smooth, "stroke": color, "stroke_width": width, "arrow": arrow,
         "trim_end": keys((0, 0), (draw_time, 100), ease="in_out_sine")}
    if dash:
        s["dash"] = dash
    return {"type": "shape", "start": start, "duration": duration, "shapes": [s],
            "effects": [{"type": "drop_shadow", "opacity": 0.45, "distance": 2, "softness": 6}]}


def title_card(title, start=0, duration=4, size=140, font="Cinzel", color="#f3e3c3", extrude=None, three_d=False):
    L = {"type": "text", "text": title, "start": start, "duration": duration, "font": font, "bold": True,
         "size": size, "color": color, "tracking": 120,
         "scale": keys((0, 92), (duration, 100), ease="out_cubic"),
         "animator": {"type": "fade", "unit": "char", "stagger": 0.05, "duration": 0.8, "order": "center"},
         "effects": [{"type": "glow", "threshold": 0.5, "radius": 30, "intensity": 0.5}],
         "transition_out": {"type": "fade", "duration": 0.6}}
    if extrude:
        L["extrude"] = extrude
    if three_d:
        L["three_d"] = True
    return L
