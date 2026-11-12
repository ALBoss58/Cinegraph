"""The compositor: evaluates layers at a time and builds a premultiplied float32 RGBA frame.

Pipeline per layer:  source -> masks -> effects -> (extrude slices) -> transform/3D projection (+motion blur)
                     -> transitions -> DOF / fog -> track matte -> opacity -> blend mode
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from . import effects as FX
from . import easing, media, shapes as SH, text as TX
from .blend import composite
from .camera import Camera, euler, rot_z, clip_polygon_near, project_points
from .props import P, as_vec
from .util import parse_color, stable_seed

TIME_FX = {"grain", "flicker", "light_leak", "old_film", "wave_warp", "turbulent_displace", "god_rays"}
NEAR = 1.0


def _u8_to_premul(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float32:
        f = img.copy()
    else:
        f = img.astype(np.float32) * (1.0 / (65535.0 if img.dtype == np.uint16 else 255.0))
    f[..., :3] *= f[..., 3:4]
    return f


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 4) for x in o.ravel()]
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


class Layer:
    def __init__(self, spec: dict, comp: "Comp", index: int):
        self.spec = spec
        self.comp = comp
        self.index = index
        self.id = spec.get("id", f"layer{index}")
        self.type = spec["type"]
        self.start = float(spec.get("start", 0.0))
        self.enabled = spec.get("enabled", True)
        self.three_d = bool(spec.get("three_d", spec.get("3d", False))) or self.type in ("model", "light")
        self.blend = spec.get("blend", "normal")
        self.parent_id = spec.get("parent")
        self.parent: Layer | None = None
        self.effects = spec.get("effects", [])
        self.masks = spec.get("masks", [])
        self.matte = spec.get("matte")
        self.motion_blur = spec.get("motion_blur", True)
        self.seed = stable_seed(self.id)
        self._props: dict = {}
        self._cache_key = None
        self._cache_val = None
        self._mips: dict = {}
        self.reader = None
        self.src_info = None
        tf = spec.get("transform", spec)  # transform keys may live at top level or under "transform"
        self.tf = tf
        W, H = comp.W, comp.H
        self.fit_scale = 1.0
        self.natural = (W, H)
        if self.type in ("video", "image"):
            src = spec["src"]
            if self.type == "image" or media.is_image(src):
                img = media.load_image(src)
                self.natural = (img.shape[1], img.shape[0])
                self.src_info = {"duration": None}
            else:
                self.src_info = media.probe(src)
                self.natural = (self.src_info["width"], self.src_info["height"])
            default_fit = "cover"
            if self.three_d:
                default_fit = "none"
            elif self.type == "image" or media.is_image(src):
                if img.shape[2] == 4 and img[..., 3].min() < 250:
                    default_fit = "none"  # graphics with transparency keep native size
            fit = spec.get("fit", default_fit)
            nw, nh = self.natural
            if fit == "cover":
                self.fit_scale = max(W / nw, H / nh)
            elif fit == "contain":
                self.fit_scale = min(W / nw, H / nh)
            elif fit in ("width",):
                self.fit_scale = W / nw
            elif fit in ("height",):
                self.fit_scale = H / nh
            elif fit == "fill":
                self.fit_scale = (W / nw, H / nh)
        elif self.type == "solid":
            sz = spec.get("size", [W, H])
            self.natural = (float(sz[0]), float(sz[1]))
        elif self.type == "comp":
            sub = comp.project.comps_spec[spec["comp"]]
            self.natural = (sub["width"], sub["height"])
        elif self.type == "shape":
            self._shape_bounds = SH.bounds(self._eval_shapes(0.0))
        # duration
        if "duration" in spec:
            self.duration = float(spec["duration"])
        elif self.type == "video" and self.src_info and self.src_info.get("duration") and not spec.get("loop"):
            spd = abs(float(spec.get("speed", 1.0))) or 1.0
            self.duration = max(0.0, (self.src_info["duration"] - float(spec.get("in", 0))) / spd)
        else:
            self.duration = max(0.0, comp.duration - self.start)
        self.end = self.start + self.duration

    # ------------------------------------------------------------ props
    def prop(self, key, raw, default=0.0, color=False):
        p = self._props.get(key)
        if p is None:
            p = P(raw, default=default, color=color)
            self._props[key] = p
        return p

    def val(self, name, default, lt, ct, color=False, src=None):
        src = self.tf if src is None else src
        raw = src.get(name)
        if isinstance(raw, dict) and "track" in raw:
            v = self._tracked(raw["track"], ct)
            off = np.asarray(raw.get("offset", [0, 0]), np.float32)
            v = np.concatenate([v + off[:2], off[2:3] if off.size > 2 else np.zeros(1, np.float32)])
            return v
        if src.get(name) is None:  # defaults can be dynamic (text size) -> never cache them
            return default if not isinstance(default, list) else np.asarray(default, np.float32)
        return self.prop(("tf", name), src.get(name), default, color).at(lt, ct)

    def active(self, t):
        return self.enabled and self.start - 1e-6 <= t < self.end - 1e-6

    # ------------------------------------------------------------ transforms
    def default_anchor(self):
        if self.type == "shape":
            x0, y0, x1, y1 = self._shape_bounds
            return [(x0 + x1) / 2, (y0 + y1) / 2, 0.0]
        if self.type == "text" and "path" in self.spec:
            p = np.asarray(self.spec["path"]["points"], np.float32).reshape(-1, 2)
            c_ = (p.min(0) + p.max(0)) / 2  # path text lives in comp coords like shapes
            return [float(c_[0]), float(c_[1]), 0.0]
        if self.type == "text":
            return None  # resolved at render time from text block size
        if self.type == "particles":
            return [self.comp.W / 2, self.comp.H / 2, 0]
        if self.type in ("null", "model", "light"):
            return [0.0, 0.0, 0.0]  # rig pivot / model origin / light origin
        w, h = self.natural
        return [w / 2, h / 2, 0.0]

    def default_position(self):
        if self.type == "shape" or (self.type == "text" and "path" in self.spec):
            a = self.default_anchor()
            return [a[0], a[1], 0.0]
        if self.parent is not None:
            return [0.0, 0.0, 0.0]
        return [self.comp.W / 2, self.comp.H / 2, 0.0]

    def local_matrix(self, t: float, text_size=None) -> np.ndarray:
        lt = t - self.start
        da = self.default_anchor()
        if da is None:
            tw, th = text_size or (0, 0)
            da = [tw / 2, th / 2, 0.0]
        anchor = as_vec(self.val("anchor", da, lt, t), 3)
        pos = as_vec(self.val("position", self.default_position(), lt, t), 3)
        raw = np.atleast_1d(np.asarray(self.val("scale", 100.0, lt, t), np.float32)).ravel()
        if raw.size == 1:
            sc = np.repeat(raw, 3) / 100.0
        elif raw.size == 2:
            sc = np.array([raw[0], raw[1], 100.0], np.float32) / 100.0
        else:
            sc = raw[:3] / 100.0
        fs = self.fit_scale
        if isinstance(fs, tuple):
            sc = sc * np.array([fs[0], fs[1], 1.0], np.float32)
        else:
            sc = sc * np.array([fs, fs, 1.0], np.float32)
        if self.three_d:
            rx = float(self.val("rotation_x", 0.0, lt, t))
            ry = float(self.val("rotation_y", 0.0, lt, t))
            rz = float(self.val("rotation_z" if "rotation_z" in self.tf else "rotation", 0.0, lt, t))
            ori = as_vec(self.val("orientation", [0, 0, 0], lt, t), 3)
            R = euler(ori[0], ori[1], ori[2]) @ euler(rx, ry, rz)
        else:
            R = rot_z(math.radians(float(self.val("rotation", 0.0, lt, t))))
            pos = pos.copy()
            pos[2] = 0.0
            anchor = anchor.copy()
            anchor[2] = 0.0
        M = np.eye(4)
        M[:3, :3] = R @ np.diag(sc.astype(np.float64))
        M[:3, 3] = pos
        A = np.eye(4)
        A[:3, 3] = -anchor
        return M @ A

    def world_matrix(self, t: float, text_size=None) -> np.ndarray:
        M = self.local_matrix(t, text_size)
        p = self.parent
        depth = 0
        while p is not None and depth < 32:
            M = p.local_matrix(t, getattr(p, "_last_text_size", None)) @ M
            p = p.parent
            depth += 1
        return M

    def opacity(self, t):
        return float(self.val("opacity", 100.0, t - self.start, t)) / 100.0

    # ------------------------------------------------------------ time mapping
    def source_time(self, t: float) -> float:
        lt = t - self.start
        s = self.spec
        if "time_remap" in s:
            return float(self.prop("time_remap", s["time_remap"]).at(lt, t))
        if "freeze_at" in s:
            return float(s["freeze_at"])
        spd = float(s.get("speed", 1.0))
        tin = float(s.get("in", 0.0))
        if s.get("reverse"):
            st = tin + (self.duration - lt) * abs(spd) - 1.0 / max(1.0, self.comp.fps)
        else:
            st = tin + lt * spd
        if s.get("loop") and self.src_info and self.src_info.get("duration"):
            d = self.src_info["duration"]
            st = st % d
        return max(0.0, st)

    # ------------------------------------------------------------ source rendering
    def _eval_shapes(self, lt, ct=None):
        ct = lt + self.start if ct is None else ct
        out = []
        for i, s in enumerate(self.spec.get("shapes", [])):
            e = {}
            for k, v in s.items():
                if k in ("kind", "closed", "smooth", "arrow", "dash", "points") and not isinstance(v, dict):
                    e[k] = v
                    continue
                isc = k in ("fill", "stroke")
                if v is None:
                    e[k] = None
                    continue
                e[k] = self.prop(("shape", i, k), v, color=isc).at(lt, ct)
                if isinstance(e[k], np.ndarray) and not isc and k in ("center", "size"):
                    e[k] = [float(x) for x in e[k]]
            out.append(e)
        return out

    def _text_spec(self, lt, ct):
        s = self.spec
        out = {}
        animatable = {"size": 72, "color": "#ffffff", "stroke_color": "#000000", "stroke_width": 0, "tracking": 0,
                      "line_height": 1.2, "max_width": 0}
        for k in ("text", "font", "bold", "italic", "align", "box", "animator", "all_caps", "gradient", "box_follows_text",
                  "word_highlight"):
            if k in s:
                out[k] = s[k]
        if "path" in s:
            pth = dict(s["path"])
            if "offset" in pth:
                pth["offset"] = float(self.prop(("tpath", "offset"), pth["offset"]).at(lt, ct))
            if "baseline" in pth:
                pth["baseline"] = float(self.prop(("tpath", "baseline"), pth["baseline"]).at(lt, ct))
            out["path"] = pth
        for k, d in animatable.items():
            if k in s:
                v = self.prop(("text", k), s[k], d, color=k.endswith("color")).at(lt, ct)
                out[k] = v
        if isinstance(out.get("text"), dict):
            out["text"] = self._keyed_text(out["text"], lt)
        if "counter" in s:
            c = s["counter"]
            u = (lt - float(c.get("start", 0))) / max(1e-6, float(c.get("duration", 2)))
            u = easing.get(c.get("ease", "out_cubic"))(min(1, max(0, u)))
            v = float(c.get("from", 0)) + (float(c.get("to", 100)) - float(c.get("from", 0))) * u
            out["text"] = str(c.get("format", "{:,.0f}")).format(v)
        if isinstance(out.get("color"), np.ndarray):
            out["color"] = out["color"].tolist()
        if isinstance(out.get("stroke_color"), np.ndarray):
            out["stroke_color"] = out["stroke_color"].tolist()
        return out

    @staticmethod
    def _keyed_text(spec, lt):
        keys = sorted(spec.get("keys", []), key=lambda k: k[0])
        cur = keys[0][1] if keys else ""
        for k in keys:
            if lt >= k[0]:
                cur = k[1]
        return cur

    def raster_scale(self, t):
        if "raster_scale" in self.spec:
            return float(self.spec["raster_scale"]) * self.comp.scale
        raw = self.tf.get("scale", 100)
        mx = 100.0
        if isinstance(raw, dict) and raw.get("keys"):
            for k in raw["keys"]:
                v = k["v"] if isinstance(k, dict) else k[1]
                mx = max(mx, float(np.max(np.atleast_1d(v))))
        elif not isinstance(raw, (dict, str)):
            mx = max(mx, float(np.max(np.atleast_1d(raw))))
        rs = min(4.0, mx / 100.0)
        if self.three_d:
            rs = max(rs, float(self.spec.get("raster_3d", 1.5)))
        return max(1.0, rs) * self.comp.scale

    def source(self, t: float):
        """Source in the comp's working colour space: (img premul float32, Msrc, static)."""
        raw = self._source_raw(t)
        cs_ = self.comp.project.color
        cs_name = self.spec.get("colorspace")
        if self.type == "comp" or (not cs_.linear and not cs_name):
            return raw
        key = (id(raw[0]), cs_name, cs_.working)
        if getattr(self, "_cs_key", None) == key:
            return self._cs_val
        img = cs_.input(raw[0], cs_name)
        self._cs_key, self._cs_val = key, (img, raw[1], raw[2])
        return self._cs_val

    def _source_raw(self, t: float):
        """Return (img premul float32, Msrc 3x3 src-px -> layer coords, static: bool)."""
        lt = t - self.start
        typ = self.type
        s = self.spec
        sc = self.comp.scale
        if typ == "solid":
            c = self.prop("color", s.get("color", "#808080"), color=True).at(lt, t)
            w, h = self.natural
            key = ("solid", tuple(np.round(c, 4)), w, h)
            if key == self._cache_key:
                return self._cache_val
            pw, ph = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
            img = np.empty((ph, pw, 4), np.float32)
            img[...] = np.array([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]], np.float32)
            M = np.diag([w / pw, h / ph, 1.0])
            self._cache_key, self._cache_val = key, (img, M, not self.prop("color", None).animated)
            return self._cache_val
        if typ == "image":
            img8 = media.load_image(s["src"])
            key = ("img", s["src"])
            if key != self._cache_key:
                if self._hi_depth():
                    self._cache_key, self._cache_val = key, (_u8_to_premul(media.load_image_hi(s["src"])), np.eye(3), True)
                    return self._cache_val
                img8 = self._ai_process(img8, 0, lt, t, still=True)
                self._cache_key, self._cache_val = key, (_u8_to_premul(img8), np.eye(3), True)
            return self._cache_val
        if typ == "video":
            if self.reader is None:
                fs = self.fit_scale if not isinstance(self.fit_scale, tuple) else max(self.fit_scale)
                dec = min(1.0, max(0.05, sc * fs * float(s.get("decode_headroom", 1.35))))
                path, pscale = s["src"], dec
                if self.comp.project.proxies:
                    from .proxy import proxy_path
                    pp = proxy_path(s["src"], self.comp.project.base_dir)
                    if os.path.exists(pp):
                        pw = media.probe(pp)["width"]
                        path, pscale = pp, min(1.0, self.natural[0] * dec / pw)
                self.reader = media.VideoReader(path, scale=pscale, depth=16 if self._hi_depth() and path == s["src"] else 8)
                self._dec = (self.reader.w / self.natural[0], self.reader.h / self.natural[1])
            st = self.source_time(t)
            fps = self.reader.fps
            fpos = st * fps
            idx = int(math.floor(fpos + 1e-4))
            frac = fpos - idx
            fb = s.get("frame_blend", "none")
            interp = fb in ("flow", "pixel_motion", "blend", "frame_mix") and frac > 0.02
            key = ("vid", idx, round(frac, 3) if interp else 0)
            if key == self._cache_key:
                return self._cache_val
            img8 = self._video_frame(idx)
            if interp:
                nxt = self._video_frame(idx + 1)
                if fb in ("flow", "pixel_motion"):
                    from . import motion as MO
                    if not hasattr(self, "_flow_cache"):
                        self._flow_cache = {}
                    img8 = MO.interpolate(img8, nxt, frac, self._flow_cache, idx)
                else:
                    img8 = np.clip(img8.astype(np.float32) * (1 - frac) + nxt.astype(np.float32) * frac + 0.5, 0, 255).astype(np.uint8)
            stab = s.get("stabilize")
            if stab:
                from . import motion as MO
                if not hasattr(self, "_stab"):
                    self._stab = MO.Stabilizer(s["src"], stab if isinstance(stab, dict) else {}, self.comp.project.base_dir)
                img8 = self._stab.apply(img8, int(round(fpos)))
            img8 = self._ai_process(img8, int(round(fpos)), lt, t)
            img = _u8_to_premul(img8)
            M = np.diag([1.0 / self._dec[0], 1.0 / self._dec[1], 1.0])
            self._cache_key, self._cache_val = key, (img, M, False)
            return self._cache_val
        if typ == "text":
            spec = self._text_spec(lt, t)
            rs = self.raster_scale(t)
            anim = spec.get("animator") or spec.get("word_highlight")
            atime = lt if anim else 0.0
            key = ("text", json.dumps(_jsonable(spec), sort_keys=True), rs, round(atime, 4) if anim else 0)
            if key == self._cache_key:
                return self._cache_val
            if "path" in spec:
                img, ox, oy, nw, nh = TX.render_path(spec, atime, rs)
                self._last_text_size = (nw, nh)
                self._path_text_origin = (ox, oy)
            else:
                img, ox, oy, nw, nh = TX.render(spec, atime, rs)
                self._last_text_size = (nw, nh)
            M = np.array([[1 / rs, 0, ox], [0, 1 / rs, oy], [0, 0, 1]], np.float64)
            self._cache_key, self._cache_val = key, (img, M, not anim)
            return self._cache_val
        if typ == "shape":
            ev = self._eval_shapes(lt, t)
            rs = self.raster_scale(t)
            key = ("shape", json.dumps(_jsonable(ev), sort_keys=True), rs)
            if key == self._cache_key:
                return self._cache_val
            img, ox, oy = SH.render(ev, rs)
            M = np.array([[1 / rs, 0, ox], [0, 1 / rs, oy], [0, 0, 1]], np.float64)
            self._cache_key, self._cache_val = key, (img, M, False)
            return self._cache_val
        if typ == "gradient":
            return self._gradient(lt, t)
        if typ == "comp":
            sub = self.comp.project.get_comp(s["comp"])
            st = self.source_time(t) if ("time_remap" in s or "speed" in s or "in" in s) else lt
            img = sub.render(st)
            M = np.diag([1.0 / sub.scale, 1.0 / sub.scale, 1.0])
            return img, M, False
        if typ == "particles":
            return self._particles(lt, t)
        raise ValueError(f"Layer type {typ} has no source")

    def _gradient(self, lt, t):
        s = self.spec
        W, H = self.natural
        sc = self.comp.scale
        cols = [self.prop(("grad", i), c, color=True).at(lt, t) for i, c in enumerate(s.get("colors", ["#000", "#fff"]))]
        kind = s.get("kind", "linear")
        ang = float(self.prop("g_angle", s.get("angle", 90)).at(lt, t))
        cen = as_vec(self.prop("g_center", s.get("center", [0.5, 0.5])).at(lt, t), 2)
        rad = float(self.prop("g_radius", s.get("radius", 0.75)).at(lt, t))
        key = ("grad", tuple(np.round(np.concatenate(cols), 4)), kind, round(ang, 3), tuple(np.round(cen, 4)), round(rad, 4))
        if key == self._cache_key:
            return self._cache_val
        pw, ph = max(2, int(W * sc)), max(2, int(H * sc))
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        if kind == "radial":
            d = np.sqrt(((xx / pw - cen[0]) * (pw / ph)) ** 2 + (yy / ph - cen[1]) ** 2)
            u = np.clip(d / max(1e-3, rad), 0, 1)
        else:
            a = math.radians(ang)
            proj = (xx - pw / 2) * math.cos(a) + (yy - ph / 2) * math.sin(a)
            span = abs(pw * math.cos(a)) + abs(ph * math.sin(a))
            u = np.clip(proj / max(1, span) + 0.5, 0, 1)
        stops = s.get("stops") or list(np.linspace(0, 1, len(cols)))
        out = np.stack([np.interp(u, stops, [c[i] * c[3] for c in cols]) for i in range(3)] +
                       [np.interp(u, stops, [c[3] for c in cols])], -1).astype(np.float32)
        M = np.diag([W / pw, H / ph, 1.0])
        self._cache_key, self._cache_val = key, (out, M, False)
        return self._cache_val

    def _particles(self, lt, t):
        """Deterministic atmospheric particles (dust, embers, snow, rain, stars)."""
        s = self.spec
        W, H = self.comp.W, self.comp.H
        sc = self.comp.scale
        pw, ph = int(W * sc), int(H * sc)
        kind = s.get("kind", "dust")
        n = int(s.get("count", 150))
        size = float(s.get("size", 3 if kind != "rain" else 1.5))
        speed = float(s.get("speed", 1.0))
        col = parse_color(s.get("color", {"embers": "#ff9a3c", "snow": "#ffffff", "rain": "#a8c0ff",
                                          "stars": "#ffffff"}.get(kind, "#f5e6c8")))
        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(0, W, n)
        y0 = rng.uniform(0, H, n)
        sz = size * rng.uniform(0.4, 1.4, n)
        ph_ = rng.uniform(0, 2 * math.pi, n)
        depth = rng.uniform(0.3, 1.0, n)
        if kind == "embers":
            vx, vy = rng.normal(0, 15, n), -rng.uniform(30, 90, n)
        elif kind == "snow":
            vx, vy = rng.normal(10, 10, n), rng.uniform(30, 80, n)
        elif kind == "rain":
            vx, vy = np.full(n, 60.0), rng.uniform(900, 1300, n)
        elif kind == "stars":
            vx, vy = np.zeros(n), np.zeros(n)
        else:  # dust motes drift
            vx, vy = rng.normal(6, 8, n), rng.normal(-4, 6, n)
        T = lt * speed
        x = (x0 + vx * depth * T + 18 * np.sin(T * 0.6 + ph_) * (kind in ("dust", "snow", "embers"))) % (W + 40) - 20
        y = (y0 + vy * depth * T + 10 * np.cos(T * 0.5 + ph_) * (kind != "rain")) % (H + 40) - 20
        alpha = np.clip(0.35 + 0.65 * np.sin(T * (1.5 if kind != "stars" else 2.5) + ph_) ** 2, 0, 1) * depth
        if kind == "embers":
            alpha *= np.clip(1 - ((y0 - y) % H) / H, 0.2, 1)
        m = np.zeros((ph, pw), np.float32)
        for i in range(n):
            if kind == "rain":
                p1 = (int((x[i]) * sc * 16), int(y[i] * sc * 16))
                p2 = (int((x[i] - 0.04 * vx[i]) * sc * 16), int((y[i] - 0.03 * vy[i]) * sc * 16))
                cv2.line(m, p1, p2, float(alpha[i]), max(1, int(sz[i] * sc)), cv2.LINE_AA, 4)
            else:
                cv2.circle(m, (int(x[i] * sc * 16), int(y[i] * sc * 16)), max(16, int(sz[i] * sc * 16)),
                           float(alpha[i]), -1, cv2.LINE_AA, 4)
        if kind in ("dust", "embers", "snow"):
            m = np.maximum(m, cv2.GaussianBlur(m, (0, 0), max(0.6, size * sc * 0.6)) * 1.2)
        m = np.clip(m, 0, 1)[..., None] * col[3]
        img = np.concatenate([col[:3] * m, m], -1).astype(np.float32)
        return img, np.diag([1 / sc, 1 / sc, 1.0]), False

    def _video_frame(self, idx):
        """Decoded uint8 RGBA frame, optionally denoised (temporal = motion-compensated)."""
        dn = self.spec.get("denoise")
        f = self.reader.frame(idx)
        if not dn:
            return f
        dn = dn if isinstance(dn, dict) else {}
        if not hasattr(self, "_dn_cache"):
            self._dn_cache = {}
        if idx in self._dn_cache:
            return self._dn_cache[idx]
        from . import motion as MO
        mode = dn.get("mode", "temporal")
        strength = float(dn.get("strength", 0.5))
        if mode == "temporal":
            r = int(dn.get("radius", 2))
            nbs = [self.reader.frame(j).copy() for j in range(idx - r, idx + r + 1) if j != idx and 0 <= j < self.reader.n_frames]
            out = MO.temporal_denoise(self.reader.frame(idx).copy(), nbs, strength)
        else:
            out = MO.spatial_denoise(f, strength, "nlm" if mode == "nlm" else "bilateral")
        if len(self._dn_cache) > 4:
            self._dn_cache.clear()
        self._dn_cache[idx] = out
        return out

    def _hi_depth(self):
        """16-bit decode for log / wide-gamut sources, unless an 8-bit-only tool (AI, flow, denoise) is used."""
        s = self.spec
        want = s.get("bit_depth", 16 if s.get("colorspace") and s.get("colorspace") not in ("srgb", "sRGB") else 8)
        eight_only = any(s.get(k) for k in ("cutout", "remove", "denoise", "stabilize")) or s.get("frame_blend") in ("flow", "pixel_motion")
        return int(want) > 8 and not eight_only

    def _ai_process(self, img8, idx, lt, t, still=False):
        """Object removal (clean plate) then AI cut-out, on uint8 RGBA in decode resolution."""
        s = self.spec
        if s.get("remove"):
            from . import ai
            h, w = img8.shape[:2]
            sx, sy = w / self.natural[0], h / self.natural[1]
            mk = np.zeros((h, w), np.uint8)
            for i, r in enumerate(s["remove"]):
                ev = {k: (self.prop(("rm", i, k), v).at(lt, t) if k not in ("shape", "points", "closed", "smooth") else v)
                      for k, v in r.items()}
                shape = {"kind": ev.get("shape", "rect"), **{k: (list(map(float, v)) if isinstance(v, np.ndarray) else v)
                                                             for k, v in ev.items() if k in ("center", "size", "points", "radius", "smooth")}}
                if shape["kind"] == "path":
                    shape["closed"] = True
                pts, _ = SH.geometry(shape)
                pts = pts * [sx, sy]
                cv2.fillPoly(mk, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
                grow = int(float(ev.get("expand", 4)) * sx)
                if grow > 0:
                    mk = cv2.dilate(mk, np.ones((grow * 2 + 1,) * 2, np.uint8))
            nbs, nbm = [], []
            if not still and self.type == "video":
                if not hasattr(self, "_nb_readers"):
                    self._nb_readers = {}
                fps_ = self.reader.fps
                nbm = []
                for off in sorted(s.get("remove_window", [-8, 8, -16, 16, -32, 32, -48, 48]), key=abs):
                    j = idx + int(off)
                    if 0 <= j < self.reader.n_frames:
                        rd = self._nb_readers.get(off)
                        if rd is None:
                            rd = self._nb_readers[off] = media.VideoReader(s["src"], scale=self.reader.w / self.natural[0])
                        nbs.append(rd.frame(j).copy())
                        nbm.append(self._remove_mask(w, h, sx, sy, lt + int(off) / fps_ / max(1e-6, float(s.get("speed", 1))), t))
            img8 = ai.remove_object(np.ascontiguousarray(img8), mk.astype(np.float32) / 255.0, nbs,
                                    neighbor_masks=nbm if nbm else None)
        co = s.get("cutout")
        if co:
            from . import ai
            if not hasattr(self, "_cutout"):
                self._cutout = ai.Cutout(co if isinstance(co, dict) else {})
            m = self._cutout.mask(np.ascontiguousarray(img8), idx)
            img8 = img8.copy()
            img8[..., 3] = np.clip(img8[..., 3].astype(np.float32) * m + 0.5, 0, 255).astype(np.uint8)
        return img8

    def _remove_mask(self, w, h, sx, sy, lt, t):
        """Removal mask evaluated at another layer-local time (where the object is in a neighbour frame)."""
        mk = np.zeros((h, w), np.uint8)
        for i, r in enumerate(self.spec["remove"]):
            ev = {k: (self.prop(("rm", i, k), v).at(lt, t) if k not in ("shape", "points", "closed", "smooth") else v)
                  for k, v in r.items()}
            shape = {"kind": ev.get("shape", "rect"), **{k: (list(map(float, v)) if isinstance(v, np.ndarray) else v)
                                                         for k, v in ev.items() if k in ("center", "size", "points", "radius", "smooth")}}
            if shape["kind"] == "path":
                shape["closed"] = True
            pts, _ = SH.geometry(shape)
            cv2.fillPoly(mk, [np.round(pts * [sx, sy] * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        return mk.astype(np.float32) / 255.0

    def _tracked(self, spec, t, planar=False):
        """Evaluate a tracking-driven value in comp coordinates."""
        from . import motion as MO
        L2 = self.comp.byid.get(spec["layer"])
        if L2 is None or L2.type != "video":
            raise ValueError(f"track: layer '{spec.get('layer')}' must be a video layer in the same comp")
        t_in = float(L2.spec.get("in", 0))
        t_end = t_in + L2.duration * abs(float(L2.spec.get("speed", 1))) + 0.1
        base = self.comp.project.base_dir
        st = L2.source_time(min(max(t, L2.start), L2.end - 1e-3))
        if planar:
            tr = MO.track_planar(L2.spec["src"], spec["quad"], t_in, t_end, base)
            pts = MO.sample_track(tr, st, "corners")
        else:
            tr = MO.track_points(L2.spec["src"], [spec["point"]], t_in, t_end, base)
            pts = MO.sample_track(tr, st, "points", 0)[None]
        Wm = L2.world_matrix(t)
        out = np.array([(Wm @ np.array([p[0], p[1], 0, 1.0]))[:2] for p in pts], np.float32)
        return out if planar else out[0]

    def corner_pin(self, t):
        cp = self.spec.get("corner_pin")
        if cp is None:
            return None
        if isinstance(cp, dict) and "track_planar" in cp:
            pts = self._tracked(cp["track_planar"], t, planar=True)
        else:
            pts = np.asarray(self.prop("corner_pin", cp).at(t - self.start, t), np.float32).reshape(4, 2)
        return pts

    # ------------------------------------------------------------ 3D models
    def meshes(self, lt=0.0):
        s = self.spec
        if s.get("src") and s.get("animation", True) is not False and not s.get("primitive"):
            from . import gltf_anim
            if gltf_anim.has_animation(s["src"]):
                return self._anim_meshes(lt)
        if getattr(self, "_meshes", None) is None:
            from . import models as MD
            size = float(s.get("size", 400)) if np.isscalar(s.get("size", 400)) else 400.0
            if s.get("primitive") == "terrain":
                self._meshes = MD.terrain(s, self.comp.project.base_dir)
            elif s.get("primitive"):
                self._meshes = MD.primitive(s["primitive"], s)
            else:
                self._meshes = MD.load_file(s["src"], size, bool(s.get("normalize", True)))[0]
        return self._meshes

    def _anim_meshes(self, lt):
        """Skinned / node-animated glTF posed at layer time (anim_speed, anim_offset, loop, animation name)."""
        from . import gltf_anim
        from .models import Mesh
        s = self.spec
        g = gltf_anim.load(s["src"])
        at = float(s.get("anim_offset", 0)) + lt * float(s.get("anim_speed", 1.0))
        key = round(at, 5)
        if getattr(self, "_anim_key", None) == key:
            return self._anim_val
        R = np.diag([1.0, -1.0, -1.0])
        if getattr(self, "_anim_norm", None) is None:  # normalization from the rest pose: no size drift
            rest = g.geometry(0.0, s.get("animation"), True)
            allp = np.vstack([r[0] for r in rest]) @ R.T
            lo, hi = allp.min(0), allp.max(0)
            ext = float(np.max(hi - lo)) or 1.0
            size = float(s.get("size", 400)) if np.isscalar(s.get("size", 400)) else 400.0
            sc = size / ext if s.get("normalize", True) else 1.0
            c_ = (lo + hi) / 2
            if s.get("ground", True):  # feet on the origin plane: centre x/z, bottom at y=0 (y down)
                c_ = np.array([c_[0], hi[1], c_[2]])
            self._anim_norm = (c_, sc)
        c_, sc = self._anim_norm
        out = []
        for pos, nrm, uv, idx, tex, color, vcol in g.geometry(at, s.get("animation"), bool(s.get("loop", True))):
            p = (pos @ R.T - c_) * sc
            out.append(Mesh(p.astype(np.float32), (nrm @ R.T).astype(np.float32), uv, idx, tex,
                            color, vcol))
        self._anim_key, self._anim_val = key, out
        return out

    def _lin_tex(self, tex):
        if getattr(self, "_lin_tex_cache", (None,))[0] is not tex:
            from .color import convert_premul, convert, SRGB
            w = self.comp.project.color.working
            self._lin_tex_cache = (tex, convert_premul(tex, lambda rgb: convert(rgb, SRGB, w)))
        return self._lin_tex_cache[1]

    def _lin_meshes(self, meshes):
        if getattr(self, "_lin_mesh_cache", (None,))[0] is not meshes:
            from .models import Mesh
            from .color import convert_premul, convert, SRGB
            w = self.comp.project.color.working
            out = []
            for m in meshes:
                tex = convert_premul(m.tex, lambda rgb: convert(rgb, SRGB, w)) if m.tex is not None else None
                colr = self.comp.project.color.color(m.color) if m.color is not None else None
                vc = None
                if m.vcolor is not None:
                    vc = m.vcolor.copy()
                    vc[:, :3] = convert(vc[None, :, :3], SRGB, w)[0]
                out.append(Mesh(m.pos, m.nrm, m.uv, m.idx, tex, colr, vc))
            self._lin_mesh_cache = (meshes, out)
        return self._lin_mesh_cache[1]

    def material(self, lt, t):
        s = self.spec
        m = dict(s.get("material", {}))
        if "color" in s or "color" in m:
            m["color"] = self.prop("m_color", s.get("color", m.get("color")), color=True).at(lt, t)
        if s.get("texture") or m.get("texture"):
            if getattr(self, "_tex_img", None) is None:
                img = media.load_image(s.get("texture") or m.get("texture")).astype(np.float32) / 255.0
                img[..., :3] *= img[..., 3:4]
                self._tex_img = np.ascontiguousarray(img)
            m["_tex_img"] = self._tex_img
        if "emissive" in m:
            m["emissive"] = parse_color(m["emissive"])[:3] * float(m.get("emissive_strength", 1.0))
        if "outline_color" in m:
            m["outline_color"] = parse_color(m["outline_color"])
        return m

    def light_info(self, t, Wm):
        s = self.spec
        lt = t - self.start
        kind = s.get("light_type", "spot")
        tmap = {"ambient": 0, "parallel": 1, "directional": 1, "sun": 1, "point": 2, "spot": 3}
        if kind not in tmap:
            raise ValueError(f"light_type must be one of {list(tmap)}")
        col = parse_color(self.prop("l_color", s.get("color", "#ffffff"), color=True).at(lt, t))[:3]
        inten = float(self.prop("l_int", s.get("intensity", 100)).at(lt, t)) / 100.0
        pos = (Wm @ np.array([0, 0, 0, 1.0]))[:3]
        poi = as_vec(self.prop("l_poi", s.get("poi", [self.comp.W / 2, self.comp.H / 2, 0])).at(lt, t), 3).astype(float)
        d = poi - pos
        d = d / max(np.linalg.norm(d), 1e-9)
        ang = float(self.prop("l_cone", s.get("cone_angle", 90)).at(lt, t))
        fe = float(self.prop("l_feather", s.get("cone_feather", 50)).at(lt, t)) / 100.0
        return {"type": tmap[kind], "color": (col * inten).astype(np.float32), "pos": pos.astype(np.float32),
                "dir": d.astype(np.float32), "cos_outer": math.cos(math.radians(ang / 2)),
                "cos_inner": math.cos(math.radians(ang / 2 * (1 - fe))),
                "range": float(s.get("falloff_radius", 0)), "shadows": bool(s.get("casts_shadows", False)),
                "darkness": float(s.get("shadow_darkness", 100)) / 100.0, "softness": float(s.get("shadow_diffusion", 1.5))}

    # ------------------------------------------------------------ masks & effects
    def _fx_eval(self, lt, ct):
        def ev(i, name, raw, is_color):
            if isinstance(raw, (dict, str)) and not (isinstance(raw, str) and not raw.startswith("=")):
                return self.prop(("fx", i, name), raw, color=is_color).at(lt, ct)
            if isinstance(raw, dict):
                return self.prop(("fx", i, name), raw, color=is_color).at(lt, ct)
            return raw
        return ev

    def apply_masks(self, img, Msrc, lt, ct):
        if not self.masks:
            return img
        h, w = img.shape[:2]
        Minv = np.linalg.inv(Msrc)
        unit = 1.0 / Msrc[0, 0]
        acc = None
        for i, m in enumerate(self.masks):
            ev = {}
            for k, v in m.items():
                if k in ("shape", "kind", "mode", "invert", "closed", "smooth", "points") and not isinstance(v, dict):
                    ev[k] = v
                else:
                    ev[k] = self.prop(("mask", i, k), v).at(lt, ct)
            shape = {"kind": ev.get("shape", ev.get("kind", "rect")), **{k: v for k, v in ev.items() if k in ("center", "size", "radius", "points", "closed", "smooth", "sides", "rotation")}}
            if shape["kind"] in ("path",):
                shape["closed"] = True
            for k in ("center", "size"):
                if k in shape and isinstance(shape[k], np.ndarray):
                    shape[k] = [float(x) for x in shape[k]]
            pts, _ = SH.geometry(shape)
            ph = np.hstack([pts, np.ones((len(pts), 1))]) @ Minv.T
            mk = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mk, [np.round(ph[:, :2] * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
            mk = mk.astype(np.float32) / 255.0
            fe = float(ev.get("feather", 0)) * unit
            if fe > 0.3:
                mk = cv2.GaussianBlur(mk, (0, 0), fe / 2)
            exp = float(ev.get("expansion", 0)) * unit
            if abs(exp) >= 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(abs(exp)) * 2 + 1,) * 2)
                mk = cv2.dilate(mk, k) if exp > 0 else cv2.erode(mk, k)
            if ev.get("invert"):
                mk = 1 - mk
            mk *= float(ev.get("opacity", 1.0))
            mode = ev.get("mode", "add")
            if acc is None:
                acc = mk if mode in ("add", "intersect") else (1 - mk)
            elif mode == "add":
                acc = np.maximum(acc, mk)
            elif mode == "subtract":
                acc = acc * (1 - mk)
            elif mode == "intersect":
                acc = acc * mk
        return img * acc[..., None]

    def effects_time_dependent(self):
        for e in self.effects:
            if FX.resolve_name(e["type"]) in TIME_FX:
                return True
            for v in e.values():
                if isinstance(v, dict) or (isinstance(v, str) and v.startswith("=")):
                    return True
        for m in self.masks:
            for v in m.values():
                if isinstance(v, dict) or (isinstance(v, str) and v.startswith("=")):
                    return True
        return False

    def processed_source(self, t):
        """Source after masks + effects, with padding. Returns (img, Msrc) or None."""
        img, Msrc, static = self.source(t)
        lt = t - self.start
        dyn = self.effects_time_dependent()
        ck = ("proc", id(img), lt if dyn else 0)
        if static and not dyn and getattr(self, "_proc_key", None) == ck:
            return self._proc_val
        img = self.apply_masks(img, Msrc, lt, t)
        if self.effects:
            ev = self._fx_eval(lt, t)
            unit = float(1.0 / Msrc[0, 0])
            pad = int(math.ceil(FX.effects_pad(self.effects, ev) * unit))
            if pad > 0:
                pad = min(pad, 1024)
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                Msrc = Msrc @ np.array([[1, 0, -pad], [0, 1, -pad], [0, 0, 1]], np.float64)
            ctx = FX.FxCtx(t=lt, comp_t=t, fps=self.comp.fps, unit=unit, seed=self.seed,
                           comp_w=self.comp.W, comp_h=self.comp.H, base_dir=self.comp.project.base_dir)
            img = FX.apply_effects(img, self.effects, ev, ctx)
        self._proc_key, self._proc_val = ck, (img, Msrc)
        self._mips = {}
        return img, Msrc

    def mip(self, img, level):
        key = (id(img), level)
        if key in self._mips:
            return self._mips[key]
        out = img
        for _ in range(level):
            if min(out.shape[:2]) < 4:
                break
            out = cv2.pyrDown(out)
        if len(self._mips) > 6:
            self._mips.clear()
        self._mips[key] = out
        return out


class Comp:
    def __init__(self, spec: dict, project: "ProjectRuntime", scale: float = 1.0):
        self.spec = spec
        self.project = project
        self.W, self.H = int(spec["width"]), int(spec["height"])
        self.fps = float(spec["fps"])
        self.duration = float(spec["duration"])
        self.scale = scale
        self.OW, self.OH = max(2, int(round(self.W * scale))), max(2, int(round(self.H * scale)))
        bg = spec.get("background", "#000000")
        self.bg = parse_color(bg) if bg not in (None, "transparent") else np.zeros(4, np.float32)
        self.layers = [Layer(L, self, i) for i, L in enumerate(spec["layers"])]
        byid = {L.id: L for L in self.layers}
        for L in self.layers:
            if L.parent_id:
                if L.parent_id not in byid:
                    raise ValueError(f"Layer {L.id}: parent '{L.parent_id}' not found")
                L.parent = byid[L.parent_id]
        self.byid = byid
        self.matte_ids = {L.matte["layer"] for L in self.layers if L.matte and not L.matte.get("show_source")}
        cam = spec.get("camera")
        self.camera = Camera(cam, self.W, self.H)
        mb = spec.get("motion_blur", False)
        self.mb = ({"samples": 8, "shutter": 180} | (mb if isinstance(mb, dict) else {})) if mb else None
        self.markers = spec.get("markers", [])

    # ------------------------------------------------------------ camera
    def cam_state(self, t):
        pw = None
        if self.camera.parent:
            pw = self.byid[self.camera.parent].world_matrix(t)
        return self.camera.state(t, pw)

    # ------------------------------------------------------------ geometry
    def screen_homography(self, L: Layer, t: float, cam, Wm=None):
        """3x3 homography layer-plane coords -> output pixels and, for 3D, the camera-space quad info."""
        if Wm is None:
            Wm = L.world_matrix(t, getattr(L, "_last_text_size", None))
        S = np.diag([self.scale, self.scale, 1.0])
        if not L.three_d and L.spec.get("corner_pin") is not None:
            pts = L.corner_pin(t)
            if L.type == "text":
                w, h = getattr(L, "_last_text_size", (1, 1))
            else:
                w, h = L.natural
            rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            return S @ cv2.getPerspectiveTransform(rect, np.float32(pts)).astype(np.float64), None
        if not L.three_d:
            A = np.array([[Wm[0, 0], Wm[0, 1], Wm[0, 3]], [Wm[1, 0], Wm[1, 1], Wm[1, 3]], [0, 0, 1]])
            return S @ A, None
        VW = cam["view"] @ Wm
        K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])
        Hm = S @ K @ VW[:3][:, [0, 1, 3]]
        return Hm, VW

    def warp(self, L: Layer, img, Msrc, Hm, VW, extra_margin=0, z_off=0.0):
        """Warp processed source into an output-space patch. Returns (patch, x0, y0) or None."""
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64)
        clip_poly = None
        if VW is not None and z_off != 0:
            Tz = np.eye(4)
            Tz[2, 3] = z_off
            VW = VW @ Tz
            cam = self._cam_cache
            K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])
            Hm = np.diag([self.scale, self.scale, 1.0]) @ K @ VW[:3][:, [0, 1, 3]]
        H = Hm @ Msrc
        if VW is not None:
            lc = corners @ Msrc.T
            cam_pts = (np.hstack([lc[:, :2], np.zeros((4, 1)), np.ones((4, 1))]) @ VW.T)[:, :3]
            if np.all(cam_pts[:, 2] < NEAR):
                return None
            if np.any(cam_pts[:, 2] < NEAR):
                cp = clip_polygon_near(cam_pts, NEAR)
                if len(cp) < 3:
                    return None
                clip_poly = project_points(cp, self._cam_cache) * self.scale
        if clip_poly is not None:
            scr = clip_poly
        else:
            p = corners @ H.T
            if np.any(p[:, 2] <= 1e-9):
                return None
            scr = p[:, :2] / p[:, 2:3]
        m = 2 + extra_margin
        x0 = int(math.floor(max(0, scr[:, 0].min() - m)))
        y0 = int(math.floor(max(0, scr[:, 1].min() - m)))
        x1 = int(math.ceil(min(self.OW, scr[:, 0].max() + m)))
        y1 = int(math.ceil(min(self.OH, scr[:, 1].max() + m)))
        if x1 <= x0 or y1 <= y0:
            return None
        # mip selection for minification
        area_src = float(w * h)
        sp = scr
        area_dst = 0.5 * abs(np.dot(sp[:, 0], np.roll(sp[:, 1], 1)) - np.dot(sp[:, 1], np.roll(sp[:, 0], 1)))
        ratio = math.sqrt(max(area_dst, 1e-9) / max(area_src, 1e-9))
        level = 0
        if ratio < 0.5 and clip_poly is None:
            level = min(6, int(math.floor(math.log2(1.0 / ratio))))
        if level:
            src = L.mip(img, level)
            fx_ = w / src.shape[1]
            fy_ = h / src.shape[0]
            H = H @ np.diag([fx_, fy_, 1.0])
        else:
            src = img
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        Hr = T @ H
        size = (x1 - x0, y1 - y0)
        interp = cv2.INTER_LINEAR
        if abs(Hr[2, 0]) < 1e-12 and abs(Hr[2, 1]) < 1e-12:
            out = cv2.warpAffine(src, (Hr / Hr[2, 2])[:2], size, flags=interp, borderMode=cv2.BORDER_CONSTANT)
        else:
            out = cv2.warpPerspective(src, Hr, size, flags=interp, borderMode=cv2.BORDER_CONSTANT)
        if clip_poly is not None:
            mk = np.zeros((size[1], size[0]), np.uint8)
            cv2.fillPoly(mk, [np.round((clip_poly - [x0, y0]) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
            out = out * (mk.astype(np.float32) / 255.0)[..., None]
        if out.ndim == 2:
            out = out[..., None]
        return out, x0, y0

    # ------------------------------------------------------------ transitions
    def transition_state(self, L: Layer, t: float):
        lt = t - L.start
        st = {"opacity": 1.0, "offset": np.zeros(2), "scale": 1.0, "blur": 0.0, "dblur": (0.0, 0.0),
              "mask": None, "color": None, "flash": 0.0}
        for which in ("transition_in", "transition_out"):
            tr = L.spec.get(which)
            if not tr:
                continue
            d = float(tr.get("duration", 0.5))
            if d <= 0:
                continue
            if which == "transition_in":
                if lt >= d:
                    continue
                u = lt / d
            else:
                rem = L.duration - lt
                if rem >= d:
                    continue
                u = rem / d
            u = easing.get(tr.get("ease", "in_out_cubic" if which == "transition_in" else "in_out_cubic"))(min(1, max(0, u)))
            typ = tr.get("type", "crossfade")
            inv = 1 - u
            W, Hh = self.W, self.H
            sign = 1 if which == "transition_in" else -1
            if typ in ("crossfade", "fade", "dissolve"):
                st["opacity"] *= u
            elif typ.startswith("dip"):
                col = tr.get("color", "#ffffff" if "white" in typ else "#000000")
                st["color"] = (parse_color(col), inv)
            elif typ == "flash":
                st["flash"] = max(st["flash"], inv)
                st["opacity"] *= min(1, u * 2)
            elif typ.startswith(("slide", "push")):
                dirn = typ.split("_")[1] if "_" in typ else "left"
                v = {"left": (W, 0), "right": (-W, 0), "up": (0, Hh), "down": (0, -Hh)}[dirn]
                st["offset"] = st["offset"] + np.array(v) * inv * sign * float(tr.get("distance", 1.0))
            elif typ.startswith("whip"):
                dirn = typ.split("_")[1] if "_" in typ else "left"
                v = {"left": (W, 0), "right": (-W, 0), "up": (0, Hh), "down": (0, -Hh)}[dirn]
                st["offset"] = st["offset"] + np.array(v) * inv * sign * 0.35
                ang = 0 if dirn in ("left", "right") else 90
                st["dblur"] = (ang, 140 * math.sin(math.pi * min(1, inv)) + 80 * inv)
                st["opacity"] *= min(1, u * 1.6)
            elif typ in ("zoom", "zoom_in", "zoom_out"):
                amt = float(tr.get("amount", 0.25))
                st["scale"] *= (1 - 0.5 * amt * inv) if typ == "zoom_out" else (1 + amt * inv)
                st["opacity"] *= u
                st["blur"] = max(st["blur"], 12 * inv)
            elif typ == "blur":
                st["blur"] = max(st["blur"], float(tr.get("amount", 40)) * inv)
                st["opacity"] *= u
            elif typ.startswith("wipe") or typ == "iris":
                st["mask"] = (typ, u, tr)
            elif typ == "glitch":
                st["glitch"] = inv
                st["opacity"] *= u if u > 0.3 else u * (1 if (int(lt * 30) % 2) else 0.2)
            else:
                raise ValueError(f"Unknown transition '{typ}'")
        return st

    def _transition_mask(self, spec, size, x0, y0):
        typ, u, tr = spec
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx += x0
        yy += y0
        soft = float(tr.get("softness", 60)) * self.scale
        if typ == "iris":
            cx, cy = (np.array(tr.get("center", [self.W / 2, self.H / 2])) * self.scale)
            R = math.hypot(self.OW, self.OH) / 2 * u * 1.05 + soft
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            return np.clip((R - d) / max(1, soft), 0, 1)
        dirn = typ.split("_")[1] if "_" in typ else tr.get("direction", "right")
        ang = {"right": 0, "left": 180, "down": 90, "up": 270}.get(dirn, float(tr.get("angle", 0)))
        a = math.radians(ang)
        proj = xx * math.cos(a) + yy * math.sin(a)
        corners = np.array([[0, 0], [self.OW, 0], [0, self.OH], [self.OW, self.OH]], np.float32)
        pr = corners[:, 0] * math.cos(a) + corners[:, 1] * math.sin(a)
        lo, hi = pr.min() - soft, pr.max() + soft
        edge = lo + (hi - lo) * u
        return np.clip((edge - proj) / max(1, soft), 0, 1)

    # ------------------------------------------------------------ layer render
    def render_layer(self, L: Layer, t: float, cam):
        """Returns (patch premul, x0, y0) in output space, or None."""
        pt = L.spec.get("posterize_time")
        if pt:  # stop-motion / on-twos look: hold each pose for 1/pt s
            t = L.start + math.floor((t - L.start) * float(pt) + 1e-6) / float(pt)
        return self._render_layer(L, t, cam)

    def _render_layer(self, L: Layer, t: float, cam):
        proc = L.processed_source(t)
        if proc is None:
            return None
        img, Msrc = proc
        tr = self.transition_state(L, t)
        lt = t - L.start
        extra = 0
        dof_blur = 0.0
        Wm = L.world_matrix(t, getattr(L, "_last_text_size", None))
        self._cam_cache = cam
        depth = None
        if L.three_d:
            c = (cam["view"] @ Wm @ np.array([*self._layer_center(L, img, Msrc), 0, 1]))[:3]
            depth = c[2]
            if self.camera.dof:
                fd = float(self.camera.p_focus.at(t, t))
                ap = float(self.camera.p_aperture.at(t, t))
                if depth > 1 and ap > 0:
                    dof_blur = min(self.camera.max_blur, ap * abs(1 - fd / depth)) * self.scale
                    extra = int(dof_blur * 2.5)
        blur = tr["blur"] * self.scale
        extra = max(extra, int(blur * 2.5))

        def homog(tt):
            WmT = Wm if tt == t else L.world_matrix(tt, getattr(L, "_last_text_size", None))
            Hm, VW = self.screen_homography(L, tt, cam if tt == t else self.cam_state(tt), WmT)
            if np.any(tr["offset"]) or tr["scale"] != 1.0:
                cx, cy = self.OW / 2, self.OH / 2
                s = tr["scale"]
                ox, oy = tr["offset"] * self.scale
                Tm = np.array([[s, 0, cx - s * cx + ox], [0, s, cy - s * cy + oy], [0, 0, 1]])
                Hm = Tm @ Hm
            return Hm, VW

        slices = self._extrude_slices(L, img, Msrc, lt, t)
        res_list = []
        mb = self.mb and L.motion_blur and L.type != "adjustment"
        times = [t]
        if mb:
            sh = self.mb["shutter"] / 360.0 / self.fps
            n = int(self.mb["samples"])
            times = [t + sh * ((i + 0.5) / n - 0.5) for i in range(n)]
            H0, _ = homog(times[0])
            H1, _ = homog(times[-1])
            Hc, _ = homog(t)
            p = np.array([[0, 0, 1], [img.shape[1], img.shape[0], 1]], float) @ (Msrc.T)
            def proj(Hx):
                q = p @ Hx.T
                return q[:, :2] / q[:, 2:3]
            disp = float(np.max(np.abs(proj(H1) - proj(H0))))
            k = int(min(n, math.ceil(disp / 1.5)))  # ~1 sample per 1.5px of smear: no wasted work
            if k <= 1:
                times = [t]
            else:
                times = [t + sh * ((i + 0.5) / k - 0.5) for i in range(k)]
        for si, tt in enumerate(times):
            Hm, VW = homog(tt)
            for (simg, sM, zoff, col) in slices:
                r = self.warp(L, simg, sM, Hm, VW, extra_margin=extra, z_off=zoff)
                if r is not None:
                    p_, x_, y_ = r
                    if col is not None:  # alpha-only slice -> flat premultiplied color
                        p_ = np.concatenate([p_ * col, p_], -1) if p_.shape[2] == 1 else p_
                    res_list.append((p_, x_, y_, si))
        if not res_list:
            return None
        patch, x0, y0 = self._merge_patches(res_list, len(times))
        # post-warp effects
        if dof_blur > 0.3:
            patch = FX.gblur(patch, dof_blur / 2)
        if blur > 0.3:
            patch = FX.gblur(patch, blur / 2)
        if tr["dblur"][1] > 1:
            patch = FX.fx_directional_blur(patch, {"angle": tr["dblur"][0], "length": tr["dblur"][1]}, FX.FxCtx(unit=self.scale))
        if tr["color"] is not None:
            col, k = tr["color"]
            patch[..., :3] = patch[..., :3] * (1 - k) + col[:3] * patch[..., 3:4] * k
        if tr["flash"] > 0:
            patch[..., :3] = np.minimum(patch[..., 3:4], patch[..., :3] + tr["flash"] * patch[..., 3:4])
        if tr.get("glitch"):
            patch = self._glitch(patch, tr["glitch"], t)
        if L.three_d and self.camera.fog and depth is not None:
            fg = self.camera.fog
            k = np.clip((depth - float(fg.get("near", 1000))) / max(1, float(fg.get("far", 6000)) - float(fg.get("near", 1000))), 0, 1)
            k *= float(fg.get("amount", 1.0))
            fc = parse_color(fg.get("color", "#000000"))[:3]
            patch[..., :3] = patch[..., :3] * (1 - k) + fc * patch[..., 3:4] * k
        if tr["mask"] is not None:
            patch = patch * self._transition_mask(tr["mask"], (patch.shape[1], patch.shape[0]), x0, y0)[..., None]
        if L.matte:
            patch = self._apply_matte(L, patch, x0, y0, t, cam)
        op = L.opacity(t) * tr["opacity"]
        if op < 1:
            patch = patch * op
        return patch, x0, y0

    def _glitch(self, patch, amt, t):
        rng = np.random.default_rng(int(t * 1000))
        h, w = patch.shape[:2]
        out = patch.copy()
        for _ in range(int(6 * amt) + 1):
            y = int(rng.uniform(0, h))
            hh = int(rng.uniform(4, 40) * self.scale)
            dx = int(rng.normal(0, 40 * amt) * self.scale)
            out[y:y + hh] = np.roll(patch[y:y + hh], dx, axis=1)
        out[..., 0] = np.roll(out[..., 0], int(8 * amt * self.scale), axis=1)
        return out

    def _layer_center(self, L, img, Msrc):
        h, w = img.shape[:2]
        c = Msrc @ np.array([w / 2, h / 2, 1.0])
        return c[:2]

    def _extrude_slices(self, L, img, Msrc, lt, t):
        """Returns list of (img, Msrc, z_off, color|None). Extrusion slices are alpha-only (fast) + flat color."""
        ex = L.spec.get("extrude")
        if not ex:
            return [(img, Msrc, 0.0, None)]
        depth = float(L.prop("ex_depth", ex.get("depth", 30)).at(lt, t))
        steps = int(ex.get("steps", max(4, min(40, int(abs(depth) / (1.5 if L.three_d else 1.0))))))
        a = np.ascontiguousarray(img[..., 3])
        side = ex.get("color")
        if side is None:
            w = float(a.sum())
            base = (img[..., :3].reshape(-1, 3).sum(0) / max(w, 1e-6)) * 0.45 if w > 0 else np.zeros(3)
        else:
            base = parse_color(side)[:3]
        out = []
        ang = math.radians(float(ex.get("angle", 45)))
        dx, dy = math.cos(ang), math.sin(ang)
        for k in range(steps, 0, -1):
            shade = 0.72 + 0.28 * (1 - k / steps)
            col = (np.asarray(base, np.float32) * shade).astype(np.float32)
            if L.three_d:
                out.append((a, Msrc, depth * k / steps, col))
            else:
                off = depth * k / steps
                out.append((a, np.array([[1, 0, dx * off], [0, 1, dy * off], [0, 0, 1]]) @ Msrc, 0.0, col))
        out.append((img, Msrc, 0.0, None))
        return out

    def _merge_patches(self, items, n_times):
        """items: (patch, x0, y0, sample_idx). Slices are over-composited per sample; samples averaged."""
        if len(items) == 1:
            return items[0][:3]
        x0 = min(i[1] for i in items)
        y0 = min(i[2] for i in items)
        x1 = max(i[1] + i[0].shape[1] for i in items)
        y1 = max(i[2] + i[0].shape[0] for i in items)
        acc = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        if n_times == 1:
            for p, px, py, _ in items:
                composite(acc, p, px - x0, py - y0, "normal")
            return acc, x0, y0
        for k in range(n_times):
            group = [it for it in items if it[3] == k]
            if not group:
                continue
            if len(group) == 1:
                p, px, py, _ = group[0]
                acc[py - y0:py - y0 + p.shape[0], px - x0:px - x0 + p.shape[1]] += p
            else:
                c = np.zeros_like(acc)
                for p, px, py, _ in group:
                    composite(c, p, px - x0, py - y0, "normal")
                acc += c
        return acc / n_times, x0, y0

    def _apply_matte(self, L, patch, x0, y0, t, cam):
        m = L.matte
        src = self.byid.get(m["layer"])
        if src is None:
            raise ValueError(f"Matte layer '{m['layer']}' not found")
        h, w = patch.shape[:2]
        alpha = np.zeros((h, w), np.float32)
        if src.start - 1e-6 <= t < src.end - 1e-6:
            r = self.render_layer(src, t, cam)
            if r is not None:
                mp, mx, my = r
                buf = np.zeros((h, w, 4), np.float32)
                composite(buf, mp, mx - x0, my - y0, "normal")
                mode = m.get("mode", "alpha")
                if mode.startswith("luma"):
                    rgb = np.where(buf[..., 3:4] > 1e-6, buf[..., :3] / np.maximum(buf[..., 3:4], 1e-6), 0)
                    alpha = FX.luminance(rgb) * buf[..., 3]
                else:
                    alpha = buf[..., 3]
                if mode.endswith(("inv", "inverted")):
                    alpha = 1 - alpha
        elif m.get("mode", "alpha").endswith(("inv", "inverted")):
            alpha = np.ones((h, w), np.float32)
        return patch * alpha[..., None]

    # ------------------------------------------------------------ adjustment layers
    def apply_adjustment(self, L: Layer, canvas: np.ndarray, t: float, cam):
        lt = t - L.start
        ev = L._fx_eval(lt, t)
        ctx = FX.FxCtx(t=lt, comp_t=t, fps=self.fps, unit=self.scale, seed=L.seed, comp_w=self.W, comp_h=self.H,
                       base_dir=self.project.base_dir)
        fx = FX.apply_effects(canvas.copy(), L.effects, ev, ctx)
        cov = np.ones((self.OH, self.OW), np.float32)
        if L.masks or L.matte or L.spec.get("transition_in") or L.spec.get("transition_out"):
            ones = np.ones((self.OH, self.OW, 4), np.float32)
            M = np.diag([1 / self.scale, 1 / self.scale, 1.0])
            ones = L.apply_masks(ones, M, lt, t)
            tr = self.transition_state(L, t)
            c = ones[..., 3] * tr["opacity"]
            if tr["mask"] is not None:
                c = c * self._transition_mask(tr["mask"], (self.OW, self.OH), 0, 0)
            if L.matte:
                c = self._apply_matte(L, c[..., None].repeat(4, -1), 0, 0, t, cam)[..., 3]
            cov = c
        k = (cov * L.opacity(t))[..., None]
        return canvas + (fx - canvas) * k

    # ------------------------------------------------------------ frame
    def render(self, t: float) -> np.ndarray:
        canvas = np.empty((self.OH, self.OW, 4), np.float32)
        c = self.project.color.color(self.bg)
        canvas[...] = np.array([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]], np.float32)
        cam = self.cam_state(t)
        active = [L for L in self.layers if L.active(t) and L.type not in ("null", "light") and L.id not in self.matte_ids]
        lights = [L for L in self.layers if L.type == "light" and L.active(t)]
        use_gl = self.spec.get("renderer") == "gl" or bool(lights) or any(L.type == "model" for L in active)
        i = 0
        while i < len(active):
            L = active[i]
            if L.three_d and L.type != "adjustment":
                run = []
                while i < len(active) and active[i].three_d and active[i].type != "adjustment":
                    run.append(active[i])
                    i += 1
                if use_gl:
                    self._render_gl_run(run, lights, canvas, t, cam)
                    continue
                keyed = []
                for k, R in enumerate(run):
                    Wm = R.world_matrix(t, getattr(R, "_last_text_size", None))
                    d = (cam["view"] @ Wm @ np.array([*self._anchor_of(R, t), 0, 1.0]))[2]
                    keyed.append((-d, k, R))
                keyed.sort(key=lambda x: (x[0], x[1]))
                for _, _, R in keyed:
                    self._draw(R, canvas, t, cam)
                continue
            if L.type == "adjustment":
                canvas = self.apply_adjustment(L, canvas, t, cam)
            else:
                self._draw(L, canvas, t, cam)
            i += 1
        if self.spec.get("name") == "main" and self.project.color.linear:
            canvas = self.project.color.output_frame(canvas)
        return canvas

    def _gl_scene(self, run, light_layers, t):
        from . import gl3d
        items = []
        for R in run:
            lt = t - R.start
            tr = self.transition_state(R, t)
            op = R.opacity(t) * tr["opacity"]
            if op <= 0.001:
                continue
            Wm = R.world_matrix(t, getattr(R, "_last_text_size", None))
            if R.type == "model":
                mat = R.material(lt, t)
                meshes = R.meshes(lt)
                col = self.project.color
                if col.linear:
                    if mat.get("color") is not None:
                        mat["color"] = col.color(mat["color"])
                    if mat.get("_tex_img") is not None:
                        mat["_tex_img"] = R._lin_tex(mat["_tex_img"])
                    meshes = R._lin_meshes(meshes)
                items.append(gl3d.Item(meshes, Wm, mat, R.id, opacity=op,
                                       cast=R.spec.get("casts_shadows", True), recv=R.spec.get("accepts_shadows", True),
                                       lit=R.spec.get("accepts_lights", True)))
                continue
            img, Msrc = R.processed_source(t)
            dyn = (not R._cache_val or not R._cache_val[2]) or R.effects_time_dependent()
            for si, (simg, sM, zoff, col) in enumerate(self._extrude_slices(R, img, Msrc, lt, t)):
                if col is not None:
                    a = simg if simg.ndim == 3 else simg[..., None]
                    simg = np.concatenate([a * col, a], -1).astype(np.float32)
                Tz = np.eye(4)
                Tz[2, 3] = zoff
                key = (id(R), si, id(img), round(t, 5) if dyn else 0)
                mesh = gl3d.card_mesh(simg.shape[1], simg.shape[0], sM)
                mat = {"specular": float(R.spec.get("specular", 0.0)), "shininess": float(R.spec.get("shininess", 16))}
                items.append(gl3d.Item([mesh], Wm @ Tz, mat, key, card_tex=(key, simg), opacity=op,
                                       cast=R.spec.get("casts_shadows", False), recv=R.spec.get("accepts_shadows", True),
                                       lit=R.spec.get("accepts_lights", True)))
        lights = [L.light_info(t, L.world_matrix(t)) for L in light_layers if L.active(t)]
        if not lights and any(it.meshes and it.card_tex is None for it in items) and not self._env():
            lights = [{"type": 0, "color": np.array([0.35, 0.35, 0.38], np.float32)},
                      {"type": 1, "color": np.array([0.85, 0.83, 0.8], np.float32),
                       "dir": np.array([0.45, 0.6, 0.66], np.float32) / np.linalg.norm([0.45, 0.6, 0.66])}]
            for it in items:
                if it.card_tex is not None:
                    it.lit = False
        return items, lights

    def _env(self):
        """Environment map (equirect, LDR or .hdr/.exr) for reflections, IBL and optional sky background."""
        e = self.spec.get("environment")
        if not e:
            return None
        if getattr(self, "_env_cache", None) is None:
            src = e["src"] if os.path.isabs(e["src"]) else os.path.join(self.project.base_dir, e["src"])
            img = media.load_image_hi(src)
            if src.lower().endswith((".hdr", ".exr")):
                img[..., :3] = img[..., :3]  # already linear radiance
            elif self.project.color.linear:
                from .color import convert, SRGB
                img[..., :3] = convert(img[..., :3], SRGB, self.project.color.working)
            img[..., 3] = 1.0
            self._env_cache = dict(e, img=np.ascontiguousarray(img, np.float32))
        return self._env_cache

    def _render_gl_run(self, run, light_layers, canvas, t, cam):
        from . import gl3d
        times = [t]
        if self.mb and any(R.motion_blur for R in run):
            sh = self.mb["shutter"] / 360.0 / self.fps
            n = int(min(8, self.mb["samples"]))
            # adaptive: estimate screen motion of item centres + camera over the shutter
            def centres(tt):
                c_ = self.cam_state(tt)
                pts = []
                for R in run:
                    p = c_["view"] @ R.world_matrix(tt, getattr(R, "_last_text_size", None)) @ np.array([0, 0, 0, 1.0])
                    if p[2] > 1:
                        pts.append([c_["f"] * p[0] / p[2], c_["f"] * p[1] / p[2]])
                return np.array(pts) * self.scale
            a_, b_ = centres(t - sh / 2), centres(t + sh / 2)
            disp = float(np.max(np.abs(a_ - b_))) if len(a_) and len(a_) == len(b_) else 0.0
            k = int(min(n, math.ceil(disp / 1.5)))
            if k > 1:
                times = [t + sh * ((i + 0.5) / k - 0.5) for i in range(k)]
        cs = self.camera
        acc = None
        depth = None
        settings = dict(self.spec.get("gl", {}))
        settings["need_depth"] = bool(cs.dof or cs.fog)
        env = self._env()
        for tt in times:
            items, lights = self._gl_scene(run, light_layers, tt)
            if not items:
                return
            camt = cam if tt == t else self.cam_state(tt)
            rgba, dep = gl3d.render(items, lights, camt, self.OW, self.OH, self.scale, settings, env)
            acc = rgba if acc is None else acc + rgba
            if tt == times[len(times) // 2]:
                depth = dep
        rgba = acc / len(times)
        dof = None
        if cs.dof:
            dof = {"focus": float(cs.p_focus.at(t, t)), "aperture": float(cs.p_aperture.at(t, t)), "max": cs.max_blur}
        fog = None
        if cs.fog:
            fg = cs.fog
            fog = {"near": float(fg.get("near", 1000)), "far": float(fg.get("far", 6000)), "amount": float(fg.get("amount", 1)),
                   "color": parse_color(fg.get("color", "#000"))[:3]}
        rgba = gl3d.post_dof_fog(rgba, depth, dof, fog, self.scale)
        composite(canvas, rgba, 0, 0, "normal")

    def _anchor_of(self, L, t):
        a = L.tf.get("anchor")
        if a is not None and not isinstance(a, (dict, str)):
            return as_vec(a, 2)
        da = L.default_anchor()
        if da is None:
            ts = getattr(L, "_last_text_size", (0, 0))
            return [ts[0] / 2, ts[1] / 2]
        return da[:2]

    def _draw(self, L, canvas, t, cam):
        echo = L.spec.get("echo")
        if echo:  # trails: earlier copies drawn first, fading
            n = int(echo.get("count", 3))
            for k in range(n, 0, -1):
                tk = t - k * float(echo.get("delay", 0.05))
                if tk < L.start:
                    continue
                r = self.render_layer(L, tk, cam)
                if r is not None:
                    composite(canvas, r[0] * float(echo.get("decay", 0.5)) ** k, r[1], r[2], echo.get("blend", L.blend))
        r = self.render_layer(L, t, cam)
        if r is None:
            return
        patch, x0, y0 = r
        composite(canvas, patch, x0, y0, L.blend)


class ProjectRuntime:
    def __init__(self, normalized: dict, scale: float = 1.0, proxies: bool = False):
        self.data = normalized
        self.proxies = proxies
        self.base_dir = normalized["base_dir"]
        self.comps_spec = normalized["comps"]
        self.scale = scale
        from .color import ColorSettings
        self.color = ColorSettings(normalized["comps"]["main"].get("color"))
        self._comps: dict = {}
        self.main = self.get_comp("main")

    def get_comp(self, name) -> Comp:
        if name not in self._comps:
            if name not in self.comps_spec:
                raise ValueError(f"Unknown comp '{name}'")
            self._comps[name] = Comp(self.comps_spec[name], self, self.scale)
        return self._comps[name]
