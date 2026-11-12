"""OpenGL 3D renderer (moderngl): meshes + textured card layers in one depth buffer, up to 8 lights
(ambient / directional / point / spot), PCF shadow maps, Blinn-Phong or toon (cel) shading with
inverted-hull outlines, MSAA, and depth-based DOF + fog in post. Headless via EGL; GPU when present."""
from __future__ import annotations

import math
import os

import numpy as np

MAX_LIGHTS = 8
MAX_SHADOWS = 4
NEAR, FAR = 5.0, 60000.0

VS = """
#version 330
in vec3 in_pos; in vec3 in_normal; in vec2 in_uv; in vec4 in_col;
uniform mat4 u_model; uniform mat4 u_viewproj; uniform mat3 u_nmat; uniform float u_outline;
uniform mat4 u_lvp[4];
out vec3 v_world; out vec3 v_normal; out vec2 v_uv; out vec4 v_col; out vec4 v_lp[4];
void main(){
  vec4 w = u_model * vec4(in_pos, 1.0);
  vec3 n = normalize(u_nmat * in_normal);
  w.xyz += n * u_outline;
  v_world = w.xyz; v_normal = n; v_uv = in_uv; v_col = in_col;
  for (int i = 0; i < 4; i++) v_lp[i] = u_lvp[i] * w;
  gl_Position = u_viewproj * w;
}"""

FS = """
#version 330
in vec3 v_world; in vec3 v_normal; in vec2 v_uv; in vec4 v_col; in vec4 v_lp[4];
uniform sampler2D u_tex; uniform int u_use_tex; uniform vec4 u_color; uniform float u_opacity;
uniform int u_lit; uniform int u_nl; uniform int u_ltype[8]; uniform vec3 u_lcol[8]; uniform vec3 u_lpos[8];
uniform vec3 u_ldir[8]; uniform float u_lco[8]; uniform float u_lci[8]; uniform float u_lrange[8];
uniform int u_lsh[8]; uniform float u_ldark[8]; uniform float u_lsoft[8];
uniform sampler2D u_sh0; uniform sampler2D u_sh1; uniform sampler2D u_sh2; uniform sampler2D u_sh3;
uniform vec3 u_campos; uniform float u_spec; uniform float u_shin; uniform vec3 u_emis; uniform int u_toon;
uniform int u_pass; uniform vec4 u_ocol; uniform int u_recv; uniform float u_rim;
uniform int u_pbr; uniform float u_metal; uniform float u_rough;
uniform sampler2D u_env; uniform int u_has_env; uniform float u_env_int; uniform float u_env_rot; uniform float u_env_lod;
uniform int u_psh; uniform mat4 u_pvp[6]; uniform vec3 u_ppos; uniform float u_pdark; uniform float u_psoft;
uniform sampler2D u_pf0; uniform sampler2D u_pf1; uniform sampler2D u_pf2; uniform sampler2D u_pf3; uniform sampler2D u_pf4; uniform sampler2D u_pf5;
out vec4 f;
const float PI = 3.14159265;
vec2 dir2uv(vec3 d){ d = normalize(d); float lon = atan(d.x, d.z) + u_env_rot; float lat = asin(clamp(-d.y, -1.0, 1.0));
  return vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI); }
vec3 envAt(vec3 d, float lod){ return textureLod(u_env, dir2uv(d), lod).rgb * u_env_int; }
float pointShadow(vec3 wp){
  vec3 L = wp - u_ppos; vec3 a = abs(L); int face;
  if (a.x >= a.y && a.x >= a.z) face = L.x > 0.0 ? 0 : 1; else if (a.y >= a.z) face = L.y > 0.0 ? 2 : 3; else face = L.z > 0.0 ? 4 : 5;
  vec4 lp = u_pvp[face] * vec4(wp, 1.0); vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
  if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
  vec2 ts = 1.0 / vec2(textureSize(u_pf0, 0)); float s = 0.0; float n = 0.0;
  for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++) {
    vec2 q = p.xy + vec2(x, y) * ts * u_psoft; float d;
    if (face == 0) d = texture(u_pf0, q).r; else if (face == 1) d = texture(u_pf1, q).r; else if (face == 2) d = texture(u_pf2, q).r;
    else if (face == 3) d = texture(u_pf3, q).r; else if (face == 4) d = texture(u_pf4, q).r; else d = texture(u_pf5, q).r;
    s += (p.z - 0.0008 > d) ? 0.0 : 1.0; n += 1.0; }
  return s / n;
}
float shadowAt(int k, vec4 lp, float soft){
  vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
  if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
  vec2 ts = 1.0 / vec2(textureSize(u_sh0, 0));
  float bias = 0.0006; float s = 0.0; float n = 0.0;
  for (int x = -2; x <= 2; x++) for (int y = -2; y <= 2; y++) {
    vec2 o = vec2(x, y) * ts * soft; float d;
    if (k == 0) d = texture(u_sh0, p.xy + o).r; else if (k == 1) d = texture(u_sh1, p.xy + o).r;
    else if (k == 2) d = texture(u_sh2, p.xy + o).r; else d = texture(u_sh3, p.xy + o).r;
    s += (p.z - bias > d) ? 0.0 : 1.0; n += 1.0;
  }
  return s / n;
}
void main(){
  if (u_pass == 3) { f = u_ocol; return; }
  vec4 t = (u_use_tex == 1) ? texture(u_tex, v_uv) : vec4(1.0);
  vec4 b = vec4(t.rgb * u_color.rgb * v_col.rgb, t.a * v_col.a) * u_color.a * u_opacity;
  if (b.a < 0.002) discard;
  if (u_pass == 0 && b.a < 0.995) discard;
  if (u_pass == 1 && b.a >= 0.995) discard;
  if (u_lit == 0) { f = b; return; }
  vec3 base = b.rgb / b.a;
  vec3 V = normalize(u_campos - v_world);
  vec3 N = normalize(v_normal);
  if (dot(N, V) < 0.0) N = -N;
  vec3 diff = vec3(0.0); vec3 spec = vec3(0.0);
  float rough = clamp(u_rough, 0.04, 1.0); float a2 = rough * rough * rough * rough;
  vec3 F0 = mix(vec3(0.04), base, u_metal); float NdV = max(dot(N, V), 1e-4);
  for (int i = 0; i < 8; i++) {
    if (i >= u_nl) break;
    if (u_ltype[i] == 0) { diff += u_lcol[i]; continue; }
    vec3 L; float att = 1.0;
    if (u_ltype[i] == 1) { L = -u_ldir[i]; }
    else {
      vec3 d = u_lpos[i] - v_world; float dist = length(d); L = d / max(dist, 1e-4);
      if (u_lrange[i] > 0.0) { float q = clamp(1.0 - dist / u_lrange[i], 0.0, 1.0); att = q * q; }
      if (u_ltype[i] == 3) att *= smoothstep(u_lco[i], u_lci[i], dot(-L, u_ldir[i]));
    }
    float ndl = max(dot(N, L), 0.0);
    if (u_toon > 0) ndl = floor(ndl * float(u_toon) + 0.35) / float(u_toon);
    float vis = 1.0;
    if (u_lsh[i] >= 0 && u_recv == 1) {
      int k = u_lsh[i]; vec4 lp = (k == 0) ? v_lp[0] : (k == 1) ? v_lp[1] : (k == 2) ? v_lp[2] : v_lp[3];
      vis = 1.0 - u_ldark[i] * (1.0 - shadowAt(k, lp, u_lsoft[i]));
    }
    if (u_psh == i && u_recv == 1) vis *= 1.0 - u_pdark * (1.0 - pointShadow(v_world));
    vec3 H = normalize(L + V);
    if (u_pbr == 1) {
      float NdH = max(dot(N, H), 0.0); float VdH = max(dot(V, H), 0.0);
      float D = a2 / (PI * pow(NdH * NdH * (a2 - 1.0) + 1.0, 2.0));
      float kk = (rough + 1.0) * (rough + 1.0) / 8.0;
      float G = NdV / (NdV * (1.0 - kk) + kk) * max(ndl, 1e-4) / (max(ndl, 1e-4) * (1.0 - kk) + kk);
      vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdH, 5.0);
      vec3 sp = D * G * F / max(4.0 * NdV * max(ndl, 1e-4), 1e-4);
      diff += u_lcol[i] * ndl * att * vis * (1.0 - F) * (1.0 - u_metal);
      spec += u_lcol[i] * sp * ndl * att * vis * PI;
    } else {
      diff += u_lcol[i] * ndl * att * vis;
      float sp = pow(max(dot(N, H), 0.0), max(u_shin, 1.0));
      if (u_toon > 0) sp = step(0.5, sp);
      spec += u_lcol[i] * sp * u_spec * att * vis * step(0.0001, ndl);
    }
  }
  if (u_has_env == 1) {
    vec3 R = reflect(-V, N);
    vec3 Fe = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - NdV, 5.0);
    if (u_pbr == 1) {
      spec += envAt(R, rough * u_env_lod) * Fe;
      diff += envAt(N, u_env_lod - 1.0) * (1.0 - u_metal) * (1.0 - Fe);
    } else {
      diff += envAt(N, u_env_lod - 1.0) * 0.6;
      spec += envAt(R, 2.0) * u_spec * 0.5;
    }
  }
  float rim = u_rim * pow(1.0 - max(dot(N, V), 0.0), 3.0);
  vec3 rgb = base * diff + spec + u_emis + rim;
  f = vec4(rgb * b.a, b.a);
}"""

SKY_VS = """
#version 330
in vec2 in_pos; out vec2 v_ndc;
void main(){ v_ndc = in_pos; gl_Position = vec4(in_pos, 0.99999, 1.0); }"""
SKY_FS = """
#version 330
in vec2 v_ndc; uniform mat4 u_invvp; uniform sampler2D u_env; uniform float u_env_int; uniform float u_env_rot;
uniform float u_blur; out vec4 f; const float PI = 3.14159265;
void main(){ vec4 a = u_invvp * vec4(v_ndc, -1.0, 1.0); vec4 b = u_invvp * vec4(v_ndc, 1.0, 1.0);
  vec3 d = normalize(b.xyz / b.w - a.xyz / a.w);
  float lon = atan(d.x, d.z) + u_env_rot; float lat = asin(clamp(-d.y, -1.0, 1.0));
  f = vec4(textureLod(u_env, vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI), u_blur).rgb * u_env_int, 1.0); }"""

SH_VS = """
#version 330
in vec3 in_pos; in vec2 in_uv;
uniform mat4 u_model; uniform mat4 u_vp;
out vec2 v_uv;
void main(){ v_uv = in_uv; gl_Position = u_vp * u_model * vec4(in_pos, 1.0); }"""
SH_FS = """
#version 330
in vec2 v_uv; uniform sampler2D u_tex; uniform int u_use_tex; uniform float u_cut;
void main(){ if (u_use_tex == 1 && texture(u_tex, v_uv).a < u_cut) discard; }"""


class GL:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None or cls._inst.pid != os.getpid():
            cls._inst = GL()
        return cls._inst

    def __init__(self):
        import moderngl
        self.mgl = moderngl
        self.pid = os.getpid()
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception:
            self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.sprog = self.ctx.program(vertex_shader=SH_VS, fragment_shader=SH_FS)
        self.skyprog = self.ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], np.float32)
        self.skyvao = self.ctx.vertex_array(self.skyprog, [(self.ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.white = self.ctx.texture((1, 1), 4, np.ones(4, np.float32).tobytes(), dtype="f4")
        self.fbos = {}
        self.shadow_maps = {}
        self.tex_cache = {}
        self.mesh_cache = {}

    def fbo(self, w, h, samples):
        key = (w, h, samples)
        if key not in self.fbos:
            c = self.ctx
            ms = c.framebuffer(c.renderbuffer((w, h), 4, samples=samples, dtype="f4"),
                               c.depth_renderbuffer((w, h), samples=samples)) if samples > 1 else None
            out_tex = c.texture((w, h), 4, dtype="f4")
            dep = c.depth_texture((w, h))
            out = c.framebuffer(out_tex, dep)
            self.fbos[key] = (ms, out, out_tex, dep)
        return self.fbos[key]

    def shadow_map(self, k, size):
        key = (k, size)
        if key not in self.shadow_maps:
            d = self.ctx.depth_texture((size, size))
            d.compare_func = ""
            d.repeat_x = d.repeat_y = False
            self.shadow_maps[key] = (self.ctx.framebuffer(depth_attachment=d), d)
        return self.shadow_maps[key]

    def texture(self, key, img):
        if key in self.tex_cache:
            return self.tex_cache[key]
        if len(self.tex_cache) > 48:
            for t in self.tex_cache.values():
                t.release()
            self.tex_cache.clear()
            getattr(self, "_tex_refs", {}).clear()
        img = np.ascontiguousarray(img, np.float32)
        t = self.ctx.texture((img.shape[1], img.shape[0]), 4, img.tobytes(), dtype="f4")
        t.build_mipmaps()
        t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        t.anisotropy = 8.0
        t.repeat_x = t.repeat_y = False
        self.tex_cache[key] = t
        self._tex_refs = getattr(self, "_tex_refs", {})
        self._tex_refs[key] = img  # keep source alive so id-based keys can't be recycled
        return t

    def mesh_vao(self, key, m, prog):
        k = (id(m), id(prog))  # keyed by the Mesh object itself: layer ids can repeat across comps/precomps
        if k in self.mesh_cache:
            return self.mesh_cache[k]
        n = len(m.pos)
        col = m.vcolor if m.vcolor is not None else np.ones((n, 4), np.float32)
        if col.shape[1] == 3:
            col = np.hstack([col, np.ones((n, 1), np.float32)])
        data = np.hstack([m.pos, m.nrm, m.uv, col.astype(np.float32)]).astype(np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(m.idx.astype(np.uint32).tobytes())
        fmt = ["3f 3f 2f 4f", "in_pos", "in_normal", "in_uv", "in_col"] if prog is self.prog else ["3f 12x 2f 16x", "in_pos", "in_uv"]
        vao = self.ctx.vertex_array(prog, [(vbo, fmt[0], *fmt[1:])], ibo)
        self.mesh_cache[k] = (vao, vbo, ibo, m)  # hold a ref so id(m) can't be recycled while cached
        if len(self.mesh_cache) > 256:
            for kk in list(self.mesh_cache)[:64]:
                v = self.mesh_cache.pop(kk)
                v[0].release(); v[1].release(); v[2].release()
        return self.mesh_cache[k]


def _persp(f, cx, cy, W, H):
    P = np.zeros((4, 4))
    P[0, 0], P[0, 2] = 2 * f / W, 2 * cx / W - 1
    P[1, 1], P[1, 2] = 2 * f / H, 2 * cy / H - 1
    P[2, 2], P[2, 3] = (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)
    P[3, 2] = 1
    return P


def _look(pos, target, up=(0, -1, 0)):
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    up = np.asarray(up, float)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0, 0, 1.0])
    r = np.cross(fwd, up)
    r /= np.linalg.norm(r)
    d = np.cross(fwd, r)
    V = np.eye(4)
    V[:3, :3] = np.stack([r, d, fwd])
    V[:3, 3] = -V[:3, :3] @ pos
    return V


def _ortho(half, near, far):
    P = np.eye(4)
    P[0, 0] = P[1, 1] = 1 / half
    P[2, 2] = 2 / (far - near)
    P[2, 3] = -(far + near) / (far - near)
    return P


def _m4(a):
    return np.asarray(a, np.float32).T.copy().tobytes()  # column-major


def _m3(a):
    return np.asarray(a, np.float32).T.copy().tobytes()


class Item:
    """A drawable: mesh (list of Mesh) + model matrix + material + texture info."""

    def __init__(self, meshes, model, mat, key, card_tex=None, opacity=1.0, cast=True, recv=True, lit=True):
        self.meshes, self.model, self.mat, self.key = meshes, model, mat, key
        self.card_tex = card_tex
        self.opacity = opacity
        self.cast, self.recv, self.lit = cast, recv, lit


def card_mesh(w, h, Msrc):
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ Msrc.T
    pos = np.array([[c[0], c[1], 0] for c in corners], np.float32)
    from .models import Mesh
    return Mesh(pos, np.tile([0, 0, -1.0], (4, 1)), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                np.array([0, 1, 2, 0, 2, 3], np.uint32))


def render(items, lights, cam, OW, OH, scale, settings, env=None):
    """items: list[Item]; lights: list of dict(type,color(rgb*intensity),pos,dir,cos_outer,cos_inner,range,
    shadows,darkness,softness); cam: Comp.cam_state dict. Returns (rgba premul float32, linear depth)."""
    g = GL.get()
    ctx = g.ctx
    samples = int(settings.get("msaa", 4))
    ms, out, out_tex, dep = g.fbo(OW, OH, samples)
    P = _persp(cam["f"] * scale, cam["cx"] * scale, cam["cy"] * scale, OW, OH)
    VP = P @ cam["view"]
    campos = cam["pos"]
    # ---- bounds for shadow fitting
    pts = []
    for it in items:
        for m in it.meshes:
            lo, hi = m.pos.min(0), m.pos.max(0)
            box = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts.append((box @ it.model.T)[:, :3])
    allp = np.vstack(pts) if pts else np.zeros((1, 3))
    center = (allp.min(0) + allp.max(0)) / 2
    radius = float(np.linalg.norm(allp.max(0) - allp.min(0)) / 2) + 1.0
    # ---- shadow passes
    lvps = [np.eye(4)] * 4
    sh_tex = [g.shadow_map(0, 16)[1]] * 4
    k = 0
    size = int(settings.get("shadow_size", 2048))
    ctx.enable(g.mgl.DEPTH_TEST)
    ctx.disable(g.mgl.BLEND)
    ctx.disable(g.mgl.CULL_FACE)
    for L in lights:
        L["sh"] = -1
        if not L.get("shadows") or L["type"] not in (1, 3) or k >= MAX_SHADOWS:
            continue
        if L["type"] == 1:
            d = L["dir"]
            pos = center - d * radius * 2
            V = _look(pos, center)
            Pl = _ortho(radius, radius * 0.5, radius * 3.5)
        else:
            pos = L["pos"]
            V = _look(pos, pos + L["dir"])
            fov = math.degrees(math.acos(max(-1, min(1, L["cos_outer"])))) * 2 + 4
            fl = 1 / math.tan(math.radians(min(170, fov)) / 2)
            corners = np.array([[x, y, z, 1] for x in (allp.min(0)[0], allp.max(0)[0])
                                for y in (allp.min(0)[1], allp.max(0)[1]) for z in (allp.min(0)[2], allp.max(0)[2])])
            zl = (corners @ V.T)[:, 2]
            far = max(10.0, float(zl.max()) * 1.05)
            nr = max(far / 500.0, float(zl.min()) * 0.9, 1.0)
            Pl = np.zeros((4, 4))
            Pl[0, 0] = Pl[1, 1] = fl
            Pl[2, 2], Pl[2, 3] = (far + nr) / (far - nr), -2 * far * nr / (far - nr)
            Pl[3, 2] = 1
        lvp = Pl @ V
        fb, tex = g.shadow_map(k, size)
        fb.use()
        fb.clear(depth=1.0)
        ctx.viewport = (0, 0, size, size)
        g.sprog["u_vp"].write(_m4(lvp))
        for it in items:
            if not it.cast:
                continue
            g.sprog["u_model"].write(_m4(it.model))
            for mi, m in enumerate(it.meshes):
                tex_obj = _item_tex(g, it, mi, m)
                g.sprog["u_use_tex"].value = 1 if tex_obj is not None else 0
                g.sprog["u_cut"].value = 0.5
                if tex_obj is not None:
                    tex_obj.use(0)
                    g.sprog["u_tex"].value = 0
                vao = g.mesh_vao((it.key, mi, "s"), m, g.sprog)[0]
                vao.render()
        lvps[k] = lvp
        sh_tex[k] = tex
        L["sh"] = k
        k += 1
    # ---- point-light shadow (one light): 6 x 90deg perspective maps
    pvps = [np.eye(4)] * 6
    pf_tex = [g.shadow_map(0, 16)[1]] * 6
    psh = -1
    pinfo = None
    for li, L in enumerate(lights):
        if L["type"] == 2 and L.get("shadows"):
            psh, pinfo = li, L
            break
    if pinfo is not None:
        psize = int(settings.get("point_shadow_size", 1024))
        pos = pinfo["pos"]
        dists = np.linalg.norm(allp - pos, axis=1)
        far = max(10.0, float(dists.max()) * 1.05)
        nr = max(1.0, far / 2000.0)
        fl = 1 / math.tan(math.radians(92) / 2)
        Pp = np.zeros((4, 4))
        Pp[0, 0] = Pp[1, 1] = fl
        Pp[2, 2], Pp[2, 3] = (far + nr) / (far - nr), -2 * far * nr / (far - nr)
        Pp[3, 2] = 1
        axes = [((1, 0, 0), (0, -1, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, -1)),
                ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, -1, 0))]
        ctx.enable(g.mgl.DEPTH_TEST)
        ctx.disable(g.mgl.BLEND)
        for fi, (ax, up) in enumerate(axes):
            V = _look(pos, pos + np.array(ax, float), up)
            lvp = Pp @ V
            fb, tex = g.shadow_map(10 + fi, psize)
            fb.use()
            fb.clear(depth=1.0)
            ctx.viewport = (0, 0, psize, psize)
            g.sprog["u_vp"].write(_m4(lvp))
            for it in items:
                if not it.cast:
                    continue
                g.sprog["u_model"].write(_m4(it.model))
                for mi, m in enumerate(it.meshes):
                    tex_obj = _item_tex(g, it, mi, m)
                    g.sprog["u_use_tex"].value = 1 if tex_obj is not None else 0
                    g.sprog["u_cut"].value = 0.5
                    if tex_obj is not None:
                        tex_obj.use(0)
                        g.sprog["u_tex"].value = 0
                    g.mesh_vao((it.key, mi, "s"), m, g.sprog)[0].render()
            pvps[fi] = lvp
            pf_tex[fi] = tex
    # ---- main pass
    target = ms or out
    target.use()
    ctx.viewport = (0, 0, OW, OH)
    target.clear(0, 0, 0, 0, depth=1.0)
    p = g.prog
    p["u_viewproj"].write(_m4(VP))
    p["u_campos"].value = tuple(float(x) for x in campos)
    p["u_lvp"].write(b"".join(_m4(m) for m in lvps))
    for i, name in enumerate(["u_sh0", "u_sh1", "u_sh2", "u_sh3"]):
        sh_tex[i].use(4 + i)
        p[name].value = 4 + i
    for i in range(6):
        pf_tex[i].use(8 + i)
        p[f"u_pf{i}"].value = 8 + i
    p["u_psh"].value = psh
    p["u_pvp"].write(b"".join(_m4(m) for m in pvps))
    if pinfo is not None:
        p["u_ppos"].value = tuple(float(x) for x in pinfo["pos"])
        p["u_pdark"].value = float(pinfo.get("darkness", 1.0))
        p["u_psoft"].value = float(pinfo.get("softness", 1.5))
    env_tex = None
    if env is not None:
        env_tex = g.texture(("env", id(env["img"])), env["img"])
        env_tex.repeat_x = True
        env_tex.use(14)
        p["u_env"].value = 14
        p["u_has_env"].value = 1
        p["u_env_int"].value = float(env.get("intensity", 1.0))
        p["u_env_rot"].value = math.radians(float(env.get("rotation", 0)))
        p["u_env_lod"].value = float(math.log2(max(env["img"].shape[:2])))
        if env.get("background", False):
            sk = g.skyprog
            ctx.disable(g.mgl.DEPTH_TEST)
            env_tex.use(14)
            sk["u_env"].value = 14
            sk["u_invvp"].write(_m4(np.linalg.inv(VP)))
            sk["u_env_int"].value = float(env.get("background_intensity", env.get("intensity", 1.0)))
            sk["u_env_rot"].value = math.radians(float(env.get("rotation", 0)))
            sk["u_blur"].value = float(env.get("background_blur", 0.0))
            g.skyvao.render()
            ctx.enable(g.mgl.DEPTH_TEST)
    else:
        p["u_has_env"].value = 0
    nl = min(MAX_LIGHTS, len(lights))
    p["u_nl"].value = nl

    def arr(name, vals, fmt):
        if name in p:
            buf = np.zeros((MAX_LIGHTS,) + np.shape(vals[0]) if vals else (MAX_LIGHTS,), fmt)
            for i, v in enumerate(vals[:MAX_LIGHTS]):
                buf[i] = v
            p[name].write(buf.tobytes())

    if nl:
        arr("u_ltype", [L["type"] for L in lights], np.int32)
        arr("u_lcol", [L["color"] for L in lights], np.float32)
        arr("u_lpos", [L.get("pos", np.zeros(3)) for L in lights], np.float32)
        arr("u_ldir", [L.get("dir", np.array([0, 0, 1.0])) for L in lights], np.float32)
        arr("u_lco", [L.get("cos_outer", 0.0) for L in lights], np.float32)
        arr("u_lci", [L.get("cos_inner", 0.0) for L in lights], np.float32)
        arr("u_lrange", [L.get("range", 0.0) for L in lights], np.float32)
        arr("u_lsh", [L.get("sh", -1) for L in lights], np.int32)
        arr("u_ldark", [L.get("darkness", 1.0) for L in lights], np.float32)
        arr("u_lsoft", [L.get("softness", 1.0) for L in lights], np.float32)

    def draw(it, pass_id):
        mat = it.mat
        p["u_model"].write(_m4(it.model))
        nm = np.linalg.inv(it.model[:3, :3]).T
        p["u_nmat"].write(_m3(nm))
        p["u_opacity"].value = float(it.opacity)
        p["u_lit"].value = 1 if (it.lit and nl) else 0
        p["u_recv"].value = 1 if it.recv else 0
        p["u_spec"].value = float(mat.get("specular", 0.25))
        p["u_shin"].value = float(mat.get("shininess", 32))
        p["u_emis"].value = tuple(float(x) for x in mat.get("emissive", (0, 0, 0)))
        p["u_toon"].value = int(mat.get("toon", 0))
        p["u_rim"].value = float(mat.get("rim", 0.0))
        pbr = "metallic" in mat or "roughness" in mat
        p["u_pbr"].value = 1 if pbr else 0
        p["u_metal"].value = float(mat.get("metallic", 0.0))
        p["u_rough"].value = float(mat.get("roughness", 0.5))
        p["u_outline"].value = 0.0
        for mi, m in enumerate(it.meshes):
            col = mat.get("color")
            if col is None:
                col = m.color if m.color is not None else (1, 1, 1, 1)
            p["u_color"].value = tuple(float(x) for x in col)
            tex_obj = _item_tex(g, it, mi, m)
            p["u_use_tex"].value = 1 if tex_obj is not None else 0
            if tex_obj is not None:
                tex_obj.use(0)
                p["u_tex"].value = 0
            vao = g.mesh_vao((it.key, mi, "m"), m, p)[0]
            p["u_pass"].value = pass_id
            vao.render()
            ow = float(mat.get("outline", 0) or 0)
            if ow > 0 and pass_id == 0:
                ctx.enable(g.mgl.CULL_FACE)
                ctx.cull_face = "back"  # projection flips y -> winding reversed: this culls the hull's near side
                p["u_outline"].value = ow
                p["u_pass"].value = 3
                oc = mat.get("outline_color", (0.02, 0.02, 0.03, 1.0))
                p["u_ocol"].value = tuple(float(x) * (float(oc[3]) if i < 3 else 1) for i, x in enumerate(oc))
                vao.render()
                p["u_outline"].value = 0.0
                ctx.disable(g.mgl.CULL_FACE)

    ctx.enable(g.mgl.DEPTH_TEST)
    ctx.disable(g.mgl.BLEND)
    ctx.depth_mask = True
    for it in items:
        draw(it, 0)  # opaque pixels, depth write
    ctx.enable(g.mgl.BLEND)
    ctx.blend_func = (g.mgl.ONE, g.mgl.ONE_MINUS_SRC_ALPHA)
    ctx.depth_mask = False

    def dist(it):
        c = it.model @ np.array([0, 0, 0, 1.0])
        return -float(np.linalg.norm(c[:3] - campos))

    for it in sorted(items, key=dist):
        draw(it, 1)  # translucent pixels, back to front
    ctx.depth_mask = True
    ctx.disable(g.mgl.BLEND)
    if ms is not None:
        ctx.copy_framebuffer(out, ms)
    rgba = np.frombuffer(out_tex.read(), np.float32).reshape(OH, OW, 4).copy()
    depth = None
    if settings.get("need_depth"):
        # depth-only, non-MSAA pass for DOF/fog
        out.use()
        out.clear(depth=1.0, color=(0, 0, 0, 0))
        ctx.enable(g.mgl.DEPTH_TEST)
        g.sprog["u_vp"].write(_m4(VP))
        for it in items:
            g.sprog["u_model"].write(_m4(it.model))
            for mi, m in enumerate(it.meshes):
                tex_obj = _item_tex(g, it, mi, m)
                g.sprog["u_use_tex"].value = 1 if tex_obj is not None else 0
                g.sprog["u_cut"].value = 0.5
                if tex_obj is not None:
                    tex_obj.use(0)
                    g.sprog["u_tex"].value = 0
                g.mesh_vao((it.key, mi, "s"), m, g.sprog)[0].render()
        d = np.frombuffer(dep.read(), np.float32).reshape(OH, OW)
        zn = d * 2 - 1
        depth = (2 * FAR * NEAR / (FAR + NEAR - zn * (FAR - NEAR))).astype(np.float32)
        depth[d >= 0.99999] = np.inf
    return np.clip(rgba, 0, None), depth


def _item_tex(g, it, mi, m):
    if it.card_tex is not None:
        key, img = it.card_tex
        return g.texture(key, img)
    if m.tex is not None:
        return g.texture((id(m.tex), "meshtex"), m.tex)
    tex = it.mat.get("_tex_img")
    if tex is not None:
        return g.texture((id(tex), "mattex"), tex)
    return None


def post_dof_fog(rgba, depth, dof, fog, scale):
    """Depth-aware DOF (4-level CoC blend) + fog."""
    import cv2
    out = rgba
    if dof and depth is not None and dof.get("aperture", 0) > 0:
        fd = dof["focus"]
        coc = np.where(np.isfinite(depth), np.minimum(dof["max"], dof["aperture"] * np.abs(1 - fd / np.maximum(depth, 1))), dof["max"]) * scale
        levels = [0, dof["max"] * scale / 4, dof["max"] * scale / 2, dof["max"] * scale]
        blurred = [rgba] + [cv2.GaussianBlur(rgba, (0, 0), max(0.3, s / 2)) for s in levels[1:]]
        idx = np.clip(np.searchsorted(levels, coc, side="right") - 1, 0, 2)
        lo = np.take(levels, idx)
        hi = np.take(levels, idx + 1)
        w = np.clip((coc - lo) / np.maximum(hi - lo, 1e-6), 0, 1)[..., None]
        out = np.zeros_like(rgba)
        for i in range(3):
            m = (idx == i)[..., None]
            out += m * (blurred[i] * (1 - w) + blurred[i + 1] * w)
    if fog and depth is not None:
        k = np.clip((np.where(np.isfinite(depth), depth, fog["far"]) - fog["near"]) / max(1, fog["far"] - fog["near"]), 0, 1) * fog["amount"]
        out = out.copy()
        out[..., :3] = out[..., :3] * (1 - k[..., None]) + fog["color"] * out[..., 3:4] * k[..., None]
    return out
