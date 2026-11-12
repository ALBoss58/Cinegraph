"""3D mesh loading (OBJ, GLB/glTF, STL, PLY, OFF via trimesh), primitives and heightmap terrain.

All meshes are converted to Cinegraph world convention (y down, z into screen) by rotating 180deg about X
(model +Y up -> screen up, model +Z front -> toward camera), then normalized: centered and scaled so the
largest dimension equals `size` px (default 400) unless normalize is false.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np


class Mesh:
    __slots__ = ("pos", "nrm", "uv", "idx", "tex", "color", "vcolor")

    def __init__(self, pos, nrm, uv, idx, tex=None, color=None, vcolor=None):
        self.pos = np.ascontiguousarray(pos, np.float32)
        self.nrm = np.ascontiguousarray(nrm, np.float32)
        self.uv = np.ascontiguousarray(uv, np.float32)
        self.idx = np.ascontiguousarray(idx, np.uint32)
        self.tex = tex          # premultiplied float32 RGBA or None
        self.color = color      # straight RGBA float or None
        self.vcolor = vcolor    # (N,4) float or None


def _from_trimesh(tm, tex_img=None):
    import trimesh
    tm = tm.copy()
    if not hasattr(tm, "faces") or len(tm.faces) == 0:
        return None
    uv = None
    tex = None
    color = None
    vcolor = None
    vis = getattr(tm, "visual", None)
    if vis is not None and getattr(vis, "kind", None) == "texture":
        uv = getattr(vis, "uv", None)
        mat = getattr(vis, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            bc = getattr(mat, "baseColorFactor", None)
            if bc is not None:
                color = np.asarray(bc, np.float32) / (255.0 if np.max(bc) > 1 else 1.0)
            elif img is None:  # plain diffuse colour only matters without a texture (trimesh defaults it to 40% grey)
                dc = getattr(mat, "diffuse", None)
                if dc is not None:
                    color = np.asarray(dc, np.float32) / 255.0
        if img is not None:
            a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
            a[..., :3] *= a[..., 3:4]
            tex = np.ascontiguousarray(a[::-1])  # GL uv origin bottom-left in model files
    elif vis is not None and getattr(vis, "kind", None) == "vertex":
        vcolor = np.asarray(vis.vertex_colors, np.float32) / 255.0
    elif vis is not None and getattr(vis, "kind", None) == "face":
        fc = np.asarray(vis.face_colors, np.float32) / 255.0
        if len(fc) and np.allclose(fc, fc[0]):
            color = fc[0]
        else:  # unshare vertices so each face keeps its colour
            tm = tm.copy()
            tm.unmerge_vertices()
            vcolor = np.repeat(fc, 3, axis=0)
    if uv is None or len(uv) != len(tm.vertices):
        uv = np.zeros((len(tm.vertices), 2), np.float32)
        tex = None
    else:
        uv = np.asarray(uv, np.float32).copy()  # OpenGL convention (v up) matches the bottom-up row order above
    return Mesh(tm.vertices, tm.vertex_normals, uv, tm.faces.reshape(-1), tex, color, vcolor)


def _to_world(meshes, size, normalize, center=True):
    R = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # 180deg about X
    allp = np.vstack([m.pos for m in meshes])
    allp = allp @ R.T
    lo, hi = allp.min(0), allp.max(0)
    c = (lo + hi) / 2 if center else np.zeros(3, np.float32)
    ext = float(np.max(hi - lo)) or 1.0
    s = (size / ext) if normalize else 1.0
    out = []
    for m in meshes:
        p = (m.pos @ R.T - c) * s
        n = m.nrm @ R.T
        out.append(Mesh(p, n, m.uv, m.idx, m.tex, m.color, m.vcolor))
    return out, (lo - c) * s, (hi - c) * s


@functools.lru_cache(maxsize=32)
def load_file(path: str, size: float = 400.0, normalize: bool = True, center: bool = True):
    import trimesh
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    scene = trimesh.load(path, force="scene", process=False)
    meshes = []
    for g in scene.graph.nodes_geometry:
        T, gname = scene.graph[g]
        geom = scene.geometry[gname].copy()
        if not isinstance(geom, trimesh.Trimesh):
            continue
        geom.apply_transform(T)
        m = _from_trimesh(geom)
        if m is not None:
            meshes.append(m)
    if not meshes:
        raise ValueError(f"No triangle meshes in {path}")
    return _to_world(meshes, size, normalize, center)


def primitive(kind: str, p: dict):
    import trimesh
    c = trimesh.creation
    k = kind.lower()
    if k in ("cube", "box"):
        sz = p.get("size", [200, 200, 200])
        sz = [sz] * 3 if np.isscalar(sz) else sz
        tm = c.box(extents=sz)
    elif k == "sphere":
        tm = c.icosphere(subdivisions=int(p.get("detail", 4)), radius=float(p.get("radius", 100)))
    elif k == "cylinder":
        tm = c.cylinder(radius=float(p.get("radius", 80)), height=float(p.get("height", 200)), sections=int(p.get("detail", 64)))
        tm.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    elif k == "cone":
        tm = c.cone(radius=float(p.get("radius", 100)), height=float(p.get("height", 200)), sections=int(p.get("detail", 64)))
        tm.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    elif k == "torus":
        tm = c.torus(major_radius=float(p.get("radius", 120)), minor_radius=float(p.get("tube", 35)),
                     major_sections=int(p.get("detail", 64)), minor_sections=max(8, int(p.get("detail", 64)) // 2))
    elif k == "capsule":
        tm = c.capsule(height=float(p.get("height", 200)), radius=float(p.get("radius", 60)))
        tm.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    elif k in ("plane", "floor", "ground"):
        sz = p.get("size", [2000, 2000])
        tm = c.box(extents=[sz[0], 1.0, sz[1]])
    else:
        raise ValueError(f"Unknown primitive '{kind}' (cube, sphere, cylinder, cone, torus, capsule, plane, terrain)")
    if p.get("flat_shading"):
        tm.unmerge_vertices()
    R = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    pos = np.asarray(tm.vertices, np.float32) @ R.T
    return [Mesh(pos, np.asarray(tm.vertex_normals, np.float32) @ R.T, _box_uv(pos), tm.faces.reshape(-1))]


def _box_uv(v):
    v = np.asarray(v, np.float32)
    lo, hi = v.min(0), v.max(0)
    span = np.maximum(hi - lo, 1e-6)
    # top-down projection in world space: image top = far side (+z), reads correctly from the default camera
    return np.stack([(v[:, 0] - lo[0]) / span[0], (hi[2] - v[:, 2]) / span[2]], -1).astype(np.float32)


def terrain(p: dict, base_dir: str):
    """Heightmap grid. size [w, d] px, height px, resolution cells. Texture drapes over the top."""
    import cv2
    from .media import load_image
    res = int(p.get("resolution", 200))
    w, d = p.get("size", [1600, 1000])
    hgt = float(p.get("height", 150))
    hm = p.get("heightmap")
    if hm:
        path = hm if os.path.isabs(hm) else os.path.join(base_dir, hm)
        img = load_image(path)
        g = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = cv2.GaussianBlur(g, (0, 0), max(0.5, min(g.shape) / res / 2))
        g = cv2.resize(g, (res + 1, res + 1), interpolation=cv2.INTER_AREA)
    else:
        g = np.zeros((res + 1, res + 1), np.float32)
    zs, xs = np.mgrid[0:res + 1, 0:res + 1].astype(np.float32) / res
    X = (xs - 0.5) * w
    Z = (zs - 0.5) * d
    Y = -g * hgt  # up is -y
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3)
    uv = np.stack([xs, zs], -1).reshape(-1, 2)
    i = np.arange((res + 1) * (res + 1)).reshape(res + 1, res + 1)
    a, b, c_, d_ = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    idx = np.stack([a, c_, b, b, c_, d_], -1).reshape(-1)
    # normals from height gradient
    gy, gx = np.gradient(Y, d / res, w / res)
    n = np.stack([gx, -np.ones_like(Y), gy], -1)  # pointing up (-y)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return [Mesh(pos, n.reshape(-1, 3), uv, idx)]
