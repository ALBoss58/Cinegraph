"""glTF 2.0 animation: node TRS animation + skeletal skinning (linear blend), evaluated on the CPU per frame.

Supports LINEAR / STEP / CUBICSPLINE samplers, quaternion slerp, multiple animations (by name or index),
textured primitives (baseColorTexture + baseColorFactor), and rigid node animation for unskinned meshes."""
from __future__ import annotations

import functools
import io
import math
import os

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _q_to_m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * u
        return r / np.linalg.norm(r)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def _trs(t, r, s):
    M = np.eye(4)
    M[:3, :3] = _q_to_m(r) * s
    M[:3, 3] = t
    return M


class GLTFAnimated:
    def __init__(self, path):
        import pygltflib
        self.path = path
        g = pygltflib.GLTF2().load(path)
        self.g = g
        blob = g.binary_blob() if path.lower().endswith(".glb") else None
        self._buffers = []
        for i, b in enumerate(g.buffers):
            if blob is not None and i == 0 and (b.uri is None):
                self._buffers.append(blob)
            elif b.uri and b.uri.startswith("data:"):
                import base64
                self._buffers.append(base64.b64decode(b.uri.split(",", 1)[1]))
            else:
                with open(os.path.join(os.path.dirname(path), b.uri), "rb") as f:
                    self._buffers.append(f.read())
        n = len(g.nodes)
        self.parent = [-1] * n
        for i, nd in enumerate(g.nodes):
            for c in nd.children or []:
                self.parent[c] = i
        self.rest = []
        for nd in g.nodes:
            if nd.matrix:
                M = np.array(nd.matrix, np.float64).reshape(4, 4).T
                t = M[:3, 3]
                s = np.linalg.norm(M[:3, :3], axis=0)
                R = M[:3, :3] / np.where(s == 0, 1, s)
                # matrix -> quaternion
                w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
                q = np.array([(R[2, 1] - R[1, 2]) / (4 * w + 1e-12), (R[0, 2] - R[2, 0]) / (4 * w + 1e-12),
                              (R[1, 0] - R[0, 1]) / (4 * w + 1e-12), w])
                self.rest.append((t, q / np.linalg.norm(q), s))
            else:
                self.rest.append((np.array(nd.translation or [0, 0, 0], float), np.array(nd.rotation or [0, 0, 0, 1], float),
                                  np.array(nd.scale or [1, 1, 1], float)))
        self.order = self._topo()
        self.anims = {}
        for ai, an in enumerate(g.animations or []):
            chans = []
            for ch in an.channels:
                smp = an.samplers[ch.sampler]
                tin = self._acc(smp.input).ravel()
                tout = self._acc(smp.output)
                chans.append((ch.target.node, ch.target.path, smp.interpolation or "LINEAR", tin, tout))
            dur = max((c[3][-1] for c in chans), default=0.0)
            self.anims[an.name or f"anim{ai}"] = {"channels": chans, "duration": float(dur), "index": ai}
        self.prims = []  # (node, skin, arrays...)
        for ni, nd in enumerate(g.nodes):
            if nd.mesh is None:
                continue
            for pr in g.meshes[nd.mesh].primitives:
                if pr.mode not in (None, 4):
                    continue
                at = pr.attributes
                pos = self._acc(at.POSITION).astype(np.float64)
                nrm = self._acc(at.NORMAL).astype(np.float64) if at.NORMAL is not None else None
                uv = self._acc(at.TEXCOORD_0).astype(np.float32) if at.TEXCOORD_0 is not None else np.zeros((len(pos), 2), np.float32)
                idx = self._acc(pr.indices).ravel().astype(np.uint32) if pr.indices is not None else np.arange(len(pos), dtype=np.uint32)
                jn = self._acc(at.JOINTS_0).astype(np.int64) if at.JOINTS_0 is not None else None
                wt = self._acc(at.WEIGHTS_0).astype(np.float64) if at.WEIGHTS_0 is not None else None
                if wt is not None:
                    wt = wt / np.maximum(wt.sum(1, keepdims=True), 1e-9)
                vcol = self._acc(at.COLOR_0).astype(np.float32) if getattr(at, "COLOR_0", None) is not None else None
                tex, color = self._material(pr.material)
                if nrm is None:
                    nrm = self._normals(pos, idx)
                self.prims.append({"node": ni, "skin": nd.skin, "pos": pos, "nrm": nrm, "uv": uv, "idx": idx,
                                   "joints": jn, "weights": wt, "tex": tex, "color": color, "vcol": vcol})
        self.skins = []
        for sk in g.skins or []:
            ibm = self._acc(sk.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1) if sk.inverseBindMatrices is not None \
                else np.repeat(np.eye(4)[None], len(sk.joints), 0)
            self.skins.append((list(sk.joints), ibm.astype(np.float64)))

    def _topo(self):
        seen, order = set(), []

        def visit(i):
            if i in seen:
                return
            if self.parent[i] >= 0:
                visit(self.parent[i])
            seen.add(i)
            order.append(i)
        for i in range(len(self.parent)):
            visit(i)
        return order

    def _acc(self, i):
        g = self.g
        a = g.accessors[i]
        dt = COMP[a.componentType]
        nc = NCOMP[a.type]
        if a.bufferView is None:
            return np.zeros((a.count, nc), dt)
        bv = g.bufferViews[a.bufferView]
        buf = self._buffers[bv.buffer]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        isz = np.dtype(dt).itemsize
        stride = bv.byteStride or isz * nc
        if stride == isz * nc:
            arr = np.frombuffer(buf, dt, a.count * nc, off).reshape(a.count, nc)
        else:
            raw = np.frombuffer(buf, np.uint8, stride * (a.count - 1) + isz * nc, off)
            arr = np.stack([np.frombuffer(raw[k * stride:k * stride + isz * nc].tobytes(), dt) for k in range(a.count)])
        arr = arr.astype(np.float32) if a.normalized and dt != np.float32 else arr
        if a.normalized and dt != np.float32:
            arr = arr / float(np.iinfo(dt).max)
        return arr

    def _material(self, mi):
        if mi is None:
            return None, None
        m = self.g.materials[mi]
        pbr = m.pbrMetallicRoughness
        color = np.array(pbr.baseColorFactor, np.float32) if pbr and pbr.baseColorFactor else None
        tex = None
        if pbr and pbr.baseColorTexture is not None:
            try:
                from PIL import Image
                tx = self.g.textures[pbr.baseColorTexture.index]
                im = self.g.images[tx.source]
                if im.bufferView is not None:
                    bv = self.g.bufferViews[im.bufferView]
                    data = self._buffers[bv.buffer][(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
                    img = Image.open(io.BytesIO(data))
                else:
                    img = Image.open(os.path.join(os.path.dirname(self.path), im.uri))
                a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
                a[..., :3] *= a[..., 3:4]
                tex = np.ascontiguousarray(a)  # glTF uv origin is top-left: rows stay top-down
            except Exception:
                tex = None
        return tex, color

    @staticmethod
    def _normals(pos, idx):
        f = idx.reshape(-1, 3)
        n = np.zeros_like(pos)
        fn = np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]])
        for k in range(3):
            np.add.at(n, f[:, k], fn)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    # ------------------------------------------------------------ evaluation
    def anim_names(self):
        return list(self.anims)

    def pose(self, t, anim=None, loop=True):
        T = [list(r) for r in self.rest]
        if self.anims:
            a = self.anims.get(anim) if isinstance(anim, str) else None
            if a is None:
                keys = list(self.anims)
                a = self.anims[keys[anim if isinstance(anim, int) and anim < len(keys) else 0]]
            d = a["duration"]
            tt = (t % d) if (loop and d > 0) else min(t, d)
            for node, path, interp, tin, tout in a["channels"]:
                k = {"translation": 0, "rotation": 1, "scale": 2}.get(path)
                if k is None:
                    continue  # morph weights unsupported
                T[node][k] = self._sample(tin, tout, interp, tt, path == "rotation")
        G = [None] * len(T)
        for i in self.order:
            L = _trs(*T[i])
            G[i] = L if self.parent[i] < 0 else G[self.parent[i]] @ L
        return G

    @staticmethod
    def _sample(tin, tout, interp, t, is_rot):
        n = len(tin)
        cubic = interp == "CUBICSPLINE"
        vals = tout.reshape(n, 3, -1)[:, 1] if cubic else tout.reshape(n, -1)
        if t <= tin[0]:
            return vals[0].astype(np.float64)
        if t >= tin[-1]:
            return vals[-1].astype(np.float64)
        i = int(np.searchsorted(tin, t) - 1)
        t0, t1 = tin[i], tin[i + 1]
        u = (t - t0) / max(1e-9, t1 - t0)
        if interp == "STEP":
            return vals[i].astype(np.float64)
        if cubic:
            tt = tout.reshape(n, 3, -1)
            dt = t1 - t0
            p0, m0 = tt[i, 1], tt[i, 2] * dt
            p1, m1 = tt[i + 1, 1], tt[i + 1, 0] * dt
            u2, u3 = u * u, u * u * u
            v = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
            return v / np.linalg.norm(v) if is_rot else v
        a, b = vals[i].astype(np.float64), vals[i + 1].astype(np.float64)
        return _slerp(a, b, u) if is_rot else a + (b - a) * u

    def geometry(self, t, anim=None, loop=True):
        """Posed primitives in glTF model space: list of (pos, nrm, uv, idx, tex, color, vcol)."""
        G = self.pose(t, anim, loop)
        out = []
        for p in self.prims:
            if p["skin"] is not None and p["joints"] is not None:
                joints, ibm = self.skins[p["skin"]]
                J = np.stack([G[j] for j in joints]) @ ibm  # (nj,4,4)
                w = p["weights"]
                jn = p["joints"]
                M = np.einsum("vk,vkij->vij", w, J[jn])  # blended matrix per vertex
                ph = np.concatenate([p["pos"], np.ones((len(p["pos"]), 1))], 1)
                pos = np.einsum("vij,vj->vi", M, ph)[:, :3]
                nrm = np.einsum("vij,vj->vi", M[:, :3, :3], p["nrm"])
            else:
                Gm = G[p["node"]]
                pos = p["pos"] @ Gm[:3, :3].T + Gm[:3, 3]
                nrm = p["nrm"] @ np.linalg.inv(Gm[:3, :3])
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            out.append((pos, nrm, p["uv"], p["idx"], p["tex"], p["color"], p["vcol"]))
        return out


@functools.lru_cache(maxsize=16)
def load(path):
    return GLTFAnimated(path)


def has_animation(path):
    if not path.lower().endswith((".glb", ".gltf")):
        return False
    try:
        return bool(load(path).anims)
    except Exception:
        return False
