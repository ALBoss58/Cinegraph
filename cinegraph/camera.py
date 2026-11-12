"""3D camera & transform math (After Effects conventions).

World space: x right, y DOWN, z INTO the screen (pixels). A 3D layer at z=0 with scale 100%
seen by the default camera is pixel-identical to a 2D layer, exactly like AE.
Camera space: x right, y down, z forward. Projection: u = f*x/z + cx.
"""
from __future__ import annotations

import math

import numpy as np

from .props import P, as_vec


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def euler(rx, ry, rz):
    """Degrees. Applied X, then Y, then Z (AE order)."""
    return rot_z(math.radians(rz)) @ rot_y(math.radians(ry)) @ rot_x(math.radians(rx))


def mat4(R=None, t=None, s=None):
    M = np.eye(4)
    if s is not None:
        M[:3, :3] = np.diag(s)
    if R is not None:
        M[:3, :3] = R @ M[:3, :3]
    if t is not None:
        M[:3, 3] = t
    return M


def translate4(t):
    M = np.eye(4)
    M[:3, 3] = t
    return M


DEFAULT_HFOV = 39.5978  # AE 50mm preset


class Camera:
    """Camera spec keys (all animatable, comp time):
        position [x,y,z]        default [W/2, H/2, -zoom]
        poi      [x,y,z]        point of interest (two-node camera). Omit + set "rotation" for one-node.
        rotation [rx,ry,rz]     one-node orientation (deg) used when poi is null
        roll     deg
        fov      horizontal angle of view (deg)   | zoom (px) | focal_length (mm, 36mm film)
        dof      {"focus_distance": px, "aperture": blur-px at 2x focus distance, "max_blur": 40}
        fog      {"color": "#..", "near": px, "far": px, "amount": 0..1}
        shake    {"amount": px, "frequency": hz, "rotation": deg}
        parent   layer id (null) to attach the camera to (rig)
    """

    def __init__(self, spec: dict | None, comp_w: int, comp_h: int):
        spec = spec or {}
        self.spec = spec
        self.W, self.H = comp_w, comp_h
        z0 = self.zoom_from(spec, 0.0)
        self.p_pos = P(spec.get("position"), default=[comp_w / 2, comp_h / 2, -z0])
        self.has_poi = "poi" in spec or "rotation" not in spec
        self.p_poi = P(spec.get("poi"), default=[comp_w / 2, comp_h / 2, 0.0])
        self.p_rot = P(spec.get("rotation"), default=[0.0, 0.0, 0.0])
        self.p_roll = P(spec.get("roll"), default=0.0)
        self.p_fov = P(spec.get("fov")) if "fov" in spec else None
        self.p_zoom = P(spec.get("zoom")) if "zoom" in spec else None
        self.p_focal = P(spec.get("focal_length")) if "focal_length" in spec else None
        dof = spec.get("dof") or {}
        self.dof = bool(dof)
        self.p_focus = P(dof.get("focus_distance"), default=z0)
        self.p_aperture = P(dof.get("aperture"), default=0.0)
        self.max_blur = float(dof.get("max_blur", 40))
        fog = spec.get("fog") or {}
        self.fog = fog if fog else None
        sh = spec.get("shake") or {}
        self.shake = sh if sh else None
        self.parent = spec.get("parent")

    def zoom_from(self, spec, t):
        if "zoom" in spec:
            return float(P(spec["zoom"]).at(t, t))
        if "focal_length" in spec:
            return float(P(spec["focal_length"]).at(t, t)) / 36.0 * self.W
        fov = float(P(spec["fov"]).at(t, t)) if "fov" in spec else DEFAULT_HFOV
        return (self.W / 2) / math.tan(math.radians(fov) / 2)

    def zoom(self, t):
        if self.p_zoom is not None:
            return float(self.p_zoom.at(t, t))
        if self.p_focal is not None:
            return float(self.p_focal.at(t, t)) / 36.0 * self.W
        fov = float(self.p_fov.at(t, t)) if self.p_fov is not None else DEFAULT_HFOV
        return (self.W / 2) / math.tan(math.radians(max(0.5, min(179, fov))) / 2)

    def state(self, t: float, parent_world: np.ndarray | None = None):
        """Return dict(view 4x4 world->camera, f, cx, cy, pos)."""
        pos = as_vec(self.p_pos.at(t, t), 3).astype(np.float64)
        roll = float(self.p_roll.at(t, t))
        if self.shake:
            from .props import wiggle_value
            amt = float(self.shake.get("amount", 6))
            fq = float(self.shake.get("frequency", 1.5))
            pos = pos + np.asarray(wiggle_value(t, fq, amt, 2, 7, 3), np.float64)
            roll += float(wiggle_value(t, fq, float(self.shake.get("rotation", 0.3)), 2, 9, 1))
        if self.has_poi and not ("rotation" in self.spec and "poi" not in self.spec):
            poi = as_vec(self.p_poi.at(t, t), 3).astype(np.float64)
            if parent_world is not None:
                poi = (parent_world @ np.append(poi, 1))[:3]
                pos = (parent_world @ np.append(pos, 1))[:3]
            fwd = poi - pos
            n = np.linalg.norm(fwd)
            fwd = fwd / n if n > 1e-9 else np.array([0, 0, 1.0])
            up = np.array([0, -1.0, 0])
            if abs(np.dot(fwd, up)) > 0.999:
                up = np.array([0, 0, 1.0])
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])  # rows: camera axes in world
        else:
            r = as_vec(self.p_rot.at(t, t), 3)
            Rw = euler(r[0], r[1], r[2])  # camera->world
            if parent_world is not None:
                Rw = parent_world[:3, :3] / np.linalg.norm(parent_world[:3, :3], axis=0) @ Rw
                pos = (parent_world @ np.append(pos, 1))[:3]
            R = Rw.T
        if roll:
            R = rot_z(math.radians(roll)).T @ R
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = -R @ pos
        return {"view": V, "f": self.zoom(t), "cx": self.W / 2, "cy": self.H / 2, "pos": pos}


def project_points(P3cam: np.ndarray, cam: dict) -> np.ndarray:
    z = P3cam[:, 2:3]
    return np.hstack([cam["f"] * P3cam[:, 0:1] / z + cam["cx"], cam["f"] * P3cam[:, 1:2] / z + cam["cy"]])


def clip_polygon_near(poly: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman clip of camera-space polygon (N,3) against z >= near."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            u = (near - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * u)
    return np.array(out) if out else np.zeros((0, 3))
