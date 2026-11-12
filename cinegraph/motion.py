"""Motion analysis: point tracking, planar (corner-pin) tracking, stabilization, optical-flow frame
interpolation (smooth slow motion) and motion-compensated temporal denoise.

Analyses are cached on disk in <project>/.cinegraph_cache so renders and parallel workers reuse them."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

import cv2
import numpy as np

from . import media

ANALYSIS_W = 960  # analysis resolution (width) — fast and precise enough; results are rescaled


# ---------------------------------------------------------------- cache + decode helpers
def _cache_path(base_dir, kind, *parts):
    d = os.path.join(base_dir or ".", ".cinegraph_cache")
    os.makedirs(d, exist_ok=True)
    h = hashlib.md5(json.dumps([kind, *parts], sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(d, f"{kind}_{h}.json")


def _file_sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, int(st.st_mtime)]


def gray_frames(path, t0=0.0, t1=None, width=ANALYSIS_W):
    """Yield (frame_index, gray uint8) at analysis width from t0..t1 (source seconds)."""
    info = media.probe(path)
    fps = info["fps"]
    W, H = info["width"], info["height"]
    w = min(width, W)
    h = int(round(H * w / W / 2) * 2)
    cmd = [media.FFMPEG, "-v", "error", "-nostdin"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.6f}"]
    if t1 is not None:
        cmd += ["-t", f"{max(0.0, t1 - t0):.6f}"]
    cmd += ["-i", path, "-map", "0:v:0", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "passthrough", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    i = int(round(t0 * fps))
    n = w * h
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield i, np.frombuffer(b, np.uint8).reshape(h, w)
            i += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


# ---------------------------------------------------------------- point tracking
LK = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))


def track_points(path, points, t0=0.0, t1=None, base_dir=None, template=24):
    """Track feature points (source px) from t0 to t1. Hybrid: pyramidal LK (forward-backward checked) +
    template re-anchoring against the first frame's patch to stop drift. Returns
    {"fps", "frames": [idx...], "points": [[[x, y] ...] per frame], "confidence": [[c ...] per frame]}."""
    info = media.probe(path)
    key = _cache_path(base_dir, "track", _file_sig(path), points, t0, t1, template)
    if os.path.exists(key):
        with open(key) as f:
            return json.load(f)
    s = None
    prev = None
    pts = None
    tmpls = []
    out = {"fps": info["fps"], "frames": [], "points": [], "confidence": []}
    for idx, g in gray_frames(path, t0, t1):
        if s is None:
            s = g.shape[1] / info["width"]
            pts = np.array(points, np.float32).reshape(-1, 1, 2) * s
            r = int(template * s) or 8
            for p in pts[:, 0]:
                x, y = int(p[0]), int(p[1])
                tmpls.append(g[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].copy())
            conf = np.ones(len(pts))
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None, **LK)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            conf = np.where((st.ravel() == 1) & (st2.ravel() == 1), np.exp(-fb / 1.5), 0.0)
            # template re-anchor in a search window around the LK estimate
            for k, p in enumerate(nxt[:, 0]):
                tp = tmpls[k]
                if tp.size == 0:
                    continue
                th, tw = tp.shape
                sr = max(th, tw) + int(12 * s) + 8
                x, y = int(p[0]), int(p[1])
                x0, y0 = max(0, x - sr), max(0, y - sr)
                win = g[y0:y + sr + 1, x0:x + sr + 1]
                if win.shape[0] <= th or win.shape[1] <= tw:
                    continue
                res = cv2.matchTemplate(win, tp, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv > 0.8:
                    cand = np.array([x0 + ml[0] + tw / 2 - 0.5, y0 + ml[1] + th / 2 - 0.5], np.float32)
                    if np.linalg.norm(cand - p) < 6 * s + 2:
                        w = (mv - 0.8) / 0.2 * 0.6
                        nxt[k, 0] = p * (1 - w) + cand * w
                    conf[k] = max(conf[k], mv)
                elif conf[k] < 0.2:
                    nxt[k, 0] = pts[k, 0]  # lost: hold last good position
            pts = nxt
        prev = g
        out["frames"].append(idx)
        out["points"].append((pts.reshape(-1, 2) / s).round(3).tolist())
        out["confidence"].append(np.round(conf, 3).tolist())
    with open(key, "w") as f:
        json.dump(out, f)
    return out


def track_planar(path, quad, t0=0.0, t1=None, base_dir=None):
    """Planar tracker: features inside the quad, RANSAC homography to the reference frame each step
    (re-detecting features when they thin out). Returns {"fps", "frames", "corners": [[[x,y]*4] ...]}."""
    info = media.probe(path)
    key = _cache_path(base_dir, "planar", _file_sig(path), quad, t0, t1)
    if os.path.exists(key):
        with open(key) as f:
            return json.load(f)
    out = {"fps": info["fps"], "frames": [], "corners": []}
    s = None
    prev = None
    H_acc = np.eye(3)
    q0 = None
    feats = None
    for idx, g in gray_frames(path, t0, t1):
        if s is None:
            s = g.shape[1] / info["width"]
            q0 = np.array(quad, np.float32) * s
        cur_q = cv2.perspectiveTransform(q0.reshape(-1, 1, 2), H_acc).reshape(-1, 2)
        if prev is not None and feats is not None and len(feats) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, feats, None, **LK)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                Hs, inl = cv2.findHomography(feats[ok], nxt[ok], cv2.RANSAC, 2.0)
                if Hs is not None:
                    H_acc = Hs @ H_acc
                    feats = nxt[ok][inl.ravel() == 1].reshape(-1, 1, 2)
            cur_q = cv2.perspectiveTransform(q0.reshape(-1, 1, 2), H_acc).reshape(-1, 2)
        if feats is None or len(feats) < 40:
            mask = np.zeros_like(g)
            cq = cur_q.mean(0) + (cur_q - cur_q.mean(0)) * 0.82  # stay inside the plane: no edge/background features
            cv2.fillConvexPoly(mask, cq.astype(np.int32), 255)
            f = cv2.goodFeaturesToTrack(g, 400, 0.005, 4, mask=mask, blockSize=5)
            feats = f.astype(np.float32) if f is not None else feats
        prev = g
        out["frames"].append(idx)
        out["corners"].append((cur_q / s).round(3).tolist())
    with open(key, "w") as f:
        json.dump(out, f)
    return out


def sample_track(tr, src_t, key="points", index=0):
    """Interpolate a track at source time. Returns np.array([x, y]) (or (4,2) for corners)."""
    fps = tr["fps"]
    fr = np.asarray(tr["frames"], np.float64)
    data = np.asarray(tr[key], np.float64)
    f = src_t * fps
    if f <= fr[0]:
        v = data[0]
    elif f >= fr[-1]:
        v = data[-1]
    else:
        i = int(np.searchsorted(fr, f) - 1)
        a = (f - fr[i]) / max(1e-9, fr[i + 1] - fr[i])
        v = data[i] * (1 - a) + data[i + 1] * a
    return v[index] if key == "points" else v


# ---------------------------------------------------------------- stabilization
def stabilize_analysis(path, base_dir=None):
    """Frame-to-frame similarity motion (dx, dy, da) at analysis res, scaled to source px."""
    info = media.probe(path)
    key = _cache_path(base_dir, "stab", _file_sig(path))
    if os.path.exists(key):
        with open(key) as f:
            return json.load(f)
    prev = None
    s = None
    motion = [[0.0, 0.0, 0.0]]
    pfeat = None
    for idx, g in gray_frames(path):
        if s is None:
            s = g.shape[1] / info["width"]
        if prev is not None:
            m = [0.0, 0.0, 0.0]
            if pfeat is not None and len(pfeat) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pfeat, None, **LK)
                ok = st.ravel() == 1
                if ok.sum() >= 10:
                    M, _ = cv2.estimateAffinePartial2D(pfeat[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        m = [float(M[0, 2] / s), float(M[1, 2] / s), float(math.atan2(M[1, 0], M[0, 0]))]
            motion.append(m)
        pfeat = cv2.goodFeaturesToTrack(g, 300, 0.01, 20)
        pfeat = pfeat.astype(np.float32) if pfeat is not None else None
        prev = g
    out = {"fps": info["fps"], "w": info["width"], "h": info["height"], "motion": motion}
    with open(key, "w") as f:
        json.dump(out, f)
    return out


def _gauss_smooth(x, sigma):
    if sigma <= 0:
        return x.copy()
    r = int(sigma * 3) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


class Stabilizer:
    """Correction transforms per source frame. mode: smooth (keep intentional moves) | lock (tripod)."""

    def __init__(self, path, spec: dict, base_dir=None):
        a = stabilize_analysis(path, base_dir)
        self.fps = a["fps"]
        self.w, self.h = a["w"], a["h"]
        traj = np.cumsum(np.asarray(a["motion"], np.float64), axis=0)
        mode = spec.get("mode", "smooth")
        if mode == "lock":
            target = np.repeat(traj[:1], len(traj), 0)
        else:
            target = _gauss_smooth(traj, float(spec.get("smoothness", 1.0)) * self.fps)
        self.corr = target - traj  # dx, dy, da to apply
        z = spec.get("zoom", "auto")
        if z == "auto":
            c = self.corr
            need = 1 + 2 * np.max(np.abs(c[:, 0])) / self.w + 2 * np.max(np.abs(c[:, 1])) / self.h \
                + np.max(np.abs(c[:, 2])) * (self.w + self.h) / min(self.w, self.h)
            self.zoom = float(min(float(spec.get("max_zoom", 1.25)), need))
        else:
            self.zoom = float(z)

    def matrix(self, idx, out_w, out_h):
        i = max(0, min(len(self.corr) - 1, idx))
        dx, dy, da = self.corr[i]
        sx = out_w / self.w
        c, s = math.cos(da) * self.zoom, math.sin(da) * self.zoom
        cx, cy = out_w / 2, out_h / 2
        # rotate+scale about centre, then translate (in output px)
        return np.array([[c, -s, cx - c * cx + s * cy + dx * sx], [s, c, cy - s * cx - c * cy + dy * sx]], np.float32)

    def apply(self, img, idx):
        h, w = img.shape[:2]
        return cv2.warpAffine(img, self.matrix(idx, w, h), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- optical flow
_DIS = {}


def dis(preset="medium"):
    if preset not in _DIS:
        p = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
             "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset]
        _DIS[preset] = cv2.DISOpticalFlow_create(p)
    return _DIS[preset]


def flow(a_rgb, b_rgb, width=ANALYSIS_W, preset="medium"):
    """Dense flow a->b at full size of a (float32 HxWx2), computed at analysis width."""
    h, w = a_rgb.shape[:2]
    s = min(1.0, width / w)
    sw, sh = max(16, int(w * s)), max(16, int(h * s))
    ga = cv2.cvtColor(cv2.resize(a_rgb[..., :3], (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b_rgb[..., :3], (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    f = dis(preset).calc(ga, gb, None)
    if s < 1:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / s
    return f.astype(np.float32)


def _warp(img, fl, k):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + fl[..., 0] * k, yy + fl[..., 1] * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _splat_flow(F, k):
    """Forward-splat flow F (from frame X) to time fraction k: returns (flow at target sampled back to X,
    hit mask). Larger motions win collisions (foreground objects usually move more than background)."""
    h, w = F.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.rint(xx + F[..., 0] * k).astype(np.int64)
    ty = np.rint(yy + F[..., 1] * k).astype(np.int64)
    ok = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    mag = np.linalg.norm(F, axis=-1)
    src = np.flatnonzero(ok.ravel())
    tgt = (ty.ravel() * w + tx.ravel())[src]
    order = np.argsort(mag.ravel()[src], kind="stable")  # small first, large last -> large overwrite
    out = np.zeros((h * w, 2), np.float32)
    hit = np.zeros(h * w, bool)
    out[tgt[order]] = F.reshape(-1, 2)[src[order]]
    hit[tgt[order]] = True
    out = out.reshape(h, w, 2)
    hit = hit.reshape(h, w)
    # close 1-px splat holes
    k3 = np.ones((3, 3), np.uint8)
    hole = (~hit).astype(np.uint8)
    if hole.any():
        filled = cv2.dilate(out * hit[..., None], k3)
        cnt = cv2.dilate(hit.astype(np.float32), k3)
        small = (cv2.erode(hole, k3) == 0) & (hole == 1)
        out[small] = filled[small]
        hit = hit | small
    return out, hit


def interpolate(f0, f1, a, cache: dict | None = None, key=None):
    """Optical-flow in-between frame at fraction a (0..1) of uint8 RGBA frames (AE 'Pixel Motion').
    Forward flow splatting + bidirectional occlusion-aware blending."""
    if cache is not None and key in cache:
        F01, F10 = cache[key]
    else:
        F01 = flow(f0, f1)
        F10 = flow(f1, f0)
        if cache is not None:
            if len(cache) > 4:
                cache.clear()
            cache[key] = (F01, F10)
    S0, h0 = _splat_flow(F01, a)        # motion of frame-0 content arriving at t
    S1, h1 = _splat_flow(F10, 1 - a)    # motion of frame-1 content arriving at t
    # fallback (Super SloMo linear approx) where nothing was splatted
    Ft0_fb = -(1 - a) * a * F01 + a * a * F10
    Ft1_fb = (1 - a) * (1 - a) * F01 - a * (1 - a) * F10
    Ft0 = np.where(h0[..., None], -a * S0, Ft0_fb)
    Ft1 = np.where(h1[..., None], -(1 - a) * S1, Ft1_fb)
    w0 = _warp(f0.astype(np.float32), Ft0, 1.0)
    w1 = _warp(f1.astype(np.float32), Ft1, 1.0)
    wa = (1 - a) * (0.05 + 0.95 * h0)[..., None]
    wb = a * (0.05 + 0.95 * h1)[..., None]
    out = (w0 * wa + w1 * wb) / np.maximum(wa + wb, 1e-6)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def estimate_noise(img_u8):
    """Gaussian noise sigma estimate (Immerkaer's fast method) on luma."""
    g = cv2.cvtColor(np.ascontiguousarray(img_u8[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, -1, k)
    h, w = g.shape
    return float(np.sum(np.abs(r)) * math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)))


def temporal_denoise(center, neighbors, strength=0.5):
    """Motion-compensated temporal average (neighbors aligned to center with flow computed on pre-smoothed
    frames); per-pixel weights from a noise-aware difference test so moving/occluded areas don't ghost."""
    sig = max(1.0, estimate_noise(center))
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.6)
    c_s = blur(center)
    acc = center.astype(np.float32)
    wsum = np.ones(center.shape[:2] + (1,), np.float32)
    # after a sigma=1.6 blur the residual noise of a difference of two frames is ~ sig*0.25*sqrt(2)
    tol = sig * 0.36 * (1.0 + 2.0 * float(strength))
    for nb in neighbors:
        F = flow(c_s, blur(nb), preset="medium")
        al = _warp(nb.astype(np.float32), F, 1.0)
        d = np.abs(blur(al)[..., :3] - c_s[..., :3].astype(np.float32)).mean(-1, keepdims=True)
        w = np.exp(-(np.maximum(d - tol, 0) / tol) ** 2)
        acc += al * w
        wsum += w
    out = acc / wsum
    if strength > 0.6:  # mop up residual with a light spatial pass
        out8 = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return spatial_denoise(out8, (strength - 0.6), "bilateral")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def spatial_denoise(img_u8, strength=0.5, mode="nlm"):
    rgb = np.ascontiguousarray(img_u8[..., :3])
    if mode == "nlm":
        h = 3 + 12 * float(strength)
        out = cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 5, 15)
    else:
        d = 5 + int(6 * strength)
        out = cv2.bilateralFilter(rgb, d, 20 + 60 * strength, 5)
    r = img_u8.copy()
    r[..., :3] = out
    return r
