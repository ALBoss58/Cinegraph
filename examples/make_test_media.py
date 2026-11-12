"""Regenerate every synthetic test asset (with ground truth) used by tests/. Requires ffmpeg on PATH.
python examples/make_test_media.py [--force]"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FORCE = "--force" in sys.argv
os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
need = lambda n: FORCE or not os.path.exists(P(n))


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def writer(name, W, H, rate):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(rate),
                             "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", P(name)], stdin=subprocess.PIPE)


if need("clip.mp4"):
    ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=6", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", P("clip.mp4"))
if need("painting.png"):
    ff("-f", "lavfi", "-i", "mandelbrot=size=1920x1280:rate=24", "-frames:v", "1", P("painting.png"))
if need("voice.wav"):
    ff("-f", "lavfi", "-i", "aevalsrc=0.3*sin(2*PI*220*t)*gt(sin(2*PI*0.5*t)\\,0):d=12", "-ar", "48000", P("voice.wav"))
if need("music.wav"):
    ff("-f", "lavfi", "-i", "aevalsrc=0.25*sin(2*PI*130.8*t)+0.2*sin(2*PI*196*t)+0.15*sin(2*PI*261.6*t):d=20", "-ar", "48000", P("music.wav"))
if need("noisytone.wav"):
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-f", "lavfi", "-i", "anoisesrc=d=4:c=white:a=0.05",
       "-filter_complex", "[0][1]amix=inputs=2:normalize=0", "-ar", "48000", P("noisytone.wav"))
if need("map.jpg"):
    h, w = 1080, 1920
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (120 + 60 * np.sin(x / 200)).astype(np.uint8)
    img[..., 1] = (150 + 40 * np.cos(y / 150)).astype(np.uint8)
    img[..., 2] = 170
    cv2.putText(img, "MAP OF EUROPE 1799", (500, 540), cv2.FONT_HERSHEY_COMPLEX, 3, (40, 40, 40), 6, cv2.LINE_AA)
    cv2.imwrite(P("map.jpg"), img)
if need("emblem.png"):
    a = np.zeros((400, 400, 4), np.uint8)
    cv2.circle(a, (200, 200), 180, (30, 160, 240, 255), -1, cv2.LINE_AA)
    cv2.imwrite(P("emblem.png"), a)
if need("sky.png"):
    W, H = 2048, 1024
    v = np.linspace(0, 1, H)[:, None]
    u = np.linspace(0, 1, W)[None, :]
    lat = (0.5 - v) * np.pi
    t = np.clip(lat / (np.pi / 2), 0, 1)[..., None]
    sky = np.array([0.85, 0.8, 0.75]) * (1 - t) ** 3 + np.array([0.18, 0.35, 0.75]) * (1 - (1 - t) ** 3)
    sky = np.where((lat < 0)[..., None], np.array([0.25, 0.22, 0.18]), sky)
    lon = (u - 0.5) * 2 * np.pi
    sky = sky + np.exp(-((lon - 0.8) ** 2 + (lat - 0.5) ** 2) / 0.004)[..., None] * np.array([1.0, 0.9, 0.7]) * 2
    cv2.imwrite(P("sky.png"), cv2.cvtColor((np.clip(sky, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
if need("pawn.glb"):
    import trimesh
    from PIL import Image
    m = trimesh.creation.capsule(height=1.5, radius=0.5)
    img = Image.open(P("map.jpg")).convert("RGB").resize((512, 256))
    uv = np.stack([(np.arctan2(m.vertices[:, 0], m.vertices[:, 2]) / (2 * np.pi) + 0.5), (m.vertices[:, 1] + 1.25) / 2.5], -1)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    m.export(P("pawn.glb"))

rng = np.random.default_rng(1)
W, H, fps, n = 960, 540, 30, 90
bg = cv2.resize(rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
for i in range(40):
    cv2.circle(bg, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.uniform(5, 30)),
               tuple(int(x) for x in rng.integers(0, 255, 3)), -1, cv2.LINE_AA)
if need("track.mp4"):  # moving patch with known centre path
    patch = np.zeros((80, 80, 3), np.uint8)
    patch[:] = (240, 240, 240)
    cv2.rectangle(patch, (10, 10), (70, 70), (30, 30, 200), -1)
    cv2.circle(patch, (40, 40), 14, (250, 220, 20), -1)
    truth = []
    p = writer("track.mp4", W, H, fps)
    for i in range(n):
        t = i / fps
        cx, cy = 200 + 500 * t / 3, 270 + 120 * np.sin(t * 2)
        f = bg.copy()
        x, y = int(round(cx - 40)), int(round(cy - 40))
        f[y:y + 80, x:x + 80] = patch
        truth.append([cx, cy])
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    json.dump(truth, open(P("track_truth.json"), "w"))
if need("shaky.mp4"):
    big = cv2.resize(bg, (int(W * 1.3), int(H * 1.3)))
    p = writer("shaky.mp4", W, H, fps)
    for i in range(n):
        jx, jy = rng.normal(0, 6), rng.normal(0, 6)
        M = np.float32([[1, 0, -0.15 * W + jx - i * 0.3], [0, 1, -0.15 * H + jy]])
        p.stdin.write(cv2.warpAffine(big, M, (W, H)).tobytes())
    p.stdin.close()
    p.wait()
if need("motion30.mp4"):
    p = writer("motion30.mp4", W, H, 30)
    for i in range(60):
        f = bg.copy()
        t = i / 30
        cv2.circle(f, (int(100 + 700 * t / 2), int(270 + 150 * np.sin(t * 3))), 40, (255, 80, 40), -1, cv2.LINE_AA)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
if need("noisy.mp4"):
    p = writer("noisy.mp4", W, H, fps)
    for i in range(30):
        p.stdin.write(np.clip(bg.astype(np.float32) + rng.normal(0, 18, bg.shape), 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
if need("pan_person.mp4"):
    r2 = np.random.default_rng(3)
    big = cv2.GaussianBlur(cv2.resize(r2.integers(0, 255, (90, 160, 3), dtype=np.uint8), (W * 2, H)), (0, 0), 2)
    for i in range(60):
        cv2.circle(big, (int(r2.uniform(0, W * 2)), int(r2.uniform(0, H))), int(r2.uniform(6, 30)),
                   tuple(int(x) for x in r2.integers(0, 255, 3)), -1, cv2.LINE_AA)
    p = writer("pan_person.mp4", W, H, 30)
    for i in range(90):
        f = big[:, i * 6:i * 6 + W].copy()
        cv2.ellipse(f, (200 + i * 4, 300), (40, 110), 0, 0, 360, (20, 20, 25), -1, cv2.LINE_AA)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
if need("speech.wav"):  # real speech sample shipped inside the (Apache-2.0) sherpa-onnx English model
    try:
        from cinegraph import ai
        ai._sherpa_recognizer()
        shutil.copy(os.path.join(ai.MODEL_DIR, ai.ASR_DIR, "test_wavs", "1.wav"), P("speech.wav"))
    except Exception as e:
        print("speech.wav skipped:", e)
print("test media ready in", OUT)
