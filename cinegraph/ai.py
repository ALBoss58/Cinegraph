"""AI tools: subject cut-out (U2-Net / ISNet), object removal (flow/homography clean plate + inpaint),
speech-to-text with word timestamps (faster-whisper if installed, else sherpa-onnx Zipformer + Silero VAD)
and caption building.

Models live in ~/.cinegraph/models (override: CINEGRAPH_MODELS) and download on first use."""
from __future__ import annotations

import math
import os
import re
import tarfile
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.environ.get("CINEGRAPH_MODELS", os.path.join(os.path.expanduser("~"), ".cinegraph", "models"))
REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
SEG_MODELS = {  # name: (file, input size, mean, std)
    "human": ("u2net_human_seg.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "general": ("isnet-general-use.onnx", 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "fast": ("u2netp.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
ASR_DIR = "sherpa-onnx-zipformer-en-2023-06-26"


def _fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[cinegraph] downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def model_path(name):
    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        _fetch(REMBG + name, p)
    return p


# ---------------------------------------------------------------- segmentation
_SESS = {}


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        prov = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()]
        _SESS[path] = ort.InferenceSession(path, so, providers=prov)
    return _SESS[path]


def guided_filter(I, p, r, eps):
    """He et al. guided filter (edge-aware mask refinement). I: gray guide 0..1, p: mask 0..1."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return np.clip(box(a) * I + box(b), 0, 1)


def segment(img_rgb_u8: np.ndarray, model="general", refine=True) -> np.ndarray:
    """Foreground mask float32 (H, W) in 0..1."""
    if model not in SEG_MODELS:
        raise ValueError(f"cutout model must be one of {list(SEG_MODELS)}")
    fname, size, mean, std = SEG_MODELS[model]
    sess = _session(model_path(fname))
    h, w = img_rgb_u8.shape[:2]
    x = cv2.resize(img_rgb_u8[..., :3], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(1e-6, out.max() - out.min())
    m = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if refine:
        g = cv2.cvtColor(img_rgb_u8[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = max(2, int(round(min(h, w) / 180)))
        m = guided_filter(g, m, r, 1e-3)
    return m.astype(np.float32)


class Cutout:
    """Per-layer video cut-out with temporal smoothing (flow-warped EMA) to stop mask flicker."""

    def __init__(self, spec: dict):
        self.model = spec.get("model", "general")
        self.refine = spec.get("refine", True)
        self.temporal = float(spec.get("temporal", 0.35))
        self.choke = float(spec.get("choke", 0))
        self.feather = float(spec.get("feather", 0))
        self.invert = bool(spec.get("invert", False))
        self.threshold = spec.get("threshold")
        self.prev = None  # (idx, frame_u8, mask)
        self.cache = {}

    def mask(self, frame_u8, idx):
        if idx in self.cache:
            return self.cache[idx]
        m = segment(frame_u8, self.model, self.refine)
        if self.temporal > 0 and self.prev is not None and abs(idx - self.prev[0]) == 1:
            from .motion import flow, _warp
            F = flow(frame_u8, self.prev[1], preset="fast")
            pm = _warp(self.prev[2][..., None], F, 1.0)
            pm = pm[..., 0] if pm.ndim == 3 else pm
            m = m * (1 - self.temporal) + pm * self.temporal
        self.prev = (idx, frame_u8, m)
        out = m
        if self.threshold is not None:
            t = float(self.threshold)
            out = np.clip((out - t) / 0.1 + 0.5, 0, 1)
        if self.choke:
            k = int(abs(self.choke)) * 2 + 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            out = cv2.erode(out, se) if self.choke > 0 else cv2.dilate(out, se)
        if self.feather > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.feather / 2)
        if self.invert:
            out = 1 - out
        if len(self.cache) > 8:
            self.cache.clear()
        self.cache[idx] = out
        return out


# ---------------------------------------------------------------- object removal
def _align_h(g_src, g_dst, keep_src, keep_dst):
    """Homography mapping g_src -> g_dst from ORB matches outside the masked regions (robust to big moves)."""
    orb = cv2.ORB_create(3000, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(g_src, keep_src)
    k2, d2 = orb.detectAndCompute(g_dst, keep_dst)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    mt = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in mt if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    Hm, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
    if Hm is None or inl.sum() < 10:
        return None
    # sub-pixel refinement with LK on inliers
    a_in = a[inl.ravel() == 1].reshape(-1, 1, 2)
    pred = cv2.perspectiveTransform(a_in, Hm)
    ref, st, _ = cv2.calcOpticalFlowPyrLK(g_src, g_dst, a_in, pred.copy(), winSize=(15, 15), maxLevel=1,
                                          flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = st.ravel() == 1
    if ok.sum() >= 10:
        H2, _ = cv2.findHomography(a_in[ok], ref[ok], cv2.RANSAC, 1.5)
        if H2 is not None:
            Hm = H2
    return Hm


def remove_object(center_u8, mask, neighbors_u8, inpaint_radius=6, neighbor_masks=None):
    """Fill `mask` (H,W float, 1 = remove) with real background revealed in neighbor frames (aligned by
    ORB+RANSAC homography on pixels outside the object, LK-refined). Nearest-in-time valid pixel wins (sharp);
    anything never revealed is inpainted (Telea). neighbors_u8 should be ordered nearest first."""
    h, w = mask.shape
    hole = mask > 0.5
    if not hole.any():
        return center_u8
    g0 = cv2.cvtColor(np.ascontiguousarray(center_u8[..., :3]), cv2.COLOR_RGB2GRAY)
    keep0 = (~cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)).astype(np.uint8) * 255
    out = center_u8.copy()
    filled = np.zeros((h, w), bool)
    for k, nb in enumerate(neighbors_u8):
        if filled[hole].all():
            break
        g1 = cv2.cvtColor(np.ascontiguousarray(nb[..., :3]), cv2.COLOR_RGB2GRAY)
        nm = neighbor_masks[k] if neighbor_masks is not None else None
        keep1 = None if nm is None else (~cv2.dilate((nm > 0.5).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)).astype(np.uint8) * 255
        Hm = _align_h(g1, g0, keep1, keep0)
        if Hm is None:
            continue
        al = cv2.warpPerspective(nb[..., :3], Hm, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hm, (w, h)) > 250
        if nm is not None:
            valid &= cv2.warpPerspective((nm > 0.5).astype(np.uint8), Hm, (w, h)) == 0
        else:
            # without the neighbour's own mask, reject pixels that still look like the object (same as centre)
            diff = np.abs(al.astype(np.int16) - center_u8[..., :3].astype(np.int16)).mean(-1)
            valid &= diff > 18
        take = hole & valid & ~filled
        out[..., :3][take] = al[take]
        filled |= take
    rest = (hole & ~filled).astype(np.uint8) * 255
    if rest.any():
        rgb = np.ascontiguousarray(out[..., :3])
        out[..., :3] = cv2.inpaint(rgb, rest, inpaint_radius, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
    res = center_u8.copy()
    res[..., :3] = np.clip(out[..., :3] * soft + center_u8[..., :3] * (1 - soft) + 0.5, 0, 255).astype(np.uint8)
    return res


# ---------------------------------------------------------------- speech to text
def _sherpa_recognizer():
    import sherpa_onnx
    d = os.path.join(MODEL_DIR, ASR_DIR)
    if not os.path.isdir(d):
        tb = os.path.join(MODEL_DIR, ASR_DIR + ".tar.bz2")
        _fetch(SHERPA + ASR_DIR + ".tar.bz2", tb)
        with tarfile.open(tb) as t:
            t.extractall(MODEL_DIR)
        os.remove(tb)
    pick = lambda stem: os.path.join(d, stem + ".int8.onnx") if os.path.exists(os.path.join(d, stem + ".int8.onnx")) else os.path.join(d, stem + ".onnx")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=pick("encoder-epoch-99-avg-1"), decoder=pick("decoder-epoch-99-avg-1"), joiner=pick("joiner-epoch-99-avg-1"),
        tokens=os.path.join(d, "tokens.txt"), num_threads=max(1, (os.cpu_count() or 2) // 2), decoding_method="greedy_search")


def _vad_segments(audio, sr):
    import sherpa_onnx
    vp = os.path.join(MODEL_DIR, "silero_vad.onnx")
    if not os.path.exists(vp):
        _fetch(SHERPA + "silero_vad.onnx", vp)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vp
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.min_speech_duration = 0.2
    cfg.silero_vad.max_speech_duration = 20
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=600)
    win = cfg.silero_vad.window_size
    segs = []
    for i in range(0, len(audio), win):
        vad.accept_waveform(audio[i:i + win])
        while not vad.empty():
            s = vad.front
            segs.append((s.start / sr, np.asarray(s.samples, np.float32)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        segs.append((s.start / sr, np.asarray(s.samples, np.float32)))
        vad.pop()
    return segs


def _tokens_to_words(tokens, times, offset, seg_end):
    words = []
    for tok, tm in zip(tokens, times):
        if tok.startswith(("\u2581", " ")) or not words:  # word boundary (sentencepiece mark or space)
            words.append({"word": tok.replace("\u2581", "").strip(), "start": offset + tm, "end": None})
        else:
            words[-1]["word"] += tok
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        w["end"] = min(nxt, w["start"] + max(0.18, 0.08 * len(w["word"]) + 0.12))
        w["word"] = w["word"].strip()
    return [w for w in words if w["word"]]


def transcribe(path: str, language: str = "en", backend: str = "auto"):
    """Return list of {"word", "start", "end"} (seconds). Uses faster-whisper (best, multilingual, punctuation)
    when installed and its model is reachable; otherwise offline sherpa-onnx (English)."""
    from .media import decode_audio
    if backend in ("auto", "whisper"):
        try:
            from faster_whisper import WhisperModel
            size = os.environ.get("CINEGRAPH_WHISPER", "small")
            m = WhisperModel(size, device="auto", compute_type="int8")
            segs, _ = m.transcribe(path, language=language, word_timestamps=True, vad_filter=True)
            out = []
            for s in segs:
                for w in s.words or []:
                    out.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
            if out:
                return out
        except Exception as e:
            if backend == "whisper":
                raise
            print(f"[cinegraph] faster-whisper unavailable ({type(e).__name__}); using offline sherpa-onnx")
    sr = 16000
    audio = decode_audio(path, sr)[:, 0].copy()
    rec = _sherpa_recognizer()
    words = []
    for start, samples in _vad_segments(audio, sr):
        st = rec.create_stream()
        st.accept_waveform(sr, samples)
        rec.decode_stream(st)
        r = st.result
        words += _tokens_to_words(list(r.tokens), list(r.timestamps), start, start + len(samples) / sr)
    if words and all(w["word"].isupper() for w in words):  # this model outputs upper case: make it readable
        words = _sentence_case(words)
    return words


def _sentence_case(words):
    cap = True
    keep_upper = {"I", "I'M", "I'LL", "I'VE", "I'D"}
    for w in words:
        t = w["word"]
        lw = t.lower()
        if t in keep_upper:
            lw = t.capitalize() if t == "I" else "I" + lw[1:]
        w["word"] = lw[:1].upper() + lw[1:] if cap else lw
        cap = False
    return words


def build_captions(words, max_chars=32, max_lines=2, max_dur=3.2, min_gap_break=0.45):
    """Group words into caption cues: [{"start","end","text","words":[...]}]. Breaks on punctuation and pauses."""
    cues = []
    cur = []

    def flush():
        if cur:
            text = " ".join(w["word"] for w in cur)
            cues.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": _wrap(text, max_chars, max_lines),
                         "words": [dict(w) for w in cur]})
            cur.clear()

    for w in words:
        if cur:
            text_len = len(" ".join(x["word"] for x in cur + [w]))
            pause = w["start"] - cur[-1]["end"]
            if text_len > max_chars * max_lines or (w["end"] - cur[0]["start"]) > max_dur or pause > min_gap_break \
                    or re.search(r"[.!?]$", cur[-1]["word"]):
                flush()
        cur.append(w)
    flush()
    # merge a 1-word orphan into the previous cue when it still fits
    merged = []
    for c in cues:
        if merged and len(c["words"]) == 1 and len(merged[-1]["text"].replace("\n", " ")) + 1 + len(c["text"]) <= max_chars * max_lines \
                and c["start"] - merged[-1]["end"] < min_gap_break and c["end"] - merged[-1]["start"] <= max_dur + 1.2:
            p = merged[-1]
            p["words"] += c["words"]
            p["end"] = c["end"]
            p["text"] = _wrap(" ".join(w["word"] for w in p["words"]), max_chars, max_lines)
        else:
            merged.append(c)
    cues = merged
    for i, c in enumerate(cues):  # readability: keep each cue >= 0.7s without overlapping the next
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else c["end"] + 2
        c["end"] = min(max(c["end"] + 0.15, c["start"] + 0.7), nxt - 0.02)
    return cues


def _wrap(text, max_chars, max_lines):
    words = text.split()
    if len(text) <= max_chars:
        return text
    best = None  # balanced 2-line split
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(a) - len(b)) + (100 if max(len(a), len(b)) > max_chars else 0)
        if best is None or score < best[0]:
            best = (score, a + "\n" + b)
    return best[1]
