"""Audio engine: multitrack mixing with keyframed gain (dB), fades, pan, loops, auto-ducking
(music under voice) and EBU R128 loudness normalization (YouTube default -14 LUFS)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

from . import media
from .props import P

SR = 48000


def _gain_envelope(prop, start, n, t_offset, sr=SR, rate=200):
    """Evaluate a dB property (layer-local time) into a per-sample linear gain array."""
    if not prop.animated:
        return np.full(n, 10 ** (float(prop.at(0)) / 20.0), np.float32)
    k = max(2, int(n / sr * rate) + 2)
    ts = np.linspace(0, n / sr, k)
    db = np.array([float(prop.at(t_offset + x, start + t_offset + x)) for x in ts], np.float32)
    g = 10 ** (db / 20.0)
    return np.interp(np.arange(n) / sr, ts, g).astype(np.float32)


def _atempo(s):
    parts = []
    while s > 2.0 + 1e-9:
        parts.append(2.0)
        s /= 2.0
    while s < 0.5 - 1e-9:
        parts.append(0.5)
        s /= 0.5
    parts.append(s)
    return ",".join(f"atempo={p:.6f}" for p in parts if abs(p - 1) > 1e-6)


def _clip_from(L, info, warn):
    spd = float(L.get("speed", 1.0))
    if "time_remap" in L:
        warn.append(f"{L.get('id')}: time_remap audio not supported (muted)")
        return None
    dur = L.get("duration")
    if dur is None:
        dur = max(0.0, ((info.get("duration") or 0) - float(L.get("in", 0))) / max(abs(spd), 1e-6))
    return {"src": L["src"], "start": float(L["start"]), "in": float(L.get("in", 0)), "duration": dur,
            "volume": L.get("volume", 0.0), "fade_in": L.get("audio_fade_in", 0.02), "fade_out": L.get("audio_fade_out", 0.02),
            "role": L.get("role", "sfx"), "pan": L.get("pan", 0), "loop": L.get("loop", False), "id": L.get("id"),
            "speed": abs(spd), "reverse": bool(L.get("reverse")), "pitch": L.get("pitch", "preserve"),
            "denoise": L.get("audio_denoise"), "filters": L.get("audio_filters"), "highpass": L.get("highpass"),
            "voice": L.get("voice"), "eq": L.get("eq"), "compressor": L.get("compressor"), "deess": L.get("deess"),
            "gate": L.get("gate"), "limiter": L.get("limiter")}


def _collect(norm, comp_name, offset, w0, w1, gain_db, depth, warn):
    comp = norm["comps"][comp_name]
    out = []
    for L in comp["layers"]:
        if L.get("enabled") is False:
            continue
        if L.get("type") == "video" and not L.get("mute") and L.get("audio") is not False:
            try:
                info = media.probe(L["src"])
            except media.MediaError:
                continue
            if not info.get("has_audio"):
                continue
            c = _clip_from(L, info, warn)
            if c:
                out.append(c)
        elif L.get("type") == "comp" and depth < 8 and not L.get("mute"):
            sub_in = float(L.get("in", 0))
            if "time_remap" in L or abs(float(L.get("speed", 1)) - 1) > 1e-6 or L.get("reverse"):
                warn.append(f"{L.get('id')}: retimed precomp audio not supported (muted)")
                continue
            s0 = float(L["start"])
            d = float(L.get("duration", norm["comps"][L["comp"]]["duration"] - sub_in))
            vol = L.get("volume", 0.0)
            gd = float(vol) if not isinstance(vol, (dict, str)) else 0.0
            for c in _collect(norm, L["comp"], 0.0, -1e9, 1e9, 0.0, depth + 1, warn):
                c = dict(c)
                c["start"] = c["start"] - sub_in + s0
                c["_win"] = (max(s0, c.get("_win", (-1e9, 1e9))[0] - sub_in + s0), min(s0 + d, c.get("_win", (-1e9, 1e9))[1] - sub_in + s0))
                c["_gain"] = c.get("_gain", 0.0) + gd
                out.append(c)
    for a_ in comp.get("audio", []) if comp_name != "main" else []:
        if a_.get("enabled") is not False:
            out.append(dict(a_))
    for c in out:
        c["start"] = c["start"] + offset
        if gain_db:
            c["_gain"] = c.get("_gain", 0.0) + gain_db
    return out


def collect_clips(norm: dict, warn=None):
    warn = [] if warn is None else warn
    clips = _collect(norm, "main", 0.0, -1e9, 1e9, 0.0, 0, warn)
    for a_ in norm["audio"]:
        if a_.get("enabled") is not False:
            clips.append(a_)
    return clips


VOICE_PRESETS = {
    "clean": {"highpass": 80, "denoise": 0.4, "deess": True,
              "compressor": {"threshold_db": -20, "ratio": 3, "attack": 10, "release": 120, "makeup_db": 3}},
    "broadcast": {"highpass": 90, "denoise": 0.35, "deess": True,
                  "eq": [{"type": "lowshelf", "f": 120, "g": 2}, {"type": "peak", "f": 300, "g": -2.5, "q": 1.2},
                         {"type": "peak", "f": 3500, "g": 2.5, "q": 1}, {"type": "highshelf", "f": 10000, "g": 1.5}],
                  "compressor": {"threshold_db": -22, "ratio": 4, "attack": 6, "release": 100, "makeup_db": 5}},
    "warm": {"highpass": 70, "eq": [{"type": "lowshelf", "f": 150, "g": 3}, {"type": "highshelf", "f": 8000, "g": -2}],
             "compressor": {"threshold_db": -18, "ratio": 2.5, "attack": 15, "release": 150, "makeup_db": 2}},
    "phone": {"highpass": 400, "eq": [{"type": "lowpass", "f": 3400}, {"type": "peak", "f": 1500, "g": 4, "q": 1}]},
    "radio_1940s": {"highpass": 350, "eq": [{"type": "lowpass", "f": 4000}, {"type": "peak", "f": 1800, "g": 6, "q": 0.8}],
                    "filters": "acrusher=bits=10:mode=log:aa=1,volume=1.4"},
}


def eq_filters(bands):
    out = []
    for b in bands or []:
        t = b.get("type", "peak")
        f = float(b.get("f", 1000))
        g = float(b.get("g", 0))
        q = float(b.get("q", 0.707))
        if t == "peak":
            out.append(f"equalizer=f={f}:t=q:w={q}:g={g}")
        elif t == "lowshelf":
            out.append(f"bass=f={f}:g={g}:t=q:w={q}")
        elif t == "highshelf":
            out.append(f"treble=f={f}:g={g}:t=q:w={q}")
        elif t in ("highpass", "lowpass"):
            n = max(1, int(b.get("slope", 24)) // 12)  # 12 dB/oct per 2nd-order stage
            out += [f"{t}=f={f}:p=2"] * n
        elif t == "notch":
            out.append(f"bandreject=f={f}:t=q:w={max(q, 2)}")
        else:
            raise ValueError(f"EQ band type '{t}' (peak, lowshelf, highshelf, highpass, lowpass, notch)")
    return out


def dynamics_filters(c):
    out = []
    if c.get("gate"):
        g = c["gate"] if isinstance(c["gate"], dict) else {}
        out.append(f"agate=threshold={10 ** (float(g.get('threshold_db', -45)) / 20):.6f}:ratio={g.get('ratio', 4)}:"
                   f"attack={g.get('attack', 5)}:release={g.get('release', 150)}")
    if c.get("deess"):
        out.append("deesser=i=0.4:m=0.5:f=0.5")
    if c.get("compressor"):
        k = c["compressor"]
        out.append(f"acompressor=threshold={10 ** (float(k.get('threshold_db', -18)) / 20):.6f}:ratio={k.get('ratio', 3)}:"
                   f"attack={k.get('attack', 10)}:release={k.get('release', 120)}:makeup={10 ** (float(k.get('makeup_db', 0)) / 20):.4f}:knee=2.8")
    if c.get("limiter") is not None and c.get("limiter") is not False:
        lim = -1.0 if c["limiter"] is True else float(c["limiter"])
        out.append(f"alimiter=limit={10 ** (lim / 20):.6f}:attack=5:release=50:level=disabled")
    return out


def _filters(c):
    c = dict(c)
    vp = c.get("voice")
    if vp:
        if vp not in VOICE_PRESETS:
            raise ValueError(f"voice preset must be one of {list(VOICE_PRESETS)}")
        for k, v in VOICE_PRESETS[vp].items():
            c.setdefault(k, v)
    f = []
    if c.get("highpass"):
        f.append(f"highpass=f={float(c['highpass'])}:p=2")
    dn = c.get("denoise")
    if dn:
        st = 0.5 if dn is True else float(dn if not isinstance(dn, dict) else dn.get("strength", 0.5))
        f.append(f"afftdn=nr={6 + 24 * st:.1f}:nf=-30:tn=1")
    f += eq_filters(c.get("eq"))
    f += dynamics_filters(c)
    if c.get("filters"):
        f.append(str(c["filters"]))
    return f


def master_chain(settings):
    m = settings.get("master") or {}
    return ",".join(eq_filters(m.get("eq")) + dynamics_filters(m))


def _read_segment(c, sr, local0, seg_len, sdur):
    """Decode the source audio for layer-local window [local0, local0+seg_len] with speed/reverse/filters."""
    cin = float(c.get("in", 0))
    spd = float(c.get("speed", 1.0)) or 1.0
    dur = float(c["duration"])
    pre = _filters(c)
    if c.get("reverse"):
        s_a = cin + (dur - (local0 + seg_len)) * spd
        src_len = seg_len * spd
    else:
        s_a = cin + local0 * spd
        src_len = seg_len * spd
    chain = [f"aresample={sr}"] + list(pre)
    if c.get("reverse"):
        chain.append("areverse")
    if abs(spd - 1) > 1e-6:
        if c.get("pitch", "preserve") == "tape":
            chain.append(f"asetrate={int(sr * spd)},aresample={sr}")
        else:
            chain.append(_atempo(spd))
    seg = media.decode_audio(c["src"], sr, max(0.0, s_a), src_len, ",".join(x for x in chain if x) or None)
    m = int(round(seg_len * sr))
    if len(seg) < m:
        seg = np.vstack([seg, np.zeros((m - len(seg), 2), np.float32)])
    return seg[:m]


def mix(norm: dict, t0: float, t1: float, sr: int = SR, log=None):
    """Return (stereo float32 (n,2), stats)."""
    n = max(1, int(round((t1 - t0) * sr)))
    buses = {"voice": np.zeros((n, 2), np.float32), "music": np.zeros((n, 2), np.float32),
             "sfx": np.zeros((n, 2), np.float32)}
    warn = []
    for c in collect_clips(norm, warn):
        src = c["src"]
        start = float(c.get("start", 0))
        cin = float(c.get("in", 0))
        info = media.probe(src)
        sdur = info.get("duration") or 0
        dur = c.get("duration")
        if dur is None:
            dur = max(0.0, sdur - cin) / float(c.get("speed", 1.0) or 1.0) if not c.get("loop") else (t1 - start)
        dur = float(dur)
        c["duration"] = dur
        w0, w1 = c.get("_win", (-1e9, 1e9))
        a0, a1 = max(start, t0, w0), min(start + dur, t1, w1)
        if a1 <= a0:
            continue
        local0 = a0 - start
        if c.get("loop") and sdur > 0:
            full = media.decode_audio(src, sr, cin, None, ",".join(_filters(c)) or None)
            if len(full) == 0:
                continue
            need = int(round((local0 + (a1 - a0)) * sr)) + 1
            reps = int(np.ceil(need / len(full))) + 1
            seg = np.tile(full, (reps, 1))[int(round(local0 * sr)):int(round(local0 * sr)) + int(round((a1 - a0) * sr))]
        else:
            seg = _read_segment(c, sr, local0, a1 - a0, sdur)
        if len(seg) == 0:
            continue
        m = len(seg)
        g = _gain_envelope(P(c.get("volume", 0.0)), start, m, local0, sr) * (10 ** (float(c.get("_gain", 0.0)) / 20))
        ts = local0 + np.arange(m) / sr
        fi, fo = float(c.get("fade_in", 0) or 0), float(c.get("fade_out", 0) or 0)
        if fi > 0:
            g *= np.sin(np.clip(ts / fi, 0, 1) * np.pi / 2)
        if fo > 0:
            g *= np.sin(np.clip((dur - ts) / fo, 0, 1) * np.pi / 2)
        pan = float(c.get("pan", 0))
        lp, rp = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2), np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        seg = seg * g[:, None] * np.array([lp, rp], np.float32)
        off = int(round((a0 - t0) * sr))
        e = min(n, off + m)
        bus = c.get("role") or ("voice" if str(c.get("id") or "").startswith(("vo", "voice", "narr")) else "music")
        if bus not in buses:
            bus = "sfx"
        buses[bus][off:e] += seg[:e - off]
    settings = norm.get("audio_settings", {})
    duck = settings.get("duck", {"amount_db": -10}) if settings.get("duck", True) else None
    if duck and np.any(buses["voice"]) and np.any(buses["music"]):
        v = buses["voice"].mean(axis=1)
        win = int(sr * 0.02)
        k = len(v) // win
        rms = np.sqrt(np.mean(v[:k * win].reshape(k, win) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        on = (db > float(duck.get("threshold_db", -38))).astype(np.float32)
        hold = int(float(duck.get("hold", 0.25)) / 0.02)
        if hold > 0:
            on = np.convolve(on, np.ones(hold * 2 + 1), "same") > 0
            on = on.astype(np.float32)
        att = np.exp(-0.02 / max(1e-3, float(duck.get("attack", 0.12))))
        rel = np.exp(-0.02 / max(1e-3, float(duck.get("release", 0.5))))
        env = np.zeros_like(on)
        cur = 0.0
        for i, x in enumerate(on):
            c_ = att if x > cur else rel
            cur = c_ * cur + (1 - c_) * x
            env[i] = cur
        gain_db = env * float(duck.get("amount_db", -10))
        g = 10 ** (gain_db / 20)
        gs = np.interp(np.arange(n), np.arange(k) * win + win / 2, g).astype(np.float32)
        buses["music"] *= gs[:, None]
    out = buses["voice"] + buses["music"] + buses["sfx"]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out, {"peak": peak, "has_audio": bool(np.any(out)), "warnings": warn}


def write_wav(path, data: np.ndarray, sr: int = SR):
    d = np.clip(data, -1, 1)
    pcm = (d * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def loudnorm(in_wav: str, out_wav: str, target=-14.0, tp=-1.0, lra=11.0):
    """Two-pass EBU R128 normalization. Returns measured stats."""
    ff = media.FFMPEG
    r = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", in_wav, "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        raise RuntimeError("loudnorm measure failed: " + r.stderr[-500:])
    st = json.loads(m.group(0))
    if float(st["input_i"]) < -70:  # silence
        os.replace(in_wav, out_wav)
        return st
    af = (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={st['input_i']}:measured_TP={st['input_tp']}:"
          f"measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}:offset={st['target_offset']}:linear=true")
    r2 = subprocess.run([ff, "-hide_banner", "-nostdin", "-y", "-i", in_wav, "-af", af, "-ar", str(SR), out_wav],
                        capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError("loudnorm apply failed: " + r2.stderr[-500:])
    return st


def apply_chain(in_wav, out_wav, chain):
    if not chain:
        return in_wav
    r = subprocess.run([media.FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", in_wav, "-af", chain,
                        "-ar", str(SR), out_wav], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("master audio chain failed: " + r.stderr[-400:])
    return out_wav


# ---------------------------------------------------------------- analysis
def beats(path, sr=22050, hop=256):
    """Tempo + beat times via spectral-flux onsets and dynamic-programming beat tracking (Ellis 2007)."""
    x = media.decode_audio(path, sr).mean(1)
    if len(x) < sr:
        return {"bpm": 0.0, "beats": [], "onsets": []}
    n = 2048
    win = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    S = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
    flux = np.maximum(0, np.diff(S, axis=0)).sum(1)
    flux = np.concatenate([[0], flux])
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fps = sr / hop
    # tempo: autocorrelation of onset envelope, weighted toward 120 bpm
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    w = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-3) / 120) / 1.0) ** 2) * (bpm >= 50) * (bpm <= 220)
    lag = int(np.argmax(ac * w))
    tempo = 60 * fps / lag
    # DP beat tracking
    period = lag
    score = flux.copy()
    back = np.full(len(flux), -1)
    for i in range(len(flux)):
        lo, hi = i - 2 * period, i - period // 2
        if hi <= 0:
            continue
        cand = np.arange(max(0, lo), hi)
        pen = -100 * (np.log((i - cand) / period)) ** 2
        j = cand[np.argmax(score[cand] + pen)]
        score[i] = flux[i] + score[j] + pen[cand == j][0]
        back[i] = j
    i = int(np.argmax(score[-period * 2:]) + len(score) - period * 2)
    bts = []
    while i >= 0:
        bts.append(i)
        i = back[i]
    # frame i spans samples [i*hop, i*hop+n); flux[i] = change from frame i-1 to i, i.e. new energy entering at the
    # window's leading edge: onset time ~ (i*hop + n) / sr minus half a hop
    t_of = lambda i: (np.asarray(i, np.float64) * hop + n - 1.5 * hop) / sr  # calibrated on click tracks
    bts = t_of(sorted(bts))
    on = np.where((flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > 1.5))[0] + 1
    return {"bpm": round(float(tempo), 2), "beats": [round(float(b), 4) for b in bts if b >= 0],
            "onsets": [round(float(o), 4) for o in t_of(on)]}


def loudness_report(wav):
    r = subprocess.run([media.FFMPEG, "-hide_banner", "-nostdin", "-i", wav, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr[r.stderr.rfind("Summary:"):]
    get = lambda k: float(re.search(k + r":\s+(-?[\d.]+|-inf)", txt).group(1)) if re.search(k + r":\s+(-?[\d.]+)", txt) else None
    return {"integrated_lufs": get("I"), "lra": get("LRA"), "true_peak_dbtp": get("Peak")}
