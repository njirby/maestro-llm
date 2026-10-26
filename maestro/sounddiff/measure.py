"""Measure rendered sounds: perceptual descriptors per wav, CLAP / frozen Qwen3-Omni audio-tower embeddings, CLAP
audio-text similarity for a small tone-adjective vocabulary, and pairwise distances (log-mel, CLAP, Omni) computed later
on the host from the stored embeddings, so any pair (A vs B, current vs target, A vs A2 = render-noise floor) can be
compared without re-running the GPU step.

A measure dir holds renders.jsonl (one line per wav: {"name", "notes", "bpm", "preset", "omni": bool}), wav/<name>.wav,
and after this step desc.jsonl ({"name", "desc", "adj"}) + emb/<name>.npz (clap, omni frames). GPU step, container only
(ear-graft image: torch, transformers 5.16, CLAP + Omni weights in the HF cache):

  GPU=0 NAME=sd-measure ear_graft/docker_run.sh -m maestro.sounddiff.measure outputs/sounddiff/probe1 [more dirs]

Host side (numpy only): load_dir(dir) -> {name: {"desc", "adj"}}, pair(dir, a, b) -> distances.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
NFFT, HOP = 2048, 512
ENV_HOP = 220  # 5 ms
BANDS = [(0, 120, "sub"), (120, 500, "low"), (500, 2000, "mid"), (2000, 6000, "high_mid"), (6000, 22050, "air")]

# tone vocabulary for CLAP audio-text similarity; z-scored against the reference set (calibrate.py) before use
ADJECTIVES = ["hollow", "glassy", "buzzy", "metallic", "breathy", "warm", "harsh", "plucky", "airy", "nasal",
              "bell-like", "gritty", "smooth", "punchy", "thin", "thick", "muddy", "reedy", "lush", "percussive",
              "wobbly", "distorted", "soft", "bright", "dark", "detuned", "organ-like", "brassy", "spacey", "clean"]
ADJ_TEMPLATES = ["a {} synthesizer sound", "the sound of a {} synth", "{} synth tone"]


def _frames(x, n, hop):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i + n] for i in idx])


def spectrogram(mono):
    fr = _frames(mono, NFFT, HOP) * np.hanning(NFFT)
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2  # (T, F) power


def mel_fb(n_mels=128, fmin=30.0, fmax=16000.0):
    hz = np.fft.rfftfreq(NFFT, 1 / SR)
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = 700 * (10 ** (np.linspace(mel(fmin), mel(fmax), n_mels + 2) / 2595) - 1)
    fb = np.zeros((n_mels, len(hz)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((hz - lo) / (c - lo), (hi - hz) / (hi - c)), 0, None)
    return fb


FB = mel_fb()


def logmel(mono):
    m = 10 * np.log10(spectrogram(mono) @ FB.T + 1e-12)
    return np.maximum(m, m.max() - 80)


def envelope(mono):
    fr = _frames(mono, 2 * ENV_HOP, ENV_HOP)
    return np.sqrt((fr ** 2).mean(1) + 1e-12)


def db(x):
    return 20 * np.log10(x + 1e-12)


def octave_offset(mono, notes):
    """Median octave offset of the sounding pitch vs the MIDI notes (harmonic-sum over candidate octaves).
    Measured but never turned into a fact: misfires on distortion/FX changes (probe0)."""
    hz = np.fft.rfftfreq(8192, 1 / SR)
    offs = []
    for n in notes:
        if n["dur_s"] < 0.15:
            continue
        a = int((n["start_s"] + 0.03) * SR)
        seg = mono[a:a + 8192]
        if len(seg) < 4096:
            continue
        p = np.abs(np.fft.rfft(np.pad(seg, (0, 8192 - len(seg))) * np.hanning(8192))) ** 2
        lp = np.log(p + 1e-12)
        f0 = 440 * 2 ** ((n["pitch"] - 69) / 12)
        score = {}
        for k in (-2, -1, 0, 1, 2):
            f = f0 * 2 ** k
            hs = [h * f for h in range(1, 7) if h * f < 16000]
            score[k] = np.mean([lp[np.argmin(np.abs(hz - x))] for x in hs]) if hs else -np.inf
        offs.append(max(score, key=score.get))
    return float(np.median(offs)) if offs else 0.0


def harmonic_richness(mono, notes):
    """Energy above the ~6th harmonic vs harmonics 1-3 (dB), per note, median over notes. Distortion/saturation
    pushes energy up the harmonic series. Anchored on the lowest partial actually present among the MIDI pitch's
    octaves (2 down .. 2 up): patches transposed up an octave or two otherwise read as absurdly 'rich' (probe0 j056)."""
    vals = []
    for n in notes:
        if n["dur_s"] < 0.06:
            continue
        a = int((n["start_s"] + 0.02) * SR)
        seg = mono[a:a + int(min(n["dur_s"], 0.5) * SR)]
        if len(seg) < 1024:
            continue
        nf = 16384
        p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nf)) ** 2
        hz = np.fft.rfftfreq(nf, 1 / SR)
        f0 = 440 * 2 ** ((n["pitch"] - 69) / 12)
        cand = [f0 * 2 ** k for k in (-2, -1, 0, 1, 2) if 25 < f0 * 2 ** k < 5000]
        e = [p[(hz >= f * 0.96) & (hz <= f * 1.04)].sum() for f in cand]
        if not cand or max(e) <= 0:
            continue
        fs = next(f for f, x in zip(cand, e) if x >= max(e) * 10 ** (-25 / 10))
        lo = p[(hz >= 0.5 * fs) & (hz < 3.5 * fs)].sum()
        hi = p[(hz >= 6.5 * fs) & (hz < 16000)].sum()
        if lo > 0 and 6.5 * fs < 16000:
            vals.append(10 * np.log10((hi + 1e-20) / lo))
    return float(np.median(vals)) if vals else None


def crest(mono, env):
    """Median crest factor (peak/RMS, dB) over 50 ms windows that are within 30 dB of the loudest: drops when the
    waveform is squashed (distortion, clipping)."""
    fr = _frames(mono, 2205, 2205)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-20)
    keep = rms > rms.max() * 10 ** (-30 / 20)
    if not keep.any():
        return None
    return float(np.median(db(np.abs(fr[keep]).max(1)) - db(rms[keep])))


def modulation(S, hz, notes):
    """Clip-wide movement, robust to short/plucky notes (the old within-note wobble needed notes >= 0.4 s):
    - mod_cent: RMS of per-note linearly-detrended log2 centroid over the energetic frames (within 30 dB of the note
      peak), pooled over notes; an LFO on cutoff shows up here even on 150 ms plucks;
    - mod_amp: same for frame level in dB, on the 200 Hz - 5 kHz band only (tremolo / volume pulsing as heard; bass
      beating between detuned low partials otherwise reads as pulsing, probe1 e019);
    - mod_note: note-to-note spread of per-note mean log2 centroid not explained by pitch (a free-running LFO lands
      each note at a different brightness)."""
    pw = S.sum(1)
    cent = np.log2((S * hz).sum(1) / (pw + 1e-20) + 1)
    lev = 10 * np.log10(pw + 1e-20)
    lev_band = 10 * np.log10(S[:, (hz >= 200) & (hz < 5000)].sum(1) + 1e-20)
    rc, ra, per_note = [], [], []
    for n in notes:
        if n["dur_s"] < 0.1:
            continue
        i0 = int((n["start_s"] + 0.025) * SR / HOP)
        i1 = int((n["start_s"] + n["dur_s"]) * SR / HOP)
        if i1 - i0 < 6 or i1 > len(pw):
            continue
        sl = slice(i0, i1)
        keep = lev[sl] > lev[sl].max() - 30
        if keep.sum() < 6:
            continue
        t = np.arange(i1 - i0)[keep]
        for series, out in ((cent[sl][keep], rc), (lev_band[sl][keep], ra)):
            fit = np.polyval(np.polyfit(t, series, 1), t)
            out.extend((series - fit).tolist())
        w = pw[sl][keep]
        if w.sum() <= 0:
            continue
        per_note.append((n["pitch"], float((cent[sl][keep] * w).sum() / w.sum())))
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) >= 12 else None
    mod_note = None
    if len(per_note) >= 4:
        p, c = np.array(per_note).T
        if np.ptp(p) > 0:
            c = c - np.polyval(np.polyfit(p / 12, c, 1), p / 12)
        mod_note = float(np.std(c))
    return {"mod_cent": rms(rc), "mod_amp": rms(ra), "mod_note": mod_note}


def descriptors(audio, notes):
    mono = audio.mean(1)
    S = spectrogram(mono)
    hz = np.fft.rfftfreq(NFFT, 1 / SR)
    fr_pow = S.sum(1)
    active = fr_pow > fr_pow.max() * 1e-4  # within 40 dB of the loudest frame
    Sa = S[active] if active.any() else S
    tot = Sa.sum(1, keepdims=True) + 1e-20
    cent = (Sa * hz).sum(1) / tot[:, 0]
    agg = Sa.sum(0)
    cum = np.cumsum(Sa, 1) / tot
    roll = hz[np.argmax(cum >= 0.85, axis=1)]
    flat = np.exp(np.log(Sa + 1e-20).mean(1)) / (Sa.mean(1) + 1e-20)
    band_e = {name: float(Sa[:, (hz >= lo) & (hz < hi)].sum() / Sa.sum()) for lo, hi, name in BANDS}
    mid, side = audio.mean(1), (audio[:, 0] - audio[:, 1]) / 2
    env = envelope(mono)
    t_env = np.arange(len(env)) * ENV_HOP / SR
    peak = env.max()
    attacks, sustains, wob_c, wob_a = [], [], [], []
    for n in notes:
        i0 = int(n["start_s"] * SR / ENV_HOP)
        i1 = int((n["start_s"] + min(n["dur_s"], 1.2)) * SR / ENV_HOP)
        seg = env[i0:max(i1, i0 + 2)]
        if len(seg) < 4:
            continue
        pk = seg.max()
        attacks.append(np.argmax(seg >= 0.9 * pk) * ENV_HOP / SR * 1000)
        if n["dur_s"] >= 0.25:
            j = int((n["start_s"] + 0.8 * n["dur_s"]) * SR / ENV_HOP)
            if j < len(env):
                sustains.append(env[j] / (pk + 1e-12))
        if n["dur_s"] >= 0.4:
            a, b = int((n["start_s"] + 0.3 * n["dur_s"]) * SR / HOP), int((n["start_s"] + 0.9 * n["dur_s"]) * SR / HOP)
            seg_s = S[a:b]
            if len(seg_s) >= 4:
                c = (seg_s * hz).sum(1) / (seg_s.sum(1) + 1e-20)
                wob_c.append(float(np.std(np.log2(c + 1))))
                e = np.sqrt(seg_s.sum(1))
                wob_a.append(float(np.std(e) / (e.mean() + 1e-12)))
    last_off = max(n["start_s"] + n["dur_s"] for n in notes)
    after = (t_env > last_off) & (env > peak * 10 ** (-45 / 20))
    tail = float(t_env[after].max() - last_off) if after.any() else 0.0
    sounding = np.zeros(len(env), bool)
    for n in notes:
        sounding[int(n["start_s"] * SR / ENV_HOP):int((n["start_s"] + n["dur_s"]) * SR / ENV_HOP)] = True
    gaps = ~sounding & (t_env < last_off)
    gap_fill = float(db(np.sqrt((env[gaps] ** 2).mean())) - db(np.sqrt((env[sounding] ** 2).mean()))) \
        if gaps.sum() > 10 and sounding.any() else None
    med = lambda v: float(np.median(v)) if v else None
    return {
        "loudness_db": float(10 * np.log10(fr_pow[active].mean() / NFFT + 1e-20)),
        # level in dBFS (RMS over frames within 40 dB of the loudest): the near-silence gate in facts.py uses this
        "rms_dbfs": float(db(np.sqrt((_frames(mono, NFFT, HOP)[active] ** 2).mean())) if active.any() else -240.0),
        "peak_db": float(db(np.abs(audio).max())),
        # centroid_hz: median over frames (quiet decay frames count as much as the loud attack); centroid_w_hz: of the
        # energy-weighted average spectrum, i.e. what dominates what you hear (used for brightness facts)
        "centroid_hz": float(np.median(cent)), "centroid_w_hz": float((agg * hz).sum() / (agg.sum() + 1e-20)),
        "rolloff85_hz": float(np.median(roll)),
        "flatness": float(np.median(flat)), "bands": band_e,
        "width_db": float(10 * np.log10((side ** 2).mean() / ((mid ** 2).mean() + 1e-20) + 1e-12)),
        "attack_ms": med(attacks), "sustain_ratio": med(sustains), "tail_s": tail, "gap_fill_db": gap_fill,
        "wobble_cent_oct": med(wob_c), "wobble_amp_cv": med(wob_a), "octave_offset": octave_offset(mono, notes),
        "harm_rich_db": harmonic_richness(mono, notes), "crest_db": crest(mono, env),
        **modulation(S, hz, notes),
    }


class Embedders:
    def __init__(self, dev, omni=True):  # omni=False: CLAP only
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch, self.dev = torch, dev
        self.clap = ClapModel.from_pretrained("laion/larger_clap_general").to(dev).eval()
        self.clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_general")
        if omni:
            from transformers import WhisperFeatureExtractor
            from ear_graft.model import OMNI_PATH, load_audio_tower
            self.tower = load_audio_tower().to(dev, torch.bfloat16).eval()
            self.fe = WhisperFeatureExtractor.from_pretrained(OMNI_PATH)
        self.adj = self._adj_text()

    def _adj_text(self):
        with self.torch.no_grad():
            texts = [t.format(a) for a in ADJECTIVES for t in ADJ_TEMPLATES]
            inp = self.clap_proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
            e = self.clap.get_text_features(**inp)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            e = self.torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        e = e.reshape(len(ADJECTIVES), len(ADJ_TEMPLATES), -1).mean(1)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def clap_emb(self, mono):
        import soxr
        with self.torch.no_grad():
            x = soxr.resample(mono, SR, 48000)
            inp = self.clap_proc(audio=x, sampling_rate=48000, return_tensors="pt").to(self.dev)
            e = self.clap.get_audio_features(**inp)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            return self.torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu().numpy()

    def omni_emb(self, mono):
        import soxr
        with self.torch.no_grad():
            x = soxr.resample(mono, SR, 16000)[: 16000 * 30]
            f = self.fe(x, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
            feats, mask = f["input_features"], f["attention_mask"]
            lens = mask.sum(-1)
            packed = feats.permute(0, 2, 1)[mask.bool()].permute(1, 0)
            out = self.tower(packed.to(self.dev, self.torch.bfloat16), feature_lens=lens.to(self.dev)).last_hidden_state
            return out.float().cpu().numpy()  # (n_tok, 2048), ~13 tokens/s


def _wav_path(d: Path, r: dict) -> Path:
    p = Path(r.get("path") or f"wav/{r['name']}.wav")
    return p if p.is_absolute() else d / p


def cos_d(a, b):
    return float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def redescribe(d):
    """Recompute descriptors (numpy only, host-side) for a measured dir, keeping CLAP adjective sims: for descriptor
    changes that need no GPU."""
    d = Path(d)
    rows = {r["name"]: r for r in map(json.loads, open(d / "renders.jsonl"))}
    out = []
    for r in map(json.loads, open(d / "desc.jsonl")):
        audio = sf.read(_wav_path(d, rows[r["name"]]), dtype="float32")[0]
        r["desc"] = descriptors(audio, rows[r["name"]]["notes"])
        out.append(r)
    with open(d / "desc.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"redescribed {len(out)} in {d}")


# --- host side -------------------------------------------------------------------------------------------------------

def load_dir(d) -> dict:
    """{name: {"desc": descriptors, "adj": {adjective: CLAP cosine}}} for a measured dir."""
    d = Path(d)
    return {r["name"]: r for r in map(json.loads, open(d / "desc.jsonl"))}


@lru_cache(maxsize=256)
def _logmel(path: str):
    return logmel(sf.read(path, dtype="float32")[0].mean(1))


@lru_cache(maxsize=256)
def _emb(path: str):
    z = np.load(path)
    return z["clap"], z["omni"].astype(np.float32) if "omni" in z.files else None


def pair(d, a: str, b: str) -> dict:
    """Distances between two measured wavs of the same notes: mean |log-mel diff| (dB), CLAP cosine distance,
    Omni pooled and frame-wise cosine distance. Compare against pair(d, x, x2) (two renders of one preset) for the
    render-noise floor."""
    d = Path(d)
    la, lb = _logmel(str(d / "wav" / f"{a}.wav")), _logmel(str(d / "wav" / f"{b}.wav"))
    n = min(len(la), len(lb))
    (ca, ta), (cb, tb) = _emb(str(d / "emb" / f"{a}.npz")), _emb(str(d / "emb" / f"{b}.npz"))
    out = {"mel_db": float(np.abs(la[:n] - lb[:n]).mean()), "clap": cos_d(ca, cb)}
    if ta is not None and tb is not None:
        k = min(len(ta), len(tb))
        out["omni_pooled"] = cos_d(ta.mean(0), tb.mean(0))
        out["omni_frames"] = float(np.mean([cos_d(ta[i], tb[i]) for i in range(k)]))
    return out


# --- container side --------------------------------------------------------------------------------------------------

def measure_dir(d: Path, emb: "Embedders | None") -> "Embedders | None":
    import torch
    rows = [json.loads(l) for l in open(d / "renders.jsonl")]
    out_path = d / "desc.jsonl"
    done = {json.loads(l)["name"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["name"] not in done]
    if not todo:
        print(f"{d}: nothing to measure")
        return emb
    if emb is None:
        emb = Embedders(torch.device("cuda", 0))
    (d / "emb").mkdir(exist_ok=True)
    with open(out_path, "a") as f:
        for i, r in enumerate(todo):
            audio = sf.read(_wav_path(d, r), dtype="float32")[0]
            mono = audio.mean(1)
            clap = emb.clap_emb(mono)
            arrs = {"clap": clap}
            if r.get("omni", True):
                arrs["omni"] = emb.omni_emb(mono).astype(np.float16)
            np.savez(d / "emb" / f"{r['name']}.npz", **arrs)
            adj = {a: round(float(s), 5) for a, s in zip(ADJECTIVES, emb.adj @ clap)}
            f.write(json.dumps({"name": r["name"], "desc": descriptors(audio, r["notes"]), "adj": adj}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{d.name} {i}/{len(todo)} {r['name']}", flush=True)
    print(f"measured {len(todo)} -> {out_path}", flush=True)
    return emb


def main():
    emb = None
    for d in sys.argv[1:]:
        emb = measure_dir(Path(d), emb)

if __name__ == "__main__":
    main()
