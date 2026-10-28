"""Audio equivalence for re-rendered clips (host side; numpy/librosa/soundfile).

Vital renders are not bit-identical run to run (random oscillator phase, free-running LFOs). A clip the
rollout rendered and the replay re-rendered are equivalent because the state that produced them is
byte-identical (identical tool calls + the final-workdir check); this module only catches gross
breakage: a length difference beyond one block, a silence mismatch (one clip's peak below
SILENT_PEAK_DBFS and the other's not), and a mean |log-mel difference| above a loose cap
(DEFAULT_MAX_DB). Same-preset render noise reaches ~5.7 dB (heavy unison, random phase, some
single-wavetable probes), so mel_db is informational and cannot (and is not meant to) detect small
real edits. The metric is scripts/check_render_parity.py's mel_dist.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

DEFAULT_MAX_DB = 10.0
SILENT_PEAK_DBFS = -60.0  # a clip whose peak is below this counts as silent
SILENCE_HYSTERESIS_DB = 5.0  # silent-vs-loud only fails when the loud side is this far above the floor
BLOCK = 512  # samples: render block size; lengths may differ by at most this


def logmel(x: np.ndarray, sr: int) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x.mean(axis=1).astype(np.float32), sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=64)
    return librosa.power_to_db(m, ref=1.0, amin=1e-10, top_db=None).clip(min=-100.0)


def mel_dist(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Mean |log-mel difference| in dB over frames where either clip is above -80 dB."""
    ma, mb = logmel(a, sr), logmel(b, sr)
    n = min(ma.shape[1], mb.shape[1])
    ma, mb = ma[:, :n], mb[:, :n]
    mask = np.maximum(ma, mb) > -80.0
    return float(np.abs(ma - mb)[mask].mean()) if mask.any() else 0.0


def peak_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12)) if x.size else -240.0


def compare(expected: str | Path, got: str | Path, max_db: float = DEFAULT_MAX_DB, block: int = BLOCK) -> dict:
    """{"equivalent": bool, "reason", "mel_db", "peak_dbfs_expected", "peak_dbfs_got", "len_expected",
    "len_got", "sr"}."""
    try:
        a, sra = sf.read(str(expected), always_2d=True)
        b, srb = sf.read(str(got), always_2d=True)
    except Exception as e:  # not decodable audio: only the exact check applies
        return {"equivalent": False, "reason": f"undecodable: {e}"}
    out = {"len_expected": len(a), "len_got": len(b), "sr": sra}
    if sra != srb or a.shape[1] != b.shape[1]:
        return {**out, "equivalent": False, "reason": f"format {sra}x{a.shape[1]} vs {srb}x{b.shape[1]}"}
    if abs(len(a) - len(b)) > block:
        return {**out, "equivalent": False, "reason": f"length differs by {abs(len(a) - len(b))} samples"}
    pa, pb = peak_dbfs(a), peak_dbfs(b)
    out.update(peak_dbfs_expected=round(pa, 1), peak_dbfs_got=round(pb, 1))
    if (pa < SILENT_PEAK_DBFS) != (pb < SILENT_PEAK_DBFS) and max(pa, pb) >= SILENT_PEAK_DBFS + SILENCE_HYSTERESIS_DB:
        # the loud side is clearly audible: a true silent-vs-sounding flip
        return {**out, "equivalent": False,
                "reason": f"silence mismatch: peak {pa:.1f} vs {pb:.1f} dBFS (silent < {SILENT_PEAK_DBFS})"}
    # borderline (the loud side is within a few dB of the floor): render noise can flip such clips
    # across the threshold run to run — let the mel distance judge them
    if pa < SILENT_PEAK_DBFS and pb < SILENT_PEAK_DBFS:
        return {**out, "equivalent": True, "reason": "both silent"}  # mel_dist of near-silence is noise
    d = mel_dist(a, b, sra)
    out["mel_db"] = round(d, 3)
    if d > max_db:
        return {**out, "equivalent": False, "reason": f"mel distance {d:.2f} dB > cap {max_db} dB"}
    return {**out, "equivalent": True, "reason": "within render noise"}
