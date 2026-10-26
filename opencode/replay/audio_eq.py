"""Audio equivalence for re-rendered clips (host side; numpy/librosa/soundfile).

Vital renders are not bit-identical run to run (random oscillator phase, free-running LFOs), so a clip
the rollout rendered and the replay re-rendered match when their lengths agree (within one block)
and their mean |log-mel difference| is under the render-noise floor (0.2-0.7 dB measured for the same
preset). The metric is scripts/check_render_parity.py's mel_dist.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

DEFAULT_MAX_DB = 1.5
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


def compare(expected: str | Path, got: str | Path, max_db: float = DEFAULT_MAX_DB, block: int = BLOCK) -> dict:
    """{"equivalent": bool, "reason", "mel_db", "len_expected", "len_got", "sr"}."""
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
    d = mel_dist(a, b, sra)
    out["mel_db"] = round(d, 3)
    if d > max_db:
        return {**out, "equivalent": False, "reason": f"mel distance {d:.2f} dB > {max_db} dB"}
    return {**out, "equivalent": True, "reason": "within render noise"}
