"""Small host-side audio measures for oracle bookkeeping (which probe is nearest the GT-table probe). Not narration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

N_FFT, HOP, N_BANDS = 2048, 512, 48


def _bands(sr: int) -> np.ndarray:
    edges = np.geomspace(40.0, min(16000.0, sr / 2 - 1), N_BANDS + 1)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    return np.stack([(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges[:-1], edges[1:])]).astype(np.float32)


def log_bands(path: str | Path) -> np.ndarray:
    """[frames, N_BANDS] log band energies (dB) of the mono mix."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    spec = np.abs(np.fft.rfft(frames * np.hanning(N_FFT).astype(np.float32), axis=1)) ** 2
    return 10 * np.log10(spec @ _bands(sr).T + 1e-10)


def distance(a: str | Path, b: str | Path) -> float:
    """Mean |dB| difference of the time-averaged band spectra plus of the frame-wise spectra (aligned, cropped)."""
    A, B = log_bands(a), log_bands(b)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    return float(np.mean(np.abs(A.mean(0) - B.mean(0))) + 0.5 * np.mean(np.abs(np.maximum(A, -100) - np.maximum(B, -100))))


def peak_dbfs(path: str | Path) -> float:
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12))
