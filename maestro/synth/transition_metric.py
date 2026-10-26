"""Transition-local audio difference: what changes right after the notes where the pitch moves (glide, attack, ...).

A whole-clip mean log-mel distance dilutes an effect that lives in the first ~100 ms of some notes (a 60 ms glide
changes a few frames per note) under the render-noise floor of a busy patch (random phase, unison, free LFOs ~2 dB).
These functions average only over short windows after pitch-change onsets.

  transition_onsets(notes)                          onset times (s) where the pitch differs from the previous note
  transition_mel_d(a, b, notes, win_s=0.15)          mean |log-mel diff| (dB) over those windows; None if no onsets
  audible_vs_takes(takes, variant, notes, ...)       gate-shaped decision: variant differs from the GT takes beyond
                                                    MEL_SCALE * floor + MEL_MARGIN_DB, floor from the takes pairwise

Audio is (N, 2) or (N,) float at measure.SR; log-mel is maestro.sounddiff.measure.logmel (NFFT 2048, hop 512), the
same features as verified.mel_d.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

MEL_SCALE, MEL_MARGIN_DB = 1.25, 0.3  # the v5 audio gate's rule shape (maestro.v5.gate)
WIN_S = 0.15


def transition_onsets(notes: list[dict]) -> list[float]:
    """Onsets where a note starts later than the previous onset with a different pitch (chord tones at the same onset
    and repeated pitches are not transitions)."""
    ns = sorted(notes, key=lambda n: (n["start_s"], n["pitch"]))
    out = []
    for prev, n in zip(ns, ns[1:]):
        if n["start_s"] > prev["start_s"] + 1e-3 and n["pitch"] != prev["pitch"]:
            out.append(float(n["start_s"]))
    return out


def _frame_mask(n_frames: int, onsets: list[float], win_s: float) -> np.ndarray:
    from maestro.sounddiff.measure import HOP, NFFT, SR
    centers = (np.arange(n_frames) * HOP + NFFT / 2) / SR
    mask = np.zeros(n_frames, bool)
    for t in onsets:
        mask |= (centers >= t) & (centers < t + win_s)
    return mask


def transition_mel_d(a: np.ndarray, b: np.ndarray, notes: list[dict], win_s: float = WIN_S) -> float | None:
    from maestro.sounddiff.measure import logmel
    onsets = transition_onsets(notes)
    if not onsets:
        return None
    mono = lambda x: x.mean(1) if x.ndim == 2 else x  # noqa: E731
    la, lb = logmel(mono(a)), logmel(mono(b))
    n = min(len(la), len(lb))
    mask = _frame_mask(n, onsets, win_s)
    if not mask.any():
        return None
    return float(np.abs(la[:n][mask] - lb[:n][mask]).mean())


def audible_vs_takes(takes: list[np.ndarray], variant: np.ndarray, notes: list[dict], win_s: float = WIN_S) -> dict:
    """Is `variant` (e.g. the patch with the glide switched off) audibly different from the GT, judged on the
    transition windows? floor = mean pairwise distance of the takes; d = mean distance variant-to-takes."""
    floor_pairs = [transition_mel_d(x, y, notes, win_s) for x, y in combinations(takes, 2)]
    if any(f is None for f in floor_pairs):
        return {"audible": False, "error": "no pitch-change onsets"}
    floor = float(np.mean(floor_pairs))
    d = float(np.mean([transition_mel_d(variant, w, notes, win_s) for w in takes]))
    limit = MEL_SCALE * floor + MEL_MARGIN_DB
    return {"audible": d > limit, "d_db": round(d, 3), "floor_db": round(floor, 3), "limit_db": round(limit, 3),
            "n_onsets": len(transition_onsets(notes))}
