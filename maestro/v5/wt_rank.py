"""Exhaustive wavetable-subset ranking by superposition (docs/wavetable_search_heuristic.md).

In the stripped GT patch (FX off, mod routes cleared, random phase off; filters kept: they are linear here) with no
oscillator cross-modulation (preset_gen.NO_CROSS_MOD) the output is the sum of the oscillators rendered one at a time.
So each candidate table is rendered once per GT oscillator slot (that slot's own settings, the other slots off), and
every assignment of candidates (or nothing) to the slots is scored by ONE metric without rendering it: the STFT is
linear, so a combination's spectrum is the sum of its slots' complex spectra. Frames come from the steady middle of each
note (frame 0, no modulation: the sound is steady there). The GT assignment scores exactly 0; subset sizes, smaller
ones included, fall out of the metric alone (no per-table cost, no slots the patch leaves off).

  ranked = rank_subsets(renderer, gt, candidates)   -> [Ranked(score_db, (table id or None per active slot)), ...]
"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import planner as pl

vl = pl.vitallib
SR = 44100
PATTERN = {"bpm": 120, "notes": [{"pitch": p, "velocity": 100, "start_s": i * 1.5, "dur_s": 1.0}
                                 for i, p in enumerate((36, 48, 60, 72))]}
NFFT = 2048
FRAMES_PER_NOTE = 4
FLOOR_DB = 80.0          # bins more than this far below the target's peak are clamped to the floor
P_WEIGHT = 0.5           # bin weight = power ** P_WEIGHT (see distance)
DUPES = Path(__file__).resolve().parent / "wavetable_frame0_dupes.json"   # scripts/wt_fingerprint.py


@lru_cache(maxsize=1)
def _dupe_key() -> dict[str, str]:
    """Library id -> its frame-0 duplicate group's first id (tables whose frame-0 renders are identical)."""
    out = {}
    if DUPES.exists():
        for g in json.loads(DUPES.read_text())["groups"]:
            for i in g:
                out[i] = g[0]
    return out


def same_table(a: str | None, b: str | None) -> bool:
    """Equal, or frame-0 duplicates of each other (either is equally correct)."""
    k = _dupe_key()
    return a == b or (a is not None and b is not None and k.get(a, a) == k.get(b, b))


def active_slots(gt: dict) -> list[int]:
    return [k for k in (1, 2, 3) if float(gt["settings"].get(f"osc_{k}_on", 0)) > 0.5]


def stripped(gt: dict) -> dict:
    p = copy.deepcopy(gt)
    s = p["settings"]
    for u in vl.FX:
        s[f"{u}_on"] = 0.0
    for i in range(len(s.get("modulations", []))):
        s["modulations"][i] = {"source": "", "destination": ""}
        s[f"modulation_{i + 1}_amount"] = 0.0
    for k in (1, 2, 3):
        s[f"osc_{k}_random_phase"] = 0.0
    return p


def slot_preset(base: dict, slot: int, table: dict) -> dict:
    """Only `slot` sounding (its own settings), with `table`."""
    p = copy.deepcopy(base)
    s = p["settings"]
    for k in active_slots(base):
        if k != slot:
            s[f"osc_{k}_on"] = 0.0
    s["wavetables"][slot - 1] = table
    return p


def sustain_frames(y: np.ndarray, notes: dict = PATTERN, per_note: int = FRAMES_PER_NOTE) -> np.ndarray:
    """Complex spectra of `per_note` evenly spaced frames from the middle of each note -> (frames, bins). Linear in y."""
    m = y.mean(1) if y.ndim == 2 else y
    win = np.hanning(NFFT)
    out = []
    for n in notes["notes"]:
        a, b = n["start_s"] + 0.3 * n["dur_s"], n["start_s"] + 0.9 * n["dur_s"] - NFFT / SR
        for t in np.linspace(a, b, per_note):
            i = int(t * SR)
            out.append(np.fft.rfft(m[i:i + NFFT] * win))
    return np.array(out)


def distance(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """|dB| difference between target frames T (frames, bins) and candidate frames S (..., frames, bins), averaged
    over the bins where either is within FLOOR_DB of the target's peak, each bin weighted by the louder power ** P_WEIGHT
    (= the louder magnitude). Vectorized over S's leading axes.

    P_WEIGHT = 0.5 is what the ear test chose (scripts/wt_metric_check.py, wt_drop_page.py; 23 one-oscillator drops):
    heard vs not-heard separation 0.99 (0.90 unweighted, 0.96 energy-weighted). Unweighted, thousands of quiet bins
    outvote the harmonics that carry an oscillator; energy-weighted, quiet but audible layers are nearly free."""
    ta, sa = np.abs(T), np.abs(S)
    tl, sl = 20 * np.log10(ta + 1e-9), 20 * np.log10(sa + 1e-9)
    floor = tl.max() - FLOOR_DB
    w = np.maximum(ta, sa) ** (2 * P_WEIGHT) * ((sl > floor) | (tl > floor))
    d = np.abs(np.maximum(sl, floor) - np.maximum(tl, floor)) * w
    return d.sum((-2, -1)) / np.maximum(w.sum((-2, -1)), 1e-30)


@dataclass
class Ranked:
    score_db: float           # distance to the target (0 = identical)
    tables: tuple             # table id per active GT slot (None = slot left empty)


def table_scores(ranked: list[Ranked]) -> dict[str, float]:
    """Each table's best score over the ranked assignments it appears in."""
    out: dict[str, float] = {}
    for r in ranked:
        for t in r.tables:
            if t is not None and r.score_db < out.get(t, float("inf")):
                out[t] = r.score_db
    return out


def slot_frames(renderer, gt: dict, candidates: list[dict], notes: dict = PATTERN):
    """(target frames, active slots, ids + [None], per-slot frames (options, frames, bins); the None option is 0)."""
    base = stripped(gt)
    slots = active_slots(base)
    jobs = {"target": base}
    for k in slots:
        for e in candidates:
            jobs[f"{k}/{e['id']}"] = slot_preset(base, k, vl.wt_load(e))
    ys = renderer.render(jobs, notes)
    T = sustain_frames(ys["target"], notes)
    ids = [e["id"] for e in candidates] + [None]
    X = [np.array([sustain_frames(ys[f"{k}/{i}"], notes) for i in ids[:-1]] + [np.zeros_like(T)]) for k in slots]
    return T, slots, ids, X


def rank_subsets(renderer, gt: dict, candidates: list[dict], notes: dict = PATTERN, top: int = 1000) -> list[Ranked]:
    """Every assignment of candidates (or nothing) to each active GT slot, scored exactly; the best `top`."""
    T, slots, ids, X = slot_frames(renderer, gt, candidates, notes)
    rest = list(itertools.product(range(len(ids)), repeat=len(slots) - 1))
    R = np.array([sum((X[s + 1][c] for s, c in enumerate(combo)), np.zeros_like(T)) for combo in rest])
    res = []
    for a in range(len(ids)):
        d = distance(T, X[0][a][None] + R)
        for ci in np.argsort(d)[:top]:
            tables = (ids[a],) + tuple(ids[c] for c in rest[ci])
            if any(t is not None for t in tables):
                res.append(Ranked(float(d[ci]), tables))
    res.sort(key=lambda r: r.score_db)
    return res[:top]
