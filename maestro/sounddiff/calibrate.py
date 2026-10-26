"""Rank-calibrate absolute descriptors and CLAP tone adjectives on a reference set (items.py refset, rendered and
measured), so first-impression buckets mean "darker than most generated presets" rather than an arbitrary Hz cut, and
an adjective is only used when CLAP prefers it unusually strongly for this sound.

  .venv/bin/python -m maestro.sounddiff.calibrate outputs/sounddiff/refset_v2   # -> maestro/sounddiff/calib.json

refset_v2 (2026-10-26, refset.py): fixed preset_gen + loudness normalization / level QC, rendered by the skill's
render.py; the v1 set (outputs/sounddiff/refset, render_pairs, pre-fix generator) and its calibration
(refset/calib_v1.json) are kept for reference.

Also prints adjective sanity correlations against descriptors (does "plucky" track low sustain, "dark" low centroid?);
adjectives whose sanity check fails are listed in calib.json "adj_drop" and never emitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .measure import ADJECTIVES, load_dir

CALIB_PATH = Path(__file__).with_name("calib.json")
KEYS = ["centroid_w_hz", "harm_rich_db", "flatness", "width_db", "attack_ms", "sustain_ratio", "tail_s", "gap_fill_db",
        "mod_cent", "mod_amp", "mod_note", "loudness_db", "crest_db"]
BAND_KEYS = ["sub", "low", "mid", "high_mid", "air"]
PCTS = list(range(0, 101, 5))
# adjective -> (descriptor, expected sign of the Spearman correlation); adjectives failing their check are dropped
SANITY = {"dark": ("centroid_w_hz", -1), "bright": ("centroid_w_hz", 1), "warm": ("centroid_w_hz", -1),
          "plucky": ("sustain_ratio", -1), "percussive": ("sustain_ratio", -1), "soft": ("attack_ms", 1),
          "gritty": ("harm_rich_db", 1), "distorted": ("harm_rich_db", 1), "buzzy": ("harm_rich_db", 1),
          "harsh": ("centroid_w_hz", 1), "airy": ("centroid_w_hz", 1), "spacey": ("tail_s", 1),
          "wobbly": ("mod_cent", 1), "thin": ("centroid_w_hz", 1), "muddy": ("centroid_w_hz", -1),
          "smooth": ("harm_rich_db", -1), "punchy": ("attack_ms", -1), "detuned": ("width_db", 1),
          "lush": ("width_db", 1)}


def centered(adj: dict) -> np.ndarray:
    v = np.array([adj[a] for a in ADJECTIVES])
    return v - v.mean()


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    d = Path(sys.argv[1])
    rows = list(load_dir(d).values())
    calib = {"n": len(rows), "pcts": PCTS, "q": {}, "bands_q": {}}
    for k in KEYS:
        v = np.array([r["desc"][k] for r in rows if r["desc"].get(k) is not None], float)
        calib["q"][k] = np.percentile(v, PCTS).round(5).tolist()
    for b in BAND_KEYS:
        v = np.array([r["desc"]["bands"][b] for r in rows])
        calib["bands_q"][b] = np.percentile(v, PCTS).round(6).tolist()
    C = np.stack([centered(r["adj"]) for r in rows])
    calib["adj"] = {a: {"mean": float(C[:, i].mean()), "std": float(C[:, i].std() + 1e-9)} for i, a in enumerate(ADJECTIVES)}
    Z = (C - C.mean(0)) / (C.std(0) + 1e-9)
    drop = []
    print("adjective sanity (Spearman of z vs descriptor):")
    for a, (k, sign) in SANITY.items():
        idx = [i for i, r in enumerate(rows) if r["desc"].get(k) is not None]
        rho = spearman(Z[idx, ADJECTIVES.index(a)], np.array([rows[i]["desc"][k] for i in idx]))
        ok = rho * sign > 0.1
        drop += [] if ok else [a]
        print(f"  {a:11} vs {k:14} rho {rho:+.2f} {'ok' if ok else 'DROP'}")
    calib["adj_drop"] = drop
    CALIB_PATH.write_text(json.dumps(calib, indent=1))
    print(f"{len(rows)} reference sounds -> {CALIB_PATH}; dropped adjectives: {drop}")


if __name__ == "__main__":
    main()
