"""Wavetable fingerprints for the search heuristic (first-frame POC): every library table rendered once through the init
patch on osc 1 (frame 0, or 128 for tables silent at frame 0; random phase off) with a fixed note pattern, reduced to
its average log-magnitude spectrum per note, and a table-to-table distance matrix.

  .venv/bin/python scripts/wt_fingerprint.py build            -> outputs/wt_rank/fingerprints.npz
  .venv/bin/python scripts/wt_fingerprint.py check --n 20     agreement with in-context renders on fresh GT presets
  .venv/bin/python scripts/wt_fingerprint.py dupes            -> maestro/v5/wavetable_frame0_dupes.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.v5 import planner as pl  # noqa: E402

vl = pl.vitallib
INIT = json.loads((pl.SKILL_DIR / "data" / "init_preset.json").read_text())
OUT = ROOT / "outputs/wt_rank"
SR = 44100
PITCHES = (36, 48, 60, 72)
NOTE_S, GAP_S = 1.0, 0.5
PATTERN = {"bpm": 120, "notes": [{"pitch": p, "velocity": 100, "start_s": i * (NOTE_S + GAP_S), "dur_s": NOTE_S}
                                 for i, p in enumerate(PITCHES)]}
NFFT = (1024, 4096)


def fp_preset(entry: dict) -> dict:
    p = copy.deepcopy(INIT)
    s = p["settings"]
    s["wavetables"][0] = vl.wt_load(entry)
    s["osc_1_random_phase"] = 0.0
    s["osc_1_wave_frame"] = 128.0 if entry.get("silent_frame0") else 0.0
    return p


def spectra(y: np.ndarray, notes=PATTERN["notes"]) -> np.ndarray:
    """Per note: the mean log-magnitude spectrum (dB) over the middle of the note, at each FFT size; concatenated."""
    m = y.mean(1) if y.ndim == 2 else y
    feats = []
    for n in notes:
        a = int((n["start_s"] + 0.25 * n["dur_s"]) * SR)
        b = int((n["start_s"] + 0.85 * n["dur_s"]) * SR)
        seg = m[a:b]
        for nfft in NFFT:
            hop = nfft // 2
            frames = [seg[i:i + nfft] * np.hanning(nfft) for i in range(0, max(1, len(seg) - nfft), hop)]
            mag = np.abs(np.fft.rfft(np.array(frames), axis=1)).mean(0) if frames else np.zeros(nfft // 2 + 1)
            feats.append(20 * np.log10(mag + 1e-6))
    return np.concatenate(feats)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |dB| difference, clipped at -100 dB so near-silent bins don't dominate."""
    return float(np.abs(np.maximum(a, -100) - np.maximum(b, -100)).mean())


def build():
    idx = vl.wt_index()
    with ContainerRenderer(OUT / "fp_cache", jobs=8) as r:
        r._session()
        ys = r.render({e["id"]: fp_preset(e) for e in idx}, PATTERN)
    F = np.stack([spectra(ys[e["id"]]) for e in idx])
    D = np.array([[dist(F[i], F[j]) for j in range(len(idx))] for i in range(len(idx))])
    np.savez(OUT / "fingerprints.npz", ids=np.array([e["id"] for e in idx]), F=F, D=D)
    print(f"{len(idx)} tables; median pairwise distance {np.median(D[np.triu_indices(len(idx), 1)]):.2f} dB")


def check(n: int, seed: int):
    """Fresh first-frame GT presets: for the 32-table chunk around the GT osc-1 table, rank the tables by fingerprint
    distance to the GT table and by in-context distance (GT patch with osc 1 swapped, random phase off, the fixed
    pattern) to the target. Report rank agreement and where the GT lands in-context."""
    from maestro.synth.preset_gen import ARCHETYPES, generate_preset
    from maestro.synth.wavetable_lib import load_wavetable_lib
    z = np.load(OUT / "fingerprints.npz")
    ids, D = list(z["ids"]), z["D"]
    idx, byh = vl.wt_index(), pl._wt_by_hash()
    lib = load_wavetable_lib(ROOT / "data" / "wavetable_lib.json")
    rng = random.Random(seed)
    rows = []
    with ContainerRenderer(OUT / "fp_cache", jobs=8) as r:
        r._session()
        while len(rows) < n:
            gt = generate_preset(rng.choice(ARCHETYPES), random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
            e = byh.get(pl._wt_hash(gt["settings"]["wavetables"][0]))
            if e is None:
                continue
            for k in (1, 2, 3):
                gt["settings"][f"osc_{k}_random_phase"] = 0.0
            pos = ids.index(e["id"])
            lo = pos // 32 * 32
            chunk = ids[lo:lo + 32]
            swaps = {}
            for c in chunk:
                p = copy.deepcopy(gt)
                p["settings"]["wavetables"][0] = vl.wt_load(idx[ids.index(c)])
                swaps[c] = p
            ys = r.render({"target": gt, **swaps}, PATTERN)
            ft = spectra(ys["target"])
            ctx = {c: dist(ft, spectra(ys[c])) for c in chunk}
            fp = {c: float(D[pos, ids.index(c)]) for c in chunk}
            order_ctx = sorted(chunk, key=ctx.get)
            order_fp = sorted(chunk, key=fp.get)
            rk = lambda o: {c: i for i, c in enumerate(o)}  # noqa: E731
            a, b = rk(order_ctx), rk(order_fp)
            m = len(chunk)
            rho = 1 - 6 * sum((a[c] - b[c]) ** 2 for c in chunk) / (m * (m * m - 1))
            top5 = len(set(order_ctx[:6]) & set(order_fp[:6])) - 1  # excluding the GT itself (rank 0 in both)
            rows.append({"gt": e["id"], "rho": rho, "top5_overlap": top5, "gt_ctx_rank": a[e["id"]] + 1,
                         "ctx_gt_dist": ctx[e["id"]], "ctx_next": sorted(v for c, v in ctx.items() if c != e["id"])[:3]})
            print(f"{e['id']}: spearman {rho:.2f}, top-5 overlap {top5}/5, GT in-context rank {a[e['id']] + 1} "
                  f"(dist {ctx[e['id']]:.2f}; next {[round(v, 2) for v in rows[-1]['ctx_next']]})", flush=True)
    (OUT / "fp_check.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} samples: median spearman {statistics.median(x['rho'] for x in rows):.2f}, "
          f"mean top-5 overlap {statistics.mean(x['top5_overlap'] for x in rows):.1f}/5, "
          f"GT first in-context {sum(x['gt_ctx_rank'] == 1 for x in rows)}/{len(rows)}")


def dupes():
    """Groups of library tables whose frame-0 fingerprint renders are identical (any of a group is equally correct)."""
    z = np.load(OUT / "fingerprints.npz")
    ids, D = [str(i) for i in z["ids"]], z["D"]
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if D[i, j] < 1e-6:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(ids[i])
    g = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    dst = ROOT / "maestro/v5/wavetable_frame0_dupes.json"
    dst.write_text(json.dumps({"note": "library tables whose frame-0 renders are identical (scripts/wt_fingerprint.py "
                                       "dupes); any of a group is equally correct", "groups": g}, indent=1))
    print(f"{len(g)} groups, {sum(map(len, g))} tables -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "check", "dupes"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"build": build, "dupes": dupes}.get(a.cmd, lambda: check(a.n, a.seed))()
