"""Sanity check of the wt_rank metric: drop one oscillator at a time from each target (stripped patch) and compare the
metric's cost with that oscillator's loudness share. A solid metric charges more for dropping a louder oscillator, and
dropping a big share should cost a lot. An energy-weighted variant is printed alongside for contrast (not adopted).

  .venv/bin/python scripts/wt_metric_check.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.synth.preset_gen import ARCHETYPES, generate_preset  # noqa: E402
from maestro.synth.wavetable_lib import load_wavetable_lib  # noqa: E402
from maestro.v5 import planner as pl, wt_rank as wr  # noqa: E402


def weighted(T, S, p=1.0):
    """Mean |dB| difference with each bin weighted by (the louder of the two powers) ** p: p=1 energy, p~0.3 loudness
    (Stevens), p=0 every bin within the floor equally (the current metric)."""
    tl, sl = 20 * np.log10(np.abs(T) + 1e-9), 20 * np.log10(np.abs(S) + 1e-9)
    floor = tl.max() - wr.FLOOR_DB
    pw = np.maximum(np.abs(T) ** 2, np.abs(S) ** 2)
    w = pw ** p * ((tl > floor) | (sl > floor))
    return float((w * np.abs(np.maximum(sl, floor) - np.maximum(tl, floor))).sum() / w.sum())


EXPONENTS = (0.0, 0.15, 0.3, 0.5, 1.0)


lib = load_wavetable_lib(ROOT / "data/wavetable_lib.json")
byh = pl._wt_by_hash()
targets, rng = [], random.Random(1)
while len(targets) < 8:
    gt = generate_preset(rng.choice(ARCHETYPES), random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
    if None not in [byh.get(pl._wt_hash(gt["settings"]["wavetables"][k - 1])) for k in wr.active_slots(gt)]:
        targets.append(gt)
targets += [json.loads(f.read_text()) for f in sorted((ROOT / "outputs/v5_rollouts/probe_wtrank/inputs").glob("*/gt_preset.json"))]
rows = []
with ContainerRenderer(ROOT / "outputs/wt_rank/rank_cache", jobs=8) as r:
    r._session()
    for ti, gt in enumerate(targets):
        slots = wr.active_slots(gt)
        if len(slots) < 2:
            continue
        gts = [byh[pl._wt_hash(gt["settings"]["wavetables"][k - 1])] for k in slots]
        T, sl, ids, X = wr.slot_frames(r, gt, gts)
        own = [X[s][ids.index(gts[s]["id"])] for s in range(len(slots))]
        tot = (np.abs(T) ** 2).sum()
        for s, k in enumerate(slots):
            S = sum(own[j] for j in range(len(slots)) if j != s)
            share = 10 * np.log10((np.abs(own[s]) ** 2).sum() / tot)
            costs = {p: round(weighted(T, S, p), 2) for p in EXPONENTS}
            rows.append({"t": ti, "osc": k, "share": round(float(share), 1), "costs": costs})
            print(f"target {ti:2d} drop osc {k}: share {share:6.1f} dB -> " +
                  "  ".join(f"p={p}: {c:5.2f}" for p, c in costs.items()), flush=True)
(ROOT / "outputs/wt_rank/metric_check.json").write_text(json.dumps(rows, indent=1))
sh = np.array([x["share"] for x in rows])
rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
print(f"\n{len(rows)} drops")
for p in EXPONENTS:
    c = np.array([x["costs"][p] for x in rows])
    rho = float(np.corrcoef(rank(sh), rank(c))[0, 1])
    print(f"p={p:<4}: Spearman(share, cost) {rho:.2f}; cost of dropping a >= -6 dB share: min {c[sh >= -6].min():5.2f}; "
          f"-6..-20 dB shares: {sorted(c[(sh < -6) & (sh >= -20)].round(2).tolist())}")
