"""Check maestro/v5/wt_rank on fresh POC GT presets (first frame, no cross-mod): the candidates are a 32-table chunk
plus the GT tables (so the GT assignment is reachable). Reports the GT's rank and score, the top runners-up, the
superposition error (top-3 predicted vs actually rendered), and the time.

  .venv/bin/python scripts/wt_rank_check.py --n 8
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.synth.preset_gen import ARCHETYPES, generate_preset  # noqa: E402
from maestro.synth.wavetable_lib import load_wavetable_lib  # noqa: E402
from maestro.v5 import planner as pl, wt_rank as wr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
lib = load_wavetable_lib(ROOT / "data/wavetable_lib.json")
idx, byh = pl.vitallib.wt_index(), pl._wt_by_hash()
rng = random.Random(a.seed)
rows = []
with ContainerRenderer(ROOT / "outputs/wt_rank/rank_cache", jobs=8) as r:
    r._session()
    while len(rows) < a.n:
        gt = generate_preset(rng.choice(ARCHETYPES), random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
        slots = wr.active_slots(gt)
        gts = [byh.get(pl._wt_hash(gt["settings"]["wavetables"][k - 1])) for k in slots]
        if None in gts:
            continue
        pos = next(i for i, x in enumerate(idx) if x["id"] == gts[0]["id"])
        lo = pos // 32 * 32
        cands = {e["id"]: e for e in idx[lo:lo + 32]}
        cands.update({e["id"]: e for e in gts})
        t0 = time.time()
        ranked = wr.rank_subsets(r, gt, list(cands.values()))
        dt = time.time() - t0
        want = tuple(e["id"] for e in gts)
        gi = next((i for i, x in enumerate(ranked) if x.tables == want), None)
        # superposition check: render the top-3 non-GT assignments for real (stripped patch) and re-score
        base = wr.stripped(gt)
        checks = []
        for x in [x for x in ranked if x.tables != want][:3]:
            p = json.loads(json.dumps(base))
            for k, tid in zip(slots, x.tables):
                if tid is None:
                    p["settings"][f"osc_{k}_on"] = 0.0
                else:
                    p["settings"]["wavetables"][k - 1] = pl.vitallib.wt_load(cands[tid])
            ys = r.render({"t": base, "c": p}, wr.PATTERN)
            checks.append(round(float(wr.distance(wr.sustain_frames(ys["t"]), wr.sustain_frames(ys["c"]))) - x.score_db, 3))
        row = {"slots": len(slots), "gt": want, "gt_rank": None if gi is None else gi + 1,
               "gt_score": None if gi is None else round(ranked[gi].score_db, 3),
               "next": [(round(x.score_db, 2), x.tables) for x in ranked if x.tables != want][:4],
               "render_minus_predicted": checks, "seconds": round(dt, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
(ROOT / "outputs/wt_rank/rank_check.json").write_text(json.dumps(rows, indent=1))
