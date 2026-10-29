"""Choose the training records of a v5 run: one record per agent type per rollout, so the dataset is balanced across
main / transcriber / searcher / judge (a rollout has ~12 searcher records against 1 of each other type).

Per rollout:
  main         the main record
  transcriber  the trainable one (a transcriber that ended on an uncaught residual error has train=false; then the
               re-check transcriber that catches it is the one kept)
  judge        the judge record
  searcher     one, stratified across the run ~50/50 between chunks that held a correct table and chunks that
               honestly offered their best alternatives (else ~75% would be "nothing correct here")

  .venv/bin/python scripts/select_records_v5.py --run outputs/v5_rollouts/<run>   -> <run>/selection.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.run / "index.jsonl")]
    rows = [r for r in rows if "main_id" in r]
    held_n = other_n = 0
    sel = {}
    for r in sorted(rows, key=lambda r: r["sample_id"]):
        sid = r["sample_id"]
        links = r["links"]
        tx = [l for l in links if l["agent"] == "transcriber" and l.get("train", True)]
        judge = [l for l in links if l["agent"] == "judge"]
        sev = {e["k"]: e.get("reply_holds_gt") for e in r["events"] if e.get("agent") == "searcher"}
        srch = [l for l in links if l["agent"] == "searcher"]
        held = [l for l in srch if sev.get(int(l["record_id"].rsplit("searcher", 1)[1])) is True]
        other = [l for l in srch if l not in held]
        # stratify: take from the under-represented group when this rollout has one of each
        pool = held if (held and (held_n <= other_n or not other)) else other or held
        pick = rng.choice(pool)
        held_n += pick in held
        other_n += pick not in held
        sel[sid] = [r["main_id"]] + [l["record_id"] for l in tx[-1:] + judge] + [pick["record_id"]]
    (a.run / "selection.json").write_text(json.dumps({"selection": sel, "searchers_holding_correct": held_n,
                                                      "searchers_other": other_n}, indent=1))
    print(f"{len(sel)} rollouts -> {sum(map(len, sel.values()))} records; searchers holding a correct table "
          f"{held_n}, other {other_n}")


if __name__ == "__main__":
    main()
