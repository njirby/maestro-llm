#!/usr/bin/env python
"""Assemble a pilot training set: ALL records, no train/test split.

For the tracer-bullet memorization test — a full finetune driven to ~zero
loss on every generated record, then replayed through the harness. No
holdout by design: the question is "can a small model reproduce these
conversations verbatim", which validates the data/contract/serving path
end to end before any real training investment.

Usage:
    python scripts/assemble_pilot_trainset.py \
        --records-dir outputs/pilot_oc_v3 --out outputs/pilot_oc_v3/train_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate", action="store_true", default=True)
    args = ap.parse_args()

    from scripts.agent_sft_common import validate_oc_record

    kinds = ("main", "search", "judge", "transcription")
    all_recs: list[dict] = []
    per_kind = Counter()
    for kind in kinds:
        for f in sorted(args.records_dir.glob(f"{kind}_*.jsonl")):
            for line in open(f):
                rec = json.loads(line)
                all_recs.append(rec)
                per_kind[kind] += 1

    failures = []
    if args.validate:
        for rec in all_recs:
            errs = list(validate_oc_record(rec))
            if errs:
                failures.append((rec.get("id"), errs))
    if failures:
        print(f"ABORT: {len(failures)} records fail contract validation:")
        for rid, errs in failures[:5]:
            print(f"  {rid}: {errs[:2]}")
        raise SystemExit(1)

    # audio existence check — a missing wav silently poisons training
    missing = []
    for rec in all_recs:
        for p in rec.get("audios", []):
            if not Path(p).exists():
                missing.append((rec.get("id"), p))
    if missing:
        print(f"ABORT: {len(missing)} referenced audio files missing, e.g.:")
        for rid, p in missing[:5]:
            print(f"  {rid}: {p}")
        raise SystemExit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_audio = sum(len(r.get("audios", [])) for r in all_recs)
    n_msgs = sum(len(r["messages"]) for r in all_recs)
    print(f"wrote {len(all_recs)} records to {args.out}")
    for k in kinds:
        print(f"  {k:<16} {per_kind[k]}")
    print(f"  total messages   {n_msgs}")
    print(f"  total audio refs {n_audio}")
    print("  contract validation: PASS; audio existence: PASS")


if __name__ == "__main__":
    main()
