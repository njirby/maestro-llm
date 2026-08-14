#!/usr/bin/env python
"""Score search-agent final-selection generations (teacher-forced context).

swift-infer on full search records generates the final assistant message
(all probe listens as context). This scores the generated shortlist against
the record's teacher shortlist and GT-wavetable metadata.

Metrics:
  - shortlist_parse_rate: generated message contains a parseable Shortlist line
  - gt_recall: on records where the GT wavetable was in the shard, is it in
    the generated shortlist? (teacher = 100% by construction)
  - teacher_jaccard: overlap of generated vs teacher shortlist
  - size stats + empty-shortlist behavior on GT-absent records

Usage:
    python scripts/eval_search_selection.py \
        --results <swift_result.jsonl> --slice <eval_slice.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

_SHORTLIST_RE = re.compile(r"Shortlist:\s*(\[[^\]]*\])")


def parse_shortlist(text: str) -> list | None:
    m = _SHORTLIST_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--slice", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    slice_recs = [json.loads(l) for l in open(args.slice)]
    results = [json.loads(l) for l in open(args.results)]
    rows = []
    for rec, res in zip(slice_recs, results):
        meta = rec["meta"]
        gen = parse_shortlist(res.get("response") or "")
        teacher = meta.get("final_shortlist") or []
        gt = set(meta.get("gt_in_shard") or [])
        row = {
            "id": rec["id"],
            "parsed": gen is not None,
            "gen_size": len(gen) if gen else 0,
            "teacher_size": len(teacher),
            "gt_present": bool(gt),
        }
        if gen is not None:
            gset, tset = set(gen), set(teacher)
            row["teacher_jaccard"] = len(gset & tset) / max(1, len(gset | tset))
            row["exact_match"] = gset == tset
            if gt:
                row["gt_recall"] = len(gt & gset) / len(gt)
        rows.append(row)

    parsed = [r for r in rows if r["parsed"]]
    gtr = [r["gt_recall"] for r in parsed if "gt_recall" in r]
    absent = [r for r in parsed if not r["gt_present"]]
    summary = {
        "n": len(rows),
        "parse_rate": len(parsed) / max(1, len(rows)),
        "gt_recall_mean": st.mean(gtr) if gtr else None,
        "n_gt_records": len(gtr),
        "teacher_jaccard_mean": st.mean([r["teacher_jaccard"] for r in parsed]) if parsed else None,
        "exact_match_rate": st.mean([1.0 if r["exact_match"] else 0.0 for r in parsed]) if parsed else None,
        "mean_gen_size": st.mean([r["gen_size"] for r in parsed]) if parsed else None,
        "mean_teacher_size": st.mean([r["teacher_size"] for r in parsed]) if parsed else None,
        "empty_on_gt_absent_rate": st.mean([1.0 if r["gen_size"] == 0 else 0.0 for r in absent]) if absent else None,
    }
    print(json.dumps(summary, indent=2))
    out = args.out or args.results.with_suffix(".scored.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
