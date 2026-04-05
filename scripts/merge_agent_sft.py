#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import validate_ms_swift_multiturn_record


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge agent SFT JSONL files (search/judge/main).")
    ap.add_argument("--input", action="append", required=True, help="Input JSONL path. Repeat for multiple files.")
    ap.add_argument("--output", required=True, help="Output merged JSONL path.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle merged rows.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dedupe-id", action="store_true", help="Deduplicate by row['id'] keeping first seen.")
    ap.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop rows failing MS-Swift multiturn contract validation.",
    )
    args = ap.parse_args()

    merged: list[dict] = []
    for p in args.input:
        path = Path(p)
        rows = _read_jsonl(path)
        print(f"Loaded {len(rows)} rows from {path}")
        merged.extend(rows)

    if args.dedupe_id:
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in merged:
            rid = str(r.get("id", ""))
            if rid and rid in seen:
                continue
            if rid:
                seen.add(rid)
            deduped.append(r)
        print(f"Deduped: {len(merged)} -> {len(deduped)}")
        merged = deduped

    if args.drop_invalid:
        kept: list[dict] = []
        dropped = 0
        drop_reasons: dict[str, int] = {}
        for row in merged:
            errors = validate_ms_swift_multiturn_record(row)
            if not errors:
                kept.append(row)
                continue
            dropped += 1
            for e in errors:
                drop_reasons[e] = drop_reasons.get(e, 0) + 1
        merged = kept
        print(f"Dropped invalid rows: {dropped}")
        if drop_reasons:
            top = sorted(drop_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print(f"Top drop reasons: {dict(top)}")

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(merged)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in merged:
        t = str(r.get("task_type", "unknown"))
        counts[t] = counts.get(t, 0) + 1
    print(f"Wrote {len(merged)} rows to {out}")
    print(f"Task mix: {counts}")


if __name__ == "__main__":
    main()
