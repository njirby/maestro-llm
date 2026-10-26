#!/usr/bin/env python3
"""Split agent SFT JSONL into train/val for Megatron-SWIFT training.

Produces a directory with train.jsonl, val.jsonl, and stats.json.
Val samples are chosen by stable hash on sample ID, stratified by task type.

Usage:
    python scripts/prepare_agent_sft_train.py \
        --input outputs/sft_32k/sft_train_v2.jsonl \
        --output-dir outputs/sft_32k/prepared \
        --val-count 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def _hash_to_val(sample_id: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{sample_id}:{seed}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-count", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-audio-check", action="store_true")
    p.add_argument("--exclude-task-type", action="append", default=[],
                   help="Drop rows with this task_type (repeatable)")
    args = p.parse_args()

    exclude_set = set(args.exclude_task_type)
    rows: list[dict] = []
    excluded = 0
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_type", "") in exclude_set:
                excluded += 1
                continue
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {args.input}", flush=True)
    if excluded:
        print(f"  Excluded {excluded} rows (task_type in {exclude_set})", flush=True)

    if not args.skip_audio_check:
        missing = 0
        for row in rows:
            for audio_path in row.get("audios", []):
                if not Path(audio_path).exists():
                    missing += 1
        if missing:
            print(f"ERROR: {missing} audio files missing on disk", file=sys.stderr)
            sys.exit(1)
        print(f"All audio paths verified", flush=True)

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        task = row.get("task_type", "unknown")
        by_task.setdefault(task, []).append(row)

    n_tasks = len(by_task)
    per_task_val = max(1, args.val_count // n_tasks)

    val_ids: set[str] = set()
    for task, task_rows in by_task.items():
        scored = [(r, _hash_to_val(r["id"], args.seed)) for r in task_rows]
        scored.sort(key=lambda x: x[1])
        for r, _ in scored[:per_task_val]:
            val_ids.add(r["id"])

    remaining = args.val_count - len(val_ids)
    if remaining > 0:
        candidates = [(r, _hash_to_val(r["id"], args.seed))
                      for r in rows if r["id"] not in val_ids]
        candidates.sort(key=lambda x: x[1])
        for r, _ in candidates[:remaining]:
            val_ids.add(r["id"])

    train_rows = [r for r in rows if r["id"] not in val_ids]
    val_rows = [r for r in rows if r["id"] in val_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_rows in [("train", train_rows), ("val", val_rows)]:
        path = args.output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_tasks = Counter(r.get("task_type", "?") for r in train_rows)
    val_tasks = Counter(r.get("task_type", "?") for r in val_rows)

    stats = {
        "source": str(args.input),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "seed": args.seed,
        "train_by_task": dict(train_tasks.most_common()),
        "val_by_task": dict(val_tasks.most_common()),
    }
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTrain: {len(train_rows)}", flush=True)
    for t, c in train_tasks.most_common():
        print(f"  {t}: {c}", flush=True)
    print(f"\nVal: {len(val_rows)}", flush=True)
    for t, c in val_tasks.most_common():
        print(f"  {t}: {c}", flush=True)
    print(f"\nWritten to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
