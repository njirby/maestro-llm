#!/usr/bin/env python
"""Assemble the mix-balanced POC training set from repaired legacy data.

All repaired mains + search/judge subsampled to a target audio-clip share
(legacy natural mix is 90% search by audio clips — see docs/sft-pipeline.md
data-mix notes) + all transcription records (negligible audio).

Usage:
    python scripts/assemble_poc_trainset.py \
        --repaired outputs/sft_32k_repaired/main_repaired.jsonl \
        --source outputs/sft_32k/sft_train_v3.jsonl \
        --out outputs/sft_32k_repaired/poc_train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--search-audio-ratio", type=float, default=1.5,
                    help="Target search audio clips as a multiple of main audio clips.")
    ap.add_argument("--judge-audio-ratio", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    mains = [json.loads(l) for l in open(args.repaired)]
    main_clips = sum(len(r.get("audios", [])) for r in mains)

    pools: dict[str, list] = {"search_v2": [], "judge": [], "melody_transcription": []}
    # Prefer contract-aligned subagent files (D1) beside the repaired mains;
    # fall back to the raw source otherwise.
    aligned = {"search_v2": args.repaired.parent / "search_aligned.jsonl",
               "judge": args.repaired.parent / "judge_aligned.jsonl",
               "melody_transcription": args.repaired.parent / "transcription_aligned.jsonl"}
    if all(p.exists() for p in aligned.values()):
        for t, path in aligned.items():
            pools[t] = [json.loads(l) for l in open(path)]
    else:
        for line in open(args.source):
            r = json.loads(line)
            t = r.get("task_type")
            if t in pools:
                pools[t].append(r)

    def sample_to_clips(records: list, target_clips: float) -> list:
        rng.shuffle(records)
        out, clips = [], 0
        for r in records:
            if clips >= target_clips:
                break
            out.append(r)
            clips += len(r.get("audios", []))
        return out

    picked = {
        "main": mains,
        "search_v2": sample_to_clips(pools["search_v2"], args.search_audio_ratio * main_clips),
        "judge": sample_to_clips(pools["judge"], args.judge_audio_ratio * main_clips),
        "melody_transcription": pools["melody_transcription"],
    }

    all_records = [r for rs in picked.values() for r in rs]
    rng.shuffle(all_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot_r = len(all_records)
    tot_a = sum(len(r.get("audios", [])) for r in all_records) or 1
    tot_c = sum(sum(len(str(m.get("content", ""))) for m in r["messages"]) for r in all_records) or 1
    print(f"{'type':<24}{'records':<10}{'% recs':<9}{'% chars':<9}{'% audio clips'}")
    for t, rs in picked.items():
        a = sum(len(r.get("audios", [])) for r in rs)
        c = sum(sum(len(str(m.get("content", ""))) for m in r["messages"]) for r in rs)
        print(f"{t:<24}{len(rs):<10}{100*len(rs)/tot_r:<9.1f}{100*c/tot_c:<9.1f}{100*a/tot_a:.1f}")
    print(f"wrote {tot_r} records to {args.out}")


if __name__ == "__main__":
    main()
