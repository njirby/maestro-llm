#!/usr/bin/env python3
"""Experiment: determine the CLAP cosine threshold for "GT proxy" coverage.

When a search agent misses the exact GT wavetable but returns nearest-neighbor
wavetables, how similar are they typically? This experiment measures the
distribution of "nearest non-GT similarity" across all samples.

The result informs the coverage threshold: if a pool contains a wavetable
with CLAP cosine >= threshold to a GT, treat that GT as "covered by proxy."

Usage:
    python scripts/experiment_gt_coverage_threshold.py \
        --manifest outputs/smoke_test_v10/manifest.jsonl \
        --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
        --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_wavetable_retrieval_baseline import _extract_gt_wavetable_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5, help="How many nearest non-GT neighbors to record per GT.")
    args = ap.parse_args()

    # Load index
    idx = np.load(args.index_npy)
    embeddings = idx["embeddings"].astype(np.float32)

    with open(args.index_meta) as f:
        meta = json.load(f)
    rows = meta["rows"]

    # Build name → row indices (multi-frame wavetables have multiple rows)
    name_to_row_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        name_to_row_indices[row["wavetable_name"]].append(i)

    all_names = sorted(set(r["wavetable_name"] for r in rows))

    # Per-name embedding (mean-pool across frames)
    name_to_emb: dict[str, np.ndarray] = {}
    for name, idxs in name_to_row_indices.items():
        name_to_emb[name] = embeddings[idxs].mean(axis=0)

    # Build (n_names, dim) matrix of L2-normalized embeddings
    name_emb_matrix = np.stack([name_to_emb[n] for n in all_names])
    name_norms = np.linalg.norm(name_emb_matrix, axis=1, keepdims=True) + 1e-12
    name_emb_normed = name_emb_matrix / name_norms
    name_to_pos = {n: i for i, n in enumerate(all_names)}

    # Load manifest
    entries = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f"Index: {len(rows)} rows, {len(all_names)} unique wavetable names")
    print(f"Manifest: {len(entries)} samples\n")

    # Collect per-GT top-K nearest non-GT similarities
    # Separate buckets for: all, per-archetype, per rank (1st / 3rd / 5th nearest)
    all_top1: list[float] = []
    all_top3: list[float] = []
    all_top5: list[float] = []
    by_archetype: dict[str, list[float]] = defaultdict(list)  # rank-1 per archetype
    per_gt_records: list[dict] = []

    skipped = 0
    for entry in entries:
        sample_id = entry["sample_id"]
        archetype = entry.get("archetype", "unknown")
        target_preset_path = entry.get("target_preset_path")
        if not target_preset_path:
            path_file = entry.get("path_file")
            if path_file:
                with open(path_file) as f:
                    pd = json.load(f)
                target_preset_path = pd.get("target_preset_path")
        if not target_preset_path:
            skipped += 1
            continue
        gt_names = _extract_gt_wavetable_names(Path(target_preset_path))
        if not gt_names:
            skipped += 1
            continue

        gt_set = set(gt_names)
        for gt in gt_names:
            if gt not in name_to_pos:
                continue
            gt_pos = name_to_pos[gt]
            gt_vec = name_emb_normed[gt_pos]

            # Similarities against all wavetables
            sims = name_emb_normed @ gt_vec  # (n_names,)

            # Mask out GT (and other GTs in the sample) to find nearest NON-GT
            for gt_other in gt_set:
                if gt_other in name_to_pos:
                    sims[name_to_pos[gt_other]] = -1.0

            # Top-K nearest non-GT
            top_k_idx = np.argsort(-sims)[: args.top_k]
            top_k_sims = sims[top_k_idx].tolist()
            top_k_names = [all_names[i] for i in top_k_idx]

            all_top1.append(top_k_sims[0])
            if len(top_k_sims) >= 3:
                all_top3.append(top_k_sims[2])
            if len(top_k_sims) >= 5:
                all_top5.append(top_k_sims[4])
            by_archetype[archetype].append(top_k_sims[0])

            per_gt_records.append({
                "sample_id": sample_id,
                "archetype": archetype,
                "gt": gt,
                "top_k_sims": [round(s, 4) for s in top_k_sims],
                "top_k_names": top_k_names,
            })

    if skipped:
        print(f"Skipped {skipped} entries (no target preset or GT names).\n")

    # Summary
    print("=" * 80)
    print(f"Distribution of nearest non-GT CLAP cosine (n={len(all_top1)} GT wavetables)")
    print("=" * 80)

    def _stats(label: str, xs: list[float]) -> None:
        if not xs:
            return
        xs_arr = np.array(xs)
        pct = lambda p: float(np.percentile(xs_arr, p))
        print(f"\n{label} (n={len(xs)}):")
        print(f"  mean   = {xs_arr.mean():.4f}")
        print(f"  median = {np.median(xs_arr):.4f}")
        print(f"  min    = {xs_arr.min():.4f}")
        print(f"  max    = {xs_arr.max():.4f}")
        print(f"  p10/25/50/75/90 = {pct(10):.4f} / {pct(25):.4f} / {pct(50):.4f} / {pct(75):.4f} / {pct(90):.4f}")

    _stats("Rank-1 nearest non-GT similarity (best proxy)", all_top1)
    _stats("Rank-3 nearest non-GT similarity (3rd best proxy)", all_top3)
    _stats("Rank-5 nearest non-GT similarity (5th best proxy)", all_top5)

    print("\n" + "=" * 80)
    print("Rank-1 by archetype")
    print("=" * 80)
    for arch in sorted(by_archetype):
        xs = by_archetype[arch]
        xs_arr = np.array(xs)
        print(f"  {arch:12s}: mean={xs_arr.mean():.4f}  median={np.median(xs_arr):.4f}  n={len(xs)}")

    print("\n" + "=" * 80)
    print("Coverage rate at various thresholds")
    print("=" * 80)
    print("(fraction of GTs where nearest non-GT similarity >= threshold)")
    print(f"{'Threshold':>10} | {'Coverage':>10}")
    print("-" * 25)
    for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95]:
        covered = sum(1 for s in all_top1 if s >= t)
        print(f"{t:>10.2f} | {covered / len(all_top1):>10.4f}")

    # Show a few example GT → nearest-non-GT pairs to sanity-check
    print("\n" + "=" * 80)
    print("Example nearest non-GT neighbors (first 6 samples)")
    print("=" * 80)
    for r in per_gt_records[:6]:
        print(f"\n  {r['sample_id']} ({r['archetype']}) — GT: '{r['gt']}'")
        for name, sim in zip(r["top_k_names"][:3], r["top_k_sims"][:3]):
            print(f"    {sim:.4f}  {name}")

    # Write detailed results
    out_path = args.manifest.parent / "gt_coverage_threshold_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_samples": len(entries),
            "n_gts": len(all_top1),
            "rank_1_distribution": {
                "mean": float(np.mean(all_top1)) if all_top1 else None,
                "median": float(np.median(all_top1)) if all_top1 else None,
                "p25": float(np.percentile(all_top1, 25)) if all_top1 else None,
                "p75": float(np.percentile(all_top1, 75)) if all_top1 else None,
            },
            "per_gt": per_gt_records,
        }, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
