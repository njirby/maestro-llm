#!/usr/bin/env python3
"""Experiment: determine CLAP cosine threshold for GT-wavetable-based candidate pools.

For each sample in a manifest, computes cosine similarity between the GT wavetable
embeddings (from the wavetable index) and ALL other wavetable embeddings. Reports
pool sizes at various thresholds.

This is apples-to-apples: both GT and candidate embeddings come from bare wavetable
probes rendered through the default preset with identical MIDI.  Unlike the original
retrieval baseline (R@5=4.95%), which compared fully-processed target audio against
bare probes.

Usage:
    python scripts/experiment_clap_wt_threshold.py \
        --manifest outputs/smoke_test_v10/manifest.jsonl \
        --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
        --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
        --max-samples 32
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
    args = ap.parse_args()

    # Load index
    idx = np.load(args.index_npy)
    embeddings = idx["embeddings"].astype(np.float32)  # (N_rows, dim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12

    with open(args.index_meta) as f:
        meta = json.load(f)
    rows = meta["rows"]

    # Build name → row indices mapping (one WT can have multiple frames)
    name_to_row_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        name_to_row_indices[row["wavetable_name"]].append(i)

    # Unique wavetable names
    all_wt_names = sorted(set(r["wavetable_name"] for r in rows))
    print(f"Index: {len(rows)} rows, {len(all_wt_names)} unique wavetable names")

    # Compute per-name embedding (max-pool across frames)
    name_to_emb: dict[str, np.ndarray] = {}
    for name, idxs in name_to_row_indices.items():
        embs = embeddings[idxs]
        # Max-pool: take the embedding with highest L2 norm (most "confident")
        # Actually, mean-pool is more standard for retrieval
        name_to_emb[name] = embs.mean(axis=0)

    # Normalize per-name embeddings
    name_emb_matrix = np.stack([name_to_emb[n] for n in all_wt_names])  # (N_names, dim)
    name_norms = np.linalg.norm(name_emb_matrix, axis=1, keepdims=True) + 1e-12
    name_emb_normed = name_emb_matrix / name_norms

    # Load manifest
    entries = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f"Manifest: {len(entries)} samples")

    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Per-sample results
    results_by_archetype: dict[str, list[dict]] = defaultdict(list)
    all_results: list[dict] = []

    for entry in entries:
        sample_id = entry["sample_id"]
        archetype = entry.get("archetype", "unknown")

        # Get GT wavetable names
        path_file = entry.get("path_file")
        target_preset_path = entry.get("target_preset_path")
        if not target_preset_path and path_file:
            with open(path_file) as f:
                pd = json.load(f)
            target_preset_path = pd.get("target_preset_path")
        if not target_preset_path:
            continue

        gt_names = _extract_gt_wavetable_names(Path(target_preset_path))
        if not gt_names:
            continue

        # Find GT embeddings
        gt_embs = []
        gt_found = []
        for gn in gt_names:
            if gn in name_to_emb:
                emb = name_to_emb[gn]
                emb_normed = emb / (np.linalg.norm(emb) + 1e-12)
                gt_embs.append(emb_normed)
                gt_found.append(gn)

        if not gt_embs:
            print(f"  WARNING: {sample_id}: GT wavetables {gt_names} not found in index")
            continue

        gt_emb_stack = np.stack(gt_embs)  # (n_gt, dim)

        # Cosine similarity: each GT embedding against all names
        # sim_matrix: (n_gt, n_names)
        sim_matrix = gt_emb_stack @ name_emb_normed.T

        # For each candidate name, take max similarity across all GT wavetables
        max_sim_per_name = sim_matrix.max(axis=0)  # (n_names,)

        # Pool sizes at each threshold
        pool_sizes = {}
        for t in thresholds:
            # Count names exceeding threshold (excluding GT names themselves)
            gt_name_set = set(gt_found)
            n_above = sum(
                1 for i, name in enumerate(all_wt_names)
                if max_sim_per_name[i] >= t and name not in gt_name_set
            )
            pool_sizes[t] = n_above + len(gt_found)  # + GT always included

        # Top-10 most similar (excluding GT)
        gt_name_set = set(gt_found)
        ranked = sorted(
            [(all_wt_names[i], float(max_sim_per_name[i]))
             for i in range(len(all_wt_names)) if all_wt_names[i] not in gt_name_set],
            key=lambda x: -x[1],
        )

        result = {
            "sample_id": sample_id,
            "archetype": archetype,
            "gt_names": gt_found,
            "n_gt": len(gt_found),
            "pool_sizes": pool_sizes,
            "top_10": ranked[:10],
        }
        all_results.append(result)
        results_by_archetype[archetype].append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"CLAP GT-to-Index Cosine Threshold Experiment (n={len(all_results)})")
    print(f"{'='*80}\n")

    print(f"{'Threshold':>10} | {'Mean pool':>10} | {'Min':>6} | {'Max':>6} | {'Median':>8}")
    print("-" * 55)
    for t in thresholds:
        sizes = [r["pool_sizes"][t] for r in all_results]
        if sizes:
            print(f"{t:>10.2f} | {np.mean(sizes):>10.1f} | {min(sizes):>6} | {max(sizes):>6} | {np.median(sizes):>8.1f}")

    print(f"\n--- By Archetype (pool size at threshold 0.70) ---")
    for arch in sorted(results_by_archetype):
        sizes = [r["pool_sizes"][0.70] for r in results_by_archetype[arch]]
        print(f"  {arch:12s}: mean={np.mean(sizes):.1f}  min={min(sizes)}  max={max(sizes)}  n={len(sizes)}")

    print(f"\n--- Top-10 most similar (non-GT) for first 3 samples ---")
    for r in all_results[:3]:
        print(f"\n  {r['sample_id']} ({r['archetype']}) — GT: {r['gt_names']}")
        for name, sim in r["top_10"][:5]:
            print(f"    {sim:.4f}  {name}")

    # Write full results
    out_path = args.manifest.parent / "clap_wt_threshold_results.json"
    with open(out_path, "w") as f:
        json.dump({"thresholds": thresholds, "results": all_results}, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
