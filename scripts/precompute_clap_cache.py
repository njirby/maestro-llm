#!/usr/bin/env python3
"""Pre-compute CLAP embeddings for all GT audio files on GPU.

Run this BEFORE starting the Omni server so GPUs are free for CLAP.
The cache file is then loaded by run_sft_production.py, which runs CLAP
on CPU only for new audio generated during rollouts.

Usage:
    python scripts/precompute_clap_cache.py \
        --manifest outputs/sft_16k/manifest.jsonl \
        --probe-dir outputs/agent_sft/candidate_probes \
        --output outputs/sft_16k/clap_cache.npz \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import ClapEmbedder, load_manifest_entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path,
                   default=Path("outputs/agent_sft/candidate_probes"))
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write .npz cache (e.g. outputs/sft_16k/clap_cache.npz)")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device for CLAP (use GPU here)")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    entries = load_manifest_entries(args.manifest,
                                   max_samples=args.max_samples or 999_999_999)
    print(f"Loaded {len(entries)} manifest entries.", flush=True)

    audio_paths: list[Path] = []
    for e in entries:
        for k in ("gt_wav", "gt_probe_wav", "default_wav"):
            if e.get(k):
                audio_paths.append(Path(e[k]))
    if args.probe_dir.exists():
        for pp in sorted(args.probe_dir.glob("*.wav")):
            audio_paths.append(pp)
    audio_paths = list(dict.fromkeys(audio_paths))
    print(f"Collected {len(audio_paths)} unique audio paths.", flush=True)

    print(f"Loading CLAP on {args.device}...", flush=True)
    embedder = ClapEmbedder.create(args.device)

    t0 = time.monotonic()
    done = 0
    skipped = 0
    for i, p in enumerate(audio_paths):
        if not p.exists():
            skipped += 1
            continue
        try:
            embedder.embed_audio_path(p)
            done += 1
        except Exception as exc:
            print(f"  WARNING: {p.name}: {exc}", flush=True)
            skipped += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(audio_paths):
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(audio_paths)}] {done} embedded, "
                  f"{skipped} skipped, {rate:.1f}/s", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {done} embeddings in {elapsed:.1f}s ({done/elapsed:.1f}/s)", flush=True)
    print(f"Saving cache to {args.output}...", flush=True)
    embedder.save_cache(args.output)
    print(f"Saved {len(embedder._cache)} embeddings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
