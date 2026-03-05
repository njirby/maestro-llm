#!/usr/bin/env python3
"""
Prepare canonical 10-second MusicCaps clips for training.

This script trims local source audio files into fixed 10s clips and writes:
  - manifest.jsonl
  - prepare_report.json

Examples:
  python scripts/prepare_musiccaps.py
  python scripts/prepare_musiccaps.py --limit 16 --num_workers 8
  python scripts/prepare_musiccaps.py --validate_only
"""

import argparse
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maestro.data.prepare_musiccaps import (
    DEFAULT_PREPARED_DIR,
    prepare_musiccaps,
    validate_prepared_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed 10s MusicCaps clips")
    parser.add_argument("--audio_dir", default="./data/audio", help="Directory containing source {ytid}.mp3 files")
    parser.add_argument("--out_dir", default=DEFAULT_PREPARED_DIR, help="Prepared dataset output directory")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel workers (default: CPU cores - 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (for smoke tests)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild clips even if existing output is valid")
    parser.add_argument(
        "--failure_ratio_threshold",
        type=float,
        default=0.01,
        help="Fail if failed_rows / total_rows exceeds this threshold",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate existing manifest + clips without re-trimming",
    )
    args = parser.parse_args()

    if args.validate_only:
        summary = validate_prepared_manifest(
            out_dir=args.out_dir,
            failure_ratio_threshold=args.failure_ratio_threshold,
        )
    else:
        summary = prepare_musiccaps(
            audio_dir=args.audio_dir,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
            limit=args.limit,
            overwrite=args.overwrite,
            failure_ratio_threshold=args.failure_ratio_threshold,
        )

    print(
        "Done: "
        f"total={summary.total_rows}, ok={summary.ok_rows}, failed={summary.failed_rows}, "
        f"failed_ratio={summary.failed_ratio:.2%}"
    )


if __name__ == "__main__":
    main()
