#!/usr/bin/env python3
"""
Download source MusicCaps audio files from YouTube.

This does not build the canonical training dataset. After downloading, run:
    python scripts/prepare_musiccaps.py

Examples:
    # Full dataset (5,521 clips, may take hours):
    python scripts/download_data.py

    # Quick test - grab 10 clips to verify the pipeline:
    python scripts/download_data.py --limit 10

    # Custom output dir and more workers:
    python scripts/download_data.py --out_dir /data/musiccaps --num_workers 8
"""

import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maestro.data.download import download_musiccaps


def main():
    parser = argparse.ArgumentParser(description="Download source MusicCaps audio files")
    parser.add_argument("--out_dir", default="./data/audio", help="Output directory for .mp3 files")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit clips (for testing)")
    args = parser.parse_args()

    download_musiccaps(out_dir=args.out_dir, num_workers=args.num_workers, limit=args.limit)


if __name__ == "__main__":
    main()
