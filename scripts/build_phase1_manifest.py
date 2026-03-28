#!/usr/bin/env python3
"""
Build a unified Phase 1 SFT manifest from local stem/MIDI datasets.

Usage:
  python scripts/build_phase1_manifest.py \
    --config configs/phase1_data_sources.yaml \
    --out-dir data/prepared/phase1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib as pl
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maestro.data.phase1 import build_manifest_from_sources, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 SFT data manifest from configured sources.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_data_sources.yaml",
        help="YAML file describing source datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/prepared/phase1",
        help="Output directory for manifest and summary.",
    )
    parser.add_argument(
        "--strict-missing-roots",
        action="store_true",
        help="Fail if any source root path is missing.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = pl.Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "sources" not in data or not isinstance(data["sources"], list):
        raise ValueError("Config must define `sources` as a list.")
    return data


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    records = build_manifest_from_sources(
        source_specs=cfg["sources"],
        strict_missing_roots=args.strict_missing_roots,
    )

    out_dir = pl.Path(args.out_dir)
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"

    summary = write_manifest(
        records=records,
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
