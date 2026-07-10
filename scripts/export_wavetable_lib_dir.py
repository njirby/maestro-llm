#!/usr/bin/env python
"""Export data/wavetable_lib.json as a directory of .vitaltable files.

The daw-farm containers discover wavetables by scanning a Vital data dir
(the same way the emitted snippets do). Syncing an arbitrary
~/.local/share/vital gives a library that can disagree with the one presets
were generated from — same-name wavetables with different content silently
corrupt rollout audio (e.g. "Sine to Saw"). This script materializes the
exact generation library so `--daw-farm-vital-data` can point at it.

Usage:
    python scripts/export_wavetable_lib_dir.py \
        [--lib data/wavetable_lib.json] [--out data/prepared/wavetable_lib_vital_dir]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared/wavetable_lib_vital_dir"))
    args = ap.parse_args()

    lib = json.load(open(args.lib))
    if args.out.exists():
        shutil.rmtree(args.out)
    wt_dir = args.out / "Wavetables"
    wt_dir.mkdir(parents=True)

    seen: set[str] = set()
    n = 0
    for wt in lib:
        if not isinstance(wt, dict):
            continue
        name = wt.get("name")
        if not name or "groups" not in wt or name in seen:
            continue
        seen.add(name)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or f"wt_{n}"
        with open(wt_dir / f"{slug}.vitaltable", "w") as f:
            json.dump(wt, f, separators=(",", ":"))
        n += 1
    print(f"Wrote {n} unique wavetables to {wt_dir}")


if __name__ == "__main__":
    main()
