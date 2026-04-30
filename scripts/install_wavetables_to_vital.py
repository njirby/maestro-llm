#!/usr/bin/env python3
"""Install wavetables from the static library into Vital's data directory.

Extracts unique wavetables from data/wavetable_lib.json and writes them
as .vitaltable files into ~/.local/share/vital/User/Wavetables/ so that
Vital's browser and the agent's inline FS scan can both find them.

Skips wavetables that are already discoverable from Vital's data dirs.

Usage:
    python scripts/install_wavetables_to_vital.py
    python scripts/install_wavetables_to_vital.py --dry-run
    python scripts/install_wavetables_to_vital.py --lib data/wavetable_lib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

VITAL_DATA_DIRS = [
    Path.home() / ".local" / "share" / "vital",
    Path.home() / "Library" / "Application Support" / "Vital",
]
INSTALL_DIR = Path.home() / ".local" / "share" / "vital" / "User" / "Wavetables"


def discover_existing_names() -> set[str]:
    seen: set[str] = set()
    for vd in VITAL_DATA_DIRS:
        if not vd.exists():
            continue
        for vt in glob.glob(str(vd / "**" / "*.vitaltable"), recursive=True):
            try:
                w = json.loads(Path(vt).read_text())
                n = w.get("name", "")
                if n:
                    seen.add(n)
            except Exception:
                pass
        for vp in glob.glob(str(vd / "**" / "*.vital"), recursive=True):
            try:
                for w in json.loads(Path(vp).read_text()).get("settings", {}).get("wavetables", []):
                    n = w.get("name", "") if isinstance(w, dict) else ""
                    if n:
                        seen.add(n)
            except Exception:
                pass
    return seen


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")[:80] or "unnamed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lib = json.loads(args.lib.read_text())
    seen_names: set[str] = set()
    unique: list[dict] = []
    for wt in lib:
        n = wt.get("name", "")
        if n and n not in seen_names and "groups" in wt:
            seen_names.add(n)
            unique.append(wt)

    existing = discover_existing_names()
    to_install = [wt for wt in unique if wt["name"] not in existing]

    print(f"Unique in lib: {len(unique)}")
    print(f"Already in Vital: {len(existing)}")
    print(f"To install: {len(to_install)}")

    if args.dry_run:
        for wt in to_install[:10]:
            print(f"  would write: {slug(wt['name'])}.vitaltable")
        if len(to_install) > 10:
            print(f"  ... and {len(to_install) - 10} more")
        return

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for wt in to_install:
        fname = f"{slug(wt['name'])}.vitaltable"
        dest = INSTALL_DIR / fname
        dest.write_text(json.dumps(wt, separators=(",", ":")))
        written += 1

    print(f"Installed {written} wavetables to {INSTALL_DIR}")

    # Verify
    after = discover_existing_names()
    print(f"Total discoverable wavetables: {len(after)}")


if __name__ == "__main__":
    main()
