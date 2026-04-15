#!/usr/bin/env python3
"""List wavetables from the library by index range.

Simple, no-dependency tool. Used by search agents to see which wavetables
are in their assigned slice of the library.

Usage:
    # Get total count
    $ python scripts/list_wavetables.py --total
    {"total": 568}

    # Get a slice
    $ python scripts/list_wavetables.py --start 0 --end 48
    {"wavetables": [{"idx": 0, "name": "..."}, ...], "start": 0, "end": 48, "count": 48}

    # Custom library path
    $ python scripts/list_wavetables.py --start 0 --end 10 --lib /path/to/wavetable_lib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="List wavetables from the library by index range.")
    ap.add_argument("--lib", type=Path, default=Path("data/wavetable_lib.json"),
                    help="Path to wavetable_lib.json (default: data/wavetable_lib.json)")
    ap.add_argument("--total", action="store_true",
                    help="Print only the total count of wavetables.")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    ap.add_argument("--end", type=int, default=None,
                    help="End index (exclusive). Default: end of library.")
    args = ap.parse_args()

    if not args.lib.exists():
        print(json.dumps({"status": "error", "error": f"library not found: {args.lib}"}))
        sys.exit(1)

    with open(args.lib) as f:
        lib = json.load(f)
    if not isinstance(lib, list):
        print(json.dumps({"status": "error", "error": "invalid library format"}))
        sys.exit(1)

    # Filter out entries without a name and deduplicate by name.
    # The library has ~558 entries but only ~282 unique names (e.g. "Init" appears
    # 121 times across preset banks). Dedup keeps first occurrence, so indexing
    # matches the unique-name space and matches wt_index_meta.json exactly.
    seen: set[str] = set()
    names: list[str] = []
    for wt in lib:
        if not isinstance(wt, dict) or "name" not in wt:
            continue
        name = wt["name"]
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    if args.total:
        print(json.dumps({"total": len(names)}))
        return

    end = args.end if args.end is not None else len(names)
    start = max(0, args.start)
    end = min(len(names), end)

    slice_rows = [{"idx": i, "name": name} for i, name in enumerate(names[start:end], start=start)]
    print(json.dumps({
        "wavetables": slice_rows,
        "start": start,
        "end": end,
        "count": len(slice_rows),
        "total": len(names),
    }))


if __name__ == "__main__":
    main()
