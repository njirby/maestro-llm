#!/usr/bin/env python3
"""
Diagnostic: compare target preset vs final cumulative preset for a generated path.

Usage:
    python scripts/verify_preset_path.py --archetype lead --seed 42
    python scripts/verify_preset_path.py --archetype bass --seed 7 --verbose
    python scripts/verify_preset_path.py --archetype pad  --seed 123 --top 20
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from maestro.synth.path_gen import generate_preset_path, compare_preset_path
from maestro.synth.wavetable_lib import load_wavetable_lib

_WT_LIB = Path(__file__).parent.parent / "data" / "wavetable_lib.json"
_RULE = "═" * 60


def _fmt_row(name, target, final_or_init, norm_diff=None):
    base = f"  {name:<42}  target={target:>10.4f}  other={final_or_init:>10.4f}"
    if norm_diff is not None:
        base += f"  Δnorm={norm_diff:.3f}"
    return base


def print_report(diff: dict, verbose: bool = False, top: int = 10):
    print(f"\nPreset diff: target vs final step")
    print(_RULE)

    # --- NOISY ---
    noisy = diff["noisy"]
    show = noisy if verbose else noisy[:top]
    print(f"\nNOISY — tracked & applied but final ≠ target  ({len(noisy)} params)")
    if show:
        for e in show:
            print(_fmt_row(e["name"], e["target_native"], e["final_native"], e["norm_diff"]))
        if not verbose and len(noisy) > top:
            print(f"  ... and {len(noisy) - top} more")
    else:
        print("  (none)")

    # --- BELOW THRESHOLD ---
    bt = diff["below_thresh"]
    show = bt if verbose else bt[:top]
    print(f"\nBELOW THRESHOLD — |norm_diff| ≤ 0.05, skipped by design  ({len(bt)} params)")
    if show:
        for e in show:
            print(_fmt_row(e["name"], e["target_native"], e["init_native"], e["norm_diff"]))
        if not verbose and len(bt) > top:
            print(f"  ... and {len(bt) - top} more")
    else:
        print("  (none)")

    # --- UNTRACKED ---
    ut = diff["untracked"]
    show = ut if verbose else ut[:top]
    print(f"\nUNTRACKED — not in param_ranges.json  ({len(ut)} params)")
    if show:
        for e in show:
            print(f"  {e['name']:<42}  target={e['target_native']:>10.4f}  [no range data]")
        if not verbose and len(ut) > top:
            print(f"  ... and {len(ut) - top} more")
    else:
        print("  (none)")

    # --- MISSING ---
    ms = diff["missing"]
    if ms:
        print(f"\nMISSING — in target but absent from final  ({len(ms)} params)")
        for e in (ms if verbose else ms[:top]):
            print(f"  {e['name']:<42}  target={e['target_native']:>10.4f}")
        if not verbose and len(ms) > top:
            print(f"  ... and {len(ms) - top} more")

    print(f"\n{_RULE}")
    print(f"Summary: {diff['summary']}")
    print(_RULE)


def main():
    parser = argparse.ArgumentParser(description="Verify preset path fidelity")
    parser.add_argument("--archetype", default="lead",
                        choices=["bass", "lead", "pad", "keys", "pluck", "sequence"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Show all params in each category (default: top --top)")
    parser.add_argument("--top", type=int, default=10,
                        help="Max params to show per category (default: 10)")
    parser.add_argument("--wavetable-lib", default=str(_WT_LIB))
    args = parser.parse_args()

    wt_lib = None
    if Path(args.wavetable_lib).exists():
        wt_lib = load_wavetable_lib(args.wavetable_lib)

    print(f"Generating {args.archetype} path (seed={args.seed}) …")
    rng = random.Random(args.seed)
    path_result = generate_preset_path(args.archetype, rng, wavetable_lib=wt_lib)

    print(f"  n_iterations    : {path_result['n_iterations']}")
    print(f"  n_changed_params: {path_result['n_changed_params']}")
    print(f"  has_mistake_step: {path_result['has_mistake_step']}")

    diff = compare_preset_path(path_result)
    print_report(diff, verbose=args.verbose, top=args.top)


if __name__ == "__main__":
    main()
