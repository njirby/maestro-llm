"""Dump all Vital VST parameter names, indices, and current values from a live REAPER session.

Usage:
    1. Open REAPER with Vital loaded on track 1, FX slot 0
    2. Run: python scripts/dump_vital_reaper_params.py --output maestro/synth/vital_reaper_params.json

The output is a JSON list of {idx, name, value, display, min, max} for every
parameter Vital exposes via REAPER's TrackFX API. This is the ground-truth
mapping used at SFT build time to simulate the live search tool.
"""
from __future__ import annotations

import argparse
import json
import sys


def dump_params(track_idx: int = 0, fx_idx: int = 0) -> list[dict]:
    import reapy
    from reapy import reascript_api as RPR

    params = []
    with reapy.inside_reaper():
        track = RPR.GetTrack(0, track_idx)
        n = RPR.TrackFX_GetNumParams(track, fx_idx)
        for i in range(n):
            rv = RPR.TrackFX_GetParamName(track, fx_idx, i, "", 2048)
            name = rv[4] if len(rv) > 4 else rv[3]
            rv2 = RPR.TrackFX_GetParam(track, fx_idx, i, 0.0, 0.0)
            val, min_val, max_val = float(rv2[0]), float(rv2[4]), float(rv2[5])
            rv3 = RPR.TrackFX_GetFormattedParamValue(track, fx_idx, i, "", 2048)
            display = rv3[4] if len(rv3) > 4 else rv3[3]
            params.append({
                "idx": i,
                "name": name.strip(),
                "value": round(float(val), 8),
                "display": display.strip(),
                "min": round(float(min_val), 8),
                "max": round(float(max_val), 8),
            })
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=str, default="maestro/synth/vital_reaper_params.json")
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--fx", type=int, default=0)
    args = ap.parse_args()

    params = dump_params(args.track, args.fx)
    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")

    print(f"Dumped {len(params)} parameters to {args.output}")

    # Summary stats
    names = [p["name"] for p in params]
    from collections import Counter
    dupes = {n: c for n, c in Counter(names).items() if c > 1}
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate display names: {dupes}")
    else:
        print("No duplicate display names — all unique.")


if __name__ == "__main__":
    main()
