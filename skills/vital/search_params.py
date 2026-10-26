"""Search Vital VST parameters in a live REAPER session by keyword.

Usage (from bash tool call):
    python skills/vital/search_params.py "filter cutoff"
    python skills/vital/search_params.py "delay"
    python skills/vital/search_params.py "osc 1 level"

Returns JSON list of matching parameters with index, name, current value,
display string, and min/max. The model uses the idx to set values via
TrackFX_SetParam.
"""
from __future__ import annotations

import json
import sys

import reapy
from reapy import reascript_api as RPR

TRACK_IDX = 0
FX_IDX = 0


def search(keywords: list[str]) -> list[dict]:
    results = []
    with reapy.inside_reaper():
        track = RPR.GetTrack(0, TRACK_IDX)
        n = RPR.TrackFX_GetNumParams(track, FX_IDX)
        for i in range(n):
            _, _, _, _, name, _ = RPR.TrackFX_GetParamName(track, FX_IDX, i, "", 2048)
            name_lower = name.strip().lower()
            if all(kw in name_lower for kw in keywords):
                _, _, val, min_val, max_val, _ = RPR.TrackFX_GetParam(track, FX_IDX, i)
                _, _, _, _, display, _ = RPR.TrackFX_GetFormattedParamValue(track, FX_IDX, i, "", 2048)
                results.append({
                    "idx": i,
                    "name": name.strip(),
                    "value": round(float(val), 6),
                    "display": display.strip(),
                    "min": round(float(min_val), 6),
                    "max": round(float(max_val), 6),
                })
    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip().lower()
    if not query:
        print(json.dumps({"error": "Usage: search_params.py <keywords>"}))
        sys.exit(1)

    keywords = query.split()
    results = search(keywords)
    print(json.dumps({"query": query, "count": len(results), "params": results}, indent=2))
