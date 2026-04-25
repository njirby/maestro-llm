#!/usr/bin/env python3
"""Build json_key_to_reaper.json — static mapping from Vital preset JSON keys
to REAPER VST3 parameter indices.

Chain: vital_display_names.json → vital_reaper_params.json, with manual
overrides for the 14 collision pairs where display_names maps two different
JSON keys to the same REAPER display name.

Usage:
    python scripts/build_json_key_to_reaper_map.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "maestro" / "synth"

# 14 collision pairs: json_key → correct REAPER display name.
# vital_display_names.json maps both keys in each pair to the same name;
# these overrides give the secondary key its own unique REAPER name.
_COLLISION_OVERRIDES: dict[str, str] = {
    "delay_aux_frequency": "Delay Frequency 2",
    "delay_aux_sync": "Delay Sync 2",
    "delay_aux_tempo": "Delay Tempo 2",
    "osc_1_unison_blend": "Oscillator 1 Blend",
    "osc_2_unison_blend": "Oscillator 2 Blend",
    "osc_3_unison_blend": "Oscillator 3 Blend",
    # Aliases — these json_keys control the same underlying REAPER param
    # as their collision partner.  Map to the same REAPER name so idx is correct.
    "filter_1_blend_transpose": "Filter 1 Formant Transpose",
    "filter_2_blend_transpose": "Filter 2 Formant Transpose",
    "filter_fx_blend_transpose": "Filter fx Formant Transpose",
    "filter_fx_on": "Filter fx Model",
    "osc_1_unison_stack_type": "Oscillator 1 Stack Style",
    "osc_2_unison_stack_type": "Oscillator 2 Stack Style",
    "osc_3_unison_stack_type": "Oscillator 3 Stack Style",
    "portamento_on": "Portamento Force",
}


def build_mapping() -> dict[str, dict]:
    display_names: dict[str, str] = json.loads(
        (SYNTH_DIR / "vital_display_names.json").read_text()
    )
    reaper_params: list[dict] = json.loads(
        (SYNTH_DIR / "vital_reaper_params.json").read_text()
    )
    reaper_name_to_idx = {p["name"]: p["idx"] for p in reaper_params}

    mapping: dict[str, dict] = {}
    missing: list[str] = []

    for json_key, display_name in sorted(display_names.items()):
        reaper_name = _COLLISION_OVERRIDES.get(json_key, display_name)
        idx = reaper_name_to_idx.get(reaper_name)
        if idx is None:
            missing.append(f"{json_key} → {reaper_name!r}")
            continue
        mapping[json_key] = {"idx": idx, "name": reaper_name}

    if missing:
        print(f"WARNING: {len(missing)} keys failed to resolve:")
        for m in missing:
            print(f"  {m}")

    return mapping


def main() -> None:
    mapping = build_mapping()
    out_path = SYNTH_DIR / "json_key_to_reaper.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(mapping)} entries to {out_path}")

    # Verify no duplicate indices (except for alias pairs which share an idx)
    idx_to_keys: dict[int, list[str]] = {}
    for key, val in mapping.items():
        idx_to_keys.setdefault(val["idx"], []).append(key)
    shared = {idx: keys for idx, keys in idx_to_keys.items() if len(keys) > 1}
    if shared:
        print(f"\n{len(shared)} shared indices (alias pairs):")
        for idx, keys in sorted(shared.items()):
            print(f"  idx {idx}: {keys}")


if __name__ == "__main__":
    main()
