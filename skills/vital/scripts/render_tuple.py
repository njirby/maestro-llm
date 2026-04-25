#!/usr/bin/env python3
"""Render a wavetable tuple (1-3 wavetables assigned to oscillators) through the Vital default preset.

Used by the main agent to audition candidate wavetable combinations before
applying one. Each oscillator gets a different wavetable from the library;
inactive oscillators are muted.

Usage:
    # Single wavetable on osc 1
    $ python skills/vital/scripts/render_tuple.py \
        --osc1 "01 Basic Shapes" --out /tmp/tuple.wav

    # 3-oscillator tuple
    $ python skills/vital/scripts/render_tuple.py \
        --osc1 "Bell" --osc2 "1A - detuned bend bass" --osc3 "Corpusbode Phaser" \
        --out /tmp/tuple.wav

    # Custom library path
    $ python skills/vital/scripts/render_tuple.py \
        --osc1 "Sine to Saw" --out /tmp/tuple.wav --lib data/wavetable_lib.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf

from maestro.render.vital import (
    SAMPLE_RATE,
    _load_vital,
    _render_note_list,
    make_probe_notes,
    trim_silence,
)
from scripts.build_wavetable_retrieval_baseline import _load_init_preset, _load_wavetable_lib


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a wavetable tuple through Vital default preset.")
    ap.add_argument("--osc1", type=str, default=None, help="Wavetable name for oscillator 1.")
    ap.add_argument("--osc2", type=str, default=None, help="Wavetable name for oscillator 2.")
    ap.add_argument("--osc3", type=str, default=None, help="Wavetable name for oscillator 3.")
    ap.add_argument("--out", type=Path, required=True, help="Output WAV path.")
    ap.add_argument("--lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--archetype", default="lead")
    ap.add_argument("--clip-duration-s", type=float, default=10.0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    args = ap.parse_args()

    lib = _load_wavetable_lib(args.lib)
    # Dedup by name (matches list_wavetables.py)
    seen: set[str] = set()
    name_to_wt: dict[str, dict] = {}
    for wt in lib:
        if not isinstance(wt, dict) or "name" not in wt:
            continue
        if wt["name"] in seen:
            continue
        seen.add(wt["name"])
        name_to_wt[wt["name"]] = wt

    osc_assignments = [args.osc1, args.osc2, args.osc3]
    assigned = [(i, name) for i, name in enumerate(osc_assignments) if name]
    if not assigned:
        print(json.dumps({"status": "error", "error": "provide at least --osc1"}))
        sys.exit(1)

    # Validate names
    missing = [name for _, name in assigned if name not in name_to_wt]
    if missing:
        print(json.dumps({"status": "error", "error": f"wavetables not in library: {missing}"}))
        sys.exit(1)

    # Build preset
    preset = _load_init_preset()
    for osc_idx, name in assigned:
        wt = name_to_wt[name]
        if osc_idx < len(preset.get("settings", {}).get("wavetables", [])):
            preset["settings"]["wavetables"][osc_idx] = copy.deepcopy(wt)
        # Enable oscillator and set level
        preset["settings"][f"osc_{osc_idx + 1}_on"] = 1.0
        preset["settings"][f"osc_{osc_idx + 1}_level"] = 0.7

    # Mute unassigned oscillators
    for i in range(3):
        if osc_assignments[i] is None:
            preset["settings"][f"osc_{i + 1}_on"] = 0.0
            preset["settings"][f"osc_{i + 1}_level"] = 0.0

    # Render
    synth = _load_vital()
    notes = make_probe_notes(args.archetype, clip_duration_s=args.clip_duration_s)
    synth.load_json(json.dumps(preset))
    audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=args.tail_s)
    audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), audio.T, SAMPLE_RATE)

    result = {
        "status": "ok",
        "out": str(args.out),
        "wavetables": [name for _, name in assigned],
        "duration_s": round(float(audio.shape[-1] / SAMPLE_RATE), 3),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
