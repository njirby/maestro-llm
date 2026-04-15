#!/usr/bin/env python3
"""Render wavetable probe WAVs through the Vital default preset.

Used by search agents to actually hear candidate wavetables. Takes a list of
indices or names, loads each wavetable from the library, swaps it into the
init preset, renders via vita, and writes a WAV.

Usage:
    # By index range
    $ python scripts/render_wavetable_probes.py --start 0 --end 8 --out-dir /tmp/probes
    {"status": "ok", "rendered": [{"idx": 0, "name": "...", "out": "/tmp/probes/wt_0.wav"}, ...]}

    # By explicit indices
    $ python scripts/render_wavetable_probes.py --idxs 0,5,12 --out-dir /tmp/probes

    # By names
    $ python scripts/render_wavetable_probes.py --names "01 Basic Shapes,Pink Noise" --out-dir /tmp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
from scripts.build_wavetable_retrieval_baseline import (
    _build_probe_preset,
    _load_init_preset,
    _load_wavetable_lib,
)


def _slugify(s: str, max_len: int = 80) -> str:
    import re
    out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return (out or "unnamed")[:max_len]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write probe WAVs.")
    ap.add_argument("--idxs", type=str, default="",
                    help="Comma-separated wavetable indices to render (uses dense indexing from list_wavetables.py).")
    ap.add_argument("--start", type=int, default=None, help="Start index (inclusive, for range mode).")
    ap.add_argument("--end", type=int, default=None, help="End index (exclusive, for range mode).")
    ap.add_argument("--names", type=str, default="",
                    help="Comma-separated wavetable names (alternative to --idxs).")
    ap.add_argument("--archetype", default="lead",
                    help="Probe archetype for note selection (default: lead).")
    ap.add_argument("--clip-duration-s", type=float, default=10.0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-s", type=float, default=0.5)
    args = ap.parse_args()

    lib = _load_wavetable_lib(args.lib)
    # Filter to named entries, keep dense index aligned with list_wavetables.py
    named = [wt for wt in lib if isinstance(wt, dict) and "name" in wt]
    name_to_idx = {wt["name"]: i for i, wt in enumerate(named)}

    # Resolve indices to render
    if args.idxs:
        idxs = [int(x) for x in args.idxs.split(",") if x.strip()]
    elif args.start is not None and args.end is not None:
        idxs = list(range(max(0, args.start), min(len(named), args.end)))
    elif args.names:
        wanted = [n.strip() for n in args.names.split(",") if n.strip()]
        idxs = [name_to_idx[n] for n in wanted if n in name_to_idx]
        missing = [n for n in wanted if n not in name_to_idx]
        if missing:
            print(json.dumps({"status": "error", "error": f"names not in library: {missing}"}))
            sys.exit(1)
    else:
        print(json.dumps({"status": "error", "error": "provide --idxs, --start/--end, or --names"}))
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    init_preset = _load_init_preset()
    notes = make_probe_notes(args.archetype, clip_duration_s=args.clip_duration_s)
    synth = _load_vital()

    rendered = []
    for idx in idxs:
        if idx < 0 or idx >= len(named):
            rendered.append({"idx": idx, "error": "idx out of range"})
            continue
        wt = named[idx]
        name = wt["name"]
        preset = _build_probe_preset(init_preset, wt, 0)
        synth.load_json(json.dumps(preset))
        audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=args.tail_s)
        audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=args.trim_min_s)
        out_path = args.out_dir / f"wt_{idx:04d}_{_slugify(name)}.wav"
        sf.write(str(out_path), audio.T, SAMPLE_RATE)
        rendered.append({
            "idx": idx,
            "name": name,
            "out": str(out_path),
            "duration_s": round(float(audio.shape[-1] / SAMPLE_RATE), 3),
        })

    print(json.dumps({"status": "ok", "rendered": rendered}))


if __name__ == "__main__":
    main()
