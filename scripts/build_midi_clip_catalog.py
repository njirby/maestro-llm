#!/usr/bin/env python3
"""Build a catalog of usable MIDI clips from the Lakh MIDI Dataset.

For each MIDI file, walks every non-drum track and slices it into
bar-aligned, fully-contained-notes clips (4-15s by default). Emits a
JSONL catalog with per-clip metadata that downstream picks from at
sample-render time.

One catalog entry per clip:
    {
      "midi_path": "...",
      "track_idx": int,              # which track within the file
      "program": int,                # GM program (0-127)
      "track_name": str,
      "start_s": float,              # clip offset in source MIDI
      "duration_s": float,           # clip length (after trim to last note)
      "n_notes": int,
      "pitch_min": int,              # MIDI pitch
      "pitch_max": int,
      "is_monophonic": bool,         # no overlapping notes in this clip
      "avg_note_length_s": float,
      "note_density": float,         # notes per second
      "bpm": float,
    }

Usage:
    python scripts/build_midi_clip_catalog.py \\
        --source data/raw/phase1_sources/lakh_midi/lmd_full \\
        --out outputs/midi_clips/lakh_catalog.jsonl \\
        --max-files 20000 --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# Clip sizing: target 4-15s clips (matches the training pipeline's ~10s target).
MIN_CLIP_S = 4.0
MAX_CLIP_S = 15.0
MIN_NOTES = 4


def _clip_bars(rel_key: str, clip_idx: int, bpm: float) -> int:
    """Deterministic bar count whose duration stays within [MIN_CLIP_S, MAX_CLIP_S]."""
    sec_per_bar = 240.0 / max(1e-6, bpm)
    min_bars = max(1, int(math.ceil(MIN_CLIP_S / sec_per_bar)))
    max_bars = max(min_bars, int(math.floor(MAX_CLIP_S / sec_per_bar)))
    span = max_bars - min_bars + 1
    seed = hashlib.md5(f"{rel_key}_{clip_idx}_bars".encode()).digest()
    t = int.from_bytes(seed[:4], "big") / 0xFFFFFFFF
    return min_bars + min(span - 1, int(t * span))


def _is_monophonic(notes) -> bool:
    """True iff no two notes in the list overlap in time."""
    if len(notes) < 2:
        return True
    sorted_notes = sorted(notes, key=lambda n: n.start)
    for a, b in zip(sorted_notes, sorted_notes[1:]):
        if b.start < a.end - 1e-6:
            return False
    return True


def _process_file(args: tuple) -> list[dict]:
    """Return catalog entries for one MIDI file (or []  on failure)."""
    midi_path, source_root = args
    try:
        import pretty_midi as _pm
    except Exception:
        return []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = _pm.PrettyMIDI(midi_path)
    except Exception:
        return []

    tempos = pm.get_tempo_changes()
    bpm = float(tempos[1][0]) if len(tempos[1]) > 0 else 120.0
    if not (30.0 <= bpm <= 300.0):
        return []

    rel_key = str(Path(midi_path).relative_to(source_root))
    out: list[dict] = []

    for track_idx, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES:
            continue
        notes = sorted(inst.notes, key=lambda n: n.start)
        bar_len_s = 240.0 / max(1e-6, bpm)
        cursor_bar = int(math.floor(notes[0].start / bar_len_s))
        cursor = cursor_bar * bar_len_s
        track_end = notes[-1].end
        clip_idx = 0

        while cursor < track_end:
            n_bars = _clip_bars(f"{rel_key}_t{track_idx}", clip_idx, bpm)
            dur = n_bars * bar_len_s
            clip_end = cursor + dur
            clip_notes = [n for n in notes if n.start >= cursor and n.end <= clip_end]

            if len(clip_notes) >= MIN_NOTES:
                pitches = [n.pitch for n in clip_notes]
                durs = [n.end - n.start for n in clip_notes]
                # Trim end to last note to avoid trailing silence
                last_note_end_offset = max(n.end for n in clip_notes) - cursor
                effective_dur = min(dur, last_note_end_offset + 0.25)

                out.append({
                    "midi_path": str(midi_path),
                    "track_idx": track_idx,
                    "program": int(inst.program),
                    "track_name": (inst.name or "").strip(),
                    "start_s": float(cursor),
                    "duration_s": round(float(effective_dur), 3),
                    "n_notes": len(clip_notes),
                    "pitch_min": int(min(pitches)),
                    "pitch_max": int(max(pitches)),
                    "is_monophonic": _is_monophonic(clip_notes),
                    "avg_note_length_s": round(sum(durs) / len(durs), 3),
                    "note_density": round(len(clip_notes) / effective_dur, 3) if effective_dur > 0 else 0.0,
                    "bpm": round(bpm, 2),
                })

            cursor = clip_end
            cursor_bar += n_bars
            clip_idx += 1

    return out


def _worker_init():
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Root directory containing .mid files (e.g. lakh_midi/lmd_full).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path (one catalog entry per line).")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Cap number of source MIDI files processed (0 = all).")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    source_root = args.source.resolve()
    if not source_root.exists():
        print(f"ERROR: source not found: {source_root}", file=sys.stderr)
        sys.exit(1)

    midi_files = sorted(source_root.rglob("*.mid"))
    if args.max_files > 0:
        midi_files = midi_files[: args.max_files]
    print(f"Scanning {len(midi_files)} MIDI files from {source_root} ({args.workers} workers)",
          file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    files_with_clips = 0
    with open(args.out, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futs = [
                pool.submit(_process_file, (str(p), str(source_root)))
                for p in midi_files
            ]
            for i, fut in enumerate(as_completed(futs)):
                try:
                    entries = fut.result()
                except Exception:
                    continue
                if entries:
                    files_with_clips += 1
                    total_clips += len(entries)
                    for e in entries:
                        f_out.write(json.dumps(e, ensure_ascii=False) + "\n")
                if (i + 1) % 500 == 0:
                    print(
                        f"  [{i + 1}/{len(midi_files)}] "
                        f"{total_clips} clips from {files_with_clips} files",
                        file=sys.stderr,
                    )

    print(f"Wrote {total_clips} clips to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
