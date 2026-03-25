"""
Generate (midi, lua, wav) training tuples from a MIDI dataset using clip extraction.

Each MIDI track is sliced into sequential non-overlapping clips aligned to full bars.
Clip length is deterministic per source/track/clip and constrained to ~15s–2min.
Only complete notes (both start AND end within the clip window) are included —
no partial notes cut off at boundaries. WAV rendering uses Vita (no REAPER).
The Lua script is the training target: notes re-offset to t=0.

Long tracks yield multiple clips; short tracks yield one or zero.

Usage:
    python scripts/generate_reaper_tuples.py \\
        --source lakh_midi \\
        --workers 16 \\
        --out data/processed/reaper_tuples_lakh
"""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from maestro.render.reaper import generate_lua

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "phase1_sources"

SOURCES = {
    "melodic_mind": DATA_ROOT / "midi" / "melodic_mind",
    "lakh_midi":    DATA_ROOT / "lakh_midi" / "lmd_full",
    "slakh2100":    DATA_ROOT / "slakh2100" / "slakh2100_flac_redux",
}

MIN_NOTES  = 4
MIN_CLIP_S = 15.0
MAX_CLIP_S = 120.0
SAMPLE_RATE = 44100
TAIL_S = 2.0

# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------

def _clip_bars(rel: str, track_idx: int, clip_idx: int, bpm: float) -> int:
    """Deterministic bar count whose duration stays within [MIN_CLIP_S, MAX_CLIP_S]."""
    sec_per_bar = 240.0 / max(1e-6, bpm)  # 4/4 bars
    min_bars = max(1, int(math.ceil(MIN_CLIP_S / sec_per_bar)))
    max_bars = max(min_bars, int(math.floor(MAX_CLIP_S / sec_per_bar)))
    span = max_bars - min_bars + 1
    seed = hashlib.md5(f"{rel}_{track_idx}_{clip_idx}_bars".encode()).digest()
    t = int.from_bytes(seed[:4], "big") / 0xFFFFFFFF  # [0, 1)
    return min_bars + min(span - 1, int(t * span))


def _extract_clips(notes, rel: str, track_idx: int, bpm: float):
    """
    Slice a note list into sequential non-overlapping clips.

    Only notes fully contained within a clip window (both start AND end inside)
    are included — no partial notes. Clips with fewer than MIN_NOTES complete
    notes are skipped.

    Returns list of dicts:
        {clip_idx, notes (re-offset to t=0), clip_start_s, clip_dur_s}
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.start)
    bar_len_s = 240.0 / max(1e-6, bpm)  # assume 4/4
    cursor_bar = int(math.floor(notes[0].start / bar_len_s))
    cursor = cursor_bar * bar_len_s
    track_end = notes[-1].end
    clips = []
    clip_idx = 0

    while cursor < track_end:
        clip_bars = _clip_bars(rel, track_idx, clip_idx, bpm)
        dur = clip_bars * bar_len_s
        clip_end = cursor + dur

        # Only fully-contained notes
        clip_notes = [n for n in notes if n.start >= cursor and n.end <= clip_end]

        if len(clip_notes) >= MIN_NOTES:
            import pretty_midi as _pm
            reoffset = [
                _pm.Note(velocity=n.velocity, pitch=n.pitch,
                         start=n.start - cursor, end=n.end - cursor)
                for n in clip_notes
            ]
            clips.append({
                "clip_idx":    clip_idx,
                "notes":       reoffset,
                "clip_start_s": cursor,
                "clip_dur_s":  dur,
                "clip_start_bar": cursor_bar + 1,
                "clip_bars": clip_bars,
            })

        cursor = clip_end
        cursor_bar += clip_bars
        clip_idx += 1

    return clips


def _pad_audio_to_duration(audio: np.ndarray, sample_rate: int, min_duration_s: float) -> np.ndarray:
    """Right-pad stereo audio with silence up to min_duration_s."""
    target_samples = int(max(0.0, float(min_duration_s)) * sample_rate)
    if audio.shape[1] >= target_samples:
        return audio
    pad = target_samples - audio.shape[1]
    return np.pad(audio, ((0, 0), (0, pad)), mode="constant")


# ---------------------------------------------------------------------------
# Worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

_vital_instance = None


def _render_midi_file(args: tuple) -> tuple[list[dict], int]:
    """
    Process one MIDI file: extract clips from all usable tracks, render WAV/MIDI/Lua.

    Returns (entries, error_count).
    """
    global _vital_instance
    (midi_path, source_name, source_dir, midis_dir, luas_dir, wavs_dir, rpps_dir, overwrite) = args

    try:
        import pretty_midi as _pm
        from maestro.render.vital import _load_vital, _render_note_list
    except Exception:
        return [], 1

    rel = str(Path(midi_path).relative_to(source_dir))
    entries = []
    errors = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pm = _pm.PrettyMIDI(midi_path)
        except Exception:
            return [], 0  # corrupt — silent skip

    tempos = pm.get_tempo_changes()
    bpm = float(tempos[1][0]) if len(tempos[1]) > 0 else 120.0
    file_hash = hashlib.md5(rel.encode()).hexdigest()[:8]

    for track_idx, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES:
            continue

        track_name = inst.name.strip() or "Vital Synth"
        clips = _extract_clips(list(inst.notes), rel, track_idx, bpm)

        for clip in clips:
            ci = clip["clip_idx"]
            tid = f"{source_name}_{file_hash}_t{track_idx}_c{ci}"
            render_dur_s = float(clip["clip_dur_s"]) + TAIL_S

            midi_out = str(Path(midis_dir) / f"{tid}.mid")
            lua_out  = str(Path(luas_dir)  / f"{tid}.lua")
            wav_out  = str(Path(wavs_dir)  / f"{tid}.wav")
            rpp_out  = str(Path(rpps_dir)  / f"{tid}.rpp")

            if Path(wav_out).exists() and not overwrite:
                entries.append({
                    "id": tid, "source": source_name, "source_path": rel,
                    "track_name": track_name, "track_idx": track_idx,
                    "clip_idx": ci, "clip_start_s": clip["clip_start_s"],
                    "clip_dur_s": clip["clip_dur_s"], "bpm": bpm,
                    "clip_start_bar": clip["clip_start_bar"], "clip_bars": clip["clip_bars"],
                    "render_dur_s": render_dur_s,
                    "tail_s": TAIL_S, "wav_engine": "vital",
                    "midi": midi_out, "lua": lua_out, "wav": wav_out, "wav_ok": True,
                })
                continue

            try:
                notes = clip["notes"]

                # Write single-track MIDI (re-offset notes)
                new_pm = _pm.PrettyMIDI(initial_tempo=bpm)
                new_inst = _pm.Instrument(program=0, name=track_name)
                new_inst.notes = notes
                new_pm.instruments.append(new_inst)
                new_pm.write(midi_out)

                # Write Lua script (notes already at t=0)
                lua_src = generate_lua(
                    notes=notes, bpm=bpm,
                    wav_path=wav_out, rpp_path=rpp_out, track_name=track_name,
                    tail_s=TAIL_S, min_duration_s=render_dur_s,
                )
                Path(lua_out).write_text(lua_src)

                # Render WAV via Vita
                if _vital_instance is None:
                    _vital_instance = _load_vital()

                audio = _render_note_list(_vital_instance, notes, SAMPLE_RATE, TAIL_S)
                audio = _pad_audio_to_duration(audio, SAMPLE_RATE, render_dur_s)
                sf.write(wav_out, audio.T, SAMPLE_RATE, subtype="PCM_24")

                entries.append({
                    "id": tid, "source": source_name, "source_path": rel,
                    "track_name": track_name, "track_idx": track_idx,
                    "clip_idx": ci, "clip_start_s": clip["clip_start_s"],
                    "clip_dur_s": clip["clip_dur_s"], "bpm": bpm,
                    "clip_start_bar": clip["clip_start_bar"], "clip_bars": clip["clip_bars"],
                    "render_dur_s": render_dur_s,
                    "tail_s": TAIL_S, "wav_engine": "vital",
                    "midi": midi_out, "lua": lua_out, "wav": wav_out, "wav_ok": True,
                })
            except Exception as exc:
                errors += 1
                entries.append({
                    "id": tid, "source": source_name, "source_path": rel,
                    "track_name": track_name, "track_idx": track_idx,
                    "clip_idx": ci, "clip_start_s": clip["clip_start_s"],
                    "clip_dur_s": clip["clip_dur_s"], "bpm": bpm,
                    "clip_start_bar": clip["clip_start_bar"], "clip_bars": clip["clip_bars"],
                    "render_dur_s": render_dur_s,
                    "tail_s": TAIL_S, "wav_engine": "vital",
                    "midi": midi_out, "lua": lua_out, "wav": None, "wav_ok": False,
                })

    return entries, errors


def _worker_init():
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="melodic_mind", choices=list(SOURCES.keys()))
    parser.add_argument("--limit", type=int, default=None, help="Max MIDI files to process")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="data/processed/reaper_tuples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir   = Path(args.out)
    midis_dir = out_dir / "midis"
    luas_dir  = out_dir / "luas"
    wavs_dir  = out_dir / "wavs"
    rpps_dir  = out_dir / "rpps"
    for d in [midis_dir, luas_dir, wavs_dir, rpps_dir]:
        d.mkdir(parents=True, exist_ok=True)

    source_dir = SOURCES[args.source]

    print(f"Scanning {source_dir} ...")
    midi_files = [
        p for p in sorted(source_dir.rglob("*.mid"))
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]
    if args.limit:
        midi_files = midi_files[: args.limit]
    print(f"Found {len(midi_files)} MIDI files — submitting to {args.workers} workers...")

    def _job(p):
        return (str(p), args.source, str(source_dir),
                str(midis_dir), str(luas_dir), str(wavs_dir), str(rpps_dir),
                args.overwrite)

    all_entries = []
    rendered = errors = 0
    start = time.time()
    max_in_flight = args.workers * 8

    def _process(f):
        nonlocal rendered, errors
        entries, err_count = f.result()
        errors += err_count
        for e in entries:
            if e["wav_ok"]:
                rendered += 1
            all_entries.append(e)
        n = rendered + errors
        if n % 500 == 0 or n == 1:
            elapsed = time.time() - start
            rate = rendered / elapsed if elapsed > 0 else 0
            print(f"  [{rendered} clips] {rate:.1f} WAV/s", flush=True)

    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
            active: set = set()
            for midi_path in midi_files:
                active.add(ex.submit(_render_midi_file, _job(midi_path)))
                if len(active) >= max_in_flight:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for f in done:
                        _process(f)
            for f in as_completed(active):
                _process(f)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min — {rendered} clips, {errors} errors")

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Manifest: {manifest_path} ({len(all_entries)} entries)")


if __name__ == "__main__":
    main()
