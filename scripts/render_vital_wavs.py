#!/usr/bin/env python3
"""
Render single-track WAV files from LMD MIDI files using Vital synth via Vita.

For each selected MIDI file, one random non-drum track is extracted.
Default behavior renders MIDI tracks × selected .vital presets (Cartesian product).
Optional --random-preset mode ignores .vital files and generates one deterministic
random synth state per render from the global seed, with audibility gating.

Usage:
    python scripts/render_vital_wavs.py [options]

Examples:
    # Smoke test: 1 WAV
    python scripts/render_vital_wavs.py --num-midi 1 --num-presets 1

    # Small batch, 8 parallel workers
    python scripts/render_vital_wavs.py --num-midi 5 --num-presets 3 --jobs 8

    # Large batch using all cores
    python scripts/render_vital_wavs.py --num-midi 100 --num-presets 1000 --jobs 48
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import signal
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.render.vital import (
    apply_preset,
    clip_notes,
    constrain_random_preset_for_audibility,
    extract_track,
    jitter_preset,
    probe_audibility,
    randomize_preset,
    render_probe_note,
    transpose_notes,
    trim_silence,
)

DEFAULT_MIDI_DIR = Path("data/raw/phase1_sources/lakh_midi/lmd_full")
DEFAULT_PRESETS_DIR = Path("~/Downloads/vital_presets").expanduser()
DEFAULT_OUTPUT_DIR = Path("outputs/vital_renders")
SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Worker (module-level so it's picklable)
# ---------------------------------------------------------------------------

_vital_instance = None  # cached per worker process — loaded once, reused for all jobs
_wavetable_lib: list | None = None  # cached wavetable library for gen-preset mode


def _render_job(args: tuple) -> tuple[str | None, str | None, dict]:
    """
    Render one (midi_path, track_idx, preset) pair to a WAV file.
    MIDI is parsed inside the worker to avoid pickling large note lists.
    Vital is cached per worker unless random/fresh modes force a new synth per job.
    Returns (wav_path, None, meta) on success or (None, error_str, {}) on failure.
    """
    global _vital_instance
    (
        midi_path,
        track_idx,
        preset_path,
        wav_path,
        tail_s,
        jitter_std,
        transpose_range,
        transpose_prob,
        random_preset,
        random_preset_amount,
        random_preset_max_attempts,
        random_preset_min_peak,
        random_preset_probe_note,
        random_preset_probe_velocity,
        random_preset_probe_dur,
        fresh_synth_per_job,
        job_seed,
        gen_preset_archetype,
        gen_preset_wavetable_lib_path,
        gen_preset_save_dir,
    ) = args
    gen_preset = bool(gen_preset_archetype)
    meta = {}
    try:
        import pretty_midi
        from maestro.render.vital import _load_vital, _render_note_list

        # ── Parse MIDI (gracefully skip corrupt/non-MIDI files) ───────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return None, None, {}  # corrupt file — silent skip

        if track_idx is None:
            candidates = [i for i, inst in enumerate(pm.instruments)
                          if not inst.is_drum and len(inst.notes) > 0]
            if not candidates:
                return None, None, {}  # no melodic tracks — silent skip
            track_idx = random.Random(job_seed).choice(candidates)

        notes = clip_notes(list(pm.instruments[track_idx].notes))
        if not notes:
            return None, f"{Path(wav_path).name}: no notes after clipping", {}
        meta["job_seed"] = int(job_seed)
        meta["clip_dur_s"] = round(max(n.end for n in notes), 3)

        job_rng = random.Random(job_seed)

        if transpose_range and job_rng.random() < transpose_prob:
            semitones = job_rng.randint(-transpose_range, transpose_range)
            notes = transpose_notes(notes, semitones)
            meta["transpose_semitones"] = semitones

        # ── Load Vital and apply/generate preset ───────────────────────────
        if random_preset:
            attempts = max(1, int(random_preset_max_attempts))
            min_peak = max(0.0, float(random_preset_min_peak))
            probe_note = int(random_preset_probe_note)
            probe_velocity = max(0.01, min(1.0, float(random_preset_probe_velocity)))
            probe_dur = max(0.01, float(random_preset_probe_dur))

            def _attempt_seed(attempt_idx: int) -> int:
                return _stable_32bit_seed(job_seed, "random_preset_attempt", attempt_idx)

            def _build_random_synth(seed: int):
                synth = _load_vital()
                randomize_preset(synth, random.Random(seed), amount=random_preset_amount)
                constrain_random_preset_for_audibility(synth)

                jitter_seed = None
                if jitter_std:
                    jitter_seed = _stable_32bit_seed(seed, "random_preset_jitter")
                    jitter_preset(synth, "", random.Random(jitter_seed), jitter_std)
                    constrain_random_preset_for_audibility(synth)
                return synth, jitter_seed

            selected_attempt = None
            selected_seed = None
            selected_peak = -1.0
            selected_mode = "best_effort"
            selected_jitter_seed = None

            for attempt_idx in range(attempts):
                attempt_seed = _attempt_seed(attempt_idx)
                synth_probe, jitter_seed = _build_random_synth(attempt_seed)
                probe_audio = render_probe_note(
                    synth_probe,
                    midi_note=probe_note,
                    midi_velocity=probe_velocity,
                    note_dur=probe_dur,
                )
                peak = float(abs(probe_audio).max()) if probe_audio.size else 0.0

                if peak > selected_peak:
                    selected_peak = peak
                    selected_attempt = attempt_idx
                    selected_seed = attempt_seed
                    selected_jitter_seed = jitter_seed

                if peak >= min_peak:
                    selected_peak = peak
                    selected_attempt = attempt_idx
                    selected_seed = attempt_seed
                    selected_jitter_seed = jitter_seed
                    selected_mode = "threshold_pass"
                    break

            _vital_instance, _ = _build_random_synth(selected_seed)

            meta["preset_mode"] = "random_generated"
            meta["preset_seed"] = int(selected_seed)
            meta["preset_random_amount"] = float(random_preset_amount)
            meta["random_preset_min_peak"] = min_peak
            meta["random_preset_max_attempts"] = attempts
            meta["selected_attempt_index"] = int(selected_attempt)
            meta["probe_peak_selected"] = float(selected_peak)
            meta["random_preset_selection_mode"] = selected_mode
            meta["random_preset_below_threshold"] = selected_peak < min_peak
            if selected_jitter_seed is not None:
                meta["jitter_seed"] = int(selected_jitter_seed)
        elif gen_preset:
            global _wavetable_lib
            from maestro.synth.preset_gen import ARCHETYPES, generate_preset
            from maestro.synth.wavetable_lib import load_wavetable_lib

            if _vital_instance is None:
                _vital_instance = _load_vital()

            if _wavetable_lib is None and gen_preset_wavetable_lib_path:
                try:
                    _wavetable_lib = load_wavetable_lib(Path(gen_preset_wavetable_lib_path))
                except Exception as e:
                    print(
                        f"WARNING: failed to load wavetable lib: {e}",
                        file=sys.stderr,
                    )
                    _wavetable_lib = []

            archetype = gen_preset_archetype
            if archetype == "random":
                archetype = job_rng.choice(ARCHETYPES)

            # Multi-attempt with probe gating
            attempts = max(1, int(random_preset_max_attempts))
            min_peak = max(0.0, float(random_preset_min_peak))
            selected_preset = None
            selected_peak = -1.0

            for attempt_idx in range(attempts):
                attempt_seed = _stable_32bit_seed(job_seed, "gen_preset_attempt", attempt_idx)
                preset_dict = generate_preset(
                    archetype, random.Random(attempt_seed),
                    wavetable_lib=_wavetable_lib or [],
                )
                _vital_instance.load_json(json.dumps(preset_dict))
                constrain_random_preset_for_audibility(_vital_instance)
                probe = probe_audibility(_vital_instance)
                if probe["max_peak"] > selected_peak:
                    selected_peak = probe["max_peak"]
                    selected_preset = preset_dict
                if probe["pass"] and probe["max_peak"] >= min_peak:
                    selected_peak = probe["max_peak"]
                    selected_preset = preset_dict
                    break

            # Apply best preset
            _vital_instance.load_json(json.dumps(selected_preset))
            constrain_random_preset_for_audibility(_vital_instance)

            # Optionally save .vital file for provenance
            if gen_preset_save_dir:
                save_path = Path(gen_preset_save_dir) / f"{selected_preset['preset_name']}.vital"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(selected_preset, f, separators=(",", ":"))
                meta["preset_file"] = str(save_path)

            meta["preset_mode"] = "gen_preset"
            meta["preset_archetype"] = archetype
            meta["preset_name"] = selected_preset["preset_name"]
            meta["probe_peak"] = float(selected_peak)
        else:
            if fresh_synth_per_job:
                _vital_instance = _load_vital()
            elif _vital_instance is None:
                _vital_instance = _load_vital()

            apply_preset(_vital_instance, preset_path)
            meta["preset_mode"] = "file"
            meta["preset_file"] = str(preset_path)

            if jitter_std:
                jitter_preset(_vital_instance, preset_path, job_rng, jitter_std)
                meta["jitter_seed"] = job_seed

        audio = _render_note_list(_vital_instance, notes, SAMPLE_RATE, tail_s)
        audio = trim_silence(audio, SAMPLE_RATE)

        sf.write(wav_path, audio.T, SAMPLE_RATE, subtype="PCM_24")
        return wav_path, None, meta
    except Exception as e:
        return None, f"{Path(wav_path).name}: {e}", {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num-wavs", type=int, default=None, metavar="N",
                   help="Random-sample mode: render N WAVs, each with a randomly chosen MIDI+preset")
    p.add_argument("--num-midi", type=int, default=None, metavar="N",
                   help="Number of MIDI files to select (default: all)")
    p.add_argument("--num-presets", type=int, default=None, metavar="M",
                   help="Number of Vital presets to select (default: all)")
    p.add_argument("--midi-dir", type=Path, action="append", dest="midi_dirs",
                   metavar="DIR", help="MIDI directory (repeatable)")
    p.add_argument("--presets-dir", type=Path, default=DEFAULT_PRESETS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tail", type=float, default=2.0, metavar="SECS",
                   help="Release tail in seconds (default: 2.0)")
    p.add_argument("--jitter", type=float, default=0.0, metavar="STD",
                   help="Gaussian std dev for preset parameter noise, e.g. 0.05 (default: 0 = off)")
    p.add_argument("--transpose-range", type=int, default=0, metavar="SEMITONES",
                   help="Max semitones to shift when transposing (default: 0 = off)")
    p.add_argument("--transpose-prob", type=float, default=0.5, metavar="PROB",
                   help="Probability [0-1] that any given render gets transposed (default: 0.5)")
    _safe_jobs = min(multiprocessing.cpu_count(), 10)  # ~2.4 GB/worker; 10 × 2.4 = 24 GB < 30 GB
    p.add_argument("--jobs", type=int, default=_safe_jobs,
                   help=f"Parallel workers (default: {_safe_jobs}; each Vital instance ~2.4 GB)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing WAV files instead of skipping (disables resume mode)")
    p.add_argument("--fresh-synth-per-job", action="store_true",
                   help="Create a fresh Vita synth for every render job (debug mode; slower, more memory churn)")
    p.add_argument("--random-preset", action="store_true",
                   help="Ignore .vital files and generate a random preset per render (deterministic via seed)")
    p.add_argument("--random-preset-amount", type=float, default=1.0, metavar="AMOUNT",
                   help="Blend from init->random in [0,1] when --random-preset is set (default: 1.0)")
    p.add_argument("--random-preset-max-attempts", type=int, default=4, metavar="N",
                   help="Max generation attempts in random preset mode (default: 4)")
    p.add_argument("--random-preset-min-peak", type=float, default=0.01, metavar="PEAK",
                   help="Audibility gate for random preset probe peak (default: 0.01)")
    p.add_argument("--random-preset-probe-note", type=int, default=60, metavar="MIDI",
                   help="Probe note for audibility test in random preset mode (default: 60)")
    p.add_argument("--random-preset-probe-velocity", type=float, default=0.9, metavar="VEL",
                   help="Probe note velocity in [0,1] (default: 0.9)")
    p.add_argument("--random-preset-probe-dur", type=float, default=0.35, metavar="SECS",
                   help="Probe note duration seconds (default: 0.35)")
    # gen-preset mode
    p.add_argument("--gen-preset", type=str, default=None, metavar="ARCHETYPE",
                   help="Generate structured presets using archetype-based sampling. "
                        "ARCHETYPE: bass|lead|pad|keys|pluck|sequence|random. "
                        "Replaces --random-preset with musically-coherent generation.")
    p.add_argument("--gen-preset-wavetable-lib", type=Path,
                   default=Path("data/wavetable_lib.json"),
                   help="Path to wavetable library JSON (default: data/wavetable_lib.json)")
    p.add_argument("--gen-preset-save-dir", type=Path, default=None,
                   help="If set, save each generated .vital file here for provenance")
    return p.parse_args()


def resolve_dir(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def collect_files(directory: Path, pattern: str) -> list[Path]:
    files = [p for p in directory.rglob(pattern) if p.is_file()]
    if not files:
        print(f"ERROR: No {pattern} files found in {directory}", file=sys.stderr)
        sys.exit(1)
    return sorted(files)


def safe_stem(path: Path, max_len: int = 40) -> str:
    stem = path.stem[:max_len]
    for ch in ' /\\:*?"<>|':
        stem = stem.replace(ch, "_")
    return stem


def _stable_32bit_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _stable_job_seed(base_seed: int, spec: dict) -> int:
    """
    Derive a deterministic 32-bit seed from render identity.
    Stable across resume/overwrite, worker count, and scheduling order.
    """
    return _stable_32bit_seed(
        base_seed,
        spec.get("midi_path", ""),
        spec.get("track_idx", ""),
        spec.get("preset_path", ""),
        spec.get("wav_path", ""),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if (args.random_preset or args.gen_preset) and args.num_presets is not None:
        print("NOTE: --num-presets is ignored when --random-preset/--gen-preset is enabled.")
    if args.random_preset and args.random_preset_max_attempts < 1:
        print("ERROR: --random-preset-max-attempts must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.random_preset and args.random_preset_probe_dur <= 0:
        print("ERROR: --random-preset-probe-dur must be > 0", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    midi_dirs = [resolve_dir(d, repo_root) for d in (args.midi_dirs or [DEFAULT_MIDI_DIR])]
    output_dir = resolve_dir(args.output_dir, repo_root)

    # ── Collect files ──────────────────────────────────────────────────────
    print("Collecting files...")
    all_midi = []
    for midi_dir in midi_dirs:
        all_midi.extend(collect_files(midi_dir, "*.mid"))
    if args.random_preset or args.gen_preset:
        all_presets = []
    else:
        all_presets = collect_files(args.presets_dir, "*.vital")

    rng.shuffle(all_midi)
    rng.shuffle(all_presets)

    midi_files = all_midi[: args.num_midi] if args.num_midi is not None else all_midi
    preset_files = all_presets[: args.num_presets] if args.num_presets is not None else all_presets

    # ── Resolve gen-preset paths ───────────────────────────────────────────
    gen_preset_wavetable_lib_path = (
        str(resolve_dir(args.gen_preset_wavetable_lib, repo_root))
        if args.gen_preset and args.gen_preset_wavetable_lib
        else None
    )
    gen_preset_save_dir = (
        str(resolve_dir(args.gen_preset_save_dir, repo_root))
        if args.gen_preset and args.gen_preset_save_dir
        else None
    )

    # ── Create output dir ──────────────────────────────────────────────────
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # ── Build job specs (mode-dependent) ──────────────────────────────────
    # Each spec: {"midi_path", "track_idx", "track_name", "program",
    #             "preset_path", "wav_path"}
    if args.num_wavs is not None:
        job_specs = _build_random_specs(args, rng, midi_files, preset_files, wavs_dir)
    else:
        job_specs = _build_cartesian_specs(args, rng, midi_files, preset_files, wavs_dir)

    if not job_specs:
        print("ERROR: No usable job specs built.", file=sys.stderr)
        sys.exit(1)

    total = len(job_specs)
    if args.overwrite:
        already_done = 0
        pending_specs = job_specs
    else:
        already_done = sum(1 for s in job_specs if Path(s["wav_path"]).exists())
        pending_specs = [s for s in job_specs if not Path(s["wav_path"]).exists()]
    pending_total = len(pending_specs)

    print(f"  Workers    : {args.jobs}")
    if args.gen_preset:
        print(f"  Preset mode: gen_preset (archetype={args.gen_preset})")
        print(f"  Gate       : peak>={args.random_preset_min_peak:.4f} "
              f"within {args.random_preset_max_attempts} attempts")
        if gen_preset_wavetable_lib_path:
            print(f"  Wavetable  : {gen_preset_wavetable_lib_path}")
    elif args.random_preset:
        print(f"  Preset mode: random generated (amount={args.random_preset_amount:.3f})")
        print(f"  Gate       : peak>={args.random_preset_min_peak:.4f} "
              f"within {args.random_preset_max_attempts} attempts")
    else:
        print("  Preset mode: file presets")
    if args.random_preset:
        print("  Synth mode : fresh per job (required for random preset generation)")
    elif args.fresh_synth_per_job:
        print("  Synth mode : fresh per job (no worker cache)")
    else:
        print("  Synth mode : cached per worker")
    if args.overwrite:
        print("  Output mode: overwrite existing WAVs")
    if already_done:
        print(f"  {already_done:,} already rendered — skipping (resume mode)")
        print("  NOTE: If these came from older code, rerun with --overwrite or a new --output-dir.")
    if pending_total == 0:
        print("Nothing to render — all WAVs already exist.")
        _write_manifest(output_dir, job_specs, {})
        return

    print(f"\nRendering {pending_total:,} WAVs across {args.jobs} workers...")

    # ── Job tuple generator ────────────────────────────────────────────────
    def _pending_jobs():
        for spec in pending_specs:
            job_seed = _stable_job_seed(args.seed, spec)
            yield (
                spec["midi_path"],
                spec["track_idx"],
                spec["preset_path"],
                spec["wav_path"],
                args.tail,
                args.jitter,
                args.transpose_range,
                args.transpose_prob,
                args.random_preset,
                args.random_preset_amount,
                args.random_preset_max_attempts,
                args.random_preset_min_peak,
                args.random_preset_probe_note,
                args.random_preset_probe_velocity,
                args.random_preset_probe_dur,
                args.fresh_synth_per_job,
                job_seed,
                args.gen_preset,  # gen_preset_archetype (None if not set)
                gen_preset_wavetable_lib_path,
                gen_preset_save_dir,
            )

    # ── Bounded parallel render ────────────────────────────────────────────
    rendered = errors = 0
    start = time.time()
    max_in_flight = args.jobs * 4
    wav_meta: dict[str, dict] = {}

    def _worker_init():
        os.setpgrp()
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _process_future(f):
        nonlocal rendered, errors
        wav_path, err, meta = f.result()
        if err:
            errors += 1
            if errors <= 10:
                print(f"  ERROR: {err}")
        elif wav_path is not None:
            rendered += 1
            if meta:
                wav_meta[wav_path] = meta
        # wav_path=None, err=None → silent skip (corrupt MIDI / no melodic tracks)
        n = rendered + errors
        if n % 100 == 0 or n == 1:
            elapsed = time.time() - start
            rate = n / elapsed
            rem = (pending_total - n) / rate if rate > 0 else float("inf")
            print(f"  [{n + already_done:,}/{total:,}] "
                  f"{rate:.1f} WAV/s — ETA {rem/60:.0f} min")

    try:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init) as executor:
            active: set = set()
            for job in _pending_jobs():
                active.add(executor.submit(_render_job, job))
                if len(active) >= max_in_flight:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for f in done:
                        _process_future(f)
            for f in as_completed(active):
                _process_future(f)
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    rate = rendered / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed/60:.1f} min  ({rate:.1f} WAV/s)")
    print(f"  Rendered:     {rendered:,}")
    print(f"  Pre-existing: {already_done:,}")
    print(f"  Errors:       {errors:,}")
    print(f"  WAVs in:      {wavs_dir}")

    _write_manifest(output_dir, job_specs, wav_meta)


def _build_cartesian_specs(args, rng, midi_files, preset_files, wavs_dir) -> list[dict]:
    """Cartesian product: every MIDI track × every preset."""
    if args.random_preset or args.gen_preset:
        print(f"  Mode       : cartesian-{'gen' if args.gen_preset else 'random'}"
              f" ({len(midi_files):,} MIDI × 1 generated preset = {len(midi_files):,} WAVs)")
    else:
        print(f"  Mode       : cartesian ({len(midi_files):,} MIDI × {len(preset_files):,} presets"
              f" = {len(midi_files) * len(preset_files):,} WAVs)")

    print("\nExtracting MIDI tracks...")
    specs = []
    skipped = 0
    for midi_path in midi_files:
        notes, track_idx, track_name, program, _ = extract_track(str(midi_path), rng)
        if notes is None:
            skipped += 1
            continue
        notes = clip_notes(notes)
        if not notes:
            skipped += 1
            continue
        midi_stem = safe_stem(midi_path)
        track_label = f"t{track_idx}"
        preset_iter = [None] if (args.random_preset or args.gen_preset) else preset_files
        for preset_path in preset_iter:
            if preset_path is None:
                preset_label = f"gen_{args.gen_preset}" if args.gen_preset else "random_preset"
            else:
                preset_label = safe_stem(preset_path)
            wav_path = wavs_dir / f"{midi_stem}__{track_label}__{preset_label}.wav"
            specs.append({
                "midi_path": str(midi_path),
                "track_idx": track_idx,
                "track_name": track_name,
                "program": int(program),
                "preset_path": None if preset_path is None else str(preset_path),
                "wav_path": str(wav_path),
            })
    print(f"  {len(midi_files) - skipped:,} tracks extracted  ({skipped} skipped — no melodic tracks)")
    return specs


def _build_random_specs(args, rng, midi_files, preset_files, wavs_dir) -> list[dict]:
    """Random sampling: each WAV independently draws a random MIDI+preset.
    Track selection is deferred to the worker (track_idx=None) so this is instant."""
    if args.random_preset or args.gen_preset:
        label = "gen-preset" if args.gen_preset else "random-preset"
        print(f"  Mode       : random-sample-{label} ({args.num_wavs:,} WAVs"
              f" from {len(midi_files):,} MIDI × generated presets)")
    else:
        print(f"  Mode       : random-sample ({args.num_wavs:,} WAVs"
              f" from {len(midi_files):,} MIDI × {len(preset_files):,} presets)")

    specs = []
    for i in range(args.num_wavs):
        midi_path = rng.choice(midi_files)
        preset_path = None if (args.random_preset or args.gen_preset) else rng.choice(preset_files)
        if preset_path is None:
            preset_label = f"gen_{args.gen_preset}" if args.gen_preset else "random_preset"
        else:
            preset_label = safe_stem(preset_path)
        wav_path = wavs_dir / f"rnd_{i:06d}__{safe_stem(midi_path)}__{preset_label}.wav"
        specs.append({
            "midi_path": str(midi_path),
            "track_idx": None,   # worker picks a random non-drum track via job_seed
            "track_name": "",
            "program": -1,
            "preset_path": None if preset_path is None else str(preset_path),
            "wav_path": str(wav_path),
        })

    print(f"  {len(specs):,} specs built")
    return specs


def _write_manifest(output_dir: Path, job_specs: list[dict], wav_meta: dict) -> None:
    manifest_path = output_dir / "manifest.jsonl"
    written = 0
    with open(manifest_path, "w") as f:
        for spec in job_specs:
            if Path(spec["wav_path"]).exists():
                row = {
                    "source_midi": spec["midi_path"],
                    "preset_file": spec["preset_path"],
                    "wav_path": spec["wav_path"],
                }
                if spec["track_idx"] is not None:
                    row["track_idx"] = spec["track_idx"]
                    row["track_name"] = spec["track_name"]
                    row["program"] = spec["program"]
                row.update(wav_meta.get(spec["wav_path"], {}))
                f.write(json.dumps(row) + "\n")
                written += 1
    print(f"Manifest: {manifest_path} ({written:,} entries)")


if __name__ == "__main__":
    main()
