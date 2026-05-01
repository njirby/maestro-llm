#!/usr/bin/env python3
"""
Generate ground-truth targets for the unified SFT pipeline.

For each sample, produces only what build_unified_sft_v4.py needs:
  - {sample_id}_target.vital   — generated preset JSON
  - {sample_id}_gt.wav         — target audio (preset + MIDI)
  - {sample_id}_gt_probe.wav   — target probe (standard probe notes)
  - {sample_id}_default.wav    — baseline (init preset + same MIDI)
  - {sample_id}_source.mid     — MIDI clip from Lakh catalog

Outputs a manifest.jsonl consumed by run_sft_production.py.

Resumable: skips samples where gt.wav already exists (unless --overwrite).

Usage:
    python scripts/render_sft_targets.py \\
        --generate 17000 \\
        --archetypes bass lead pad keys pluck sequence \\
        --output-dir outputs/sft_16k \\
        --wavetable-lib data/wavetable_lib.json \\
        --midi-catalog outputs/midi_clips/lakh_catalog.jsonl \\
        --jobs 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.render.vital import (
    SAMPLE_RATE,
    _render_note_list,
    apply_preset,
    load_midi_clip_catalog,
    make_gt_notes,
    pick_midi_clip,
    probe_audibility,
    trim_silence,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Worker globals (one Synth per process)
# ---------------------------------------------------------------------------

_vital_instance = None
_wavetable_lib_cache = None
_init_preset_cache = None
_midi_catalog = None


def _worker_init(midi_catalog_path: str | None = None):
    global _vital_instance, _midi_catalog
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from maestro.render.vital import _load_vital
    _vital_instance = _load_vital()
    if midi_catalog_path:
        _midi_catalog = load_midi_clip_catalog(midi_catalog_path)


def _write_notes_to_midi(notes: list, out_path: Path, tempo: float = 120.0) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="source")
    for n in notes:
        vel = max(1, min(127, int(getattr(n, "velocity", 100))))
        pitch = int(getattr(n, "pitch", 60))
        start = float(getattr(n, "start", 0.0))
        end = float(getattr(n, "end", max(0.05, start + 0.25)))
        if end <= start:
            end = start + 0.05
        inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def _render_target(job: dict) -> dict | None:
    """Render one GT target sample. Returns a manifest entry or None on failure."""
    global _vital_instance, _wavetable_lib_cache, _init_preset_cache

    sample_id = job["sample_id"]
    archetype = job["archetype"]
    audio_dir = Path(job["audio_dir"])
    presets_dir = Path(job["presets_dir"])

    try:
        from maestro.synth.preset_gen import generate_preset
        from maestro.synth.wavetable_lib import load_wavetable_lib

        if _wavetable_lib_cache is None and job.get("wavetable_lib_path"):
            try:
                _wavetable_lib_cache = load_wavetable_lib(Path(job["wavetable_lib_path"]))
            except Exception as e:
                print(f"WARNING: failed to load wavetable lib: {e}", file=sys.stderr)
                _wavetable_lib_cache = []

        rng = random.Random(job["seed"])
        target_preset = generate_preset(archetype, rng, wavetable_lib=_wavetable_lib_cache or [])

        # --- Source MIDI ---
        clip_dur = float(job.get("clip_duration_s", 10.0))
        midi_clip_meta: dict | None = None
        gt_notes = None
        if _midi_catalog:
            seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
            _rng = random.Random(seed)
            picked_notes, midi_clip_meta = pick_midi_clip(archetype, _midi_catalog, _rng)
            if picked_notes:
                gt_notes = picked_notes
                clip_dur = max(n.end for n in gt_notes) + 0.25
        if gt_notes is None:
            gt_notes = make_gt_notes(clip_duration_s=clip_dur)

        _probe_tail_s = max(0.1, 1.0 * (clip_dur / 10.0))
        _gt_tail_s = max(0.2, 2.0 * (clip_dur / 10.0))

        # Write MIDI
        midi_dir = audio_dir.parent / "midi"
        source_midi_path = midi_dir / f"{sample_id}_source.mid"
        _write_notes_to_midi(gt_notes, source_midi_path)

        # Write target preset
        target_preset_path = presets_dir / f"{sample_id}_target.vital"
        presets_dir.mkdir(parents=True, exist_ok=True)
        with open(target_preset_path, "w") as f:
            json.dump(target_preset, f)

        # --- Render default/baseline ---
        if _init_preset_cache is None:
            init_path = Path(__file__).resolve().parent.parent / "maestro" / "synth" / "init_preset.json"
            with open(init_path) as f:
                _init_preset_cache = json.load(f)
        _vital_instance.load_json(json.dumps(_init_preset_cache))
        default_audio = _render_note_list(_vital_instance, gt_notes, SAMPLE_RATE, tail_s=_probe_tail_s)
        default_audio = trim_silence(default_audio, SAMPLE_RATE, min_duration_s=0.5)
        default_wav = audio_dir / f"{sample_id}_default.wav"
        sf.write(str(default_wav), default_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # --- Render GT audio ---
        _vital_instance.load_json(json.dumps(target_preset))

        gt_probe_dur = 2.0 if archetype in ("pad", "keys") else 0.3
        probe_result = probe_audibility(_vital_instance, note_dur=gt_probe_dur)
        if not probe_result["pass"]:
            # Clean up files from failed sample
            for p in [default_wav, source_midi_path, target_preset_path]:
                if p.exists():
                    p.unlink()
            return None

        gt_audio = _render_note_list(_vital_instance, gt_notes, SAMPLE_RATE, tail_s=_gt_tail_s)
        gt_audio = trim_silence(gt_audio, SAMPLE_RATE)

        # RMS gate: reject near-silent renders even if probe passed
        mono = gt_audio.mean(axis=0) if gt_audio.ndim == 2 else gt_audio
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms < 0.001:
            for p in [default_wav, source_midi_path, target_preset_path]:
                if p.exists():
                    p.unlink()
            return None

        gt_wav = audio_dir / f"{sample_id}_gt.wav"
        sf.write(str(gt_wav), gt_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # GT probe clip (same notes, for CLAP comparison)
        gt_probe_audio = _render_note_list(_vital_instance, gt_notes, SAMPLE_RATE, tail_s=_probe_tail_s)
        gt_probe_audio = trim_silence(gt_probe_audio, SAMPLE_RATE, min_duration_s=0.5)
        gt_probe_wav = audio_dir / f"{sample_id}_gt_probe.wav"
        sf.write(str(gt_probe_wav), gt_probe_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # --- Manifest entry ---
        entry = {
            "sample_id": sample_id,
            "archetype": archetype,
            "gt_wav": str(gt_wav),
            "gt_probe_wav": str(gt_probe_wav),
            "default_wav": str(default_wav),
            "source_midi_path": str(source_midi_path),
            "target_preset_path": str(target_preset_path),
            "start_type": "init",
            "gt_rms": rms,
            "gt_max_peak": float(probe_result["max_peak"]),
        }
        if midi_clip_meta is not None:
            entry["source_midi_origin"] = {
                "source": "lakh",
                "midi_path": midi_clip_meta["midi_path"],
                "track_idx": midi_clip_meta["track_idx"],
                "program": midi_clip_meta["program"],
                "start_s_in_source": midi_clip_meta["start_s"],
                "duration_s": midi_clip_meta["duration_s"],
                "n_notes": midi_clip_meta["n_notes"],
                "pitch_range": [midi_clip_meta["pitch_min"], midi_clip_meta["pitch_max"]],
                "is_monophonic": midi_clip_meta["is_monophonic"],
                "bpm": midi_clip_meta["bpm"],
            }
        return entry

    except Exception as e:
        print(f"  ERROR [{sample_id}]: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--generate", type=int, required=True, metavar="N",
                   help="Number of target samples to generate")
    p.add_argument("--archetypes", nargs="+",
                   default=["bass", "lead", "pad", "keys", "pluck", "sequence"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/sft_16k"))
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--midi-catalog", type=Path, default=None)
    p.add_argument("--jobs", type=int, default=min(multiprocessing.cpu_count(), 24))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-duration-s", type=float, default=10.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    output_dir = _resolve(args.output_dir)
    audio_dir = output_dir / "audio"
    presets_dir = output_dir / "presets"
    audio_dir.mkdir(parents=True, exist_ok=True)
    presets_dir.mkdir(parents=True, exist_ok=True)

    wavetable_lib_path = str(_resolve(args.wavetable_lib)) if args.wavetable_lib else None

    # Build job list
    rng = random.Random(args.seed)
    archetypes = args.archetypes
    jobs: list[dict] = []
    all_sample_ids: list[str] = []

    for i in range(args.generate):
        archetype = archetypes[i % len(archetypes)]
        sample_seed = rng.randint(0, 2**31)
        sample_id = f"{archetype}_{sample_seed:08x}"
        all_sample_ids.append(sample_id)

        gt_wav = audio_dir / f"{sample_id}_gt.wav"
        if gt_wav.exists() and not args.overwrite:
            continue

        jobs.append({
            "sample_id": sample_id,
            "archetype": archetype,
            "audio_dir": str(audio_dir),
            "presets_dir": str(presets_dir),
            "wavetable_lib_path": wavetable_lib_path,
            "seed": sample_seed,
            "clip_duration_s": args.clip_duration_s,
        })

    skipped = args.generate - len(jobs)
    if skipped > 0:
        print(f"  {skipped} samples already rendered — skipping (use --overwrite)")
    if not jobs:
        print("Nothing to render. Rebuilding manifest from existing files...")

    # Parallel render
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

    rendered = 0
    failed = 0
    results: list[dict] = []
    start = time.time()
    max_in_flight = args.jobs * 4

    if jobs:
        print(f"Rendering {len(jobs)} samples across {args.jobs} workers...")
        try:
            catalog_path = str(args.midi_catalog) if args.midi_catalog else None
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                initializer=_worker_init,
                initargs=(catalog_path,),
            ) as executor:
                active: set = set()
                job_iter = iter(jobs)

                pbar = tqdm(total=len(jobs), desc="Rendering", unit="sample") if tqdm else None

                def _drain(fs):
                    nonlocal rendered, failed
                    for f in fs:
                        entry = f.result()
                        if entry is not None:
                            results.append(entry)
                            rendered += 1
                        else:
                            failed += 1
                        if pbar is not None:
                            pbar.update(1)
                        elif (rendered + failed) % 50 == 0:
                            elapsed = time.time() - start
                            rate = (rendered + failed) / elapsed if elapsed > 0 else 0
                            print(f"  [{rendered + failed}/{len(jobs)}] "
                                  f"{rate:.1f} samples/s  ({rendered} ok, {failed} rejected)")

                for job in job_iter:
                    active.add(executor.submit(_render_target, job))
                    if len(active) >= max_in_flight:
                        done, active = wait(active, return_when=FIRST_COMPLETED)
                        _drain(done)

                _drain(as_completed(active))
                if pbar is not None:
                    pbar.close()

        except KeyboardInterrupt:
            print("\nInterrupted — shutting down workers...", file=sys.stderr)
            sys.exit(1)

    # Rebuild manifest: include both newly rendered and previously existing samples
    manifest_entries: list[dict] = []

    # Add newly rendered entries
    existing_ids = {e["sample_id"] for e in results}
    manifest_entries.extend(results)

    # Scan for previously rendered samples (resume support)
    for sample_id in all_sample_ids:
        if sample_id in existing_ids:
            continue
        gt_wav = audio_dir / f"{sample_id}_gt.wav"
        if not gt_wav.exists():
            continue
        # Reconstruct entry from files on disk
        archetype = sample_id.split("_")[0]
        entry = {
            "sample_id": sample_id,
            "archetype": archetype,
            "gt_wav": str(gt_wav),
            "gt_probe_wav": str(audio_dir / f"{sample_id}_gt_probe.wav"),
            "default_wav": str(audio_dir / f"{sample_id}_default.wav"),
            "source_midi_path": str(output_dir / "midi" / f"{sample_id}_source.mid"),
            "target_preset_path": str(presets_dir / f"{sample_id}_target.vital"),
            "start_type": "init",
        }
        # Verify all required files exist
        if all(Path(entry[k]).exists() for k in ("gt_wav", "gt_probe_wav", "default_wav",
                                                   "source_midi_path", "target_preset_path")):
            manifest_entries.append(entry)

    # Sort by sample_id for deterministic ordering
    manifest_entries.sort(key=lambda e: e["sample_id"])

    elapsed = time.time() - start
    rate = rendered / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s  ({rate:.1f} samples/s)")
    print(f"  Rendered: {rendered}")
    print(f"  Rejected (silent/garbage): {failed}")
    print(f"  Total in manifest: {len(manifest_entries)}")

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Manifest: {manifest_path}")

    # Summary by archetype
    from collections import Counter
    arch_counts = Counter(e["archetype"] for e in manifest_entries)
    for arch in sorted(arch_counts):
        print(f"    {arch}: {arch_counts[arch]}")


if __name__ == "__main__":
    main()
