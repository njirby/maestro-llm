#!/usr/bin/env python3
"""
Render iterative preset-recreation audio samples for SFT training data.

For each sample:
  - Generate (or load) an N-step parameter path from init to target preset
  - Render a ground-truth musical clip (target preset + make_gt_notes())
  - Render a target probe clip using the same note pattern as iteration probes
  - Render a default/init probe clip, then N iteration probe clips (one per step)
    using cumulative presets

Usage:
    python scripts/render_iter_presets.py \
        --generate 10 --archetypes bass lead pad \
        --output-dir outputs/iter_sft \
        --wavetable-lib data/wavetable_lib.json --jobs 4
"""

import argparse
import json
import multiprocessing
import os
import pretty_midi
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.render.vital import (
    SAMPLE_RATE,
    _render_note_list,
    apply_preset,
    make_gt_notes,
    make_probe_notes,
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


def _write_notes_to_midi(notes: list, out_path: Path, tempo: float = 120.0) -> None:
    """Persist a note list to a single-track MIDI file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="source")
    for n in notes:
        vel = int(getattr(n, "velocity", 100))
        vel = max(1, min(127, vel))
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


def _worker_init():
    """Pool initializer: create ONE vita.Synth and suppress SIGINT."""
    global _vital_instance
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from maestro.render.vital import _load_vital
    _vital_instance = _load_vital()


def _render_sample(job: dict) -> dict | None:
    """Render one iter-SFT sample in a worker process.

    Returns a manifest-entry dict on success, or None if the sample should be
    skipped (audibility failure, error, etc.).
    """
    global _vital_instance, _wavetable_lib_cache, _init_preset_cache

    mode = job["mode"]
    sample_id = job["sample_id"]
    archetype = job["archetype"]
    audio_dir = Path(job["audio_dir"])
    paths_dir = Path(job["paths_dir"]) if job.get("paths_dir") else None

    try:
        # ------------------------------------------------------------------
        # 1. Generate or load the parameter path
        # ------------------------------------------------------------------
        if mode == "generate":
            from maestro.synth.path_gen import compare_preset_path, generate_preset_path
            from maestro.synth.wavetable_lib import load_wavetable_lib

            if _wavetable_lib_cache is None and job.get("wavetable_lib_path"):
                try:
                    _wavetable_lib_cache = load_wavetable_lib(
                        Path(job["wavetable_lib_path"])
                    )
                except Exception as e:
                    print(
                        f"WARNING: failed to load wavetable lib: {e}",
                        file=sys.stderr,
                    )
                    _wavetable_lib_cache = []

            rng = random.Random(job["seed"])
            path_data = generate_preset_path(
                archetype,
                rng,
                wavetable_lib=_wavetable_lib_cache or [],
                output_dir=paths_dir,
                sample_id=sample_id,
            )
            compare = compare_preset_path(path_data)
        else:
            # --paths-dir mode: load existing path JSON
            path_json_path = Path(job["path_file"])
            with open(path_json_path) as f:
                path_data = json.load(f)
            sample_id = path_data["sample_id"]
            archetype = path_data["archetype"]
            compare = None

        steps = path_data["iterations"]
        n_iterations = path_data["n_iterations"]

        # ------------------------------------------------------------------
        # 2. Build source MIDI + render default/init probe clip for baseline
        # ------------------------------------------------------------------
        clip_dur = float(job.get("clip_duration_s", 10.0))
        probe_notes = make_probe_notes(archetype, clip_duration_s=clip_dur)
        gt_notes = make_gt_notes(clip_duration_s=clip_dur)
        _probe_tail_s = max(0.1, 1.0 * (clip_dur / 10.0))
        _gt_tail_s = max(0.2, 2.0 * (clip_dur / 10.0))
        midi_dir = audio_dir.parent / "midi"
        source_midi_path = midi_dir / f"{sample_id}_source.mid"
        _write_notes_to_midi(gt_notes, source_midi_path)

        if mode == "generate" and paths_dir is not None:
            path_json_path = paths_dir / f"{sample_id}.json"
            target_preset_path = paths_dir / f"{sample_id}_target.vital"
            with open(target_preset_path, "w") as f:
                json.dump(path_data["target_preset"], f)
            save_data = {
                k: v
                for k, v in path_data.items()
                if k != "target_preset"
            }
            save_data["target_preset_path"] = str(target_preset_path)
            save_data["source_midi_path"] = str(source_midi_path)
            save_data["final_vs_target_summary"] = compare["summary"]
            save_data["final_vs_target_top_noisy"] = compare["noisy"][:12]
            save_data["iterations"] = [
                {k: v for k, v in it.items() if k != "cumulative_preset"}
                for it in save_data["iterations"]
            ]
            with open(path_json_path, "w") as f:
                json.dump(save_data, f, indent=2)

        if _init_preset_cache is None:
            init_path = Path(__file__).resolve().parent.parent / "maestro" / "synth" / "init_preset.json"
            with open(init_path) as f:
                _init_preset_cache = json.load(f)
        _vital_instance.load_json(json.dumps(_init_preset_cache))
        default_audio = _render_note_list(
            _vital_instance, probe_notes, SAMPLE_RATE, tail_s=_probe_tail_s
        )
        default_audio = trim_silence(default_audio, SAMPLE_RATE, min_duration_s=0.5)
        default_wav = audio_dir / f"{sample_id}_default.wav"
        sf.write(str(default_wav), default_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # ------------------------------------------------------------------
        # 3. Render GT audio (last cumulative preset = target)
        # ------------------------------------------------------------------
        last_step = steps[-1]
        gt_preset_path = last_step.get("cumulative_preset_path")
        if gt_preset_path is None:
            # Fallback: use in-memory preset dict (generate mode only)
            if "cumulative_preset" in last_step:
                _vital_instance.load_json(json.dumps(last_step["cumulative_preset"]))
            else:
                return None
        else:
            apply_preset(_vital_instance, gt_preset_path)

        # Pad/keys have slow attacks (up to 1.5s) — use a longer probe note so
        # the sound has time to bloom before the amplitude check.
        gt_probe_dur = 2.0 if archetype in ("pad", "keys") else 0.3
        if not probe_audibility(_vital_instance, note_dur=gt_probe_dur)["pass"]:
            return None

        gt_audio = _render_note_list(_vital_instance, gt_notes, SAMPLE_RATE, tail_s=_gt_tail_s)
        gt_audio = trim_silence(gt_audio, SAMPLE_RATE)

        gt_wav = audio_dir / f"{sample_id}_gt.wav"
        sf.write(str(gt_wav), gt_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # Target probe clip: same note pattern as iter probes for apples-to-apples A/B.
        gt_probe_audio = _render_note_list(
            _vital_instance, probe_notes, SAMPLE_RATE, tail_s=_probe_tail_s
        )
        gt_probe_audio = trim_silence(gt_probe_audio, SAMPLE_RATE, min_duration_s=0.5)
        gt_probe_wav = audio_dir / f"{sample_id}_gt_probe.wav"
        sf.write(str(gt_probe_wav), gt_probe_audio.T, SAMPLE_RATE, subtype="PCM_24")

        # ------------------------------------------------------------------
        # 4. Render N iteration probe clips (steps 0 .. N-1)
        # ------------------------------------------------------------------
        iter_wavs: list[str] = []

        for step_idx in range(n_iterations):
            step = steps[step_idx]
            step_preset_path = step.get("cumulative_preset_path")
            if step_preset_path is None:
                if "cumulative_preset" in step:
                    _vital_instance.load_json(json.dumps(step["cumulative_preset"]))
                else:
                    return None
            else:
                apply_preset(_vital_instance, step_preset_path)

            # Note: no audibility gate on iter clips — early steps are intentionally
            # thin (only a few params set), and that's valid training signal.
            iter_audio = _render_note_list(
                _vital_instance, probe_notes, SAMPLE_RATE, tail_s=_probe_tail_s
            )
            iter_audio = trim_silence(iter_audio, SAMPLE_RATE, min_duration_s=0.5)

            iter_wav = audio_dir / f"{sample_id}_iter{step_idx + 1}.wav"
            sf.write(str(iter_wav), iter_audio.T, SAMPLE_RATE, subtype="PCM_24")
            iter_wavs.append(str(iter_wav))

        # ------------------------------------------------------------------
        # 5. Return manifest entry
        # ------------------------------------------------------------------
        entry = {
            "sample_id": sample_id,
            "archetype": archetype,
            "n_iterations": n_iterations,
            "gt_wav": str(gt_wav),
            "gt_probe_wav": str(gt_probe_wav),
            "default_wav": str(default_wav),
            "iter_wavs": iter_wavs,
            "source_midi_path": str(source_midi_path),
        }
        if isinstance(compare, dict):
            entry["final_vs_target_summary"] = compare.get("summary")
            entry["final_vs_target_top_noisy"] = compare.get("noisy", [])[:12]
        elif isinstance(path_data.get("final_vs_target_summary"), str):
            entry["final_vs_target_summary"] = path_data["final_vs_target_summary"]
            entry["final_vs_target_top_noisy"] = path_data.get("final_vs_target_top_noisy", [])
        if path_data.get("target_preset_path"):
            entry["target_preset_path"] = str(path_data["target_preset_path"])
        elif mode == "generate" and paths_dir is not None:
            entry["target_preset_path"] = str(paths_dir / f"{sample_id}_target.vital")
        if mode == "generate" and paths_dir is not None:
            entry["path_file"] = str(paths_dir / f"{sample_id}.json")
        elif mode == "paths_dir":
            entry["path_file"] = str(job["path_file"])
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
    p.add_argument(
        "--generate", type=int, default=None, metavar="N",
        help="Generate N samples on the fly via path_gen",
    )
    p.add_argument(
        "--archetypes", nargs="+", default=["bass"],
        help="Which archetypes to use (round-robin when generating)",
    )
    p.add_argument(
        "--paths-dir", type=Path, default=None,
        help="Read existing path JSONs from this directory",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/iter_sft"),
        help="Where to write audio/ subdir + manifest.jsonl",
    )
    p.add_argument(
        "--wavetable-lib", type=Path, default=None,
        help="Path to wavetable library JSON",
    )
    _default_jobs = min(multiprocessing.cpu_count(), 16)
    p.add_argument(
        "--jobs", type=int, default=_default_jobs,
        help=f"Parallel workers (default: {_default_jobs})",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-render existing files",
    )
    p.add_argument(
        "--clip-duration-s", type=float, default=10.0,
        help="Duration of GT and probe audio clips in seconds (default 10.0, recommend 5.0 "
             "to halve audio token budget for training)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.generate is None and args.paths_dir is None:
        print("ERROR: must specify --generate N or --paths-dir DIR", file=sys.stderr)
        sys.exit(1)

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    output_dir = _resolve(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wavetable_lib_path = (
        str(_resolve(args.wavetable_lib)) if args.wavetable_lib else None
    )

    # ------------------------------------------------------------------
    # Build job list
    # ------------------------------------------------------------------
    jobs: list[dict] = []

    if args.generate is not None:
        paths_dir = output_dir / "paths"
        paths_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)
        archetypes = args.archetypes

        for i in range(args.generate):
            archetype = archetypes[i % len(archetypes)]
            # Deterministic per-sample seed
            sample_seed = rng.randint(0, 2**31)
            sample_id = f"{archetype}_{sample_seed:08x}"

            gt_wav = audio_dir / f"{sample_id}_gt.wav"
            default_wav = audio_dir / f"{sample_id}_default.wav"
            if gt_wav.exists() and default_wav.exists() and not args.overwrite:
                continue

            jobs.append({
                "mode": "generate",
                "sample_id": sample_id,
                "archetype": archetype,
                "audio_dir": str(audio_dir),
                "paths_dir": str(paths_dir),
                "wavetable_lib_path": wavetable_lib_path,
                "seed": sample_seed,
                "clip_duration_s": args.clip_duration_s,
            })
    else:
        paths_dir_in = _resolve(args.paths_dir)
        path_files = sorted(paths_dir_in.glob("*.json"))
        if not path_files:
            print(f"ERROR: no .json files in {paths_dir_in}", file=sys.stderr)
            sys.exit(1)

        for pf in path_files:
            with open(pf) as f:
                meta = json.load(f)
            sample_id = meta["sample_id"]
            archetype = meta["archetype"]

            gt_wav = audio_dir / f"{sample_id}_gt.wav"
            default_wav = audio_dir / f"{sample_id}_default.wav"
            if gt_wav.exists() and default_wav.exists() and not args.overwrite:
                continue

            jobs.append({
                "mode": "paths_dir",
                "sample_id": sample_id,
                "archetype": archetype,
                "audio_dir": str(audio_dir),
                "paths_dir": None,
                "path_file": str(pf),
                "seed": args.seed,
                "clip_duration_s": args.clip_duration_s,
            })

    total = args.generate if args.generate else len(
        sorted(_resolve(args.paths_dir).glob("*.json")) if args.paths_dir else []
    )
    skipped = total - len(jobs)
    if skipped > 0:
        print(f"  {skipped} samples already rendered — skipping (use --overwrite)")
    if not jobs:
        print("Nothing to render.")
        return

    print(f"Rendering {len(jobs)} samples across {args.jobs} workers...")

    # ------------------------------------------------------------------
    # Parallel render
    # ------------------------------------------------------------------
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

    rendered = 0
    failed = 0
    results: list[dict] = []
    start = time.time()
    max_in_flight = args.jobs * 4

    try:
        with ProcessPoolExecutor(
            max_workers=args.jobs, initializer=_worker_init
        ) as executor:
            active: set = set()
            job_iter = iter(jobs)
            done_count = 0

            # Use tqdm if available
            pbar = None
            if tqdm is not None:
                pbar = tqdm(total=len(jobs), desc="Rendering", unit="sample")

            def _drain(fs):
                nonlocal rendered, failed, done_count
                for f in fs:
                    entry = f.result()
                    if entry is not None:
                        results.append(entry)
                        rendered += 1
                    else:
                        failed += 1
                    done_count += 1
                    if pbar is not None:
                        pbar.update(1)
                    elif done_count % 10 == 0 or done_count == len(jobs):
                        elapsed = time.time() - start
                        rate = done_count / elapsed if elapsed > 0 else 0
                        print(
                            f"  [{done_count}/{len(jobs)}] "
                            f"{rate:.1f} samples/s  "
                            f"({rendered} ok, {failed} skipped)"
                        )

            for job in job_iter:
                active.add(executor.submit(_render_sample, job))
                if len(active) >= max_in_flight:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    _drain(done)

            # Drain remaining
            _drain(as_completed(active))

            if pbar is not None:
                pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    rate = rendered / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s  ({rate:.1f} samples/s)")
    print(f"  Rendered: {rendered}")
    print(f"  Skipped:  {failed}")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"  Manifest: {manifest_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
