"""Proof-of-concept: batch-render MIDI through Vital via DawDreamer.

Demonstrates:
  1. Loading Vital VST3 in DawDreamer
  2. Applying a preset (.vital JSON file)
  3. Rendering MIDI notes to audio
  4. Parallel rendering via multiprocessing (N workers, each with own engine)
  5. Swapping presets between renders (same MIDI, different sounds)

Usage:
    python scripts/poc_dawdreamer_vital.py
    python scripts/poc_dawdreamer_vital.py --workers 4 --presets 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

VITAL_VST3 = os.path.expanduser("~/.vst3/Vital.vst3")
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
OUTPUT_DIR = Path("/tmp/dawdreamer_poc")

# Simple MIDI pattern: C major arpeggios across 2 octaves (10s)
MIDI_NOTES = [
    # (pitch, velocity, start_sec, duration_sec)
    (48, 100, 0.0, 2.0),   # C3
    (52, 90,  0.5, 1.5),   # E3
    (55, 85,  1.0, 1.5),   # G3
    (60, 100, 2.0, 2.0),   # C4
    (64, 90,  2.5, 1.5),   # E4
    (67, 85,  3.0, 1.5),   # G4
    (72, 100, 4.0, 2.0),   # C5
    (76, 90,  4.5, 1.5),   # E5
    (79, 85,  5.0, 1.5),   # G5
    (60, 100, 6.0, 2.5),   # C4 chord
    (64, 90,  6.0, 2.5),   # E4
    (67, 85,  6.0, 2.5),   # G4
    (72, 80,  6.0, 2.5),   # C5
]


def _apply_preset(synth, preset_path: str | None, preset_json: dict | None) -> str:
    """Apply a preset to a DawDreamer PluginProcessor.

    Vital's .vital files are JSON and both load_preset() and load_state()
    accept them.  For in-memory dicts, write a temp file and load_state().
    Returns the preset name for metadata.
    """
    import tempfile

    if preset_path:
        synth.load_state(preset_path)
        return Path(preset_path).stem

    if preset_json is not None:
        with tempfile.NamedTemporaryFile(suffix=".vital", mode="w", delete=False) as f:
            json.dump(preset_json, f, separators=(",", ":"))
            tmp = f.name
        try:
            synth.load_state(tmp)
        finally:
            os.unlink(tmp)
        return preset_json.get("preset_name", "generated")

    return "init"


def render_single(
    preset_path: str | None,
    output_path: str,
    *,
    preset_json: dict | None = None,
    midi_notes: list[tuple] = MIDI_NOTES,
    render_duration: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Render MIDI through Vital with given preset. Returns metadata dict."""
    import dawdreamer as daw

    t0 = time.perf_counter()

    engine = daw.RenderEngine(sample_rate, BLOCK_SIZE)
    synth = engine.make_plugin_processor("vital", VITAL_VST3)

    preset_applied = _apply_preset(synth, preset_path, preset_json)

    # Add MIDI notes
    for pitch, vel, start, dur in midi_notes:
        synth.add_midi_note(pitch, vel, start, dur)

    # Build graph: synth -> output
    graph = [
        (synth, []),
    ]
    engine.load_graph(graph)

    # Render
    engine.render(render_duration)
    audio = synth.get_audio()  # shape: (channels, samples)

    t_render = time.perf_counter() - t0

    # Check audio stats
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Write WAV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.T, sample_rate)

    return {
        "output_path": output_path,
        "preset": preset_applied,
        "channels": audio.shape[0],
        "samples": audio.shape[1],
        "duration_s": audio.shape[1] / sample_rate,
        "peak": peak,
        "rms": rms,
        "render_time_s": round(t_render, 3),
        "has_audio": peak > 0.001,
    }


def _worker_render(args: tuple) -> dict:
    """Multiprocessing target — each worker creates its own engine."""
    preset_path, output_path, preset_json_str = args
    preset_json = json.loads(preset_json_str) if preset_json_str else None
    try:
        return render_single(
            preset_path, output_path,
            preset_json=preset_json,
        )
    except Exception as e:
        return {"error": str(e), "preset_path": preset_path, "output_path": output_path}


def render_batch_parallel(
    jobs: list[tuple[str | None, str, dict | None]],
    workers: int = 4,
) -> list[dict]:
    """Render multiple presets in parallel.

    jobs: list of (preset_path, output_path, preset_json_or_None)
    """
    # Serialize preset_json for pickling
    serialized = [
        (p, o, json.dumps(j) if j else None)
        for p, o, j in jobs
    ]

    # Must use 'spawn' — JUCE plugin state doesn't survive fork()
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker_render, serialized)
    wall_time = time.perf_counter() - t0

    succeeded = sum(1 for r in results if "error" not in r and r.get("has_audio"))
    total_render = sum(r.get("render_time_s", 0) for r in results if "error" not in r)

    print(f"\n--- Batch results ({workers} workers) ---")
    print(f"  Jobs: {len(jobs)}")
    print(f"  Succeeded (with audio): {succeeded}/{len(jobs)}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Total render time: {total_render:.2f}s")
    print(f"  Speedup: {total_render / wall_time:.1f}x" if wall_time > 0 else "")
    return results


def main():
    parser = argparse.ArgumentParser(description="DawDreamer + Vital PoC")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--presets", type=int, default=8, help="Number of different presets to render")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--preset-dir", type=str, default=os.path.expanduser("~/Downloads/vital_presets"),
                        help="Directory with .vital preset files for batch test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Test 1: Single render with init preset ---
    print("=== Test 1: Single render (init preset) ===")
    result = render_single(
        None,
        str(output_dir / "test_init.wav"),
    )
    print(f"  Peak: {result['peak']:.4f}  RMS: {result['rms']:.4f}  "
          f"Duration: {result['duration_s']:.1f}s  Time: {result['render_time_s']:.3f}s  "
          f"Audio: {'YES' if result['has_audio'] else 'NO (silent)'}")

    # --- Test 2: Single render with a real .vital preset ---
    preset_dir = Path(args.preset_dir)
    vital_files = sorted(preset_dir.glob("*.vital"))[:args.presets] if preset_dir.exists() else []

    if vital_files:
        print(f"\n=== Test 2: Single render ({vital_files[0].stem}) ===")
        result = render_single(
            str(vital_files[0]),
            str(output_dir / f"test_{vital_files[0].stem}.wav"),
        )
        print(f"  Preset: {result['preset']}")
        print(f"  Peak: {result['peak']:.4f}  RMS: {result['rms']:.4f}  "
              f"Duration: {result['duration_s']:.1f}s  Time: {result['render_time_s']:.3f}s  "
              f"Audio: {'YES' if result['has_audio'] else 'NO (silent)'}")
    else:
        print("\n=== Test 2: Skipped (no .vital files found) ===")

    # --- Test 3: Single render with a generated preset (from preset_gen) ---
    print(f"\n=== Test 3: Single render (generated preset) ===")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from maestro.synth.preset_gen import generate_preset
        from maestro.synth.wavetable_lib import load_wavetable_lib
        import random

        wt_lib_path = Path(__file__).resolve().parent.parent / "data" / "wavetable_lib.json"
        wt_lib = load_wavetable_lib(wt_lib_path) if wt_lib_path.exists() else None
        rng = random.Random(42)

        preset_json = generate_preset("pad", rng, wavetable_lib=wt_lib)
        result = render_single(
            None,
            str(output_dir / "test_generated_pad.wav"),
            preset_json=preset_json,
        )
        print(f"  Preset: {result['preset']}")
        print(f"  Peak: {result['peak']:.4f}  RMS: {result['rms']:.4f}  "
              f"Duration: {result['duration_s']:.1f}s  Time: {result['render_time_s']:.3f}s  "
              f"Audio: {'YES' if result['has_audio'] else 'NO (silent)'}")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Test 4: Parallel batch render ---
    if vital_files:
        print(f"\n=== Test 4: Parallel batch ({len(vital_files)} presets, {args.workers} workers) ===")
        jobs = [
            (str(vf), str(output_dir / f"batch_{vf.stem}.wav"), None)
            for vf in vital_files
        ]
        results = render_batch_parallel(jobs, workers=args.workers)
        for r in results:
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  {r['preset']:40s}  peak={r['peak']:.4f}  rms={r['rms']:.4f}  "
                      f"time={r['render_time_s']:.3f}s  audio={'YES' if r['has_audio'] else 'NO'}")

    # --- Test 5: Same MIDI, many generated presets in parallel ---
    print(f"\n=== Test 5: Parallel generated presets ({args.presets} presets, {args.workers} workers) ===")
    try:
        from maestro.synth.preset_gen import generate_preset
        from maestro.synth.wavetable_lib import load_wavetable_lib

        wt_lib_path = Path(__file__).resolve().parent.parent / "data" / "wavetable_lib.json"
        wt_lib = load_wavetable_lib(wt_lib_path) if wt_lib_path.exists() else None

        archetypes = ["bass", "lead", "pad", "keys", "pluck", "sequence"]
        gen_jobs = []
        for i in range(args.presets):
            rng = random.Random(1000 + i)
            arch = archetypes[i % len(archetypes)]
            preset = generate_preset(arch, rng, wavetable_lib=wt_lib)
            gen_jobs.append((
                None,
                str(output_dir / f"gen_{arch}_{i:03d}.wav"),
                preset,
            ))

        results = render_batch_parallel(gen_jobs, workers=args.workers)
        for r in results:
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                tag = "YES" if r["has_audio"] else "NO"
                print(f"  {Path(r['output_path']).stem:40s}  peak={r['peak']:.4f}  "
                      f"rms={r['rms']:.4f}  time={r['render_time_s']:.3f}s  audio={tag}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
