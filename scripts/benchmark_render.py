#!/usr/bin/env python3
"""
Throughput benchmark for the gen-preset render pipeline.

Runs timed batches at varying worker counts and extrapolates to 1M renders.

Usage:
    python scripts/benchmark_render.py \\
        --midi-dir data/raw/phase1_sources/lakh_midi/lmd_full \\
        --workers 1 2 4 8 \\
        --n-per-config 20
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_batch(
    n: int,
    workers: int,
    midi_dir: Path,
    wavetable_lib: Path,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run a render batch and return timing stats."""
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "render_vital_wavs.py"),
        "--midi-dir", str(midi_dir),
        "--num-wavs", str(n),
        "--gen-preset", "random",
        "--gen-preset-wavetable-lib", str(wavetable_lib),
        "--output-dir", str(output_dir),
        "--jobs", str(workers),
        "--seed", str(seed),
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # Count actual rendered files
    wavs = list((output_dir / "wavs").glob("*.wav")) if (output_dir / "wavs").exists() else []
    rendered = len(wavs)

    # Parse rate from stdout if available
    rate = rendered / elapsed if elapsed > 0 else 0.0

    return {
        "workers": workers,
        "n_rendered": rendered,
        "elapsed_s": round(elapsed, 2),
        "rate_wav_per_s": round(rate, 3),
        "stdout_tail": result.stdout.strip().split("\n")[-3:] if result.stdout else [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--midi-dir", type=Path,
                   default=Path("data/raw/phase1_sources/lakh_midi/lmd_full"))
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n-per-config", type=int, default=20)
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark_results.json"))
    p.add_argument("--target-renders", type=int, default=1_000_000,
                   help="Target scale for extrapolation (default: 1,000,000)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    midi_dir = args.midi_dir if args.midi_dir.is_absolute() else repo_root / args.midi_dir
    wavetable_lib = args.wavetable_lib if args.wavetable_lib.is_absolute() else repo_root / args.wavetable_lib

    if not midi_dir.exists():
        print(f"ERROR: MIDI dir not found: {midi_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking gen-preset render pipeline")
    print(f"  n per config : {args.n_per_config}")
    print(f"  worker counts: {args.workers}")
    print(f"  target scale : {args.target_renders:,} renders")
    print()

    results = []
    for workers in args.workers:
        print(f"Testing {workers} worker(s)...", end=" ", flush=True)
        tmp_dir = repo_root / f"outputs/_benchmark_tmp_w{workers}"
        stats = run_batch(
            n=args.n_per_config,
            workers=workers,
            midi_dir=midi_dir,
            wavetable_lib=wavetable_lib,
            seed=42 + workers,
            output_dir=tmp_dir,
        )
        results.append(stats)

        rate = stats["rate_wav_per_s"]
        if rate > 0:
            total_s = args.target_renders / rate
            hours = total_s / 3600
            days = hours / 24
            if days >= 1:
                eta_str = f"{days:.1f} days"
            elif hours >= 1:
                eta_str = f"{hours:.1f} hours"
            else:
                eta_str = f"{total_s/60:.0f} min"
            print(f"{stats['n_rendered']}/{args.n_per_config} rendered in "
                  f"{stats['elapsed_s']:.1f}s → {rate:.2f} WAV/s  "
                  f"[ETA for {args.target_renders:,}: {eta_str}]")
        else:
            print(f"FAILED (0 rendered)")

        # Clean up temp dir
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    # Best rate (max workers)
    best = max(results, key=lambda r: r["rate_wav_per_s"])
    best_rate = best["rate_wav_per_s"]
    if best_rate > 0:
        best_eta_s = args.target_renders / best_rate
        best_eta_h = best_eta_s / 3600
        print(f"\nBest throughput: {best_rate:.2f} WAV/s at {best['workers']} workers")
        print(f"Estimated time for {args.target_renders:,} renders: "
              f"{best_eta_h:.1f} hours ({best_eta_h/24:.1f} days)")

    report = {
        "n_per_config": args.n_per_config,
        "target_renders": args.target_renders,
        "results": results,
        "best_rate_wav_per_s": best_rate,
        "best_workers": best["workers"],
        "estimated_hours_at_best_rate": round(args.target_renders / best_rate / 3600, 1) if best_rate > 0 else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
