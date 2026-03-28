#!/usr/bin/env python3
"""
Filter quiet/no-audio files from generated REAPER tuples.

The script computes lightweight amplitude heuristics for each audio file:
  - peak amplitude
  - RMS amplitude
  - 95th percentile absolute amplitude
  - active frame ratio above an amplitude floor

Thresholds are learned from a reference set of known-good audio.
If no reference set is provided, the current tuple audio set is used.

Examples:
    # Print a few quiet candidates for manual check (no deletion)
    python scripts/filter_quiet_reaper_tuples.py --tuples-dir data/processed/reaper_tuples --smoke-test

    # Learn thresholds from a known-good manifest and delete quiet audio files
    python scripts/filter_quiet_reaper_tuples.py \
        --tuples-dir data/processed/reaper_tuples \
        --reference-manifest data/processed/reaper_tuples_known_good/manifest.jsonl \
        --delete
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_AUDIO_GLOBS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.opus", "*.m4a", "*.aac")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tuples-dir",
        type=Path,
        default=Path("data/processed/reaper_tuples"),
        help="Tuple directory containing manifest.jsonl and wavs/ (default: data/processed/reaper_tuples)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path to read target audio paths from (default: <tuples-dir>/manifest.jsonl if present)",
    )
    p.add_argument(
        "--reference-manifest",
        type=Path,
        action="append",
        default=[],
        help="Known-good manifest(s) used to learn thresholds (repeatable).",
    )
    p.add_argument(
        "--reference-wav",
        type=Path,
        action="append",
        default=[],
        help="Known-good audio file or directory (repeatable). Directories are scanned recursively for common audio extensions.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 1), 32)),
        help="Parallel workers for audio scanning.",
    )
    p.add_argument(
        "--max-target-files",
        type=int,
        default=None,
        help="If set, randomly sample at most this many target WAVs before scanning.",
    )
    p.add_argument(
        "--max-reference-files",
        type=int,
        default=None,
        help="If set, randomly sample at most this many reference WAVs before scanning.",
    )
    p.add_argument(
        "--activity-floor",
        type=float,
        default=1e-3,
        help="Amplitude floor for active frame ratio (default: 1e-3, approx -60 dBFS).",
    )
    p.add_argument(
        "--low-quantile",
        type=float,
        default=0.01,
        help="Lower quantile from reference metrics used as cutoff base (default: 0.01).",
    )
    p.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Multiply learned thresholds by this value. >1.0 is stricter, <1.0 is looser.",
    )
    p.add_argument(
        "--min-peak",
        type=float,
        default=1e-4,
        help="Hard lower bound for learned peak threshold.",
    )
    p.add_argument(
        "--min-rms",
        type=float,
        default=5e-5,
        help="Hard lower bound for learned RMS threshold.",
    )
    p.add_argument(
        "--min-p95",
        type=float,
        default=7e-5,
        help="Hard lower bound for learned p95 threshold.",
    )
    p.add_argument(
        "--min-active-ratio",
        type=float,
        default=5e-4,
        help="Hard lower bound for learned active-ratio threshold.",
    )
    p.add_argument(
        "--quiet-list-out",
        type=Path,
        default=None,
        help="Optional output path to write all quiet audio paths.",
    )
    p.add_argument(
        "--smoke-count",
        type=int,
        default=20,
        help="Number of quiet paths to print in smoke-test mode.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed for smoke-test output.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--delete",
        action="store_true",
        help="Delete quiet audio files.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only print sample quiet candidates for manual inspection.",
    )
    return p.parse_args()


def _resolve_candidate_path(path_str: str, manifest_dir: Path, tuples_dir: Path) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    candidates = [raw, manifest_dir / raw, tuples_dir / raw]
    for c in candidates:
        if c.exists():
            return c
    return raw


def _read_manifest_wavs(manifest_path: Path, tuples_dir: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    wavs: list[Path] = []
    manifest_dir = manifest_path.parent
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            wav = entry.get("wav")
            if not wav:
                continue
            if entry.get("wav_ok") is False:
                continue
            wavs.append(_resolve_candidate_path(str(wav), manifest_dir, tuples_dir))
    return wavs


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _sample_paths(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    if limit is None or limit <= 0 or len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, limit)


def collect_target_wavs(tuples_dir: Path, manifest_path: Path | None) -> list[Path]:
    if manifest_path is not None and manifest_path.exists():
        manifest_wavs = _read_manifest_wavs(manifest_path, tuples_dir)
        if manifest_wavs:
            return _dedupe_paths(manifest_wavs)
    wavs_dir = tuples_dir / "wavs"
    if not wavs_dir.exists():
        return []
    paths: list[Path] = []
    for pat in _AUDIO_GLOBS:
        paths.extend(wavs_dir.rglob(pat))
    return sorted(paths)


def collect_reference_wavs(args: argparse.Namespace, tuples_dir: Path) -> list[Path]:
    ref_paths: list[Path] = []
    for manifest in args.reference_manifest:
        ref_paths.extend(_read_manifest_wavs(manifest, tuples_dir))
    for p in args.reference_wav:
        if p.is_dir():
            for pat in _AUDIO_GLOBS:
                ref_paths.extend(sorted(p.rglob(pat)))
        elif p.is_file():
            ref_paths.append(p)
    return _dedupe_paths(ref_paths)


def _scan_wav_metrics(job: tuple[str, float]) -> tuple[str, dict | None, str | None]:
    path_str, activity_floor = job
    path = Path(path_str)
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return path_str, None, str(e)

    if sr <= 0:
        return path_str, None, "invalid_sample_rate"

    frames = int(audio.shape[0])
    if frames == 0:
        return path_str, {
            "peak": 0.0,
            "rms": 0.0,
            "p95_abs": 0.0,
            "active_ratio": 0.0,
            "duration_s": 0.0,
        }, None

    abs_audio = np.abs(audio)
    frame_abs = np.max(abs_audio, axis=1)
    peak = float(frame_abs.max())
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64), dtype=np.float64)))
    p95_abs = float(np.percentile(frame_abs, 95))
    active_ratio = float(np.mean(frame_abs >= activity_floor))
    duration_s = float(frames / sr)
    return path_str, {
        "peak": peak,
        "rms": rms,
        "p95_abs": p95_abs,
        "active_ratio": active_ratio,
        "duration_s": duration_s,
    }, None


def scan_wavs(paths: list[Path], activity_floor: float, workers: int) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    metrics_by_path: dict[str, dict] = {}
    errors: list[tuple[str, str]] = []
    if not paths:
        return metrics_by_path, errors

    total = len(paths)
    start = time.time()
    max_in_flight = max(8, workers * 8)
    done_count = 0

    def _progress() -> None:
        elapsed = time.time() - start
        rate = done_count / elapsed if elapsed > 0 else 0.0
        print(f"  [{done_count:,}/{total:,}] scanned ({rate:.1f} files/s)", flush=True)

    progress = None
    if tqdm is not None:
        progress = tqdm(total=total, desc="Scanning audio", unit="file", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        active: set = set()
        idx = 0

        while idx < total and len(active) < max_in_flight:
            p = paths[idx]
            active.add(ex.submit(_scan_wav_metrics, (str(p), activity_floor)))
            idx += 1

        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            batch_n = len(done)
            for future in done:
                path_str, metrics, err = future.result()
                done_count += 1
                if err is None and metrics is not None:
                    metrics_by_path[path_str] = metrics
                else:
                    errors.append((path_str, err or "unknown_error"))
            if progress is not None and batch_n:
                progress.update(batch_n)

            while idx < total and len(active) < max_in_flight:
                p = paths[idx]
                active.add(ex.submit(_scan_wav_metrics, (str(p), activity_floor)))
                idx += 1

            if progress is None and (done_count % 1000 == 0 or done_count == total):
                _progress()

    if progress is not None:
        progress.close()

    return metrics_by_path, errors


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def infer_thresholds(ref_metrics: list[dict], args: argparse.Namespace) -> dict[str, float]:
    peaks = np.array([m["peak"] for m in ref_metrics], dtype=np.float64)
    rms = np.array([m["rms"] for m in ref_metrics], dtype=np.float64)
    p95 = np.array([m["p95_abs"] for m in ref_metrics], dtype=np.float64)
    active = np.array([m["active_ratio"] for m in ref_metrics], dtype=np.float64)

    q = max(0.0, min(0.5, float(args.low_quantile)))
    scale = max(1e-6, float(args.threshold_scale))

    t_peak = max(args.min_peak, _quantile(peaks, q) * scale)
    t_rms = max(args.min_rms, _quantile(rms, q) * scale)
    t_p95 = max(args.min_p95, _quantile(p95, q) * scale)
    t_active = max(args.min_active_ratio, _quantile(active, q) * scale)

    return {
        "peak": t_peak,
        "rms": t_rms,
        "p95_abs": t_p95,
        "active_ratio": t_active,
    }


def is_quiet(metrics: dict, thresholds: dict[str, float]) -> bool:
    primary = (
        metrics["rms"] < thresholds["rms"]
        and metrics["p95_abs"] < thresholds["p95_abs"]
        and metrics["active_ratio"] < thresholds["active_ratio"]
    )
    near_silent_with_spike = (
        metrics["peak"] < thresholds["peak"]
        and metrics["rms"] < (thresholds["rms"] * 1.5)
    )
    return primary or near_silent_with_spike


def fmt_dbfs(x: float) -> str:
    if x <= 0:
        return "-inf"
    return f"{20.0 * math.log10(x):.1f}"


def main() -> None:
    args = parse_args()
    tuples_dir = args.tuples_dir
    manifest_path = args.manifest if args.manifest is not None else (tuples_dir / "manifest.jsonl")

    targets = collect_target_wavs(tuples_dir, manifest_path)
    if not targets:
        print("No target audio files found. Check --tuples-dir/--manifest.", file=sys.stderr)
        sys.exit(1)
    if tqdm is None:
        print("NOTE: tqdm is not installed; using periodic text progress updates.")
    targets = _sample_paths(targets, args.max_target_files, args.seed)

    print(f"Scanning target audio files: {len(targets):,}")
    target_metrics, target_errors = scan_wavs(targets, args.activity_floor, args.workers)
    if not target_metrics:
        print("Failed to scan target audio files.", file=sys.stderr)
        if target_errors:
            print(f"Sample error: {target_errors[0][0]} -> {target_errors[0][1]}", file=sys.stderr)
        sys.exit(1)

    ref_paths = collect_reference_wavs(args, tuples_dir)
    ref_paths = _sample_paths(ref_paths, args.max_reference_files, args.seed + 1)
    if ref_paths:
        print(f"Scanning reference audio files: {len(ref_paths):,}")
        ref_metrics_by_path, ref_errors = scan_wavs(ref_paths, args.activity_floor, args.workers)
        ref_metrics = list(ref_metrics_by_path.values())
        if not ref_metrics:
            print("Reference audio scan produced no metrics; cannot infer thresholds.", file=sys.stderr)
            sys.exit(1)
    else:
        ref_metrics = list(target_metrics.values())
        ref_errors = []
        print("No reference audio set provided; using target corpus to infer thresholds.")

    thresholds = infer_thresholds(ref_metrics, args)
    print("Learned thresholds (linear / dBFS):")
    print(f"  peak        : {thresholds['peak']:.7f} ({fmt_dbfs(thresholds['peak'])} dBFS)")
    print(f"  rms         : {thresholds['rms']:.7f} ({fmt_dbfs(thresholds['rms'])} dBFS)")
    print(f"  p95_abs     : {thresholds['p95_abs']:.7f} ({fmt_dbfs(thresholds['p95_abs'])} dBFS)")
    print(f"  active_ratio: {thresholds['active_ratio']:.7f}")

    quiet_paths: list[Path] = []
    for path_str, metrics in target_metrics.items():
        if is_quiet(metrics, thresholds):
            quiet_paths.append(Path(path_str))

    quiet_paths = sorted(quiet_paths)
    quiet_count = len(quiet_paths)
    total_scanned = len(target_metrics)
    quiet_ratio = quiet_count / total_scanned if total_scanned else 0.0

    print("\nResults:")
    print(f"  Target files scanned : {total_scanned:,}")
    print(f"  Quiet candidates     : {quiet_count:,} ({quiet_ratio:.2%})")
    if target_errors:
        print(f"  Target scan errors   : {len(target_errors):,}")
    if ref_errors:
        print(f"  Reference scan errors: {len(ref_errors):,}")

    if args.quiet_list_out:
        args.quiet_list_out.parent.mkdir(parents=True, exist_ok=True)
        with args.quiet_list_out.open("w") as f:
            for p in quiet_paths:
                f.write(str(p) + "\n")
        print(f"  Wrote quiet list     : {args.quiet_list_out}")

    if args.smoke_test:
        print(f"\nSmoke test sample ({min(args.smoke_count, quiet_count)} paths):")
        if quiet_count == 0:
            print("  (none)")
            return
        rng = random.Random(args.seed)
        sample = quiet_paths if quiet_count <= args.smoke_count else rng.sample(quiet_paths, args.smoke_count)
        for p in sorted(sample):
            print(str(p))
        return

    if args.delete:
        deleted = 0
        missing = 0
        failed = 0
        delete_progress = None
        if tqdm is not None:
            delete_progress = tqdm(quiet_paths, desc="Deleting quiet audio", unit="file", dynamic_ncols=True)
            delete_iter = delete_progress
        else:
            delete_iter = quiet_paths

        for i, p in enumerate(delete_iter, start=1):
            try:
                p.unlink()
                deleted += 1
            except FileNotFoundError:
                missing += 1
            except Exception:
                failed += 1
            if tqdm is None and (i % 10000 == 0 or i == len(quiet_paths)):
                print(f"  [{i:,}/{len(quiet_paths):,}] deleted pass")
        if delete_progress is not None:
            delete_progress.close()
        print("\nDelete summary:")
        print(f"  Deleted: {deleted:,}")
        print(f"  Missing: {missing:,}")
        print(f"  Failed : {failed:,}")
    else:
        print("\nNo files deleted. Use --delete to remove quiet WAVs.")


if __name__ == "__main__":
    main()
