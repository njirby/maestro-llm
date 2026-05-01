#!/usr/bin/env python3
"""Production batch runner for 16K unified SFT rollouts.

Wraps build_unified_sft_v4.build_record() with:
  - Batch processing (default 200 per batch, ~85 batches for 17K)
  - Streaming JSONL output (one record at a time, append mode)
  - Resume via .done sentinel files and per-sample dedup
  - RMS quality gate on GT audio before rollout generation
  - Rich per-record metadata
  - Progress tracking and ETA

Also supports --merge mode to concatenate batch outputs into final files.

Usage:
    # Build rollouts (resumable)
    python scripts/run_sft_production.py \
        --manifest outputs/sft_16k/manifest.jsonl \
        --out-dir outputs/sft_16k/rollouts \
        --batch-size 200 --workers 24 \
        --omni-server http://localhost:8000 \
        --clap-device cpu --seed 1337 --suffix v1 --resume

    # Merge batches into final files
    python scripts/run_sft_production.py \
        --merge --out-dir outputs/sft_16k/rollouts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

MIN_GT_RMS = 0.001


def _check_gt_audio(gt_wav: str) -> tuple[bool, str]:
    """Validate GT audio. Returns (pass, reason)."""
    p = Path(gt_wav)
    if not p.exists():
        return False, "gt_wav missing"
    if p.stat().st_size < 1024:
        return False, "gt_wav too small"
    try:
        audio, sr = sf.read(str(p))
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
        else:
            mono = audio
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms < MIN_GT_RMS:
            return False, f"gt_wav near-silent (rms={rms:.6f})"
    except Exception as e:
        return False, f"gt_wav read error: {e}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer
# ---------------------------------------------------------------------------

class StreamingJsonlWriter:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line)

    def read_existing_ids(self, id_key: str = "id") -> set[str]:
        ids: set[str] = set()
        if not self._path.exists():
            return ids
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if id_key in rec:
                        ids.add(str(rec[id_key]))
                except json.JSONDecodeError:
                    continue
        return ids


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _build_batches(entries: list[dict], batch_size: int) -> list[list[dict]]:
    batches = []
    for i in range(0, len(entries), batch_size):
        batches.append(entries[i:i + batch_size])
    return batches


def _batch_dir(out_dir: Path, batch_idx: int) -> Path:
    return out_dir / f"batch_{batch_idx:04d}"


def _is_batch_done(out_dir: Path, batch_idx: int) -> bool:
    return (_batch_dir(out_dir, batch_idx) / ".done").exists()


def _mark_batch_done(out_dir: Path, batch_idx: int, stats: dict) -> None:
    sentinel = _batch_dir(out_dir, batch_idx) / ".done"
    with open(sentinel, "w") as f:
        json.dump(stats, f)


def run_batch(
    batch_idx: int,
    batch_entries: list[dict],
    out_dir: Path,
    args: argparse.Namespace,
    ctx: dict,
) -> dict:
    """Process one batch of entries. Returns stats dict."""
    from scripts.build_unified_sft_v4 import build_record
    from scripts.build_main_agent_sft_v2 import _check_server_reachable, llm_post_stats

    bdir = _batch_dir(out_dir, batch_idx)
    bdir.mkdir(parents=True, exist_ok=True)

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    main_writer = StreamingJsonlWriter(bdir / "main.jsonl")
    search_writer = StreamingJsonlWriter(bdir / "search.jsonl")
    judge_writer = StreamingJsonlWriter(bdir / "judge.jsonl")
    trans_writer = StreamingJsonlWriter(bdir / "transcription.jsonl")
    progress_writer = StreamingJsonlWriter(bdir / "progress.jsonl")
    rejected_writer = StreamingJsonlWriter(bdir / "rejected.jsonl")

    already_done = main_writer.read_existing_ids("id")

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()

    entries_to_process = []
    skipped_resume = 0
    skipped_quality = 0

    for entry in batch_entries:
        sid = str(entry["sample_id"])
        if sid in already_done:
            skipped_resume += 1
            continue
        gt_ok, gt_reason = _check_gt_audio(entry.get("gt_wav", ""))
        if not gt_ok:
            rejected_writer.append({"sample_id": sid, "reason": gt_reason, "batch_idx": batch_idx})
            skipped_quality += 1
            continue
        entries_to_process.append(entry)

    if skipped_resume:
        print(f"  batch {batch_idx:04d}: {skipped_resume} already done (resume)", flush=True)
    if skipped_quality:
        print(f"  batch {batch_idx:04d}: {skipped_quality} rejected (quality gate)", flush=True)

    rendered = 0
    failed = 0
    batch_t0 = time.monotonic()

    def _process_one(entry: dict) -> tuple[str, dict | None, list, list, list]:
        sid = str(entry["sample_id"])
        try:
            main_rec, srecs, jrecs, trecs = build_record(
                entry=entry,
                args=args,
                embedder=ctx["embedder"],
                shortlist_data=ctx["shortlist_data"],
                selected_by_name=ctx["selected_by_name"],
                wavetable_lib=ctx["wavetable_lib"],
                index_rows=ctx["index_rows"],
                candidate_audio=candidate_audio,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                serial_lock=serial_lock,
                notes=ctx["notes"],
            )
            return sid, main_rec, srecs, jrecs, trecs
        except Exception as exc:
            import traceback
            print(f"  ERROR [{sid}]: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return sid, None, [], [], []

    def _write_result(sid: str, main_rec, srecs, jrecs, trecs):
        nonlocal rendered, failed
        if main_rec is None:
            failed += 1
            progress_writer.append({
                "sample_id": sid, "status": "failed",
                "batch_idx": batch_idx, "wall_time_s": 0,
            })
            return
        main_writer.append(main_rec)
        for r in srecs:
            search_writer.append(r)
        for r in jrecs:
            judge_writer.append(r)
        for r in trecs:
            trans_writer.append(r)
        wall = main_rec.get("meta", {}).get("wall_time_s", 0)
        progress_writer.append({
            "sample_id": sid, "status": "ok",
            "batch_idx": batch_idx, "wall_time_s": wall,
            "n_turns": main_rec.get("meta", {}).get("n_turns", 0),
            "n_search_rounds": main_rec.get("meta", {}).get("n_search_rounds", 0),
        })
        rendered += 1

    if args.workers > 1 and len(entries_to_process) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_one, e): e for e in entries_to_process}
            for fut in as_completed(futs):
                sid, main_rec, srecs, jrecs, trecs = fut.result()
                _write_result(sid, main_rec, srecs, jrecs, trecs)
                total = rendered + failed
                if total % 10 == 0 or total == len(entries_to_process):
                    elapsed = time.monotonic() - batch_t0
                    rate = total / elapsed if elapsed > 0 else 0
                    print(
                        f"  batch {batch_idx:04d}: [{total}/{len(entries_to_process)}] "
                        f"{rate:.2f}/s ({rendered} ok, {failed} fail)",
                        flush=True,
                    )
    else:
        for entry in entries_to_process:
            sid, main_rec, srecs, jrecs, trecs = _process_one(entry)
            _write_result(sid, main_rec, srecs, jrecs, trecs)

    batch_elapsed = time.monotonic() - batch_t0
    stats = {
        "batch_idx": batch_idx,
        "total_entries": len(batch_entries),
        "processed": len(entries_to_process),
        "rendered": rendered,
        "failed": failed,
        "skipped_resume": skipped_resume,
        "skipped_quality": skipped_quality,
        "elapsed_s": round(batch_elapsed, 1),
    }
    return stats


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_batches(out_dir: Path, suffix: str = "v1") -> None:
    """Concatenate all batch outputs into final JSONL files."""
    batch_dirs = sorted(out_dir.glob("batch_*"))
    if not batch_dirs:
        print("No batch directories found.", file=sys.stderr)
        return

    agent_names = ["main", "search", "judge", "transcription"]
    counts: dict[str, int] = {}

    for agent in agent_names:
        out_path = out_dir / f"{agent}_final_{suffix}.jsonl"
        n = 0
        with open(out_path, "w") as out_f:
            for bd in batch_dirs:
                src = bd / f"{agent}.jsonl"
                if not src.exists():
                    continue
                with open(src) as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            n += 1
        counts[agent] = n
        print(f"  {agent}: {n} records -> {out_path}")

    progress_path = out_dir / f"progress_{suffix}.jsonl"
    rejected_path = out_dir / f"rejected_{suffix}.jsonl"
    for name, filename in [("progress", "progress.jsonl"), ("rejected", "rejected.jsonl")]:
        merged = out_dir / f"{name}_{suffix}.jsonl"
        n = 0
        with open(merged, "w") as out_f:
            for bd in batch_dirs:
                src = bd / filename
                if not src.exists():
                    continue
                with open(src) as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            n += 1
        print(f"  {name}: {n} entries -> {merged}")

    # Summary stats
    total_main = counts.get("main", 0)
    done_batches = sum(1 for bd in batch_dirs if (bd / ".done").exists())
    print(f"\nMerge complete: {total_main} main records from {done_batches}/{len(batch_dirs)} completed batches")

    if total_main > 0:
        archetypes = Counter()
        first_file = out_dir / f"main_final_{suffix}.jsonl"
        with open(first_file) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    arch = rec.get("meta", {}).get("archetype", "unknown")
                    archetypes[arch] += 1
                except (json.JSONDecodeError, KeyError):
                    pass
        print("\nArchetype distribution:")
        for arch in sorted(archetypes):
            print(f"  {arch}: {archetypes[arch]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode
    p.add_argument("--merge", action="store_true",
                   help="Merge batch outputs into final files (no building)")

    # Infrastructure
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--resume", action="store_true",
                   help="Skip completed batches and already-written samples")
    p.add_argument("--suffix", default="v1")

    # Index / CLAP
    p.add_argument("--index-npy", type=Path,
                   default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    p.add_argument("--index-meta", type=Path,
                   default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--probe-dir", type=Path,
                   default=Path("outputs/agent_sft/candidate_probes"))
    p.add_argument("--clap-device", default="cpu")

    # Builder params (passed through to build_record via args namespace)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--pool-top-k", type=int, default=48)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--candidates-per-slice", type=int, default=48)
    p.add_argument("--candidates-per-batch", type=int, default=8)
    p.add_argument("--max-search-rounds", type=int, default=3)
    p.add_argument("--force-research-rate", type=float, default=0.20)
    p.add_argument("--no-audio-rate", type=float, default=0.05)
    p.add_argument("--per-param-mistake-rate", type=float, default=0.10)
    p.add_argument("--max-correction-turns", type=int, default=3)
    p.add_argument("--transcription-mistake-rate", type=float, default=0.15)
    p.add_argument("--random-init-rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)

    # LLM servers
    p.add_argument("--omni-server", default="")
    p.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--stage2-server", default="")
    p.add_argument("--stage2-model", default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge:
        print(f"Merging batches in {args.out_dir}...")
        merge_batches(args.out_dir, suffix=args.suffix)
        return

    if not args.manifest:
        print("ERROR: --manifest required when not using --merge", file=sys.stderr)
        sys.exit(1)

    from scripts.build_unified_sft_v4 import setup_build_context
    from scripts.build_main_agent_sft_v2 import _check_server_reachable, llm_post_stats

    if args.omni_server:
        _check_server_reachable(args.omni_server, "Omni")
        stage2 = args.stage2_server or args.omni_server
        if stage2 and stage2 != args.omni_server:
            _check_server_reachable(stage2, "Stage2")

    print(f"Loading build context...", flush=True)
    ctx = setup_build_context(
        manifest_path=args.manifest,
        index_npy=args.index_npy,
        index_meta=args.index_meta,
        wavetable_lib_path=args.wavetable_lib,
        probe_dir=args.probe_dir,
        clap_device=args.clap_device,
    )
    all_entries = ctx.pop("entries")
    print(f"Loaded {len(all_entries)} manifest entries.", flush=True)

    # Split into batches
    batches = _build_batches(all_entries, args.batch_size)
    n_batches = len(batches)

    # Check which are already done
    done_count = 0
    if args.resume:
        done_count = sum(1 for i in range(n_batches) if _is_batch_done(args.out_dir, i))
        if done_count:
            print(f"{done_count}/{n_batches} batches already complete (resume mode).", flush=True)

    print(f"\nPlan: {len(all_entries)} entries in {n_batches} batches of {args.batch_size}")
    print(f"  workers={args.workers}  seed={args.seed}  suffix={args.suffix}")
    print(f"  out_dir={args.out_dir}")
    print(flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_t0 = time.monotonic()
    total_rendered = 0
    total_failed = 0
    total_skipped_quality = 0
    recent_times: list[float] = []

    for batch_idx, batch_entries in enumerate(batches):
        if args.resume and _is_batch_done(args.out_dir, batch_idx):
            continue

        batch_t0 = time.monotonic()
        print(f"\n{'='*60}", flush=True)
        print(f"Batch {batch_idx:04d}/{n_batches-1}  ({len(batch_entries)} entries)", flush=True)
        print(f"{'='*60}", flush=True)

        stats = run_batch(
            batch_idx=batch_idx,
            batch_entries=batch_entries,
            out_dir=args.out_dir,
            args=args,
            ctx=ctx,
        )

        total_rendered += stats["rendered"]
        total_failed += stats["failed"]
        total_skipped_quality += stats["skipped_quality"]

        _mark_batch_done(args.out_dir, batch_idx, stats)

        batch_elapsed = time.monotonic() - batch_t0
        recent_times.append(batch_elapsed)
        if len(recent_times) > 5:
            recent_times = recent_times[-5:]

        # Progress + ETA
        completed_batches = done_count + batch_idx + 1 - (done_count if not args.resume else 0)
        remaining_batches = n_batches - batch_idx - 1
        if args.resume:
            remaining_batches = sum(
                1 for i in range(batch_idx + 1, n_batches)
                if not _is_batch_done(args.out_dir, i)
            )

        avg_batch_s = sum(recent_times) / len(recent_times)
        eta_s = remaining_batches * avg_batch_s
        global_elapsed = time.monotonic() - global_t0

        print(f"\n  Batch {batch_idx:04d} done in {batch_elapsed:.0f}s "
              f"({stats['rendered']} ok, {stats['failed']} fail, "
              f"{stats['skipped_quality']} quality-rejected)", flush=True)
        print(f"  Running total: {total_rendered} rendered, {total_failed} failed", flush=True)
        print(f"  Elapsed: {global_elapsed/3600:.1f}h  "
              f"ETA: {eta_s/3600:.1f}h  ({remaining_batches} batches left)", flush=True)
        print(f"  LLM stats: {llm_post_stats.summary()}", flush=True)

    global_elapsed = time.monotonic() - global_t0
    print(f"\n{'='*60}", flush=True)
    print(f"ALL BATCHES COMPLETE", flush=True)
    print(f"  Total rendered: {total_rendered}", flush=True)
    print(f"  Total failed:   {total_failed}", flush=True)
    print(f"  Quality rejected: {total_skipped_quality}", flush=True)
    print(f"  Wall time: {global_elapsed/3600:.2f}h", flush=True)
    if total_rendered > 0:
        print(f"  Throughput: {total_rendered/global_elapsed*3600:.0f} rollouts/hr "
              f"(at {args.workers} workers)", flush=True)
    print(f"\nRun --merge to concatenate into final files:", flush=True)
    print(f"  python scripts/run_sft_production.py --merge "
          f"--out-dir {args.out_dir} --suffix {args.suffix}", flush=True)


if __name__ == "__main__":
    main()
