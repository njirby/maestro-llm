#!/usr/bin/env python3
"""
Sweep LoRA rank capacity for Qwen2.5-Omni-7B with Megatron-SWIFT.

For each target context length, runs a 4-step viability search and reports the
maximum verified passing rank with a configurable boundary resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunResult:
    rank: int
    rc: int
    checkpoint4: bool
    memory_gib: str | None
    run_dir: str
    log: str
    finished_at: str
    context_length: int
    lora_alpha: int

    def to_json(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rc": self.rc,
            "checkpoint4": self.checkpoint4,
            "memory_gib": self.memory_gib,
            "run_dir": self.run_dir,
            "log": self.log,
            "finished_at": self.finished_at,
            "context_length": self.context_length,
            "lora_alpha": self.lora_alpha,
        }


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    hf_local = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
    parser.add_argument("--model", default=str(hf_local) if hf_local.is_dir() else "Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prepared/omni_lua_sft_probe_repeat256"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qwen25_omni_lora_megatron_probe"))
    parser.add_argument("--launcher", type=Path, default=Path("scripts/train_qwen25_omni_lora_megatron.py"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--contexts", default="49152,65536")
    parser.add_argument("--coarse-ranks", default="256,512,768,896,1024")
    parser.add_argument("--down-ranks", default="128,64,32,16,8")
    parser.add_argument("--start-rank", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-rank-cap", type=int, default=2048)
    parser.add_argument("--train-iters", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=4)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", default="true", choices=["true", "false"])
    parser.add_argument("--packing", default="true", choices=["true", "false"])
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_preflight(args: argparse.Namespace, contexts: list[int]) -> None:
    launcher = args.launcher.expanduser().resolve()
    if not launcher.exists():
        raise FileNotFoundError(f"Missing launcher: {launcher}")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    for split in ("train.jsonl", "val.jsonl"):
        p = dataset_dir / split
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset split: {p}")
    if args.resolution <= 0:
        raise ValueError("--resolution must be > 0")
    if any(c <= 0 for c in contexts):
        raise ValueError("All context lengths must be > 0")
    if args.max_rank_cap < args.start_rank:
        raise ValueError("--max-rank-cap must be >= --start-rank")
    if args.tensor_model_parallel_size <= 0:
        raise ValueError("--tensor-model-parallel-size must be > 0")
    if args.global_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("--global-batch-size and --micro-batch-size must be > 0")


def check_gpu_environment(args: argparse.Namespace) -> None:
    if args.skip_gpu_check or args.dry_run:
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError("nvidia-smi unavailable; cannot verify 4-GPU profile.") from exc
    gpu_count = len([line for line in out.splitlines() if line.strip()])
    visible = [t.strip() for t in args.cuda_visible_devices.split(",") if t.strip()]
    if len(visible) < args.tensor_model_parallel_size:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES has {len(visible)} device(s), but tp={args.tensor_model_parallel_size}."
        )
    if gpu_count < args.tensor_model_parallel_size:
        raise RuntimeError(f"Detected {gpu_count} GPU(s), but tp={args.tensor_model_parallel_size}.")


def append_jsonl(path: Path, payload: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload.rstrip("\n") + "\n")
        else:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_memory_gib(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    mem_pat = re.compile(r"memory\(GiB\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = mem_pat.findall(text)
    if matches:
        return matches[-1]
    return None


def find_run_dir(run_root: Path) -> Path | None:
    candidates = sorted(run_root.glob("v*"))
    if not candidates:
        return None
    return candidates[-1]


def has_checkpoint4(run_root: Path) -> bool:
    for marker in run_root.glob("v*/checkpoint-4/latest_checkpointed_iteration.txt"):
        if marker.exists():
            return True
    return False


def build_rank_cmd(args: argparse.Namespace, context_length: int, rank: int, sweep_dir: Path) -> tuple[list[str], str]:
    run_name = f"rank{rank}_len{context_length}_{args.train_iters}steps"
    cmd = [
        args.python_bin,
        str(args.launcher.expanduser().resolve()),
        "--model",
        args.model,
        "--dataset-dir",
        str(args.dataset_dir.expanduser().resolve()),
        "--output-root",
        str(sweep_dir),
        "--run-name",
        run_name,
        "--max-length",
        str(context_length),
        "--tuner-type",
        "lora",
        "--train-iters",
        str(args.train_iters),
        "--save-steps",
        "999999",
        "--eval-steps",
        "999999",
        "--dataloader-num-workers",
        "0",
        "--dataset-num-proc",
        "1",
        "--global-batch-size",
        str(args.global_batch_size),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--tensor-model-parallel-size",
        str(args.tensor_model_parallel_size),
        "--context-parallel-size",
        str(args.context_parallel_size),
        "--sequence-parallel",
        args.sequence_parallel,
        "--packing",
        args.packing,
        "--lora-rank",
        str(rank),
        "--lora-alpha",
        str(rank * 2),
        "--target-modules",
        args.target_modules,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
    ]
    return cmd, run_name


def run_rank_once(
    args: argparse.Namespace,
    context_length: int,
    rank: int,
    sweep_dir: Path,
) -> RunResult:
    cmd, run_name = build_rank_cmd(args, context_length, rank, sweep_dir)
    log_path = sweep_dir / f"{run_name}.launcher.log"
    run_root = sweep_dir / run_name

    print(f"=== START {run_name} {now_iso()} ===", flush=True)
    print(" ".join(shlex.quote(x) for x in cmd), flush=True)
    print(f"log: {log_path}", flush=True)

    if args.dry_run:
        rc = 0
        checkpoint4 = True
        memory = None
        run_dir = str(run_root)
    else:
        with log_path.open("w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        checkpoint4 = has_checkpoint4(run_root)
        memory = parse_memory_gib(log_path)
        detected = find_run_dir(run_root)
        run_dir = str(detected if detected else run_root)

    result = RunResult(
        rank=rank,
        rc=rc,
        checkpoint4=checkpoint4,
        memory_gib=memory,
        run_dir=run_dir,
        log=str(log_path),
        finished_at=now_iso(),
        context_length=context_length,
        lora_alpha=rank * 2,
    )
    print(json.dumps(result.to_json(), ensure_ascii=False), flush=True)
    return result


def midpoint_rounded(low: int, high: int, step: int) -> int:
    raw = (low + high) // 2
    rounded = (raw // step) * step
    if rounded <= low:
        rounded = low + step
    if rounded >= high:
        rounded = high - step
    return rounded


def dedupe_keep_order(vals: list[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def run_context_sweep(args: argparse.Namespace, context_length: int) -> dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_root.expanduser().resolve() / f"omni7b_rank_sweep_len{context_length}_{args.train_iters}steps_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "sweep_summary.jsonl"
    append_jsonl(
        summary_path,
        {
            "started_at": now_iso(),
            "context_length": context_length,
            "output_root": str(sweep_dir),
            "strategy": "bracket_binary_resolution",
            "resolution": args.resolution,
            "coarse_ranks": parse_int_list(args.coarse_ranks),
            "down_ranks": parse_int_list(args.down_ranks),
            "start_rank": args.start_rank,
            "max_rank_cap": args.max_rank_cap,
            "viability": f"rc=0 and checkpoint-{args.train_iters}",
        },
    )

    cache: dict[int, RunResult] = {}

    def evaluate(rank: int) -> RunResult:
        if rank in cache:
            return cache[rank]
        res = run_rank_once(args, context_length, rank, sweep_dir)
        cache[rank] = res
        append_jsonl(summary_path, res.to_json())
        return res

    start_res = evaluate(args.start_rank)
    if not (start_res.rc == 0 and start_res.checkpoint4):
        append_jsonl(summary_path, f"Stopping sweep: start rank {args.start_rank} failed viability.")
        return {
            "context_length": context_length,
            "sweep_dir": str(sweep_dir),
            "max_viable_rank": None,
            "first_failing_rank": args.start_rank,
            "boundary_resolution": args.resolution,
            "stable": False,
        }

    highest_pass = args.start_rank
    lowest_fail: int | None = None

    coarse = dedupe_keep_order([r for r in parse_int_list(args.coarse_ranks) if r <= args.max_rank_cap and r > 0])
    for rank in coarse:
        res = evaluate(rank)
        if res.rc == 0 and res.checkpoint4:
            highest_pass = max(highest_pass, rank)
            continue
        lowest_fail = rank
        break

    if lowest_fail is None:
        rank = highest_pass + args.resolution
        while rank <= args.max_rank_cap:
            res = evaluate(rank)
            if res.rc == 0 and res.checkpoint4:
                highest_pass = rank
                rank += args.resolution
                continue
            lowest_fail = rank
            break

    if lowest_fail is not None and lowest_fail == coarse[0]:
        down = dedupe_keep_order([r for r in parse_int_list(args.down_ranks) if 0 < r < lowest_fail])
        for rank in down:
            res = evaluate(rank)
            if res.rc == 0 and res.checkpoint4:
                highest_pass = max(highest_pass, rank)
                break

    if lowest_fail is not None:
        while (lowest_fail - highest_pass) > args.resolution:
            mid = midpoint_rounded(highest_pass, lowest_fail, args.resolution)
            res = evaluate(mid)
            if res.rc == 0 and res.checkpoint4:
                highest_pass = mid
            else:
                lowest_fail = mid

    # Boundary confirmation
    confirm_pass = run_rank_once(args, context_length, highest_pass, sweep_dir)
    append_jsonl(summary_path, {"confirm_pass": confirm_pass.to_json()})
    confirm_fail_rank = highest_pass + args.resolution
    confirm_fail: RunResult | None = None
    if lowest_fail is not None and confirm_fail_rank <= args.max_rank_cap:
        confirm_fail = run_rank_once(args, context_length, confirm_fail_rank, sweep_dir)
        append_jsonl(summary_path, {"confirm_fail": confirm_fail.to_json()})

    stable = False
    if confirm_fail is not None:
        stable = bool(
            confirm_pass.rc == 0
            and confirm_pass.checkpoint4
            and not (confirm_fail.rc == 0 and confirm_fail.checkpoint4)
        )

    final_payload = {
        "finished_at": now_iso(),
        "context_length": context_length,
        "max_viable_rank": highest_pass if stable else None,
        "first_failing_rank": confirm_fail_rank if confirm_fail is not None else lowest_fail,
        "boundary_resolution": args.resolution,
        "stable": stable,
        "sweep_dir": str(sweep_dir),
    }
    append_jsonl(summary_path, {"final": final_payload})
    return final_payload


def main() -> None:
    args = parse_args()
    contexts = dedupe_keep_order(parse_int_list(args.contexts))
    ensure_preflight(args, contexts)
    check_gpu_environment(args)

    all_results = []
    for context_length in contexts:
        print("\n" + "=" * 88, flush=True)
        print(f"Context length sweep: {context_length}", flush=True)
        result = run_context_sweep(args, context_length)
        all_results.append(result)

    report = {
        "created_at": now_iso(),
        "model": args.model,
        "contexts": contexts,
        "results": all_results,
    }
    report_path = args.output_root.expanduser().resolve() / f"omni7b_rank_sweep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 88, flush=True)
    print(f"Report: {report_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
