#!/usr/bin/env python3
"""
Monitor an active training run and relaunch a comparison run on overfitting.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing import event_accumulator


@dataclass
class ScalarPoint:
    step: int
    value: float


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)


def find_latest_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")
    return files[-1]


def parse_rank0_pid_from_event_file(path: Path) -> int | None:
    # Pattern is usually events.out.tfevents.<ts>.<host>.<pid>.<worker_idx>
    parts = path.name.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def _read_cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _read_ppid(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return 0
    parts = stat.split()
    if len(parts) < 4:
        return 0
    try:
        return int(parts[3])
    except ValueError:
        return 0


def _pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def find_training_stop_pid(rank0_pid: int) -> int | None:
    if rank0_pid <= 0 or not _pid_alive(rank0_pid):
        return None

    current = rank0_pid
    fallback = rank0_pid
    visited = set()
    while current > 1 and current not in visited:
        visited.add(current)
        cmd = _read_cmdline(current)
        if "torchrun" in cmd and "run_experiment.py audio_caption" in cmd:
            return current
        if "run_experiment.py audio_caption" in cmd:
            fallback = current
        current = _read_ppid(current)
    return fallback


def _safe_scalar_points(acc: event_accumulator.EventAccumulator, tag: str) -> list[ScalarPoint]:
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [ScalarPoint(step=int(v.step), value=float(v.value)) for v in acc.Scalars(tag)]


def load_metrics(event_file: Path) -> tuple[list[ScalarPoint], list[ScalarPoint]]:
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()
    eval_points = _safe_scalar_points(acc, "eval/loss")
    train_points = _safe_scalar_points(acc, "train/loss")
    return eval_points, train_points


def _train_values_at_eval_steps(
    train_points: Iterable[ScalarPoint],
    eval_steps: Iterable[int],
) -> list[float]:
    train_points = list(train_points)
    out: list[float] = []
    idx = 0
    current = None
    for eval_step in eval_steps:
        while idx < len(train_points) and train_points[idx].step <= eval_step:
            current = train_points[idx].value
            idx += 1
        if current is None:
            return []
        out.append(current)
    return out


def should_stop_for_overfitting(
    eval_points: list[ScalarPoint],
    train_points: list[ScalarPoint],
    min_eval_rise: float,
) -> tuple[bool, str]:
    if len(eval_points) < 3:
        return False, "need at least 3 eval points"

    last3 = eval_points[-3:]
    d_prev = last3[1].value - last3[0].value
    d_last = last3[2].value - last3[1].value
    if d_prev < min_eval_rise or d_last < min_eval_rise:
        return False, (
            f"eval rise not sustained: deltas=({d_prev:+.6f}, {d_last:+.6f}), "
            f"threshold={min_eval_rise:.6f}"
        )

    eval_steps = [pt.step for pt in last3]
    train_vals = _train_values_at_eval_steps(train_points, eval_steps)
    if len(train_vals) < 3:
        return False, "insufficient train points aligned to eval steps"

    if train_vals[2] <= train_vals[1] and train_vals[1] <= train_vals[0]:
        return True, (
            "triggered: eval_loss rose in 2 consecutive evals while train_loss did not rise "
            f"(eval_deltas=({d_prev:+.6f}, {d_last:+.6f}), "
            f"train_values=({train_vals[0]:.6f}, {train_vals[1]:.6f}, {train_vals[2]:.6f}))"
        )
    return False, (
        "eval rises detected but train_loss also rose in the same window "
        f"(train_values=({train_vals[0]:.6f}, {train_vals[1]:.6f}, {train_vals[2]:.6f}))"
    )


def stop_training_process(stop_pid: int, timeout_s: int = 180) -> bool:
    if stop_pid <= 0 or not _pid_alive(stop_pid):
        return False

    _log(f"Sending SIGINT to training process pid={stop_pid}")
    os.kill(stop_pid, signal.SIGINT)
    waited = 0
    while waited < timeout_s and _pid_alive(stop_pid):
        time.sleep(1)
        waited += 1
    if not _pid_alive(stop_pid):
        _log(f"Training process stopped cleanly after {waited}s")
        return True

    _log(f"Training process still alive after {timeout_s}s; sending SIGTERM to pid={stop_pid}")
    os.kill(stop_pid, signal.SIGTERM)
    time.sleep(5)
    return not _pid_alive(stop_pid)


def launch_comparison_run(repo_root: Path, config_path: str, output_root: str):
    ts = datetime.now().strftime("%b%d_%H-%M-%S")
    output_dir = f"{output_root}_{ts}"
    cmd = [
        "bash",
        "scripts/train_2gpu_zero3.sh",
        "--config",
        config_path,
        "--output_dir",
        output_dir,
        # Enforce comparison intent even if the YAML is edited later.
        "--train_clap_encoder",
        "--train_projection",
        "--freeze_llm",
        "--clap_device",
        "cuda",
    ]
    log_dir = repo_root / "outputs" / "monitor_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launch_log = log_dir / f"encoder_unfrozen_launch_{ts}.log"
    with launch_log.open("a", encoding="utf-8") as f:
        _log(f"Launching comparison run: {' '.join(cmd)}")
        _log(f"Launcher stdout/stderr: {launch_log}")
        subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor eval loss and relaunch on overfitting.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-root", default="outputs/audio_caption_qwen4b/runs")
    parser.add_argument("--check-interval-minutes", type=int, default=30)
    parser.add_argument("--min-eval-rise", type=float, default=0.005)
    parser.add_argument(
        "--comparison-config",
        default="configs/audio_caption_qwen4b_encoder_unfrozen.yaml",
    )
    parser.add_argument(
        "--comparison-output-root",
        default="./outputs/audio_caption_qwen4b_encoder_unfrozen",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / args.runs_root / args.run_name
    if not run_dir.is_dir():
        _log(f"Run directory not found: {run_dir}")
        return 2

    _log(
        f"Monitoring run={args.run_name} every {args.check_interval_minutes} minutes "
        f"with min_eval_rise={args.min_eval_rise:.6f}"
    )
    last_eval_count = -1

    while True:
        try:
            event_file = find_latest_event_file(run_dir)
            eval_points, train_points = load_metrics(event_file)
        except Exception as exc:
            _log(f"Metric read failed: {exc}")
            time.sleep(args.check_interval_minutes * 60)
            continue

        eval_count = len(eval_points)
        if eval_count == 0:
            _log("No eval/loss points yet; sleeping.")
            time.sleep(args.check_interval_minutes * 60)
            continue

        latest_eval = eval_points[-1]
        _log(
            f"Latest eval/loss step={latest_eval.step} value={latest_eval.value:.6f} "
            f"(eval_points={eval_count})"
        )
        if eval_count == last_eval_count:
            _log("No new eval point since last check; sleeping.")
            time.sleep(args.check_interval_minutes * 60)
            continue
        last_eval_count = eval_count

        should_stop, reason = should_stop_for_overfitting(
            eval_points=eval_points,
            train_points=train_points,
            min_eval_rise=args.min_eval_rise,
        )
        _log(f"Decision: {'STOP' if should_stop else 'CONTINUE'} ({reason})")
        if not should_stop:
            time.sleep(args.check_interval_minutes * 60)
            continue

        if args.dry_run:
            _log("Dry run enabled: skipping stop + relaunch.")
            return 0

        rank0_pid = parse_rank0_pid_from_event_file(event_file)
        if rank0_pid is None:
            _log("Could not parse rank0 pid from event file; aborting stop/relaunch.")
            return 3
        stop_pid = find_training_stop_pid(rank0_pid)
        if stop_pid is None:
            _log(f"Could not resolve a live training process from rank0 pid={rank0_pid}.")
            return 3

        stopped = stop_training_process(stop_pid)
        if not stopped:
            _log(f"Failed to stop pid={stop_pid}; aborting relaunch.")
            return 4

        launch_comparison_run(
            repo_root=repo_root,
            config_path=args.comparison_config,
            output_root=args.comparison_output_root,
        )
        _log("Comparison run launch requested; monitor exiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
