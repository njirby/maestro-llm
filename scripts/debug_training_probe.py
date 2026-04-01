#!/usr/bin/env python3
"""Short, telemetry-heavy training probe for SSH/crash RCA."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT_DIR / "outputs" / "debug_training_probes"


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fresh", "resume"], required=True, help="Probe mode.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Required when --mode=resume.")
    parser.add_argument("--gpus", default="1,2,3", help='CUDA visible devices CSV (e.g. "0,1,2,3").')
    parser.add_argument("--duration-sec", type=int, default=240, help="Probe runtime cap in seconds.")
    parser.add_argument("--max-steps", type=int, default=16, help="Training max_steps for probe runs.")
    parser.add_argument("--save-steps", type=int, default=999999, help="Training save_steps override.")
    parser.add_argument("--eval-steps", type=int, default=999999, help="Training eval_steps override.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Training logging_steps override.")
    parser.add_argument("--run-name", default=None, help="Run name passed to training launcher via env.")
    parser.add_argument("--master-port", type=int, default=None, help="Torch rendezvous master port (auto if unset).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Probe output directory.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional DATASET_DIR override.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="Probe worker count.")
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Extra passthrough arg token for train launcher (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write command/env and exit.")
    parser.add_argument(
        "--detach",
        dest="detach",
        action="store_true",
        default=True,
        help="Run probe worker in detached background process (default).",
    )
    parser.add_argument(
        "--no-detach",
        dest="detach",
        action="store_false",
        help="Run probe worker in current terminal.",
    )
    return parser.parse_args()


def _gpu_list(gpus_csv: str) -> list[str]:
    devices = [x.strip() for x in gpus_csv.split(",") if x.strip()]
    if not devices:
        raise ValueError("--gpus must contain at least one CUDA device id.")
    return devices


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    env_out = os.environ.get("PROBE_OUT_DIR")
    if args.out_dir is not None:
        return args.out_dir.expanduser().resolve()
    if env_out:
        return Path(env_out).expanduser().resolve()
    safe_gpus = args.gpus.replace(",", "-")
    name = f"{_now_ts()}_{args.mode}_g{safe_gpus}"
    return (DEFAULT_OUT_ROOT / name).resolve()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _run_capture(cmd: Sequence[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


@dataclass
class PollTask:
    name: str
    interval_sec: float
    cmd: list[str]
    output_path: Path


def _poll_loop(task: PollTask, stop_event: threading.Event, cwd: Path) -> None:
    while not stop_event.is_set():
        rc, out = _run_capture(task.cmd, cwd=cwd)
        stamp = _iso_now()
        header = f"\n## {stamp} | rc={rc} | {' '.join(shlex.quote(c) for c in task.cmd)}\n"
        _append_text(task.output_path, header + (out or "<no output>\n"))
        stop_event.wait(task.interval_sec)


def _snapshot(path: Path, title: str, cmd: list[str], cwd: Path) -> None:
    rc, out = _run_capture(cmd, cwd=cwd)
    stamp = _iso_now()
    content = (
        f"## {title}\n"
        f"timestamp: {stamp}\n"
        f"command: {' '.join(shlex.quote(c) for c in cmd)}\n"
        f"rc: {rc}\n\n"
        f"{out or '<no output>'}\n"
    )
    _write_text(path, content)


def _extract_global_steps(train_log: Path) -> tuple[int | None, int | None]:
    if not train_log.exists():
        return None, None
    pattern = re.compile(r"[\"']global_step/max_steps[\"']:\s*[\"'](\d+)/(\d+)[\"']")
    first: int | None = None
    last: int | None = None
    with train_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if first is None:
                first = step
            last = step
    return first, last


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
    train_script = ROOT_DIR / "scripts" / "train_qwen25_omni_lora_full_9k.sh"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training launcher: {train_script}")
    cmd = [
        "bash",
        str(train_script),
        "--max_steps",
        str(args.max_steps),
        "--save_steps",
        str(args.save_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--logging_steps",
        str(args.logging_steps),
        # Probe runs should avoid expensive save-on-interrupt behaviors.
        "--save_strategy",
        "no",
        "--load_best_model_at_end",
        "false",
    ]
    if args.mode == "resume":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --mode=resume.")
        cmd.extend(["--resume_from_checkpoint", str(args.checkpoint.expanduser().resolve())])
    if args.train_arg:
        cmd.extend(args.train_arg)
    return cmd


def _build_env(args: argparse.Namespace, out_dir: Path, run_name: str) -> dict[str, str]:
    env = os.environ.copy()
    devices = _gpu_list(args.gpus)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["NPROC_PER_NODE"] = str(len(devices))
    env["DATALOADER_NUM_WORKERS"] = str(args.dataloader_num_workers)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["RUN_NAME"] = run_name
    env["OUTPUT_ROOT"] = str((out_dir / "train_outputs").resolve())
    if args.dataset_dir is not None:
        env["DATASET_DIR"] = str(args.dataset_dir.expanduser().resolve())
    return env


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def _launch_detached(args: argparse.Namespace, out_dir: Path) -> int:
    launch_log = out_dir / "launcher.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    child_cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    if "--no-detach" not in child_cmd:
        child_cmd.append("--no-detach")
    if "--out-dir" not in child_cmd and not any(token.startswith("--out-dir=") for token in child_cmd):
        child_cmd.extend(["--out-dir", str(out_dir)])
    env = os.environ.copy()
    env["PROBE_WORKER"] = "1"
    env["PROBE_OUT_DIR"] = str(out_dir)
    with launch_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            child_cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(ROOT_DIR),
            start_new_session=True,
        )
    print("Probe worker launched")
    print(f"  pid: {proc.pid}")
    print(f"  out_dir: {out_dir}")
    print(f"  launcher_log: {launch_log}")
    return 0


def main() -> int:
    args = _parse_args()
    out_dir = _resolve_out_dir(args)
    run_name = args.run_name or f"probe_{args.mode}_{_now_ts()}"

    if args.mode == "resume" and args.checkpoint is None:
        raise ValueError("--checkpoint is required with --mode=resume.")
    if args.checkpoint is not None and not args.checkpoint.expanduser().resolve().exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    if args.detach and os.environ.get("PROBE_WORKER") != "1":
        return _launch_detached(args, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = _iso_now()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    train_cmd = _build_train_cmd(args)
    env = _build_env(args, out_dir, run_name)
    env["MASTER_PORT"] = str(args.master_port if args.master_port is not None else _pick_free_port())

    env_snapshot = {
        "RUN_NAME": env.get("RUN_NAME"),
        "OUTPUT_ROOT": env.get("OUTPUT_ROOT"),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
        "NPROC_PER_NODE": env.get("NPROC_PER_NODE"),
        "DATALOADER_NUM_WORKERS": env.get("DATALOADER_NUM_WORKERS"),
        "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM"),
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS"),
        "MASTER_PORT": env.get("MASTER_PORT"),
        "DATASET_DIR": env.get("DATASET_DIR"),
    }
    _write_text(out_dir / "cmd.txt", " ".join(shlex.quote(c) for c in train_cmd) + "\n")
    _write_text(out_dir / "env.json", json.dumps(env_snapshot, indent=2) + "\n")
    _write_text(
        out_dir / "meta_start.json",
        json.dumps(
            {
                "mode": args.mode,
                "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
                "gpus": args.gpus,
                "duration_sec": args.duration_sec,
                "max_steps": args.max_steps,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
                "logging_steps": args.logging_steps,
                "run_name": run_name,
                "started_at": start_time,
                "boot_id": boot_id,
                "out_dir": str(out_dir),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
        + "\n",
    )

    _snapshot(out_dir / "snapshot_before_uptime.log", "uptime_before", ["uptime"], ROOT_DIR)
    _snapshot(out_dir / "snapshot_before_last.log", "last_before", ["bash", "-lc", "last -x | head -n 60"], ROOT_DIR)
    _snapshot(
        out_dir / "snapshot_before_services.log",
        "services_before",
        ["bash", "-lc", "systemctl is-active ssh || systemctl is-active sshd; systemctl is-active tailscaled"],
        ROOT_DIR,
    )

    if args.dry_run:
        print(f"Dry run only; wrote artifacts to: {out_dir}")
        return 0

    timeout_cmd = [
        "timeout",
        "--signal=INT",
        "--kill-after=45",
        f"{args.duration_sec}s",
        *train_cmd,
    ]
    _write_text(out_dir / "timeout_cmd.txt", " ".join(shlex.quote(c) for c in timeout_cmd) + "\n")

    poll_tasks = [
        PollTask(
            name="gpu_stats",
            interval_sec=1.0,
            cmd=[
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw,pstate",
                "--format=csv,noheader,nounits",
            ],
            output_path=out_dir / "telemetry_gpu.log",
        ),
        PollTask(
            name="gpu_procs",
            interval_sec=1.0,
            cmd=[
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            output_path=out_dir / "telemetry_gpu_procs.log",
        ),
        PollTask(
            name="vmstat",
            interval_sec=5.0,
            cmd=["bash", "-lc", "vmstat -SM 1 2 | tail -n 1"],
            output_path=out_dir / "telemetry_vmstat.log",
        ),
        PollTask(
            name="iostat",
            interval_sec=5.0,
            cmd=["bash", "-lc", "iostat -x 1 2 | tail -n +7"],
            output_path=out_dir / "telemetry_iostat.log",
        ),
        PollTask(
            name="mpstat",
            interval_sec=5.0,
            cmd=["bash", "-lc", "mpstat -P ALL 1 1"],
            output_path=out_dir / "telemetry_mpstat.log",
        ),
        PollTask(
            name="ss",
            interval_sec=5.0,
            cmd=["ss", "-s"],
            output_path=out_dir / "telemetry_ss.log",
        ),
        PollTask(
            name="top_ps",
            interval_sec=5.0,
            cmd=["bash", "-lc", "ps -eo pid,ppid,pcpu,pmem,cmd --sort=-pcpu | head -n 40"],
            output_path=out_dir / "telemetry_top_ps.log",
        ),
    ]

    stop_event = threading.Event()
    poll_threads = [
        threading.Thread(target=_poll_loop, args=(task, stop_event, ROOT_DIR), daemon=True) for task in poll_tasks
    ]
    for th in poll_threads:
        th.start()

    train_log = out_dir / "train.log"
    with train_log.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            timeout_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(ROOT_DIR),
            start_new_session=True,
        )
        while proc.poll() is None:
            time.sleep(0.5)
        exit_code = proc.returncode

    stop_event.set()
    for th in poll_threads:
        th.join(timeout=3)

    end_time = _iso_now()
    _snapshot(out_dir / "snapshot_after_uptime.log", "uptime_after", ["uptime"], ROOT_DIR)
    _snapshot(out_dir / "snapshot_after_last.log", "last_after", ["bash", "-lc", "last -x | head -n 80"], ROOT_DIR)
    _snapshot(
        out_dir / "snapshot_after_services.log",
        "services_after",
        ["bash", "-lc", "systemctl is-active ssh || systemctl is-active sshd; systemctl is-active tailscaled"],
        ROOT_DIR,
    )

    _snapshot(
        out_dir / "journal_system.log",
        "journal_system_window",
        ["journalctl", "--since", start_time, "--until", end_time, "--no-pager"],
        ROOT_DIR,
    )
    _snapshot(
        out_dir / "journal_kernel.log",
        "journal_kernel_window",
        ["journalctl", "-k", "--since", start_time, "--until", end_time, "--no-pager"],
        ROOT_DIR,
    )

    first_step, last_step = _extract_global_steps(train_log)
    summary = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "gpus": args.gpus,
        "duration_sec": args.duration_sec,
        "run_name": run_name,
        "out_dir": str(out_dir),
        "started_at": start_time,
        "ended_at": end_time,
        "boot_id_start": boot_id,
        "exit_code": exit_code,
        "timed_out": exit_code == 124,
        "first_logged_global_step": first_step,
        "last_logged_global_step": last_step,
        "train_cmd": timeout_cmd,
        "env_snapshot": env_snapshot,
    }
    _write_text(out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")

    print("Probe complete")
    print(f"  out_dir: {out_dir}")
    print(f"  summary: {out_dir / 'summary.json'}")
    print(f"  exit_code: {exit_code}")
    print(f"  timed_out: {exit_code == 124}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
