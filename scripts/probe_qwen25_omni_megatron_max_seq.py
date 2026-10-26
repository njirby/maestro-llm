#!/usr/bin/env python3
"""
Probe max trainable sequence length for Qwen2.5-Omni with Megatron-SWIFT.

Runs short training jobs over a list of max_length candidates and reports
the highest successful value.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prepared/omni_lua_sft_tiny_repro"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qwen25_omni_lora_megatron_probe"))
    parser.add_argument("--launcher", type=Path, default=Path("scripts/train_qwen25_omni_lora_megatron.py"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-lengths", default="4096,6144,8192,10240,12288,14336,16384,18432,20480,24576,28672,32768")
    parser.add_argument("--tuner-type", default="lora", choices=["lora", "full"])
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--stop-on-fail", action="store_true", default=True)
    parser.add_argument("--no-stop-on-fail", dest="stop_on_fail", action="store_false")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra launcher arg token (repeatable).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_lengths(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("No max lengths provided.")
    return vals


def _tail(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


def _looks_oom(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    pats = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "nccl error",
    ]
    t = text.lower()
    return any(p in t for p in pats)


def main() -> None:
    args = parse_args()
    lengths = _parse_lengths(args.max_lengths)

    launcher = args.launcher.expanduser().resolve()
    if not launcher.exists():
        raise FileNotFoundError(f"Missing launcher: {launcher}")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not (dataset_dir / "train.jsonl").exists() or not (dataset_dir / "val.jsonl").exists():
        raise FileNotFoundError(f"Missing train/val jsonl in: {dataset_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_out = args.output_root.expanduser().resolve() / f"probe_{ts}"
    probe_out.mkdir(parents=True, exist_ok=True)

    print(f"Probe output: {probe_out}")
    print(f"Lengths: {lengths}")
    print(f"Dataset: {dataset_dir}")

    rows: list[dict] = []
    max_success = None

    for max_length in lengths:
        run_name = f"len{max_length}_{datetime.now().strftime('%H%M%S')}"
        log_path = probe_out / f"{run_name}.log"
        cmd = [
            args.python_bin,
            str(launcher),
            "--model",
            args.model,
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(probe_out / "runs"),
            "--run-name",
            run_name,
            "--max-length",
            str(max_length),
            "--tuner-type",
            args.tuner_type,
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
        ]
        cmd.extend(args.extra_arg)

        print("\n" + "=" * 80)
        print(f"Testing max_length={max_length}")
        print(" ".join(shlex.quote(x) for x in cmd))
        print(f"log: {log_path}")

        if args.dry_run:
            rc = 0
            dur = 0.0
        else:
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            dur = time.time() - t0

        row = {
            "max_length": max_length,
            "rc": rc,
            "seconds": round(dur, 2),
            "status": "ok" if rc == 0 else "failed",
            "oom_suspected": _looks_oom(log_path) if not args.dry_run else False,
            "log_path": str(log_path),
        }
        rows.append(row)

        if rc == 0:
            max_success = max_length
            print(f"PASS max_length={max_length} ({dur:.1f}s)")
        else:
            print(f"FAIL max_length={max_length} ({dur:.1f}s)")
            tail = _tail(log_path, 30)
            if tail:
                print("--- log tail ---")
                for line in tail:
                    print(line)
            if args.stop_on_fail:
                break

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tuner_type": args.tuner_type,
        "dataset_dir": str(dataset_dir),
        "train_iters": args.train_iters,
        "max_lengths": lengths,
        "max_success_length": max_success,
        "results": rows,
    }
    report_path = probe_out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Report: {report_path}")
    if max_success is None:
        print("No successful run found.")
        raise SystemExit(1)
    print(f"Max successful max_length: {max_success}")


if __name__ == "__main__":
    main()
