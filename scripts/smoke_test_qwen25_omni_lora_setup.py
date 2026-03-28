#!/usr/bin/env python3
"""
Smoke-test Omni LoRA training setup without running full training.

Checks:
  - train/val JSONL files exist and parse
  - swift executable is discoverable (optional)
  - training launcher builds a valid command in --dry-run mode

Usage:
    python scripts/smoke_test_qwen25_omni_lora_setup.py \
      --dataset-dir data/prepared/omni_lua_sft \
      --max-rows 64
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/prepared/omni_lua_sft"),
        help="Directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=64,
        help="Maximum rows to parse per split during smoke validation.",
    )
    parser.add_argument(
        "--require-swift",
        action="store_true",
        help="Fail if swift CLI is unavailable.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional report JSON path.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        default=Path("configs/deepspeed_zero3.json"),
        help="DeepSpeed config expected by the training launcher.",
    )
    return parser.parse_args()


def _parse_jsonl(path: Path, max_rows: int) -> tuple[int, int]:
    rows = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_rows > 0 and rows >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                json.loads(line)
            except Exception:
                bad += 1
    return rows, bad


def _resolve_swift_bin() -> str | None:
    found = shutil.which("swift")
    if found:
        return found
    sibling = Path(sys.executable).parent / "swift"
    if sibling.exists() and os.access(sibling, os.X_OK):
        return str(sibling.resolve())
    return None


def main() -> None:
    args = parse_args()
    train_path = args.dataset_dir / "train.jsonl"
    val_path = args.dataset_dir / "val.jsonl"
    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}")

    train_rows, train_bad = _parse_jsonl(train_path, args.max_rows)
    val_rows, val_bad = _parse_jsonl(val_path, args.max_rows)

    swift_bin = _resolve_swift_bin()
    swift_ok = swift_bin is not None
    if args.require_swift and not swift_ok:
        raise FileNotFoundError("swift executable not found in PATH")

    ds_config_path = args.deepspeed_config.expanduser().resolve()
    if not ds_config_path.exists():
        raise FileNotFoundError(f"Missing DeepSpeed config: {ds_config_path}")
    ds_config = json.loads(ds_config_path.read_text(encoding="utf-8"))
    ds_has_optimizer = isinstance(ds_config, dict) and "optimizer" in ds_config
    ds_has_scheduler = isinstance(ds_config, dict) and "scheduler" in ds_config

    cmd = [
        sys.executable,
        "scripts/train_qwen25_omni_lora.py",
        "--profile",
        "smoke",
        "--dataset-dir",
        str(args.dataset_dir),
        "--nproc-per-node",
        "2",
        "--deepspeed-config",
        str(ds_config_path),
        "--swift-bin",
        "echo",
        "--dry-run",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    launcher_ok = run.returncode == 0
    launcher_stdout = run.stdout.splitlines()
    launcher_stderr = run.stderr.splitlines()
    launcher_has_deepspeed_arg = "--deepspeed" in run.stdout
    launcher_has_max_length_8192 = "--max_length 8192" in run.stdout
    launcher_has_packing_true = "--packing true" in run.stdout

    invalid_ds_cmd = [
        sys.executable,
        "scripts/train_qwen25_omni_lora.py",
        "--profile",
        "smoke",
        "--dataset-dir",
        str(args.dataset_dir),
        "--nproc-per-node",
        "2",
        "--swift-bin",
        "echo",
        "--dry-run",
        "--deepspeed",
        "zero3",
    ]
    invalid_ds_run = subprocess.run(invalid_ds_cmd, capture_output=True, text=True)
    invalid_ds_guard_ok = (invalid_ds_run.returncode != 0) and ("preset configs" in invalid_ds_run.stderr)

    report = {
        "dataset_dir": str(args.dataset_dir),
        "train_rows_checked": train_rows,
        "train_bad_json": train_bad,
        "val_rows_checked": val_rows,
        "val_bad_json": val_bad,
        "swift_ok": swift_ok,
        "swift_bin": swift_bin,
        "deepspeed_config": str(ds_config_path),
        "deepspeed_has_optimizer": ds_has_optimizer,
        "deepspeed_has_scheduler": ds_has_scheduler,
        "launcher_ok": launcher_ok,
        "launcher_has_deepspeed_arg": launcher_has_deepspeed_arg,
        "launcher_has_max_length_8192": launcher_has_max_length_8192,
        "launcher_has_packing_true": launcher_has_packing_true,
        "launcher_invalid_deepspeed_guard_ok": invalid_ds_guard_ok,
        "launcher_returncode": run.returncode,
        "launcher_stdout_tail": launcher_stdout[-20:],
        "launcher_stderr_tail": launcher_stderr[-20:],
        "invalid_ds_returncode": invalid_ds_run.returncode,
        "invalid_ds_stderr_tail": invalid_ds_run.stderr.splitlines()[-20:],
        "ok": (
            train_rows > 0
            and val_rows > 0
            and train_bad == 0
            and val_bad == 0
            and ds_has_optimizer
            and ds_has_scheduler
            and launcher_ok
            and launcher_has_deepspeed_arg
            and launcher_has_max_length_8192
            and launcher_has_packing_true
            and invalid_ds_guard_ok
        ),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
        print(f"Report written to: {args.report_out}")

    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
