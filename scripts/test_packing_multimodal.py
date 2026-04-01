#!/usr/bin/env python3
"""
Demonstrate that sequence packing works for multimodal (Qwen2.5-Omni) data
by running two smoke training runs on the tiny repro dataset:
  - Run A: packing=false  (baseline)
  - Run B: packing=true   (packed)

Evidence: packed run completes in fewer steps per epoch because multiple
samples are combined into each training sequence.

Usage:
    python scripts/test_packing_multimodal.py [--model Qwen/Qwen2.5-Omni-3B] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_JSONL = ROOT / "data/prepared/omni_lua_sft_tiny_repro/train.jsonl"
VAL_JSONL = ROOT / "data/prepared/omni_lua_sft_tiny_repro/val.jsonl"
OUTPUT_ROOT = ROOT / "outputs/qwen25_omni_lora"

# Sequence length chosen so that 2 short samples (~2500 tokens each) can pack
# together (2500*2 = 5000 < 5120) while staying within VRAM budget.
# - 3B model base: ~18.6 GiB; long-sequence activation: ~2.5 GiB → ~21.1 GiB total.
# - Long samples (>5120 tokens) are truncated; short samples still pack 2-at-a-time.
MAX_LENGTH = 5120


def _resolve_swift_bin() -> str:
    found = shutil.which("swift")
    if found:
        return found
    sibling = Path(sys.executable).parent / "swift"
    if sibling.exists() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise FileNotFoundError("swift executable not found in PATH or venv bin/")


def _build_cmd(
    swift_bin: str,
    model: str,
    run_name: str,
    packing: bool,
    dry_run: bool = False,
) -> list[str]:
    output_dir = OUTPUT_ROOT / run_name
    cmd = [
        swift_bin, "sft",
        "--model", model,
        "--model_type", "qwen2_5_omni",
        "--template", "qwen2_5_omni",
        "--train_type", "lora",
        "--dataset", str(TRAIN_JSONL.resolve()),
        "--val_dataset", str(VAL_JSONL.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--target_modules", "all-linear",
        "--warmup_ratio", "0.03",
        # ── profile ───────────────────────────────────────────────
        "--num_train_epochs", "1",      # full epoch → step count reveals packing
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "1e-4",
        "--max_length", str(MAX_LENGTH),
        "--logging_steps", "1",
        "--save_steps", "999",          # skip intermediate saves to save time
        "--eval_steps", "999",
        "--dataloader_num_workers", "0",
        # ── packing ───────────────────────────────────────────────
        "--packing", "true" if packing else "false",
        "--attn_impl", "flash_attn",    # required for packing
    ]
    return cmd


def _run(cmd: list[str], run_name: str, dry_run: bool) -> Path:
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    env = os.environ.copy()
    # Restrict to one GPU — prevents device_map='auto' from triggering
    # distributed mode when multiple GPUs are visible.
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["NPROC_PER_NODE"] = "1"
    env["ENABLE_AUDIO_OUTPUT"] = "0"
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print("Command:")
    print("  " + " ".join(shlex.quote(t) for t in cmd))
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    if dry_run:
        print("  [dry-run] skipping execution")
        return log_path

    with open(log_path, "w") as fout:
        proc = subprocess.run(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[WARN] run exited with code {proc.returncode} — check {log_path}")
    return log_path


def _parse_log(log_path: Path) -> dict:
    """Extract key metrics from a training log."""
    if not log_path.exists():
        return {}

    text = log_path.read_text(errors="replace")

    # Count training steps by counting logged step dicts with global_step
    step_reports = re.findall(r"'global_step/max_steps': '(\d+)/(\d+)'", text)
    total_steps = max((int(s) for s, _ in step_reports), default=0)
    max_steps_from_log = max((int(m) for _, m in step_reports), default=None)

    # Packing count from progress bar: "Packing: 100%|... 16/16"
    pack_match = re.search(r"Packing.*?(\d+)/(\d+)", text)
    n_input_samples = int(pack_match.group(2)) if pack_match else None
    n_packed_seqs = None  # set below

    # After packing, swift reports new dataset size
    # e.g. "train dataset samples: 9" or from packed_length
    ds_size_match = re.search(r"train_dataset.*?num_rows.*?(\d+)", text)
    if not ds_size_match:
        # Try "Num train samples: N"
        ds_size_match = re.search(r"[Nn]um.{0,10}train.{0,10}samples.*?(\d+)", text)
    if ds_size_match:
        n_packed_seqs = int(ds_size_match.group(1))

    # Extract peak memory
    mem_matches = re.findall(r"'memory\(GiB\)': ([\d.]+)", text)
    peak_mem = max((float(m) for m in mem_matches), default=None)

    # Extract final train_loss
    loss_matches = re.findall(r"'train_loss': ([\d.]+)", text)
    final_loss = float(loss_matches[-1]) if loss_matches else None

    # tokens per step: look for steps_per_second + total tokens indirectly
    # swift doesn't log tokens/step directly, but we can estimate from packed lengths

    return {
        "total_steps": total_steps,
        "max_steps_from_log": max_steps_from_log,
        "n_input_samples": n_input_samples,
        "n_packed_seqs": n_packed_seqs,
        "peak_mem_gib": peak_mem,
        "final_loss": final_loss,
    }


def _report(results: dict[str, dict], n_train_samples: int) -> None:
    print("\n" + "=" * 60)
    print("PACKING COMPARISON REPORT")
    print("=" * 60)
    print(f"Train dataset: {TRAIN_JSONL} ({n_train_samples} samples)")
    print(f"MAX_LENGTH / packing_length: {MAX_LENGTH}")
    print()

    for run_name, m in results.items():
        packing_label = "PACKING" if "pack_" in run_name and "nopack" not in run_name else "NO-PACK"
        print(f"  [{packing_label}]  {run_name}")
        if not m:
            print("    (no log data)")
            continue
        steps = m["total_steps"]
        n_in = m["n_input_samples"] or n_train_samples
        n_packed = m["n_packed_seqs"]
        print(f"    Training steps completed  : {steps}")
        if n_packed is not None:
            print(f"    Dataset size after packing: {n_packed} sequences  (from {n_in} samples)")
            ratio = n_in / n_packed if n_packed > 0 else 1.0
            print(f"    Packing ratio             : {ratio:.2f}x  ({n_in} → {n_packed})")
        if steps and n_in:
            spe = steps  # steps per epoch ≈ total_steps for 1 epoch
            print(f"    Steps/epoch               : {spe}  (expect ~{n_in} without packing)")
        if m["peak_mem_gib"] is not None:
            print(f"    Peak VRAM                 : {m['peak_mem_gib']:.2f} GiB")
        if m["final_loss"] is not None:
            print(f"    Final train loss          : {m['final_loss']:.5f}")
        print()

    # Side-by-side comparison
    keys = list(results.keys())
    if len(keys) == 2:
        nopack_key = next((k for k in keys if "nopack" in k), keys[0])
        pack_key   = next((k for k in keys if "nopack" not in k and "pack_" in k), keys[1])
        mp = results.get(pack_key, {})
        mn = results.get(nopack_key, {})
        sp = mp.get("total_steps", 0)
        sn = mn.get("total_steps", 0)
        if sp and sn:
            speedup = sn / sp
            print(f"Packing reduced steps by {speedup:.2f}x  ({sn} → {sp} steps/epoch)")
            print()
            if speedup > 1.2:
                print("✓  EVIDENCE: packing is combining multiple sequences per step.")
            elif speedup > 0.8:
                print("~  Steps roughly equal — samples may already exceed half packing_length,")
                print("   preventing 2-at-a-time combining.  Try a larger max_length or shorter samples.")
            else:
                print("✗  Packed run took MORE steps — something unexpected; check logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for p in (TRAIN_JSONL, VAL_JSONL):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    n_train = sum(1 for _ in TRAIN_JSONL.open())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    swift_bin = _resolve_swift_bin()

    run_names = {
        "nopack": f"packtest_nopack_{ts}",
        "pack":   f"packtest_pack_{ts}",
    }

    results: dict[str, dict] = {}

    # ── Run A: no packing ─────────────────────────────────────────
    cmd_nopack = _build_cmd(swift_bin, args.model, run_names["nopack"], packing=False)
    log_nopack = _run(cmd_nopack, run_names["nopack"], args.dry_run)

    # ── Run B: packing ────────────────────────────────────────────
    cmd_pack = _build_cmd(swift_bin, args.model, run_names["pack"], packing=True)
    log_pack = _run(cmd_pack, run_names["pack"], args.dry_run)

    if not args.dry_run:
        results[run_names["nopack"]] = _parse_log(log_nopack)
        results[run_names["pack"]]   = _parse_log(log_pack)
        _report(results, n_train)
    else:
        print("\n[dry-run] Would compare:")
        for label, name in run_names.items():
            print(f"  {label}: {OUTPUT_ROOT / name}/train.log")


if __name__ == "__main__":
    main()
