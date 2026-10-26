#!/usr/bin/env python3
"""
Launch DeepSpeed ZeRO-3 SFT for Qwen2.5-Omni via MS-Swift (non-Megatron path).

Uses HuggingFace Trainer + DeepSpeed ZeRO Stage 3, which shards model params,
optimizer states, and gradients across GPUs. Much lower CPU RAM overhead than
the Megatron TP path (no heavy per-rank framework buffers).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--cached-dataset", type=Path, default=None,
                        help="Pre-computed HF Dataset dir (skips AddLengthPreprocessor)")
    parser.add_argument("--cached-val-dataset", type=Path, default=None,
                        help="Pre-computed HF Dataset dir for val split")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/deepspeed_sft"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)

    # Model / sequence config
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--rope-scaling", type=str, default=None,
                        help="RoPE scaling type: 'dynamic', 'linear', 'yarn', or JSON dict")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Target model length for RoPE scaling factor computation")

    # Training config
    parser.add_argument("--packing", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--train-iters", type=int, default=None,
                        help="Max training steps (overrides epochs)")
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataset-num-proc", type=int, default=8)

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--target-modules", default="all-linear")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # DeepSpeed
    parser.add_argument("--deepspeed", default="zero3",
                        help="DeepSpeed config: zero2, zero3, zero3_offload, or path to JSON")
    parser.add_argument("--bf16", action="store_true", default=True)

    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _infer_nproc(cuda_visible_devices: str | None) -> int:
    if not cuda_visible_devices:
        return 1
    toks = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    return max(1, len(toks))


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def _nvidia_lib_paths() -> list[str]:
    try:
        import nvidia
    except Exception:
        return []
    root = Path(nvidia.__file__).resolve().parent
    return sorted(str(p) for p in root.glob("*/lib") if p.is_dir())


def _build_swift_cmd(args: argparse.Namespace, output_dir: Path, passthrough: list[str]) -> list[str]:
    swift_bin = shutil.which("swift") or str(Path(sys.executable).parent / "swift")
    cmd = [
        swift_bin, "sft",
        "--model", args.model,
        "--tuner_type", "lora",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--target_modules", args.target_modules,
        "--freeze_llm", "false",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
    ]

    # Dataset: cached_dataset OR raw jsonl, never both
    if args.cached_dataset:
        cmd.extend(["--cached_dataset", str(args.cached_dataset.resolve())])
    else:
        cmd.extend(["--dataset", str(args.train_jsonl.resolve())])
    if args.cached_val_dataset:
        cmd.extend(["--cached_val_dataset", str(args.cached_val_dataset.resolve())])
    elif args.val_jsonl:
        cmd.extend(["--val_dataset", str(args.val_jsonl.resolve())])

    cmd += [
        "--load_from_cache_file", "true",
        "--max_length", str(args.max_length),
        "--packing", args.packing,
        "--attn_impl", "flash_attn",

        # DeepSpeed
        "--deepspeed", args.deepspeed,

        # Training
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--gradient_checkpointing", str(args.gradient_checkpointing).lower(),
        "--learning_rate", str(args.lr),
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--warmup_ratio", str(args.warmup_ratio),
        "--weight_decay", str(args.weight_decay),

        # Saving / eval
        "--save_strategy", "steps",
        "--save_steps", str(args.save_steps),
        "--eval_strategy", args.eval_strategy,
        "--eval_steps", str(args.eval_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--output_dir", str(output_dir.resolve()),

        # Data loading
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--dataset_num_proc", str(args.dataset_num_proc),

        # Precision
        "--bf16", str(args.bf16).lower(),

        "--logging_steps", "1",
    ]

    # RoPE scaling
    if args.rope_scaling:
        cmd.extend(["--rope_scaling", args.rope_scaling])
    if args.max_model_len:
        cmd.extend(["--max_model_len", str(args.max_model_len)])

    # Epochs / iters
    if args.train_iters is not None:
        cmd.extend(["--max_steps", str(args.train_iters)])
    elif args.num_train_epochs is not None:
        cmd.extend(["--num_train_epochs", str(args.num_train_epochs)])
    else:
        cmd.extend(["--num_train_epochs", "1"])

    cmd.extend(passthrough)
    return cmd


def main() -> None:
    args, passthrough = parse_args()

    for p in [args.train_jsonl]:
        if p and not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    run_name = args.run_name or f"ds_z3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    nproc = args.nproc_per_node or _infer_nproc(args.cuda_visible_devices)
    port = args.master_port or _pick_free_port()

    # swift sft auto-launches torchrun when NPROC_PER_NODE is set
    cmd = _build_swift_cmd(args, output_dir, passthrough)

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["NPROC_PER_NODE"] = str(nproc)
    env["MASTER_PORT"] = str(port)
    env["ENABLE_AUDIO_OUTPUT"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    libs = _nvidia_lib_paths()
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs)

    print(f"model: {args.model}")
    print(f"train: {args.train_jsonl}")
    print(f"val:   {args.val_jsonl}")
    print(f"out:   {output_dir}")
    print(f"deepspeed: {args.deepspeed}")
    print(f"max_length={args.max_length} lora_rank={args.lora_rank}")
    print(f"rope_scaling={args.rope_scaling} max_model_len={args.max_model_len}")
    print(f"packing={args.packing} gradient_checkpointing={args.gradient_checkpointing}")
    print(f"nproc={nproc} master_port={port}")
    print(f"\nCommand:\n{' '.join(shlex.quote(x) for x in cmd)}")

    if args.dry_run:
        print("\nDry run only; not launching.")
        return

    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    pgid = proc.pid
    try:
        proc.wait()
    except KeyboardInterrupt:
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except Exception:
            os.killpg(pgid, signal.SIGKILL)
    if proc.returncode and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


if __name__ == "__main__":
    main()
