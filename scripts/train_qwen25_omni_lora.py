#!/usr/bin/env python3
"""
Launch LoRA SFT for Qwen2.5-Omni via MS-Swift.

Profiles:
  - smoke:         short sanity run (120 steps, 7B default)
  - full:          normal 1-epoch training (7B default)
  - oom_fallback:  reduced memory footprint (7B, gradient accum 16)

Model notes:
  - Qwen2.5-Omni-7B  requires 4×24GB with DeepSpeed ZeRO-3 for packing
  - Qwen2.5-Omni-3B  fits on 1×24GB at max_length≤5120; use for packing tests

Sequence packing (packing=true):
  - Requires --attn_impl flash_attn (enforced by ms-swift)
  - packing_length defaults to max_length; set max_length large enough that
    2 typical samples fit (max_length ≥ 2 × avg_sample_tokens)
  - Swift drops samples that exceed max_length rather than truncating them
    (truncation breaks packing alignment); expect filtered dataset < full size
  - With these samples (~2500 token average): max_length=5120 packs ~2 per step

Usage:
    python scripts/train_qwen25_omni_lora.py --profile smoke --dry-run
    python scripts/train_qwen25_omni_lora.py --profile full
    # single-card packing test on 3B:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_packing_multimodal.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEEPSPEED_PRESETS = {
    "zero0",
    "zero1",
    "zero2",
    "zero3",
    "zero2_offload",
    "zero3_offload",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--profile",
        choices=["smoke", "full", "oom_fallback"],
        default="smoke",
        help="Training profile preset.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Model ID or local model path. Use Qwen/Qwen2.5-Omni-3B for single-card packing tests.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/prepared/omni_lua_sft"),
        help="Directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Optional explicit train jsonl path (default: <dataset-dir>/train.jsonl).",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
        help="Optional explicit val jsonl path (default: <dataset-dir>/val.jsonl).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/qwen25_omni_lora"),
        help="Root output directory for checkpoints/logs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to <profile>_<timestamp>.",
    )
    parser.add_argument(
        "--swift-bin",
        default="swift",
        help="Swift CLI executable (default: swift).",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="NPROC_PER_NODE value (defaults to number of CUDA_VISIBLE_DEVICES if set, else 1).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--model-type",
        default="qwen2_5_omni",
        help="MS-Swift model_type (default: qwen2_5_omni).",
    )
    parser.add_argument(
        "--template",
        default="qwen2_5_omni",
        help="MS-Swift template (default: qwen2_5_omni).",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        default=Path("configs/deepspeed_zero3.json"),
        help="DeepSpeed config JSON used by default for multi-GPU runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and exit without running.",
    )
    parser.add_argument(
        "--enable-audio-output",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set ENABLE_AUDIO_OUTPUT env for Qwen2.5-Omni (default: 0 for text-only training).",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _infer_nproc(cuda_visible_devices: str | None) -> int:
    if not cuda_visible_devices:
        return 1
    tokens = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    return max(1, len(tokens))


def _profile_args(profile: str) -> dict[str, str]:
    if profile == "smoke":
        return {
            "num_train_epochs": "1",
            "max_steps": "120",
            "per_device_train_batch_size": "1",
            "per_device_eval_batch_size": "1",
            "gradient_accumulation_steps": "8",
            "learning_rate": "1e-4",
            "max_length": "8192",
            "packing": "true",
            "attn_impl": "flash_attn",
            "logging_steps": "5",
            "save_steps": "60",
            "eval_steps": "30",
            "dataloader_num_workers": "2",
        }
    if profile == "oom_fallback":
        return {
            "num_train_epochs": "1",
            "per_device_train_batch_size": "1",
            "per_device_eval_batch_size": "1",
            "gradient_accumulation_steps": "16",
            "learning_rate": "8e-5",
            "max_length": "6144",
            "packing": "true",
            "attn_impl": "flash_attn",
            "logging_steps": "20",
            "save_steps": "200",
            "eval_steps": "100",
            "dataloader_num_workers": "2",
        }
    return {
        "num_train_epochs": "1",
        "per_device_train_batch_size": "1",
        "per_device_eval_batch_size": "1",
        "gradient_accumulation_steps": "8",
        "learning_rate": "1e-4",
        "max_length": "8192",
        "packing": "true",
        "attn_impl": "flash_attn",
        "logging_steps": "20",
        "save_steps": "200",
        "eval_steps": "100",
        "dataloader_num_workers": "4",
    }


def _resolve_swift_bin(candidate: str) -> str | None:
    if os.path.sep in candidate:
        p = Path(candidate)
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
        return None

    found = shutil.which(candidate)
    if found:
        return found

    if candidate == "swift":
        sibling = Path(sys.executable).parent / "swift"
        if sibling.exists() and os.access(sibling, os.X_OK):
            return str(sibling.resolve())
    return None


def _normalize_model_arg(model: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    for i, token in enumerate(argv):
        if token == flag:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for {flag}")
            return argv[i + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _load_deepspeed_config(deepspeed_value: str) -> tuple[str, dict]:
    candidate = Path(deepspeed_value).expanduser()
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as f:
            return str(candidate.resolve()), json.load(f)
    try:
        parsed = json.loads(deepspeed_value)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Could not parse --deepspeed value. Use an existing JSON file path or valid JSON string."
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError("DeepSpeed config must decode to a JSON object.")
    return "<inline-json>", parsed


def _validate_deepspeed_config(deepspeed_value: str) -> str:
    if deepspeed_value in DEEPSPEED_PRESETS:
        raise ValueError(
            "DeepSpeed preset configs (e.g., 'zero3') are disabled for this launcher because they can trigger "
            "HF/DS scheduler param-group mismatches with this stack. Use a JSON config with both "
            "'optimizer' and 'scheduler' blocks (default: configs/deepspeed_zero3.json)."
        )
    source, config = _load_deepspeed_config(deepspeed_value)
    if "optimizer" not in config:
        raise ValueError(f"DeepSpeed config '{source}' is missing required 'optimizer' block.")
    if "scheduler" not in config:
        raise ValueError(f"DeepSpeed config '{source}' is missing required 'scheduler' block.")
    return source


def _build_command(
    swift_exec: str,
    args: argparse.Namespace,
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    passthrough: list[str],
) -> list[str]:
    prof = _profile_args(args.profile)
    cmd = [
        swift_exec,
        "sft",
        "--model",
        _normalize_model_arg(args.model),
        "--model_type",
        args.model_type,
        "--template",
        args.template,
        "--train_type",
        "lora",
        "--dataset",
        str(train_jsonl.resolve()),
        "--val_dataset",
        str(val_jsonl.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--lora_rank",
        "16",
        "--lora_alpha",
        "32",
        "--target_modules",
        "all-linear",
        "--warmup_ratio",
        "0.03",
    ]
    for key, value in prof.items():
        cmd.extend([f"--{key}", value])
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    args, passthrough = parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    train_jsonl = (args.train_jsonl.expanduser().resolve() if args.train_jsonl else (dataset_dir / "train.jsonl"))
    val_jsonl = (args.val_jsonl.expanduser().resolve() if args.val_jsonl else (dataset_dir / "val.jsonl"))
    for p in (train_jsonl, val_jsonl):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset split: {p}")

    run_name = args.run_name or f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    swift_path = _resolve_swift_bin(args.swift_bin)
    if swift_path is None:
        raise FileNotFoundError(
            f"Could not find Swift executable '{args.swift_bin}'. "
            "Install ms-swift in the active environment first."
        )

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    nproc = args.nproc_per_node or _infer_nproc(cuda_visible)
    env["NPROC_PER_NODE"] = str(nproc)
    env["ENABLE_AUDIO_OUTPUT"] = str(args.enable_audio_output)

    deepspeed_value = _extract_flag_value(passthrough, "--deepspeed")
    deepspeed_source = None
    if deepspeed_value is None and nproc > 1:
        default_ds = args.deepspeed_config.expanduser().resolve()
        if not default_ds.exists():
            raise FileNotFoundError(
                f"Expected DeepSpeed config at {default_ds} for multi-GPU training. "
                "Pass --deepspeed-config <path> or provide --deepspeed in passthrough args."
            )
        passthrough = [*passthrough, "--deepspeed", str(default_ds)]
        deepspeed_value = str(default_ds)
    if deepspeed_value is not None:
        deepspeed_source = _validate_deepspeed_config(deepspeed_value)

    # Helps stability on some 30xx multi-GPU rigs.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = _build_command(swift_path, args, train_jsonl, val_jsonl, output_dir, passthrough)

    print(f"swift binary: {swift_path}")
    print(f"profile: {args.profile}")
    print(f"train: {train_jsonl}")
    print(f"val:   {val_jsonl}")
    print(f"out:   {output_dir}")
    print(f"NPROC_PER_NODE={env['NPROC_PER_NODE']}")
    print(f"ENABLE_AUDIO_OUTPUT={env['ENABLE_AUDIO_OUTPUT']}")
    if deepspeed_source:
        print(f"deepspeed: {deepspeed_source}")
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    print("\nCommand:")
    print(" ".join(shlex.quote(token) for token in cmd))

    if args.dry_run:
        print("\nDry run only; not launching training.")
        return

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
