#!/usr/bin/env python3
"""
Recover a PEFT LoRA adapter from a DeepSpeed ZeRO-3 checkpoint whose adapter_model.safetensors is empty.

This reconstructs trainable parameters from optimizer shards, filters LoRA keys, normalizes key names
from "...lora_A.default.weight" -> "...lora_A.weight", and writes adapter_model.safetensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint_dir", type=Path, help="Checkpoint dir containing global_step*/ and adapter_config.json")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output adapter directory (default: <checkpoint_dir>/recovered_adapter).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional DeepSpeed tag (e.g. global_step2048). If omitted, auto-detected.",
    )
    return p.parse_args()


def _normalize_lora_key(key: str) -> str:
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")
    key = key.replace(".lora_embedding_A.default", ".lora_embedding_A")
    key = key.replace(".lora_embedding_B.default", ".lora_embedding_B")
    key = key.replace(".bias.default", ".bias")
    return key


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint_dir.expanduser().resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt}")

    out_dir = (args.output_dir.expanduser().resolve() if args.output_dir else (ckpt / "recovered_adapter"))
    out_dir.mkdir(parents=True, exist_ok=True)

    state = get_fp32_state_dict_from_zero_checkpoint(str(ckpt), tag=args.tag, exclude_frozen_parameters=True)
    lora_state = {}
    for k, v in state.items():
        if "lora_" not in k and "bias" not in k:
            continue
        lora_state[_normalize_lora_key(k)] = v.cpu().contiguous()

    if not lora_state:
        raise RuntimeError("No LoRA tensors found after reconstruction.")

    safetensors.torch.save_file(
        lora_state,
        str(out_dir / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )

    for name in ("adapter_config.json", "additional_config.json", "README.md"):
        src = ckpt / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    report = {
        "checkpoint_dir": str(ckpt),
        "output_dir": str(out_dir),
        "tensor_count": len(lora_state),
        "example_keys": sorted(list(lora_state.keys()))[:10],
    }
    (out_dir / "recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
