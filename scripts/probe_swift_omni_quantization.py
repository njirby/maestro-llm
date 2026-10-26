#!/usr/bin/env python3
"""
Probe effective quantization behavior for the plain ms-swift / HF model path.

This script loads a Qwen Omni model through Swift's non-Megatron loader,
optionally applies LoRA adapters, and reports:
  - the effective Swift/BaseArguments quantization state
  - the loaded model's quantization config
  - module class counts (including bitsandbytes classes if present)
  - parameter dtype counts
  - single-GPU memory deltas during model load

Examples:
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_swift_omni_quantization.py \
    --model ~/.cache/modelscope/hub/models/Qwen/Qwen2___5-Omni-3B \
    --model-type qwen2_5_omni

  CUDA_VISIBLE_DEVICES=0 python scripts/probe_swift_omni_quantization.py \
    --model ~/.cache/modelscope/hub/models/Qwen/Qwen2___5-Omni-3B \
    --model-type qwen2_5_omni \
    --quant-method bnb \
    --quant-bits 4
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from swift import Swift
from swift.arguments.base_args.base_args import BaseArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="Model ID or local path.")
    parser.add_argument("--model-type", required=True, help="Swift model_type, e.g. qwen2_5_omni.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quant-method", default=None, choices=["bnb", "hqq", "quanto", "eetq", "fp8", None])
    parser.add_argument("--quant-bits", type=int, default=None)
    parser.add_argument("--attn-impl", default=None)
    parser.add_argument("--device-map", default="single", choices=["single", "auto", "cpu"])
    parser.add_argument("--max-memory-gib", type=int, default=None, help="Optional single-device memory cap.")
    parser.add_argument("--enable-audio-output", type=int, choices=[0, 1], default=0)
    parser.add_argument("--apply-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="all-linear")
    return parser.parse_args()


def _resolve_dtype(raw: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[raw]


def _build_device_map(raw: str) -> Any:
    if raw == "auto":
        return "auto"
    if raw == "cpu":
        return "cpu"
    return {"": 0}


def _build_max_memory(raw_gib: int | None, device_map: Any) -> dict[int, str] | None:
    if raw_gib is None:
        return None
    if device_map == "cpu":
        return None
    if device_map == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        n = len([x for x in (visible or "0").split(",") if x.strip()])
        return {i: f"{raw_gib}GiB" for i in range(n)}
    return {0: f"{raw_gib}GiB"}


def _cuda_memory_snapshot() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "free_gib": round(free_bytes / (1024**3), 3),
        "total_gib": round(total_bytes / (1024**3), 3),
        "reserved_gib": round(torch.cuda.memory_reserved(0) / (1024**3), 3),
        "allocated_gib": round(torch.cuda.memory_allocated(0) / (1024**3), 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(0) / (1024**3), 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
    }


def _collect_module_stats(model: torch.nn.Module) -> dict[str, Any]:
    class_counts = Counter(type(module).__name__ for module in model.modules())
    fqcn_counts = Counter(f"{type(module).__module__}.{type(module).__name__}" for module in model.modules())
    interesting = {}
    for key, count in fqcn_counts.items():
        lname = key.lower()
        if any(tok in lname for tok in ("bitsandbytes", "linear4bit", "linear8bit", "lora", "quanto", "hqq")):
            interesting[key] = count
    return {
        "top_classes": class_counts.most_common(25),
        "interesting_classes": dict(sorted(interesting.items())),
    }


def _collect_param_dtypes(model: torch.nn.Module) -> dict[str, int]:
    counts = Counter()
    for _, param in model.named_parameters():
        counts[str(param.dtype)] += param.numel()
    return dict(sorted(counts.items()))


def _collect_buffer_dtypes(model: torch.nn.Module) -> dict[str, int]:
    counts = Counter()
    for _, buf in model.named_buffers():
        counts[str(buf.dtype)] += buf.numel()
    return dict(sorted(counts.items()))


def _find_quantization_config(model: torch.nn.Module) -> dict[str, Any] | None:
    cfg = getattr(model, "config", None)
    qcfg = getattr(cfg, "quantization_config", None)
    if qcfg is None:
        return None
    if hasattr(qcfg, "to_dict"):
        return qcfg.to_dict()
    return {"type": type(qcfg).__name__}


def _trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"total_params": total, "trainable_params": trainable}


def main() -> None:
    args = parse_args()
    os.environ["ENABLE_AUDIO_OUTPUT"] = str(args.enable_audio_output)

    device_map = _build_device_map(args.device_map)
    max_memory = _build_max_memory(args.max_memory_gib, device_map)
    dtype = _resolve_dtype(args.torch_dtype)

    base_args = BaseArguments(
        model=str(Path(args.model).expanduser()),
        model_type=args.model_type,
        torch_dtype=dtype,
        quant_method=args.quant_method,
        quant_bits=args.quant_bits,
        attn_impl=args.attn_impl,
        device_map=device_map,
        max_memory=max_memory,
    )

    quant_cfg = base_args.get_quantization_config()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    mem_before = _cuda_memory_snapshot()

    model, processor = base_args.get_model_processor()

    lora_error = None
    if args.apply_lora:
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
        )
        try:
            model = Swift.prepare_model(model, lora_cfg)
        except Exception as e:
            lora_error = repr(e)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_after = _cuda_memory_snapshot()

    report = {
        "model": args.model,
        "model_type": args.model_type,
        "device_map": device_map,
        "max_memory": max_memory,
        "base_args_quant_method": base_args.quant_method,
        "base_args_quant_bits": base_args.quant_bits,
        "base_args_model_info": {
            "model_dir": base_args.model_info.model_dir,
            "quant_method": base_args.model_info.quant_method,
            "quant_bits": base_args.model_info.quant_bits,
            "torch_dtype": str(base_args.model_info.torch_dtype),
        },
        "requested_quantization_config_type": type(quant_cfg).__name__ if quant_cfg is not None else None,
        "loaded_model_quantization_config": _find_quantization_config(model),
        "memory_before": mem_before,
        "memory_after": mem_after,
        "module_stats": _collect_module_stats(model),
        "parameter_dtypes": _collect_param_dtypes(model),
        "buffer_dtypes": _collect_buffer_dtypes(model),
        "trainable_summary": _trainable_summary(model),
        "apply_lora": args.apply_lora,
        "lora_error": lora_error,
        "processor_type": type(processor).__name__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
