from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_rank() -> int:
    try:
        import torch.distributed as dist
    except Exception:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        return 0
    try:
        return int(dist.get_rank())
    except Exception:
        return 0


def _collect_audit_payload(trainer) -> dict:
    args = trainer.args
    models = getattr(trainer, "peft_models", None) or getattr(trainer, "unwrapped_models", None) or []

    module_counter: Counter[str] = Counter()
    param_dtype_counter: Counter[str] = Counter()
    trainable_dtype_counter: Counter[str] = Counter()
    for model in models:
        for module in model.modules():
            cls = type(module)
            module_counter[f"{cls.__module__}.{cls.__name__}"] += 1
        for param in model.parameters():
            dtype = str(param.dtype)
            param_dtype_counter[dtype] += param.numel()
            if param.requires_grad:
                trainable_dtype_counter[dtype] += param.numel()

    interesting_module_counts = {
        name: count for name, count in sorted(module_counter.items())
        if "Linear4bit" in name
        or "LoraParallelLinear" in name
        or "BnbTE" in name
        or "transformer_engine" in name and "Linear" in name
    }

    model_info = getattr(args, "model_info", None)
    return {
        "rank": _get_rank(),
        "requested_quant_method": getattr(args, "quant_method", None),
        "requested_quant_bits": getattr(args, "quant_bits", None),
        "effective_model_info_quant_method": getattr(model_info, "quant_method", None),
        "effective_model_info_quant_bits": getattr(model_info, "quant_bits", None),
        "params_dtype": str(getattr(args, "params_dtype", None)),
        "module_counts": interesting_module_counts,
        "param_numel_by_dtype": dict(sorted(param_dtype_counter.items())),
        "trainable_param_numel_by_dtype": dict(sorted(trainable_dtype_counter.items())),
    }


def _write_payload(payload: dict) -> None:
    output_dir = os.environ.get("MAESTRO_SWIFT_MEGATRON_AUDIT_DIR")
    if not output_dir:
        return
    path = Path(output_dir).expanduser().resolve() / f"effective_quantization_rank{payload['rank']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _has_live_bnb_modules(payload: dict) -> bool:
    return any("Linear4bit" in name or "BnbTE" in name for name in payload.get("module_counts", {}))


def _has_uint8_params(payload: dict) -> bool:
    try:
        return int(payload.get("param_numel_by_dtype", {}).get("torch.uint8", 0)) > 0
    except Exception:
        return False


def _uint8_ratio(payload: dict) -> float:
    d = payload.get("param_numel_by_dtype", {}) or {}
    try:
        total = sum(int(v) for v in d.values())
        uint8_n = int(d.get("torch.uint8", 0))
    except Exception:
        return 0.0
    if total <= 0:
        return 0.0
    return float(uint8_n) / float(total)


def _write_bnb_effective_summary(payload: dict) -> None:
    output_dir = os.environ.get("MAESTRO_SWIFT_MEGATRON_AUDIT_DIR")
    if not output_dir or payload.get("rank") != 0:
        return
    patch_path = Path(output_dir).expanduser().resolve() / "bnb_tp_patch_rank0.json"
    patch_payload = {}
    if patch_path.exists():
        try:
            patch_payload = json.loads(patch_path.read_text())
        except Exception:
            patch_payload = {}
    replaced_total = int(patch_payload.get("replaced_total", 0) or 0)
    has_modules = _has_live_bnb_modules(payload)
    has_uint8 = _has_uint8_params(payload)
    ratio = _uint8_ratio(payload)
    try:
        min_ratio = float(os.environ.get("MAESTRO_SWIFT_MEGATRON_BNB_MIN_UINT8_RATIO", "0") or 0.0)
    except Exception:
        min_ratio = 0.0
    meets_ratio = ratio >= min_ratio
    ok = replaced_total > 0 and has_modules and has_uint8 and meets_ratio
    reason = "ok" if ok else (
        "replaced_total="
        f"{replaced_total}, has_live_bnb_modules={has_modules}, has_uint8_params={has_uint8}, "
        f"uint8_ratio={ratio:.6f}, min_uint8_ratio={min_ratio:.6f}"
    )
    summary = {
        "ok": ok,
        "reason": reason,
        "replaced_total": replaced_total,
        "has_live_bnb_modules": has_modules,
        "has_uint8_params": has_uint8,
        "uint8_ratio": ratio,
        "min_uint8_ratio": min_ratio,
        "meets_uint8_ratio": meets_ratio,
    }
    summary_path = Path(output_dir).expanduser().resolve() / "bnb_effective_summary_rank0.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _should_fail(payload: dict) -> bool:
    requested = str(payload.get("requested_quant_method") or "").strip().lower()
    effective = str(payload.get("effective_model_info_quant_method") or "").strip().lower()
    expect_bnb_effective = _env_flag("MAESTRO_SWIFT_EXPECT_BNB_EFFECTIVE")
    if requested == "bnb" and expect_bnb_effective:
        output_dir = os.environ.get("MAESTRO_SWIFT_MEGATRON_AUDIT_DIR")
        replaced_total = 0
        if output_dir:
            patch_path = Path(output_dir).expanduser().resolve() / f"bnb_tp_patch_rank{payload.get('rank', 0)}.json"
            if patch_path.exists():
                try:
                    patch_payload = json.loads(patch_path.read_text())
                    replaced_total = int(patch_payload.get("replaced_total", 0) or 0)
                except Exception:
                    replaced_total = 0
        if replaced_total <= 0:
            return True
        if not _has_live_bnb_modules(payload):
            return True
        if not _has_uint8_params(payload):
            return True
        try:
            min_ratio = float(os.environ.get("MAESTRO_SWIFT_MEGATRON_BNB_MIN_UINT8_RATIO", "0") or 0.0)
        except Exception:
            min_ratio = 0.0
        if _uint8_ratio(payload) < min_ratio:
            return True
        return False
    if _has_live_bnb_modules(payload):
        return False
    return bool(requested) and requested not in {"fp8"} and requested != effective


def _should_enable_bnb_tp_patch() -> bool:
    # Avoid recursive bitsandbytes backend probing in package-manager subprocesses
    # (e.g. "pip list | grep habana-torch-plugin"), which can explode host RAM.
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    argv0_name = Path(argv0).name.lower()
    if argv0_name in {"pip", "pip3", "python-pip"}:
        return False
    if not _env_flag("MAESTRO_SWIFT_MEGATRON_ENABLE_BNB_TP"):
        return False
    quant_method = os.environ.get("MAESTRO_SWIFT_REQUESTED_QUANT_METHOD", "").strip().lower()
    return quant_method == "bnb"


def _write_bnb_patch_payload(payload: dict) -> None:
    output_dir = os.environ.get("MAESTRO_SWIFT_MEGATRON_AUDIT_DIR")
    if not output_dir:
        return
    path = Path(output_dir).expanduser().resolve() / f"bnb_tp_patch_rank{payload['rank']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _install_bnb_tp_patch() -> None:
    if not _should_enable_bnb_tp_patch():
        return

    import torch
    import swift.megatron.trainers.base as trainer_base_mod
    import swift.megatron.utils.utils as utils_mod

    from maestro_megatron_bnb import replace_te_tp_linears_with_bnb

    original_prepare_mcore_model = utils_mod.prepare_mcore_model
    original_find_all_linears = utils_mod.find_all_linears

    def patched_find_all_linears(model, extra_layers=None):
        # Exclude child BNB implementation details from LoRA target discovery.
        # Otherwise PEFT can enqueue keys like "*.quant_linear", then fail after
        # the parent module is replaced by LoraParallelLinear.
        names = original_find_all_linears(model, extra_layers=extra_layers)
        filtered = []
        for name in names:
            if not name:
                continue
            if name == "quant_linear" or name.endswith(".quant_linear"):
                continue
            # Grouped wrappers store per-gemm BNB linears under ModuleList
            # "quant_linears"; these are implementation internals and should
            # not be LoRA replacement targets.
            if (
                name == "quant_linears"
                or name.startswith("quant_linears.")
                or ".quant_linears." in name
                or name.endswith(".quant_linears")
            ):
                continue
            filtered.append(name)
        return filtered

    def patched_prepare_mcore_model(args, model):
        compute_dtype = getattr(args, "params_dtype", None) or torch.bfloat16
        try:
            min_replaced_total = int(os.environ.get("MAESTRO_SWIFT_MEGATRON_BNB_MIN_REPLACED_TOTAL", "0") or 0)
        except Exception:
            min_replaced_total = 0
        patch_stats = replace_te_tp_linears_with_bnb(
            model,
            quant_type=os.environ.get("MAESTRO_SWIFT_MEGATRON_BNB_QUANT_TYPE", "nf4"),
            compute_dtype=compute_dtype,
            min_replaced_total=min_replaced_total,
        )
        payload = {
            "rank": _get_rank(),
            "requested_quant_method": getattr(args, "quant_method", None),
            "requested_quant_bits": getattr(args, "quant_bits", None),
            **patch_stats,
        }
        if payload["rank"] == 0:
            print("MAESTRO_MEGATRON_BNB_TP_PATCH " + json.dumps(payload, sort_keys=True), flush=True)
        _write_bnb_patch_payload(payload)
        return original_prepare_mcore_model(args, model)

    utils_mod.prepare_mcore_model = patched_prepare_mcore_model
    utils_mod.find_all_linears = patched_find_all_linears
    trainer_base_mod.prepare_mcore_model = patched_prepare_mcore_model


def _install_patch() -> None:
    _install_bnb_tp_patch()
    if not _env_flag("MAESTRO_SWIFT_MEGATRON_AUDIT"):
        return

    from swift.megatron.trainers.base import BaseMegatronTrainer

    original_prepare_model = BaseMegatronTrainer.prepare_model

    def patched_prepare_model(self):
        original_prepare_model(self)
        payload = _collect_audit_payload(self)
        if payload["rank"] == 0:
            print("MAESTRO_MEGATRON_AUDIT " + json.dumps(payload, sort_keys=True), flush=True)
        _write_payload(payload)
        _write_bnb_effective_summary(payload)
        if _env_flag("MAESTRO_SWIFT_MEGATRON_AUDIT_STRICT") and _should_fail(payload):
            raise RuntimeError(
                "Megatron effective quantization audit failed: "
                f"requested quant_method={payload['requested_quant_method']!r}, "
                "effective constraints not satisfied (model_info/module_counts/uint8/patch replacement). "
                "See bnb_effective_summary_rank0.json and effective_quantization_rank*.json for details."
            )

    BaseMegatronTrainer.prepare_model = patched_prepare_model


try:
    _install_patch()
except Exception as exc:
    print(f"MAESTRO_MEGATRON_AUDIT install failed: {exc!r}", flush=True)
