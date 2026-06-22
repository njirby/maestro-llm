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


def _gpu_mem_mb() -> int:
    try:
        import torch
        return torch.cuda.memory_allocated() // (1024 * 1024)
    except Exception:
        return -1


def _log_ckpt(rank: int, tag: str) -> None:
    if rank == 0:
        import time
        print(f"MAESTRO_CKPT [{tag}] gpu_mem={_gpu_mem_mb()}MB t={time.time():.1f}", flush=True)


def _move_state_to_cpu(obj):
    """Deep-copy a state dict tree, moving all tensors to CPU."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_state_to_cpu(x) for x in obj)
    return obj


def _move_state_to_device(obj, device):
    """Move all tensors in a state dict tree to device, in place."""
    import torch
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                obj[k] = v.to(device)
            elif isinstance(v, (dict, list)):
                _move_state_to_device(v, device)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor):
                obj[i] = v.to(device)
            elif isinstance(v, (dict, list)):
                _move_state_to_device(v, device)


def _install_checkpoint_fix() -> None:
    """Bypass dist_checkpointing for LoRA optimizer state to avoid DCP crashes.

    With TP>1 + LoRA, dist_checkpointing.save() either deadlocks on NCCL
    (model shards) or hard-crashes the machine (optimizer ShardedTensors going
    through DCP's pinned-memory + fork pipeline).

    This patch splits checkpoint saving into two phases:
      Phase 1: LoRA safetensors + tiny common.pt via the proven no_save_optim
               path (same as v7 — no NCCL, no DCP ShardedTensors).
      Phase 2: Optimizer + RNG + scheduler state saved per-rank via plain
               torch.save() — no DCP, no NCCL, no pinned memory, no fork.
    """
    if not _env_flag("MAESTRO_SWIFT_CHECKPOINT_LORA_FIX"):
        return

    import torch
    from swift.megatron.trainers.base import BaseMegatronTrainer

    original_save_checkpoint = BaseMegatronTrainer.save_checkpoint

    def patched_save_checkpoint(self):
        args = self.args
        if not (args.save_safetensors and args.tuner_type == "lora" and not args.merge_lora):
            return original_save_checkpoint(self)

        rank = _get_rank()
        _log_ckpt(rank, "save_start")

        # --- Phase 1: LoRA safetensors + tiny common.pt (same as v7) ---
        orig_no_save_optim = args.no_save_optim
        orig_no_save_rng = args.no_save_rng
        args.no_save_optim = True
        args.no_save_rng = True
        try:
            original_save_checkpoint(self)
        finally:
            args.no_save_optim = orig_no_save_optim
            args.no_save_rng = orig_no_save_rng

        _log_ckpt(rank, "phase1_done")

        # --- Phase 2: optimizer + RNG state via torch.save ---
        if orig_no_save_optim or self.optimizer is None:
            _log_ckpt(rank, "skip_phase2_no_optim")
            return

        iteration = self.state.iteration
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{iteration}")
        extra_path = os.path.join(ckpt_dir, f"maestro_extra_rank{rank}.pt")

        extra = {}

        if not orig_no_save_rng:
            try:
                from swift.megatron.utils.megatron_lm_utils import _get_rng_state
                rng = _get_rng_state()
                extra["rng_state"] = rng.data if hasattr(rng, "data") else rng
            except Exception as exc:
                if rank == 0:
                    print(f"MAESTRO_CKPT: rng_state capture failed: {exc!r}", flush=True)

        chained = getattr(self.optimizer, "chained_optimizers", [self.optimizer])

        extra["opt_metadata"] = [opt.state_dict() for opt in chained]

        inner_states = []
        for opt in chained:
            raw = opt.optimizer.state_dict()
            inner_states.append(_move_state_to_cpu(raw))
        extra["inner_opt_states"] = inner_states

        if self.opt_param_scheduler is not None:
            extra["opt_param_scheduler"] = self.opt_param_scheduler.state_dict()

        _log_ckpt(rank, "phase2_cpu_done")

        torch.save(extra, extra_path)
        del extra

        _log_ckpt(rank, "phase2_saved")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        _log_ckpt(rank, "save_complete")

    BaseMegatronTrainer.save_checkpoint = patched_save_checkpoint

    # --- Load-side: restore adapter + optimizer/RNG from maestro checkpoint ---
    original_load_checkpoint = BaseMegatronTrainer._load_checkpoint

    def patched_load_checkpoint(self):
        resume_dir = os.environ.get("MAESTRO_RESUME_FROM")
        if not resume_dir:
            return original_load_checkpoint(self)

        args = self.args
        rank = _get_rank()
        extra_path = os.path.join(resume_dir, f"maestro_extra_rank{rank}.pt")

        if not os.path.exists(extra_path):
            _log_ckpt(rank, f"load_no_extra_at {resume_dir}")
            return original_load_checkpoint(self)

        _log_ckpt(rank, "load_resume_start")

        # 1) Load adapter weights via bridge (safetensors, no DCP)
        self.bridge.load_weights(
            self.wrapped_models, resume_dir,
            is_peft_format=True, adapter_name="default")
        _log_ckpt(rank, "load_adapter_done")

        # 2) Sync model params into optimizer buffers
        if self.optimizer is not None:
            self.optimizer.reload_model_params()

        # 3) Load optimizer + RNG + scheduler from extra file
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)

        if not args.no_load_optim and self.optimizer is not None and "inner_opt_states" in extra:
            chained = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
            for opt, saved_inner in zip(chained, extra["inner_opt_states"]):
                device = next(iter(opt.optimizer.param_groups[0]["params"])).device
                _move_state_to_device(saved_inner, device)
                opt.optimizer.load_state_dict(saved_inner)
            _log_ckpt(rank, "load_inner_opt_done")

        if not args.no_load_optim and self.opt_param_scheduler is not None:
            if "opt_param_scheduler" in extra:
                self.opt_param_scheduler.load_state_dict(extra["opt_param_scheduler"])
                _log_ckpt(rank, "load_scheduler_done")

        if not args.no_load_rng and "rng_state" in extra:
            import random
            import numpy as np
            import megatron.core.tensor_parallel as tensor_parallel
            from megatron.core import mpu

            rng_state = extra["rng_state"]
            if hasattr(rng_state, "data"):
                rng_state = rng_state.data
            if args.data_parallel_random_init:
                rng_state = rng_state[mpu.get_data_parallel_rank()]
            else:
                rng_state = rng_state[0]
            random.setstate(rng_state["random_rng_state"])
            np.random.set_state(rng_state["np_rng_state"])
            torch.set_rng_state(rng_state["torch_rng_state"])
            torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
            tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])
            _log_ckpt(rank, "load_rng_done")

        # 4) Restore iteration
        tracker = os.path.join(resume_dir, "latest_checkpointed_iteration.txt")
        if os.path.exists(tracker):
            with open(tracker) as f:
                self.state.iteration = int(f.read().strip())
            _log_ckpt(rank, f"load_iteration={self.state.iteration}")

        del extra

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        _log_ckpt(rank, "load_resume_complete")

    BaseMegatronTrainer._load_checkpoint = patched_load_checkpoint


def _install_patch() -> None:
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    argv0_name = Path(argv0).name.lower()
    if argv0_name in {"pip", "pip3", "python-pip"}:
        return
    _install_bnb_tp_patch()
    _install_checkpoint_fix()
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
