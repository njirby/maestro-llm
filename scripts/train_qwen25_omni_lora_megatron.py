#!/usr/bin/env python3
"""
Launch Megatron-SWIFT SFT for Qwen2.5-Omni.

Supports both LoRA and full fine-tuning modes on 4x24GB GPUs for long-context
probing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prepared/omni_lua_sft_full_90_10"))
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qwen25_omni_lora_megatron"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--megatron-bin", default="megatron")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--packing", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=4)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--gradient-accumulation-fusion", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--tuner-type", default="lora", choices=["lora", "full"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument(
        "--allow-unsupported-quantization",
        action="store_true",
        help="Allow non-FP8 quantization flags to pass through even though the Megatron path here does not "
        "apply them to the actual model build.",
    )
    parser.add_argument(
        "--disable-swift-audit",
        action="store_true",
        help="Disable the injected Megatron subprocess audit that records effective quantization state and live "
        "module-class counts after model construction.",
    )
    parser.add_argument(
        "--strict-effective-quantization",
        action="store_true",
        help="Abort inside the Megatron subprocess if quantization is requested but the effective Megatron model "
        "metadata still shows an unquantized base model.",
    )
    parser.add_argument(
        "--enable-experimental-bnb-tp",
        action="store_true",
        help="Enable the repo-local experimental patch that replaces Megatron TE row/column TP linear layers with "
        "BNB 4-bit wrappers after checkpoint load and before LoRA preparation.",
    )
    parser.add_argument(
        "--bnb-min-replaced-total",
        type=int,
        default=1,
        help="Minimum TP linear replacements required for BNB patch runs before failing.",
    )
    parser.add_argument(
        "--bnb-min-uint8-ratio",
        type=float,
        default=0.01,
        help="Minimum uint8 param fraction required for BNB runs before failing (0..1).",
    )
    parser.add_argument("--save-optim", action="store_true",
                        help="Save optimizer state in checkpoints (required for resume)")
    parser.add_argument("--save-rng", action="store_true",
                        help="Save RNG state in checkpoints (required for resume)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint dir (e.g. .../checkpoint-4) to resume from")
    parser.add_argument("--enable-audio-output", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _find_megatron_bin(candidate: str) -> str:
    if os.path.sep in candidate:
        p = Path(candidate)
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
        raise FileNotFoundError(f"Megatron executable not found: {candidate}")
    found = shutil.which(candidate)
    if found:
        return found
    local_venv = Path.cwd() / ".venv" / "bin" / candidate
    if local_venv.exists() and os.access(local_venv, os.X_OK):
        return str(local_venv.resolve())
    sibling = Path(sys.executable).parent / candidate
    if sibling.exists() and os.access(sibling, os.X_OK):
        return str(sibling.resolve())
    raise FileNotFoundError(f"Megatron executable '{candidate}' not found in PATH or venv/bin.")


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
        import nvidia  # type: ignore
    except Exception:
        return []
    root = Path(nvidia.__file__).resolve().parent
    return sorted(str(p) for p in root.glob("*/lib") if p.is_dir())


def _get_passthrough_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for i, tok in enumerate(tokens):
        if tok == flag:
            if i + 1 >= len(tokens):
                return ""
            return tokens[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


def _validate_quantization_passthrough(args: argparse.Namespace, passthrough: list[str]) -> None:
    quant_method = _get_passthrough_value(passthrough, "--quant_method")
    quant_bits = _get_passthrough_value(passthrough, "--quant_bits")
    if quant_method is None and quant_bits is None:
        return
    normalized = (quant_method or "").strip().lower()
    if normalized in {"", "fp8"}:
        return
    if args.allow_unsupported_quantization:
        print(
            "warning: allowing non-FP8 quantization flags to pass through. "
            "In this repo's Megatron path they are expected to be ineffective for base-weight memory."
        )
        return
    raise ValueError(
        "Non-FP8 quantization flags were passed to the Megatron launcher "
        f"(quant_method={quant_method!r}, quant_bits={quant_bits!r}), but this backend currently builds native "
        "Megatron/TransformerEngine modules plus LoRA adapters rather than a BNB/HQQ/Quanto quantized base model. "
        "Use a non-Megatron backend for real QLoRA, or rerun with --allow-unsupported-quantization if you only "
        "want to reproduce the no-op behavior."
    )


def _requested_quant_method(passthrough: list[str]) -> str:
    return (_get_passthrough_value(passthrough, "--quant_method") or "").strip().lower()


def _requested_quant_bits(passthrough: list[str]) -> str:
    return (_get_passthrough_value(passthrough, "--quant_bits") or "").strip()


def _is_bnb_request(passthrough: list[str]) -> bool:
    return _requested_quant_method(passthrough) == "bnb"


def _with_pythonpath_prefix(env: dict[str, str], prefix: str) -> None:
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"


def _build_command(
    megatron_bin: str,
    args: argparse.Namespace,
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        megatron_bin,
        "sft",
        "--model",
        args.model,
        "--save_safetensors",
        "true",
        "--dataset",
        str(train_jsonl.resolve()),
        "--val_dataset",
        str(val_jsonl.resolve()),
        "--load_from_cache_file",
        "true",
        "--tuner_type",
        args.tuner_type,
        "--tensor_model_parallel_size",
        str(args.tensor_model_parallel_size),
        "--context_parallel_size",
        str(args.context_parallel_size),
        "--sequence_parallel",
        args.sequence_parallel,
        "--freeze_llm",
        "false",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--packing",
        args.packing,
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--global_batch_size",
        str(args.global_batch_size),
        "--recompute_granularity",
        "full",
        "--recompute_method",
        "uniform",
        "--recompute_num_layers",
        "1",
        "--finetune",
        "true",
        "--cross_entropy_loss_fusion",
        "true",
        "--gradient_accumulation_fusion",
        args.gradient_accumulation_fusion,
        "--lr",
        str(args.lr),
        "--lr_warmup_fraction",
        str(args.lr_warmup_fraction),
        "--min_lr",
        str(args.min_lr),
        "--eval_iters",
        str(args.eval_iters),
        "--save_steps",
        str(args.save_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--output_dir",
        str(output_dir.resolve()),
        "--max_length",
        str(args.max_length),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--dataset_num_proc",
        str(args.dataset_num_proc),
        "--attention_backend",
        "flash",
    ]
    if not args.save_optim:
        cmd.extend(["--no_save_optim", "true"])
    if not args.save_rng:
        cmd.extend(["--no_save_rng", "true"])
    if args.tuner_type == "lora":
        cmd.extend([
            "--merge_lora",
            "false",
            "--lora_rank",
            str(args.lora_rank),
            "--lora_alpha",
            str(args.lora_alpha),
            "--target_modules",
            args.target_modules,
        ])
    if args.train_iters is not None:
        cmd.extend(["--train_iters", str(args.train_iters)])
    elif args.num_train_epochs is not None:
        cmd.extend(["--num_train_epochs", str(args.num_train_epochs)])
    else:
        cmd.extend(["--num_train_epochs", "1"])
    cmd.extend(passthrough)
    return cmd


def _terminate_process_group(pgid: int, *, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _latest_logging_jsonl(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("v*/logging.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def _has_completion_marker(output_dir: Path) -> bool:
    log_path = _latest_logging_jsonl(output_dir)
    if log_path is None or not log_path.exists():
        return False
    try:
        with log_path.open("r", encoding="utf-8") as f:
            tail = f.readlines()[-8:]
    except Exception:
        return False
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        last_ckpt = payload.get("last_model_checkpoint")
        best_ckpt = payload.get("best_model_checkpoint")
        if (isinstance(last_ckpt, str) and last_ckpt.strip()) or (
            isinstance(best_ckpt, str) and best_ckpt.strip()
        ):
            return True
    return False


def _run_and_reap(cmd: list[str], env: dict[str, str], output_dir: Path) -> None:
    # Use a dedicated process group so we can reliably clean up torchrun/megatron
    # children if the launcher exits early or leaves idle worker processes behind.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    pgid = proc.pid
    saw_completion = False
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if _has_completion_marker(output_dir):
                saw_completion = True
                break
            time.sleep(1.0)
        if saw_completion and proc.poll() is None:
            _terminate_process_group(pgid, force=False)
            try:
                proc.wait(timeout=8)
            except Exception:
                _terminate_process_group(pgid, force=True)
                proc.wait(timeout=5)
            rc = proc.returncode if proc.returncode is not None else 0
            if rc in (-signal.SIGTERM, -signal.SIGKILL):
                rc = 0
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        _terminate_process_group(pgid, force=False)
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        _terminate_process_group(pgid, force=True)


def _get_compute_gpu_pids() -> set[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.add(int(line))
        except ValueError:
            continue
    return pids


def _check_orphan_workers(run_name: str) -> list[int]:
    suspicious_tokens = (
        run_name,
        "swift/cli/_megatron/sft.py",
        "torch.distributed.run",
        "megatron sft",
    )
    gpu_pids = _get_compute_gpu_pids()
    if not gpu_pids:
        return []
    survivors: list[int] = []
    for pid in sorted(gpu_pids):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if any(tok in cmdline for tok in suspicious_tokens):
            survivors.append(pid)
    return survivors


def _reap_orphan_workers(run_name: str, *, max_wait_s: float = 20.0) -> list[int]:
    # Best-effort cleanup for detached worker processes that can retain VRAM
    # after torchrun/megatron exits.
    deadline = time.time() + max_wait_s
    survivors = _check_orphan_workers(run_name)
    while survivors and time.time() < deadline:
        for pid in survivors:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            except Exception:
                pgid = None
            if pgid is not None and pgid > 1:
                _terminate_process_group(pgid, force=False)
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                except Exception:
                    pass
        time.sleep(1.0)
        survivors = _check_orphan_workers(run_name)
    if survivors:
        for pid in survivors:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            except Exception:
                pgid = None
            if pgid is not None and pgid > 1:
                _terminate_process_group(pgid, force=True)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except Exception:
                    pass
        time.sleep(0.8)
        survivors = _check_orphan_workers(run_name)
    return survivors


def main() -> None:
    args, passthrough = parse_args()
    _validate_quantization_passthrough(args, passthrough)
    requested_quant_method = _requested_quant_method(passthrough)
    requested_quant_bits = _requested_quant_bits(passthrough)
    bnb_requested = _is_bnb_request(passthrough)
    auto_enable_bnb_tp = bnb_requested
    if auto_enable_bnb_tp and not args.enable_experimental_bnb_tp:
        args.enable_experimental_bnb_tp = True
    if bnb_requested and args.disable_swift_audit:
        raise ValueError("BNB Megatron runs require swift audit enabled; remove --disable-swift-audit.")
    effective_strict_quant = bool(args.strict_effective_quantization or bnb_requested)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    train_jsonl = args.train_jsonl.expanduser().resolve() if args.train_jsonl else dataset_dir / "train.jsonl"
    val_jsonl = args.val_jsonl.expanduser().resolve() if args.val_jsonl else dataset_dir / "val.jsonl"
    for p in (train_jsonl, val_jsonl):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset split: {p}")

    megatron_bin = _find_megatron_bin(args.megatron_bin)

    run_name = args.run_name or f"omni_megatron_{args.tuner_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_env = [
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "MODELSCOPE_CACHE",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]
    env = {k: os.environ[k] for k in keep_env if k in os.environ}
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    nproc = args.nproc_per_node or _infer_nproc(env.get("CUDA_VISIBLE_DEVICES"))
    env["NPROC_PER_NODE"] = str(nproc)
    env["MASTER_PORT"] = str(args.master_port or env.get("MASTER_PORT") or _pick_free_port())
    env["ENABLE_AUDIO_OUTPUT"] = str(args.enable_audio_output)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not args.disable_swift_audit or args.enable_experimental_bnb_tp:
        audit_dir = str((Path.cwd() / "tools" / "swift_megatron_audit").resolve())
        _with_pythonpath_prefix(env, audit_dir)
    if not args.disable_swift_audit:
        env["MAESTRO_SWIFT_MEGATRON_AUDIT"] = "1"
        env["MAESTRO_SWIFT_MEGATRON_AUDIT_DIR"] = str(output_dir.resolve())
        env["MAESTRO_SWIFT_MEGATRON_AUDIT_STRICT"] = "1" if effective_strict_quant else "0"
    if args.enable_experimental_bnb_tp:
        env["MAESTRO_SWIFT_MEGATRON_ENABLE_BNB_TP"] = "1"
        env["MAESTRO_SWIFT_REQUESTED_QUANT_METHOD"] = requested_quant_method
        env["MAESTRO_SWIFT_MEGATRON_BNB_QUANT_TYPE"] = "nf4"
        env["MAESTRO_SWIFT_MEGATRON_BNB_MIN_REPLACED_TOTAL"] = str(max(0, int(args.bnb_min_replaced_total)))
        env["MAESTRO_SWIFT_MEGATRON_BNB_MIN_UINT8_RATIO"] = str(max(0.0, float(args.bnb_min_uint8_ratio)))
    if bnb_requested:
        env["MAESTRO_SWIFT_EXPECT_BNB_EFFECTIVE"] = "1"
    if (args.save_optim or args.resume_from) and args.tuner_type == "lora":
        env["MAESTRO_SWIFT_CHECKPOINT_LORA_FIX"] = "1"
    if args.resume_from:
        env["MAESTRO_RESUME_FROM"] = str(Path(args.resume_from).resolve())

    libs = _nvidia_lib_paths()
    if libs:
        # Keep runtime deterministic: use only the venv CUDA/NCCL/cuDNN libs.
        # Inherited host/user-site CUDA libs can cause TE/cuDNN symbol mismatches.
        env["LD_LIBRARY_PATH"] = ":".join(libs)

    cmd = _build_command(megatron_bin, args, train_jsonl, val_jsonl, output_dir, passthrough)

    print(f"megatron binary: {megatron_bin}")
    print(f"model: {args.model}")
    print(f"train: {train_jsonl}")
    print(f"val:   {val_jsonl}")
    print(f"out:   {output_dir}")
    print(f"tuner_type={args.tuner_type}")
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"NPROC_PER_NODE={env['NPROC_PER_NODE']}")
    print(f"MASTER_PORT={env['MASTER_PORT']}")
    print(f"ENABLE_AUDIO_OUTPUT={env['ENABLE_AUDIO_OUTPUT']}")
    print(f"swift_audit={'disabled' if args.disable_swift_audit else 'enabled'}")
    print(f"strict_effective_quantization={effective_strict_quant}")
    print(f"experimental_bnb_tp={args.enable_experimental_bnb_tp}")
    print(
        "requested_quant="
        f"{requested_quant_method or 'none'}:{requested_quant_bits or 'none'} "
        f"auto_enable_bnb_tp={auto_enable_bnb_tp}"
    )
    if bnb_requested:
        print(
            "bnb_contract=must_replace_tp_linears_and_expose_linear4bit_uint8_or_run_fails"
        )
        print(f"bnb_min_replaced_total={max(0, int(args.bnb_min_replaced_total))}")
        print(f"bnb_min_uint8_ratio={max(0.0, float(args.bnb_min_uint8_ratio)):.6f}")
    print(f"max_length={args.max_length} tp={args.tensor_model_parallel_size} cp={args.context_parallel_size}")
    print(f"sequence_parallel={args.sequence_parallel} packing={args.packing}")
    print("\nCommand:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        print("\nDry run only; not launching training.")
        return

    _run_and_reap(cmd, env, output_dir)
    survivors = _reap_orphan_workers(run_name)
    if survivors:
        raise RuntimeError(
            "Megatron launcher orphan check failed; compute workers still alive after run: "
            + ",".join(str(pid) for pid in survivors)
        )


if __name__ == "__main__":
    main()
