#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feasibility spike for shard-local BNB 4-bit on Megatron TP/SP row-parallel layers.",
        allow_abbrev=False,
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/megatron_bnb_row_parallel_spike"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    parser.add_argument("--sequence-parallel", action="store_true", default=True)
    parser.add_argument("--parallel-mode", choices=["row", "column"], default="row")
    parser.add_argument("--reference-impl", choices=["te", "torch"], default="te")
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=96)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--with-lora", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _infer_nproc(cuda_visible_devices: str | None) -> int:
    if not cuda_visible_devices:
        return 1
    devices = [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]
    return max(1, len(devices))


def _nvidia_lib_paths() -> list[str]:
    try:
        import nvidia  # type: ignore
    except Exception:
        return []
    root = Path(nvidia.__file__).resolve().parent
    return sorted(str(path) for path in root.glob("*/lib") if path.is_dir())


def _launch_worker(args: argparse.Namespace) -> None:
    run_name = args.run_name or f"spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    env = {key: os.environ[key] for key in keep_env if key in os.environ}
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["SPIKE_OUTPUT_DIR"] = str(output_dir)

    libs = _nvidia_lib_paths()
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs)

    nproc = args.nproc_per_node or _infer_nproc(args.cuda_visible_devices)
    master_port = args.master_port or _pick_free_port()
    worker_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--master_port",
        str(master_port),
        str(Path(__file__).resolve()),
        "--worker",
        "--output-root",
        str(args.output_root),
        "--run-name",
        run_name,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--tensor-model-parallel-size",
        str(args.tensor_model_parallel_size),
        "--parallel-mode",
        args.parallel_mode,
        "--reference-impl",
        args.reference_impl,
        "--sequence-len",
        str(args.sequence_len),
        "--batch-size",
        str(args.batch_size),
        "--input-size",
        str(args.input_size),
        "--output-size",
        str(args.output_size),
        "--dtype",
        args.dtype,
        "--quant-type",
        args.quant_type,
        "--weight-std",
        str(args.weight_std),
        "--seed",
        str(args.seed),
    ]
    if args.sequence_parallel:
        worker_cmd.append("--sequence-parallel")
    if args.with_lora:
        worker_cmd.append("--with-lora")

    print(f"output_dir={output_dir}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"nproc_per_node={nproc}")
    print(f"master_port={master_port}")
    print("command=" + " ".join(worker_cmd))
    if args.dry_run:
        print("Dry run only; not launching spike.")
        return

    subprocess.run(worker_cmd, check=True, env=env)


def _worker_main(args: argparse.Namespace) -> None:
    import bitsandbytes as bnb
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from swift.megatron.tuners.lora import LoraParallelLinear as SwiftLoraParallelLinear

    class BnbRowParallelLinear(torch.nn.Module):

        def __init__(self, base_layer, *, quant_type: str, compute_dtype: torch.dtype):
            super().__init__()
            self.input_size = getattr(base_layer, "input_size", getattr(base_layer, "in_features"))
            self.output_size = getattr(base_layer, "output_size", getattr(base_layer, "out_features"))
            self.input_is_parallel = getattr(base_layer, "input_is_parallel", True)
            self.skip_bias_add = getattr(
                base_layer, "skip_bias_add", getattr(base_layer, "te_return_bias", False)
            )
            self.sequence_parallel = base_layer.sequence_parallel
            self.tp_group = getattr(base_layer, "tp_group", getattr(base_layer, "_tp_group"))
            self.input_size_per_partition = base_layer.weight.shape[1]

            device = torch.device(torch.cuda.current_device())
            self.quant_linear = bnb.nn.Linear4bit(
                self.input_size_per_partition,
                self.output_size,
                bias=False,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                device=device,
            )
            self.quant_linear.weight = bnb.nn.Params4bit(
                base_layer.weight.detach().cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                module=self.quant_linear,
            ).to(device)

            if base_layer.bias is not None:
                self.register_buffer("bias", base_layer.bias.detach().clone(), persistent=True)
            else:
                self.bias = None

        def forward(self, input_):
            if self.input_is_parallel:
                input_parallel = input_
            else:
                input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

            original_shape = input_parallel.shape[:-1]
            flat_input = input_parallel.reshape(-1, input_parallel.shape[-1])
            flat_output = self.quant_linear(flat_input)
            output_parallel = flat_output.reshape(*original_shape, self.output_size)

            if self.sequence_parallel:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_parallel, group=self.tp_group
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_parallel, group=self.tp_group
                )

            if not self.skip_bias_add:
                output = output_ if self.bias is None else output_ + self.bias
                output_bias = None
            else:
                output = output_
                output_bias = self.bias
            return output, output_bias

    class BnbColumnParallelLinear(torch.nn.Module):

        def __init__(self, base_layer, *, quant_type: str, compute_dtype: torch.dtype):
            super().__init__()
            self.input_size = getattr(base_layer, "input_size", getattr(base_layer, "in_features"))
            self.output_size = getattr(base_layer, "output_size", getattr(base_layer, "out_features"))
            self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
            self.skip_bias_add = getattr(
                base_layer, "skip_bias_add", getattr(base_layer, "te_return_bias", False)
            )
            self.gather_output = getattr(base_layer, "gather_output", False)
            self.tp_group = getattr(base_layer, "tp_group", getattr(base_layer, "_tp_group"))
            self.output_size_per_partition = base_layer.weight.shape[0]

            device = torch.device(torch.cuda.current_device())
            self.quant_linear = bnb.nn.Linear4bit(
                self.input_size,
                self.output_size_per_partition,
                bias=False,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                device=device,
            )
            self.quant_linear.weight = bnb.nn.Params4bit(
                base_layer.weight.detach().cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                module=self.quant_linear,
            ).to(device)

            if base_layer.bias is not None:
                self.register_buffer("bias", base_layer.bias.detach().clone(), persistent=True)
            else:
                self.bias = None

        def forward(self, input_):
            if self.sequence_parallel:
                input_parallel = gather_from_sequence_parallel_region(
                    input_, tensor_parallel_output_grad=True, group=self.tp_group
                )
            else:
                input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

            original_shape = input_parallel.shape[:-1]
            flat_input = input_parallel.reshape(-1, input_parallel.shape[-1])
            flat_output = self.quant_linear(flat_input)
            output_parallel = flat_output.reshape(*original_shape, self.output_size_per_partition)

            if self.gather_output:
                raise RuntimeError("The spike expects gather_output=False.")

            if not self.skip_bias_add:
                output = output_parallel if self.bias is None else output_parallel + self.bias
                output_bias = None
            else:
                output = output_parallel
                output_bias = self.bias
            return output, output_bias

    class BnbTERowParallelLinear(TERowParallelLinear):

        def __init__(self, base_layer, *, quant_type: str, compute_dtype: torch.dtype):
            torch.nn.Module.__init__(self)
            self.config = base_layer.config
            self.input_size = getattr(base_layer, "input_size", getattr(base_layer, "in_features"))
            self.output_size = getattr(base_layer, "output_size", getattr(base_layer, "out_features"))
            self.in_features = getattr(base_layer, "in_features", self.input_size)
            self.out_features = getattr(base_layer, "out_features", self.output_size)
            self.input_is_parallel = getattr(base_layer, "input_is_parallel", True)
            self.skip_bias_add = getattr(
                base_layer, "skip_bias_add", getattr(base_layer, "te_return_bias", False)
            )
            self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
            self.parallel_mode = getattr(base_layer, "parallel_mode", "row")
            self.is_expert = getattr(base_layer, "is_expert", False)
            self.expert_parallel = getattr(base_layer, "expert_parallel", False)
            self.tp_group = getattr(base_layer, "tp_group", getattr(base_layer, "_tp_group"))
            self._tp_group = self.tp_group
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.te_return_bias = getattr(base_layer, "te_return_bias", False)
            self.use_bias = base_layer.bias is not None
            self.input_size_per_partition = base_layer.weight.shape[1]

            device = torch.device(torch.cuda.current_device())
            self.quant_linear = bnb.nn.Linear4bit(
                self.input_size_per_partition,
                self.output_size,
                bias=False,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                device=device,
            )
            self.quant_linear.weight = bnb.nn.Params4bit(
                base_layer.weight.detach().cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                module=self.quant_linear,
            ).to(device)
            self.quant_linear.weight.sequence_parallel = self.sequence_parallel
            self.quant_linear.weight.allreduce = False

            if base_layer.bias is not None:
                self.register_parameter(
                    "_bias",
                    torch.nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False),
                )
            else:
                self.register_parameter("_bias", None)

        @property
        def weight(self):
            return self.quant_linear.weight

        @property
        def bias(self):
            return self._bias

        def forward(self, input_):
            if self.input_is_parallel:
                input_parallel = input_
            else:
                input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

            original_shape = input_parallel.shape[:-1]
            flat_input = input_parallel.reshape(-1, input_parallel.shape[-1])
            flat_output = self.quant_linear(flat_input)
            output_parallel = flat_output.reshape(*original_shape, self.output_size)

            if self.sequence_parallel:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_parallel, group=self.tp_group
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_parallel, group=self.tp_group
                )

            if not self.skip_bias_add:
                output = output_ if self.bias is None else output_ + self.bias
                output_bias = None
            else:
                output = output_
                output_bias = self.bias
            return output, output_bias

    class BnbTEColumnParallelLinear(TEColumnParallelLinear):

        def __init__(self, base_layer, *, quant_type: str, compute_dtype: torch.dtype):
            torch.nn.Module.__init__(self)
            self.config = base_layer.config
            self.input_size = getattr(base_layer, "input_size", getattr(base_layer, "in_features"))
            self.output_size = getattr(base_layer, "output_size", getattr(base_layer, "out_features"))
            self.in_features = getattr(base_layer, "in_features", self.input_size)
            self.out_features = getattr(base_layer, "out_features", self.output_size)
            self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
            self.skip_bias_add = getattr(
                base_layer, "skip_bias_add", getattr(base_layer, "te_return_bias", False)
            )
            self.gather_output = getattr(base_layer, "gather_output", False)
            self.parallel_mode = getattr(base_layer, "parallel_mode", "column")
            self.is_expert = getattr(base_layer, "is_expert", False)
            self.expert_parallel = getattr(base_layer, "expert_parallel", False)
            self.tp_group = getattr(base_layer, "tp_group", getattr(base_layer, "_tp_group"))
            self._tp_group = self.tp_group
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.te_return_bias = getattr(base_layer, "te_return_bias", False)
            self.use_bias = base_layer.bias is not None
            self.output_size_per_partition = base_layer.weight.shape[0]

            device = torch.device(torch.cuda.current_device())
            self.quant_linear = bnb.nn.Linear4bit(
                self.input_size,
                self.output_size_per_partition,
                bias=False,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                device=device,
            )
            self.quant_linear.weight = bnb.nn.Params4bit(
                base_layer.weight.detach().cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
                quant_storage=torch.uint8,
                module=self.quant_linear,
            ).to(device)
            self.quant_linear.weight.sequence_parallel = self.sequence_parallel
            self.quant_linear.weight.allreduce = False

            if base_layer.bias is not None:
                self.register_parameter(
                    "_bias",
                    torch.nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False),
                )
            else:
                self.register_parameter("_bias", None)

        @property
        def weight(self):
            return self.quant_linear.weight

        @property
        def bias(self):
            return self._bias

        def forward(self, input_):
            if self.sequence_parallel:
                input_parallel = gather_from_sequence_parallel_region(
                    input_, tensor_parallel_output_grad=True, group=self.tp_group
                )
            else:
                input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

            original_shape = input_parallel.shape[:-1]
            flat_input = input_parallel.reshape(-1, input_parallel.shape[-1])
            flat_output = self.quant_linear(flat_input)
            output_parallel = flat_output.reshape(*original_shape, self.output_size_per_partition)

            if self.gather_output:
                raise RuntimeError("The spike expects gather_output=False.")

            if not self.skip_bias_add:
                output = output_parallel if self.bias is None else output_parallel + self.bias
                output_bias = None
            else:
                output = output_parallel
                output_bias = self.bias
            return output, output_bias

    def _sync_lora_weights(reference_layer, quantized_layer) -> None:
        adapter_name = "default"
        quantized_layer.lora_A[adapter_name].load_state_dict(
            reference_layer.lora_A[adapter_name].state_dict(), strict=True
        )
        quantized_layer.lora_B[adapter_name].load_state_dict(
            reference_layer.lora_B[adapter_name].state_dict(), strict=True
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = TransformerConfig(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        use_cpu_initialization=False,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        num_layers=1,
        hidden_size=args.input_size,
        num_attention_heads=args.tensor_model_parallel_size,
    )
    if args.parallel_mode == "row":
        base_cls = TERowParallelLinear if args.reference_impl == "te" else RowParallelLinear
        base_layer = base_cls(
            input_size=args.input_size,
            output_size=args.output_size,
            config=config,
            init_method=lambda *_args, **_kwargs: None,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )
    else:
        base_cls = TEColumnParallelLinear if args.reference_impl == "te" else ColumnParallelLinear
        base_layer = base_cls(
            input_size=args.input_size,
            output_size=args.output_size,
            config=config,
            init_method=lambda *_args, **_kwargs: None,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
    base_layer.to(device)
    base_layer.eval()

    rank = dist.get_rank()
    shard_generator = torch.Generator(device="cpu")
    shard_generator.manual_seed(args.seed + rank)
    local_weight = torch.randn(
        base_layer.weight.shape, generator=shard_generator, dtype=torch.float32
    ).to(device=device, dtype=dtype) * args.weight_std
    base_layer.weight.data.copy_(local_weight)
    base_layer.weight.requires_grad_(False)

    bias_generator = torch.Generator(device="cpu")
    bias_generator.manual_seed(args.seed + 99)
    local_bias = torch.randn(
        base_layer.bias.shape, generator=bias_generator, dtype=torch.float32
    ).to(device=device, dtype=dtype) * args.weight_std
    base_layer.bias.data.copy_(local_bias)
    base_layer.bias.requires_grad_(False)

    if args.with_lora and args.reference_impl != "te":
        raise ValueError("--with-lora currently requires --reference-impl te.")

    if args.with_lora:
        if args.parallel_mode == "row":
            bnb_base_layer = BnbTERowParallelLinear(
                base_layer, quant_type=args.quant_type, compute_dtype=dtype
            ).to(device)
        else:
            bnb_base_layer = BnbTEColumnParallelLinear(
                base_layer, quant_type=args.quant_type, compute_dtype=dtype
            ).to(device)
        reference_layer = SwiftLoraParallelLinear(
            base_layer=base_layer,
            adapter_name="default",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            init_lora_weights=True,
            lora_bias=False,
        ).to(device)
        bnb_layer = SwiftLoraParallelLinear(
            base_layer=bnb_base_layer,
            adapter_name="default",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            init_lora_weights=True,
            lora_bias=False,
        ).to(device)
        _sync_lora_weights(reference_layer, bnb_layer)
        reference_layer.eval()
        bnb_layer.eval()
        ref_module = reference_layer
        bnb_module = bnb_layer
        quantized_weight = bnb_base_layer.weight
        effective_base_layer_class = type(bnb_base_layer).__name__
    else:
        if args.parallel_mode == "row":
            bnb_layer = BnbRowParallelLinear(
                base_layer, quant_type=args.quant_type, compute_dtype=dtype
            ).to(device)
        else:
            bnb_layer = BnbColumnParallelLinear(
                base_layer, quant_type=args.quant_type, compute_dtype=dtype
            ).to(device)
        bnb_layer.eval()
        ref_module = base_layer
        bnb_module = bnb_layer
        quantized_weight = bnb_layer.quant_linear.weight
        effective_base_layer_class = type(bnb_layer).__name__

    local_sequence_len = args.sequence_len
    if args.parallel_mode == "column" and args.sequence_parallel:
        local_sequence_len = args.sequence_len // args.tensor_model_parallel_size

    input_hidden_size = base_layer.weight.shape[1]
    if args.parallel_mode == "column":
        input_hidden_size = args.input_size
    local_input = torch.randn(
        local_sequence_len,
        args.batch_size,
        input_hidden_size,
        device=device,
        dtype=dtype,
    )
    ref_input = local_input.detach().clone().requires_grad_(True)
    bnb_input = local_input.detach().clone().requires_grad_(True)

    ref_output, ref_bias = ref_module(ref_input)
    bnb_output, bnb_bias = bnb_module(bnb_input)
    if ref_bias is not None or bnb_bias is not None:
        raise RuntimeError("The spike expects skip_bias_add=False on both layers.")

    ref_loss = ref_output.float().pow(2).mean()
    bnb_loss = bnb_output.float().pow(2).mean()
    ref_loss.backward()
    bnb_loss.backward()

    output_diff = (ref_output - bnb_output).float()
    grad_diff = (ref_input.grad - bnb_input.grad).float()

    result = {
        "rank": rank,
        "parallel_mode": args.parallel_mode,
        "reference_impl": args.reference_impl,
        "with_lora": args.with_lora,
        "sequence_parallel": args.sequence_parallel,
        "input_shape": list(ref_input.shape),
        "output_shape": list(ref_output.shape),
        "output_dtype": str(ref_output.dtype),
        "reference_module_class": type(ref_module).__name__,
        "quantized_module_class": type(bnb_module).__name__,
        "quantized_base_layer_class": effective_base_layer_class,
        "max_abs_output_diff": float(output_diff.abs().max().item()),
        "mean_abs_output_diff": float(output_diff.abs().mean().item()),
        "rms_output_diff": float(output_diff.pow(2).mean().sqrt().item()),
        "max_abs_input_grad_diff": float(grad_diff.abs().max().item()),
        "mean_abs_input_grad_diff": float(grad_diff.abs().mean().item()),
        "ref_output_finite": bool(torch.isfinite(ref_output).all().item()),
        "bnb_output_finite": bool(torch.isfinite(bnb_output).all().item()),
        "ref_input_grad_finite": bool(torch.isfinite(ref_input.grad).all().item()),
        "bnb_input_grad_finite": bool(torch.isfinite(bnb_input.grad).all().item()),
        "bnb_weight_quantized": bool(getattr(quantized_weight, "quant_state", None) is not None),
        "bnb_weight_class": type(quantized_weight).__name__,
    }

    gathered_results: list[dict] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
    dist.all_gather_object(gathered_results, result)

    if rank == 0:
        output_dir = Path(os.environ["SPIKE_OUTPUT_DIR"]).expanduser().resolve()
        summary = {
            "config": {
                "reference_impl": args.reference_impl,
                "parallel_mode": args.parallel_mode,
                "tensor_model_parallel_size": args.tensor_model_parallel_size,
                "sequence_parallel": args.sequence_parallel,
                "sequence_len": args.sequence_len,
                "batch_size": args.batch_size,
                "input_size": args.input_size,
                "output_size": args.output_size,
                "dtype": args.dtype,
                "quant_type": args.quant_type,
                "with_lora": args.with_lora,
            },
            "results": gathered_results,
            "pass": all(
                item["ref_output_finite"]
                and item["bnb_output_finite"]
                and item["ref_input_grad_finite"]
                and item["bnb_input_grad_finite"]
                and item["bnb_weight_quantized"]
                for item in gathered_results
            ),
        }
        summary_path = output_dir / "row_parallel_spike_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _launch_worker(args)


if __name__ == "__main__":
    main()
