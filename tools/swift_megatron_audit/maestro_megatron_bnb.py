from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReplacementStats:
    replaced_row_parallel: int = 0
    replaced_column_parallel: int = 0
    replaced_grouped: int = 0
    replaced_row_grouped: int = 0
    replaced_column_grouped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "replaced_row_parallel": self.replaced_row_parallel,
            "replaced_column_parallel": self.replaced_column_parallel,
            "replaced_grouped": self.replaced_grouped,
            "replaced_row_grouped": self.replaced_row_grouped,
            "replaced_column_grouped": self.replaced_column_grouped,
            "replaced_total": (
                self.replaced_row_parallel
                + self.replaced_column_parallel
                + self.replaced_grouped
                + self.replaced_row_grouped
                + self.replaced_column_grouped
            ),
        }


def _copy_parallel_attrs(dst: Any, src: Any) -> None:
    for name in (
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "sequence_parallel",
        "allreduce",
    ):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _build_wrappers():
    import bitsandbytes as bnb
    import torch
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEGroupedLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
    from megatron.core.tensor_parallel.layers import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )

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
            has_bias = base_layer.bias is not None and base_layer.bias.numel() > 0
            self.use_bias = has_bias
            self.input_size_per_partition = base_layer.weight.shape[1]

            device = base_layer.weight.device
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
            _copy_parallel_attrs(self.quant_linear.weight, base_layer.weight)

            if has_bias:
                bias = torch.nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
                _copy_parallel_attrs(bias, base_layer.bias)
                self.register_parameter("_bias", bias)
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
            has_bias = base_layer.bias is not None and base_layer.bias.numel() > 0
            self.use_bias = has_bias
            self.output_size_per_partition = base_layer.weight.shape[0]

            device = base_layer.weight.device
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
            _copy_parallel_attrs(self.quant_linear.weight, base_layer.weight)

            if has_bias:
                bias = torch.nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
                _copy_parallel_attrs(bias, base_layer.bias)
                self.register_parameter("_bias", bias)
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
                raise RuntimeError("Experimental BNB TP patch expects gather_output=False.")

            if not self.skip_bias_add:
                output = output_parallel if self.bias is None else output_parallel + self.bias
                output_bias = None
            else:
                output = output_parallel
                output_bias = self.bias
            return output, output_bias

    class _BnbTEGroupedBase(TEGroupedLinear):

        def __init__(self, base_layer, *, quant_type: str, compute_dtype: torch.dtype):
            torch.nn.Module.__init__(self)
            self.config = base_layer.config
            self.num_gemms = int(getattr(base_layer, "num_gemms", 1))
            # Swift LoRA wrapper expects these attributes on grouped TE modules.
            self.in_features = getattr(base_layer, "in_features", None)
            self.out_features = getattr(base_layer, "out_features", None)
            self.input_size = getattr(base_layer, "input_size", self.in_features)
            self.output_size = getattr(base_layer, "output_size", self.out_features)
            self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
            self.skip_bias_add = getattr(base_layer, "skip_bias_add", False)
            self.te_return_bias = getattr(base_layer, "te_return_bias", False)
            self.is_expert = getattr(base_layer, "is_expert", True)
            self.expert_parallel = getattr(base_layer, "expert_parallel", False)
            self._tp_group = getattr(base_layer, "_tp_group", None)
            self.tp_group = getattr(base_layer, "tp_group", self._tp_group)
            self.parallel_mode = getattr(base_layer, "parallel_mode", None)
            self.explicit_expert_comm = getattr(base_layer, "explicit_expert_comm", True)
            self.is_first_microbatch = False
            self.disable_parameter_transpose_cache = True

            self.quant_linears = torch.nn.ModuleList()
            self._biases = torch.nn.ParameterList()
            self.use_bias = False

            for i in range(self.num_gemms):
                w = getattr(base_layer, f"weight{i}")
                b = getattr(base_layer, f"bias{i}", None)
                has_bias = b is not None and getattr(b, "numel", lambda: 0)() > 0
                self.use_bias = self.use_bias or has_bias
                device = w.device
                out_features = int(w.shape[0])
                in_features = int(w.shape[1])

                quant_linear = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=False,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    quant_storage=torch.uint8,
                    device=device,
                )
                quant_linear.weight = bnb.nn.Params4bit(
                    w.detach().cpu(),
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type=quant_type,
                    quant_storage=torch.uint8,
                    module=quant_linear,
                ).to(device)
                _copy_parallel_attrs(quant_linear.weight, w)
                self.quant_linears.append(quant_linear)
                setattr(self, f"weight{i}", quant_linear.weight)

                if has_bias:
                    bias = torch.nn.Parameter(b.detach().clone(), requires_grad=False)
                    _copy_parallel_attrs(bias, b)
                else:
                    bias = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=device), requires_grad=False)
                self._biases.append(bias)
                setattr(self, f"bias{i}", bias)

        def forward(self, x, m_splits):
            if m_splits is None:
                raise RuntimeError("GroupedLinear BNB wrapper requires m_splits.")
            if len(m_splits) != self.num_gemms:
                raise RuntimeError(
                    f"GroupedLinear BNB wrapper expected {self.num_gemms} splits, got {len(m_splits)}."
                )
            outputs = []
            start = 0
            for i, split in enumerate(m_splits):
                n = int(split)
                end = start + n
                xi = x[start:end]
                yi = self.quant_linears[i](xi)
                if not self.skip_bias_add and self._biases[i].numel() > 0:
                    yi = yi + self._biases[i]
                outputs.append(yi)
                start = end
            out = torch.cat(outputs, dim=0) if outputs else x.new_empty((0, 0))
            # TE GroupedLinear returns tuple when used in Megatron wrappers.
            return out, None

    class BnbTEGroupedLinear(_BnbTEGroupedBase, TEGroupedLinear):
        pass

    class BnbTEColumnParallelGroupedLinear(_BnbTEGroupedBase, TEColumnParallelGroupedLinear):
        pass

    class BnbTERowParallelGroupedLinear(_BnbTEGroupedBase, TERowParallelGroupedLinear):
        pass

    return (
        BnbTEColumnParallelLinear,
        BnbTERowParallelLinear,
        BnbTEGroupedLinear,
        BnbTEColumnParallelGroupedLinear,
        BnbTERowParallelGroupedLinear,
    )


def replace_te_tp_linears_with_bnb(
    model,
    *,
    quant_type: str = "nf4",
    compute_dtype=None,
    min_replaced_total: int = 0,
) -> dict[str, int]:
    import torch
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEGroupedLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    (
        BnbTEColumnParallelLinear,
        BnbTERowParallelLinear,
        BnbTEGroupedLinear,
        BnbTEColumnParallelGroupedLinear,
        BnbTERowParallelGroupedLinear,
    ) = _build_wrappers()

    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    stats = ReplacementStats()

    def _replace(module):
        for name, child in list(module.named_children()):
            if isinstance(
                child,
                (
                    BnbTERowParallelLinear,
                    BnbTEColumnParallelLinear,
                    BnbTEGroupedLinear,
                    BnbTEColumnParallelGroupedLinear,
                    BnbTERowParallelGroupedLinear,
                ),
            ):
                continue
            if isinstance(child, TERowParallelGroupedLinear):
                setattr(
                    module,
                    name,
                    BnbTERowParallelGroupedLinear(
                        child, quant_type=quant_type, compute_dtype=compute_dtype
                    ).to(child.weight0.device),
                )
                stats.replaced_row_grouped += 1
                continue
            if isinstance(child, TEColumnParallelGroupedLinear):
                setattr(
                    module,
                    name,
                    BnbTEColumnParallelGroupedLinear(
                        child, quant_type=quant_type, compute_dtype=compute_dtype
                    ).to(child.weight0.device),
                )
                stats.replaced_column_grouped += 1
                continue
            if isinstance(child, TEGroupedLinear):
                setattr(
                    module,
                    name,
                    BnbTEGroupedLinear(child, quant_type=quant_type, compute_dtype=compute_dtype).to(
                        child.weight0.device
                    ),
                )
                stats.replaced_grouped += 1
                continue
            if isinstance(child, TERowParallelLinear):
                setattr(
                    module,
                    name,
                    BnbTERowParallelLinear(child, quant_type=quant_type, compute_dtype=compute_dtype).to(
                        child.weight.device
                    ),
                )
                stats.replaced_row_parallel += 1
                continue
            if isinstance(child, TEColumnParallelLinear):
                setattr(
                    module,
                    name,
                    BnbTEColumnParallelLinear(child, quant_type=quant_type, compute_dtype=compute_dtype).to(
                        child.weight.device
                    ),
                )
                stats.replaced_column_parallel += 1
                continue
            _replace(child)

    _replace(model)
    payload = stats.to_dict()
    if min_replaced_total > 0 and payload["replaced_total"] < min_replaced_total:
        raise RuntimeError(
            "BNB TP replacement coverage below threshold: "
            f"replaced_total={payload['replaced_total']} < min_replaced_total={min_replaced_total}"
        )
    return payload
