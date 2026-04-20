# Megatron QLoRA Investigation

Date: 2026-04-19

## Executive Summary

- **Root cause (default path):** In the repo's default Megatron-Swift path, `--quant_method bnb --quant_bits 4` is accepted by CLI but not applied to Megatron base-weight module construction. Effective Megatron model metadata remains unquantized (`model_info.quant_method=None`), and model classes stay TE bf16 + LoRA wrappers.
- **Why "QLoRA ~= LoRA memory" happened:** Those runs were effectively LoRA-on-bf16 Megatron, not true BNB QLoRA on base weights.
- **What was fixed in this repo:** Added an **experimental runtime TP patch** that replaces TP row layers with BNB-backed 4-bit wrappers before LoRA preparation, plus compatibility fixes for LoRA target discovery and bias edge-cases.
- **Can we run QLoRA with Megatron now?**
  - **Default/unpatched Megatron path:** No (still no native BNB QLoRA application).
  - **Patched experimental path in this repo:** Yes (real `Linear4bit` modules present and measurable memory savings).
- **Measured savings (2 GPU, TP=2, SP=true):**
  - **8k context:** LoRA `~7.60-7.62 GiB` vs patched QLoRA `~7.18 GiB` (clear savings).
  - **32k context:** LoRA `~12.86 GiB` vs patched QLoRA `~12.72 GiB` (smaller savings; activation-dominated regime).
- **Evidence of real quantization in patched path:** module census includes `bitsandbytes.nn.modules.Linear4bit` and parameter dtypes include substantial `torch.uint8`:
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/effective_quantization_rank0.json:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/effective_quantization_rank0.json:1)
- **Stability fix:** Added launcher process-group reaping + completion-marker shutdown to prevent orphaned Megatron/torch workers holding VRAM/RAM after training completes:
  - [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py)

## Scope

This note investigates why QLoRA showed effectively no VRAM savings versus LoRA in the repo's Qwen Omni + Megatron experiments, especially on Qwen3-Omni-30B-A3B-Instruct where the observed stable context stayed around 17k tokens.

The working hypothesis was:
- either the model is being quantized and then silently cast back to bf16 somewhere, or
- the `--quant_method bnb --quant_bits 4` flags are not actually reaching the Megatron model path at all.

Current conclusion: the second explanation is the one supported by the code and by the saved experiment artifacts.

## What the existing experiment artifacts already show

The A/B run in `outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/` is already enough to show the symptom:

- LoRA run memory: `21.21 GiB`
- "QLoRA" run memory: `21.23 GiB`

References:
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/lora_len16000_1step/v0-20260408-204403/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/lora_len16000_1step/v0-20260408-204403/logging.jsonl:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/logging.jsonl:1)

More importantly, the "QLoRA" run's saved args contain a contradiction:

- the top-level args say `quant_method = "bnb"` and `quant_bits = 4`
- but the persisted `model_info` still says `quant_method=None, quant_bits=None`
- and `params_dtype` is still `bfloat16`

Reference:
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:62`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:62)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:377`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:377)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:387`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/v0-20260408-204652/args.json:387)

That is the first concrete sign that the quantization flags are present in the CLI state but are not the effective model state used by Megatron.

## Code-path trace

### 1. Swift's generic HF loader does know about BNB quantization

The normal Swift model-kwargs path includes `quantization_config=self.get_quantization_config()`.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/arguments/base_args/model_args.py:226`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/arguments/base_args/model_args.py:226)

And `get_model_info_meta(...)` is capable of deriving `ModelInfo.quant_method` from a passed `quantization_config`.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/model/model_meta.py:246`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/model/model_meta.py:246)

So the front-end argument system is not the problem by itself.

### 2. Megatron throws away that quantization state when building `args.model_info`

In Megatron argument initialization, Swift calls `get_model_info_meta(...)` without passing `quantization_config`.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/arguments/megatron_args.py:620`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/arguments/megatron_args.py:620)

That explains why the saved Megatron run still records:
- `quant_method=None`
- `quant_bits=None`

This is not a cosmetic issue. It means the effective Megatron-side model metadata is still describing the original bf16 checkpoint, not a 4-bit-loaded model.

### 3. The Megatron trainer builds a native MCore model, not a BNB model

The actual trainer path is:
- build MCore model with `get_mcore_model(args, self.template.config)`
- load base weights into that MCore model
- optionally wrap with LoRA adapters

References:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/trainers/base.py:177`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/trainers/base.py:177)
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/trainers/base.py:184`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/trainers/base.py:184)
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/register.py:161`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/register.py:161)

The loader constructs Megatron/TransformerEngine modules from config. There is no BNB `Linear4bit` replacement in this path.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/register.py:72`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/register.py:72)

### 4. The Megatron config path maps to bf16/fp16 pipeline dtypes, not BNB 4-bit weights

`get_mcore_model_config(...)` copies HF config into `MegatronModelConfig`, then explicitly sets `pipeline_dtype = args.torch_dtype` and `fp8_param = args.fp8_param_gather`.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/model_config.py:566`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/model_config.py:566)

There is no `quant_method`, `quant_bits`, `BitsAndBytesConfig`, or BNB-backed module branch here.

The only quantization-specific path that is obvious in Megatron is FP8.

### 5. Adapter preparation is LoRA-only on top of the MCore model

When `tuner_type == 'lora'`, the Megatron utility builds a `LoraConfig` and applies `Swift.prepare_model(...)` to the already-created MCore model.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/utils/utils.py:173`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/utils/utils.py:173)

Again, there is no QLoRA-specific branch here. No BNB module injection happens before or during adapter insertion.

### 6. The bridge path also uses an unquantized dummy HF model

The Megatron bridge's `_init_meta_hf_model()` calls `get_model_processor(...)` with only `model_dir`, `model_type`, and `return_dummy_model=True`.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/gpt_bridge.py:124`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/gpt_bridge.py:124)

No `quantization_config` is passed there either.

### 7. Megatron save/export logic special-cases FP8 and strips generic quantization metadata otherwise

When saving a full Megatron model, the bridge preserves only blockwise FP8 quantization metadata. Otherwise it deletes `hf_config.quantization_config` if present.

Reference:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/gpt_bridge.py:1794`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/model/gpt_bridge.py:1794)

This is consistent with the rest of the Megatron path: FP8 is a known native mode; BNB 4-bit is not.

## Bottom line

For the Megatron backend in this environment, the issue does not look like:
- "the model loaded in 4-bit and then got silently cast back to bf16"

It looks like:
- the `bnb` QLoRA flags are accepted by the generic argument layer,
- but the actual Megatron model build/load path never instantiates a 4-bit model,
- so training still happens on bf16 Megatron/TransformerEngine weights plus LoRA adapters.

That fully explains:
- no VRAM savings,
- no context-length increase,
- the saved `params_dtype: bfloat16`, and
- the `model_info.quant_method=None` contradiction in the supposed QLoRA run.

## What this does and does not explain

This investigation explains the repo's Megatron behavior.

We now also have direct evidence for the non-Megatron Swift path, so the remaining question is narrower:
- if someone still sees no memory savings in plain ms-swift training, the issue is not "Swift cannot load Qwen Omni in BNB 4-bit at all"
- it is more likely in a later training-stage behavior, measurement method, distributed wrapper, optimizer state, or a different model/backend combination

## Non-Megatron Reproducer Findings (April 19, 2026)

I added a dedicated probe script:
- [`scripts/probe_swift_omni_quantization.py`](/home/nate/Documents/maestro-llm/scripts/probe_swift_omni_quantization.py)

Saved probe artifacts:
- [`outputs/swift_quant_probes_20260419/swift_omni_3b_bf16_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_omni_3b_bf16_probe.json:1)
- [`outputs/swift_quant_probes_20260419/swift_omni_3b_bnb4_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_omni_3b_bnb4_probe.json:1)
- [`outputs/swift_quant_probes_20260419/swift_omni_3b_bf16_lora_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_omni_3b_bf16_lora_probe.json:1)
- [`outputs/swift_quant_probes_20260419/swift_omni_3b_bnb4_lora_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_omni_3b_bnb4_lora_probe.json:1)
- [`outputs/swift_quant_probes_20260419/swift_qwen3_omni_30b_bnb4_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_qwen3_omni_30b_bnb4_probe.json:1)
- [`outputs/swift_quant_probes_20260419/swift_qwen3_omni_30b_bnb4_lora_probe.json`](/home/nate/Documents/maestro-llm/outputs/swift_quant_probes_20260419/swift_qwen3_omni_30b_bnb4_lora_probe.json:1)

### Qwen2.5-Omni-3B, base load

Plain bf16 Swift load on 1x3090:
- no BNB classes
- `memory_after.reserved_gib ~= 8.842`

BNB 4-bit Swift load on 1x3090:
- `base_args.model_info.quant_method = "bnb"`
- model config has `quant_method = "bitsandbytes"`
- `bitsandbytes.nn.modules.Linear4bit`: `252`
- `memory_after.reserved_gib ~= 5.1`

This is a real VRAM reduction on the plain Swift path.

### Qwen2.5-Omni-3B, after LoRA wrapping

Plain bf16 + LoRA:
- `peft.tuners.lora.layer.Linear`: `672`
- `reserved_gib ~= 9.062`
- trainable params: `56,987,648`

BNB 4-bit + LoRA:
- `bitsandbytes.nn.modules.Linear4bit`: `252`
- `peft.tuners.lora.bnb.Linear4bit`: `252`
- `reserved_gib ~= 5.305`
- trainable params: `56,987,648`

So for 3B, moving from LoRA to QLoRA on the plain Swift path absolutely does preserve quantized base modules and still gives a large load-memory drop.

### Qwen3-Omni-30B-A3B-Instruct, base load

BNB 4-bit Swift load on 4x3090 (`device_map=auto`):
- `base_args.model_info.quant_method = "bnb"`
- model config has `quant_method = "bitsandbytes"`
- `bitsandbytes.nn.modules.Linear4bit`: `18,672`

Important detail:
- Swift skips MoE gates plus multimodal towers from 4-bit conversion:
  - `mlp.gate`
  - `mlp.shared_expert_gate`
  - `thinker.audio_tower*`
  - `thinker.visual*`
  - `lm_head`

That skip list is expected and does not invalidate the result. The core text model is still overwhelmingly loaded as `Linear4bit`.

### Qwen3-Omni-30B-A3B-Instruct, after LoRA wrapping

BNB 4-bit + LoRA on 4x3090:
- `bitsandbytes.nn.modules.Linear4bit`: `18,672`
- `peft.tuners.lora.bnb.Linear4bit`: `18,672`
- `peft.tuners.lora.layer.Linear`: `312`
- trainable params: `868,728,320`

This is the strongest non-Megatron evidence gathered so far:
- for the actual Qwen3-Omni-30B family, plain Swift does load a real 4-bit base model
- and LoRA wrapping remains on the BNB-specific path rather than collapsing back to ordinary bf16 `Linear`

## 2-GPU Megatron Train-Step Probe (April 19, 2026)

The next controlled check was a short train-step A/B on the Megatron backend itself, using 2 GPUs, tensor parallelism across both cards, and sequence parallelism enabled.

Configuration:
- model: `Qwen2.5-Omni-3B`
- backend: `ms-swift` + Megatron
- GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- `tensor_model_parallel_size=2`
- `context_parallel_size=1`
- `sequence_parallel=true`
- `micro_batch_size=1`
- `global_batch_size=2`
- `train_iters=4`
- `max_length=8192`

Artifacts:
- LoRA run:
  - [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/args.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/args.json:1)
  - [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:1)
- "QLoRA" flags run:
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:1)

Observed memory:
- LoRA:
  - step 1: `7.6 GiB`
  - eval at step 4: `7.62 GiB`
- "QLoRA" flags:
  - step 1: `7.6 GiB`
  - eval at step 4: `7.62 GiB`

References:
- LoRA memory:
  - [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:1)
  - [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:2`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe/v0-20260419-124010/logging.jsonl:2)
- "QLoRA" memory:
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:2`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/logging.jsonl:2)

This is the definitive result for the backend the repo is using:
- sequence parallelism does not change the conclusion
- moving from LoRA to `--quant_method bnb --quant_bits 4` on Megatron produces no train-step memory savings

The saved args repeat the same contradiction seen in the older 30B run:
- the top-level "QLoRA" run args say `quant_method = "bnb"` and `quant_bits = 4`
- but `model_info` still says `quant_method=None, quant_bits=None`

References:
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:62`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:62)
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:63`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:63)
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:377`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe/v0-20260419-124128/args.json:377)

So even after an actual 4-step distributed train run with sequence parallelism turned on, the evidence still points to the same root cause:
- Megatron accepts the quantization flags at the argument layer
- but the effective Megatron model remains an ordinary bf16 model with LoRA adapters on top
- therefore there is no base-weight VRAM reduction to observe during training

## Runtime Audit And Fail-Fast Check (April 19, 2026)

To make this failure mode explicit at startup rather than after a full training job, I added a local Megatron subprocess audit:
- launcher wiring in [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py)
- worker-process audit hook in [`tools/swift_megatron_audit/sitecustomize.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/sitecustomize.py)

The audit runs after live model construction, writes per-rank JSON payloads, prints a rank-0 summary, and can optionally abort if requested quantization never becomes effective model state.

Strict audited reproducer:
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/v0-20260419-124950/args.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/v0-20260419-124950/args.json:1)
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/effective_quantization_rank0.json:1)
- [`outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/effective_quantization_rank1.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_audit_strict_1step_len8192_v2/effective_quantization_rank1.json:1)

The saved runtime audit is the cleanest evidence gathered so far because it captures the live Megatron module classes on both ranks:
- `requested_quant_method = "bnb"`
- `effective_model_info_quant_method = None`
- `params_dtype = "torch.bfloat16"`
- module counts show only TransformerEngine linears plus `swift.megatron.tuners.lora.LoraParallelLinear`
- there are no `bitsandbytes.nn.modules.Linear4bit` modules at all

Rank-0 audit payload summary:
- `TEColumnParallelLinear`: `72`
- `TELayerNormColumnParallelLinear`: `72`
- `TELinear`: `144`
- `TERowParallelLinear`: `144`
- `LoraParallelLinear`: `144`
- total params by dtype: `torch.bfloat16 = 3,012,765,696`
- trainable params by dtype: `torch.bfloat16 = 7,778,304`

The strict run aborts immediately with:
- `Megatron effective quantization audit failed: requested quant_method='bnb', but effective model_info.quant_method=None.`

This matters because it removes ambiguity from future experiments:
- if someone passes `--allow-unsupported-quantization --strict-effective-quantization --quant_method bnb --quant_bits 4`
- the run now fails at model construction instead of silently burning GPU time on a fake QLoRA configuration

## TP/SP Feasibility Spike: Quantized Row-Parallel Layer (April 19, 2026)

The next question was narrower than full-model QLoRA:
- can a shard-local BNB 4-bit weight participate in Megatron tensor parallelism and sequence parallelism at all?

To answer that, I added a standalone distributed spike:
- [`scripts/megatron_bnb_row_parallel_spike.py`](/home/nate/Documents/maestro-llm/scripts/megatron_bnb_row_parallel_spike.py)

What it does:
- launches `torch.distributed.run` on 2 GPUs
- initializes Megatron tensor parallelism
- instantiates a real `TERowParallelLinear`
- wraps its local weight shard in `bitsandbytes` `Params4bit`
- preserves the Megatron row-parallel forward contract, including:
  - local shard GEMM
  - `reduce_scatter_to_sequence_parallel_region(...)` when `sequence_parallel=true`
  - bias add after the TP reduction
- runs forward and backward-through-input against the original TE layer and records numerical differences

Successful artifact:
- [`outputs/megatron_bnb_row_parallel_spike/te_row_spike_2gpu_v4/row_parallel_spike_summary.json:1`](/home/nate/Documents/maestro-llm/outputs/megatron_bnb_row_parallel_spike/te_row_spike_2gpu_v4/row_parallel_spike_summary.json:1)

Configuration:
- reference implementation: `te`
- `tensor_model_parallel_size=2`
- `sequence_parallel=true`
- `dtype=bf16`
- quant type: `nf4`
- local input shape per rank: `[8, 2, 64]`
- local output shape per rank after row-parallel + sequence-parallel reduction: `[4, 2, 96]`

Result:
- `pass = true`
- both ranks produced finite outputs
- both ranks produced finite input gradients
- the quantized weight class is `Params4bit`
- output differences versus the original TE layer stayed small for a 4-bit approximation:
  - rank 0 mean abs output diff: `0.01588`
  - rank 1 mean abs output diff: `0.01565`
  - rank 0 max abs input-grad diff: `5.60e-05`
  - rank 1 max abs input-grad diff: `5.58e-05`

Interpretation:
- this is the first positive evidence that BNB 4-bit is not fundamentally incompatible with Megatron sequence-parallel execution
- at least for a `TERowParallelLinear` shard, a local 4-bit wrapper can preserve the TP/SP communication pattern and backpropagate through activations correctly

What this does not prove yet:
- it does not prove that the current `ms-swift` Megatron model-builder can use this path

The next concrete blocker is visible in Swift's LoRA code:
- [`/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/tuners/lora.py:339`](/home/nate/Documents/maestro-llm/.venv/lib/python3.12/site-packages/swift/megatron/tuners/lora.py:339)

`LoraParallelLinear.forward()` explicitly branches on:
- `TELayerNormColumnParallelLinear`
- `TELinear`
- `TEGroupedLinear`
- `TopKRouter`

So even though the row-parallel 4-bit spike works, a real QLoRA integration still needs one of:
- a quantized TP layer class that Swift recognizes as a supported base layer type
- or a Swift LoRA patch that accepts the new quantized TP wrapper type

## TP/SP Feasibility Spike: Quantized Row-Parallel Layer With Swift LoRA (April 19, 2026)

The next question was whether Swift's Megatron LoRA wrapper could sit on top of the same shard-local 4-bit base layer.

I extended the same spike script:
- [`scripts/megatron_bnb_row_parallel_spike.py`](/home/nate/Documents/maestro-llm/scripts/megatron_bnb_row_parallel_spike.py)

New behavior:
- adds `--with-lora`
- builds a `BnbTERowParallelLinear` that subclasses `TERowParallelLinear` so it satisfies Swift's `isinstance(...)` checks
- initializes Megatron's model-parallel CUDA RNG tracker before constructing Swift LoRA modules
- wraps both the reference TE layer and the quantized base layer with `swift.megatron.tuners.lora.LoraParallelLinear`
- copies LoRA weights from the reference wrapper into the quantized wrapper so the comparison isolates quantization instead of random adapter initialization

There was one setup failure on the first attempt:
- `LoraParallelLinear` construction raised `Exception: cuda rng state model-parallel-rng is not added`
- that was fixed by calling `tensor_parallel.model_parallel_cuda_manual_seed(args.seed)` after `initialize_model_parallel(...)`
- this was an environment/bootstrap issue in the spike, not evidence against quantized LoRA compatibility

Successful artifact:
- [`outputs/megatron_bnb_row_parallel_spike/te_row_spike_2gpu_lora_v2/row_parallel_spike_summary.json:1`](/home/nate/Documents/maestro-llm/outputs/megatron_bnb_row_parallel_spike/te_row_spike_2gpu_lora_v2/row_parallel_spike_summary.json:1)

Configuration:
- reference implementation: `te`
- `tensor_model_parallel_size=2`
- `sequence_parallel=true`
- `dtype=bf16`
- quant type: `nf4`
- `with_lora=true`
- local input shape per rank: `[8, 2, 64]`
- local output shape per rank: `[4, 2, 96]`

Result:
- `pass = true`
- both ranks produced finite outputs
- both ranks produced finite input gradients
- the quantized base-layer weight class is `Params4bit`
- the quantized path is still wrapped by Swift's `LoraParallelLinear`
- output differences versus the TE+LoRA reference stayed in the same small range as the non-LoRA spike:
  - rank 0 mean abs output diff: `0.01631`
  - rank 1 mean abs output diff: `0.01564`
  - rank 0 max abs input-grad diff: `5.34e-05`
  - rank 1 max abs input-grad diff: `5.82e-05`

Interpretation:
- this is the first positive evidence that BNB 4-bit plus Swift LoRA is viable on a Megatron TP/SP row-parallel shard
- the math path, TP/SP collectives, and backward-through-activations are all compatible in this minimal setting
- the remaining blocker is now much narrower: the production `ms-swift` Megatron model-builder still never instantiates these quantized TP layers
- said differently, the missing piece is integration and replacement plumbing, not proof that 4-bit plus LoRA is impossible under TP/SP

## TP/SP Feasibility Spike: Quantized Column-Parallel Layer With Sequence Parallelism (April 19, 2026)

Full-model Megatron QLoRA also needs the column-parallel path, so I extended the same spike script to support:
- `--parallel-mode column`

One implementation detail mattered immediately:
- TE / Megatron column-parallel with `sequence_parallel=true` does not just consume a local sequence shard and run a local GEMM
- it first all-gathers the sequence-parallel input before the local column-partition GEMM
- the first spike attempt missed that and failed with a shape mismatch (`ref_output` sequence length `8` vs quantized output sequence length `4`)
- the fix was to mirror Megatron's contract with `gather_from_sequence_parallel_region(..., tensor_parallel_output_grad=True, ...)` inside the quantized wrapper

Successful non-LoRA artifact:
- [`outputs/megatron_bnb_row_parallel_spike/te_column_spike_2gpu_v2/row_parallel_spike_summary.json:1`](/home/nate/Documents/maestro-llm/outputs/megatron_bnb_row_parallel_spike/te_column_spike_2gpu_v2/row_parallel_spike_summary.json:1)

Configuration:
- reference implementation: `te`
- parallel mode: `column`
- `tensor_model_parallel_size=2`
- `sequence_parallel=true`
- `dtype=bf16`
- quant type: `nf4`
- local input shape per rank: `[4, 2, 128]`
- local output shape per rank after column-parallel forward: `[8, 2, 48]`

Result:
- `pass = true`
- both ranks produced finite outputs
- both ranks produced finite input gradients
- quantized weight class: `Params4bit`
- output differences versus the TE reference stayed small:
  - rank 0 mean abs output diff: `0.01624`
  - rank 1 mean abs output diff: `0.01707`
  - both ranks max abs input-grad diff: `5.63e-05`

## TP/SP Feasibility Spike: Quantized Column-Parallel Layer With Swift LoRA (April 19, 2026)

I then layered Swift Megatron LoRA on top of the quantized column-parallel base.

Successful artifact:
- [`outputs/megatron_bnb_row_parallel_spike/te_column_spike_2gpu_lora_v1/row_parallel_spike_summary.json:1`](/home/nate/Documents/maestro-llm/outputs/megatron_bnb_row_parallel_spike/te_column_spike_2gpu_lora_v1/row_parallel_spike_summary.json:1)

Configuration:
- reference implementation: `te`
- parallel mode: `column`
- `tensor_model_parallel_size=2`
- `sequence_parallel=true`
- `dtype=bf16`
- quant type: `nf4`
- `with_lora=true`
- local input shape per rank: `[4, 2, 128]`
- local output shape per rank: `[8, 2, 48]`

Result:
- `pass = true`
- both ranks produced finite outputs
- both ranks produced finite input gradients
- the quantized base-layer weight class is `Params4bit`
- the quantized path is wrapped by Swift's `LoraParallelLinear`
- output differences versus the TE+LoRA reference stayed small:
  - rank 0 mean abs output diff: `0.01706`
  - rank 1 mean abs output diff: `0.01787`
  - both ranks max abs input-grad diff: `5.41e-05`

Interpretation:
- we now have positive TP/SP feasibility evidence for both major Megatron tensor-parallel linear families:
  - row-parallel
  - column-parallel
- and for both we have a passing Swift LoRA wrapper on top of the quantized base layers
- that materially narrows the production blocker to:
  - model construction / module replacement
  - checkpoint weight loading into quantized TP layers
  - integration with the existing Swift Megatron registration path

## Experimental Runtime Patch: Replace TE TP Linears After Checkpoint Load (April 19, 2026)

To move from isolated spikes toward the real Megatron training path, I added an opt-in runtime patch in this repo:
- [`tools/swift_megatron_audit/maestro_megatron_bnb.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/maestro_megatron_bnb.py)
- [`tools/swift_megatron_audit/sitecustomize.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/sitecustomize.py)
- [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py)

What it does:
- adds `--enable-experimental-bnb-tp` to the local Megatron launcher
- installs a repo-local monkeypatch via `sitecustomize`
- patches `prepare_mcore_model(...)` so that when `quant_method=bnb` is requested, the model is traversed after bridge/checkpoint load and before Swift LoRA preparation
- replaces:
  - `TERowParallelLinear` with `BnbTERowParallelLinear`
  - `TEColumnParallelLinear` with `BnbTEColumnParallelLinear`
- keeps the existing audit path active and relaxes strict-failure logic so a run with real `Linear4bit` submodules is not rejected just because `args.model_info.quant_method` still remains `None`

Status:
- code compiles
- launcher dry-run works
- full end-to-end training validation is still pending

What happened during first smoke attempts:
- a tiny 8/2-line subset was filtered down to zero examples by Swift dataset preprocessing, so it never reached model preparation
- a larger random subset still spent most of its wall clock in repeated Swift/model startup before reaching the replacement hook

Interpretation:
- we now have an actual integration candidate, not just a feasibility spike
- but I do not yet have a completed Megatron train-step run proving that the runtime patch survives real model preparation and step execution

## Updated Conclusion

The current evidence splits cleanly by backend:

- `Megatron-SWIFT` path in this repo:
  - `bnb` QLoRA flags are effectively a no-op for base-weight memory
  - no real 4-bit Megatron model is created
  - that remains true in a 2-GPU, 4-step train probe with sequence parallelism enabled
  - but the new TP/SP feasibility spikes show that shard-local `bitsandbytes` 4-bit plus Swift LoRA is technically viable for both `TERowParallelLinear` and `TEColumnParallelLinear`

- plain `ms-swift` / HF path:
  - BNB 4-bit loading works for `Qwen2.5-Omni-3B`
  - BNB 4-bit loading also works for `Qwen3-Omni-30B-A3B-Instruct`
  - LoRA wrapping stays on the BNB-specific path

So the strongest present hypothesis is no longer "ms-swift or Megatron is universally casting QLoRA back to bf16".

It is:
- the Megatron backend does not implement BNB QLoRA for this path
- while the plain Swift/HF backend does
- therefore any "no savings" report on plain Swift likely depends on later training/runtime behavior rather than the initial model load

## Recommended next steps

### Fast, low-risk fixes

1. Fail fast in the local Megatron launcher when `--quant_method bnb` is passed.
   - Right now the launcher can create a "QLoRA" experiment label that is functionally just LoRA on bf16 Megatron weights.
   - At minimum, the script should error unless native Megatron quant support is actually implemented.

2. Add an effective-quantization sanity check at startup.
   - Print and/or assert:
     - `args.quant_method`
     - `args.model_info.quant_method`
     - `args.params_dtype`
     - counts of modules by type (`LoraParallelLinear`, `Linear4bit`, TE linear classes)
   - If `args.quant_method == 'bnb'` but `args.model_info.quant_method is None`, abort.

Status:
- Implemented locally in [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py) as:
  - a guard that blocks non-FP8 quantization passthrough by default and requires `--allow-unsupported-quantization` to reproduce the old no-op behavior
  - an injected Megatron worker audit that records effective quantization state and live module-class counts
  - an optional strict mode that aborts at model construction when requested quantization is not reflected in `args.model_info`

### If true QLoRA is required

3. Do not expect current Megatron-SWIFT to provide BNB QLoRA for Qwen Omni.
   - Based on the installed source, that backend support is not there.

4. Use a non-Megatron HF/Accelerate/DeepSpeed path for real BNB QLoRA, or implement native quantized MCore layers.
   - The latter is a significantly larger project than a launcher tweak.

### To investigate the coworker ms-swift report

5. Move from load-time probes to train-time probes on the plain Swift path.
   - The loader-level question is now answered: BNB 4-bit is real on the non-Megatron path.
   - The next question is where training memory converges if someone still observes "no savings".
   - Instrument:
     - post-load memory
     - post-LoRA-wrap memory
     - post-optimizer-init memory
     - first-forward / first-backward / first-step memory
   - Compare LoRA vs QLoRA with:
     - same model
     - same dataset
     - same batch size
     - same sequence length
     - same DeepSpeed / Accelerate / DDP setup

6. Pay special attention to what is included in the memory measurement.
   - For multimodal Omni training, activation memory and non-quantized towers can dominate.
   - If the comparison is made at long sequence length with small LoRA rank, it is possible for activation memory to hide much of the base-weight savings.
   - That would not mean QLoRA failed to quantize; it would mean the measurement is dominated by another term.

## Practical takeaway for this repo

Until proven otherwise, treat:
- Megatron + `--quant_method bnb --quant_bits 4`

as:
- a no-op for base-weight memory on the Qwen Omni path in this repo.

That means the 30B sequence-length ceiling you observed is currently governed by bf16 Megatron weights, activations, optimizer state, LoRA overhead, and sequence-parallel topology, not by any real 4-bit base-model compression.

## Runtime Patch Progress Update (April 19, 2026, later)

After the initial "pending" status above, I ran real 2-GPU Megatron launcher jobs with the experimental TP patch enabled and captured concrete outcomes.

### Run v2: reached train step with real quantized modules, then failed in wrapper bias handling

Run:
- `outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v2`

What was verified before failure:
- patch hook fired and replaced TP row layers:
  - [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v2/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v2/bnb_tp_patch_rank0.json:1)
- effective module census showed real BNB modules in the Megatron model:
  - `bitsandbytes.nn.modules.Linear4bit: 72`
  - `maestro_megatron_bnb.BnbTERowParallelLinear: 72`
  - `swift.megatron.tuners.lora.LoraParallelLinear: 144`
  - see [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v2/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v2/effective_quantization_rank0.json:1)
- params included substantial `torch.uint8` footprint, confirming quantized storage was active.

Failure:
- first-forward error in wrapper bias addition due to empty TE bias tensors:
  - `RuntimeError: The size of tensor a (2048) must match the size of tensor b (0) ...`

### Fixes applied after v2

1. LoRA target discovery filter for quant child modules
- In [`tools/swift_megatron_audit/sitecustomize.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/sitecustomize.py), monkeypatched `find_all_linears(...)` to exclude names ending in `.quant_linear`.
- This avoids PEFT trying to resolve child targets that no longer exist once parent modules are replaced by `LoraParallelLinear`.

2. Empty-bias-safe wrapper logic
- In [`tools/swift_megatron_audit/maestro_megatron_bnb.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/maestro_megatron_bnb.py), TP wrappers now treat bias as present only when `bias is not None and bias.numel() > 0`.

### Run v3: end-to-end success (1 train iter + eval + checkpoint)

Run:
- `outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3`

Evidence:
- training and eval completed:
  - [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/v0-20260419-164917/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/v0-20260419-164917/logging.jsonl:1)
- checkpoint written:
  - [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/v0-20260419-164917/checkpoint-1/adapter_model.safetensors`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/v0-20260419-164917/checkpoint-1/adapter_model.safetensors)
- patch + quantization audits present:
  - [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/bnb_tp_patch_rank0.json:1)
  - [`outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron/exp_bnb_tp_tinyrepro_1step_v3/effective_quantization_rank0.json:1)

Interpretation update:
- The original repo-default Megatron path remains a no-op for BNB quantization.
- But with the experimental runtime TP replacement patch enabled, Megatron+SP can now execute real BNB-backed TP layers with Swift LoRA in an end-to-end train step.
- Next verification is controlled train-step memory comparison (LoRA vs patched QLoRA) under identical settings.

## Controlled 4-Step Memory Probe (2 GPU, TP=2, SP=true) (April 19, 2026, latest)

I reran a controlled pair with identical settings, differing only by quantization mode:

- LoRA baseline (no quant flags):
  - run: `lora_2gpu_4steps_probe_patchcmp_v3`
  - log: [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_v3/v0-20260419-171623/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_v3/v0-20260419-171623/logging.jsonl:1)
  - audit: [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_v3/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_v3/effective_quantization_rank0.json:1)

- Patched QLoRA (`--quant_method bnb --quant_bits 4` + `--enable-experimental-bnb-tp`):
  - run: `qlora_2gpu_4steps_probe_patchcmp_v4`
  - log: [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/v0-20260419-173546/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/v0-20260419-173546/logging.jsonl:1)
  - audit: [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/effective_quantization_rank0.json:1)
  - patch evidence: [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_v4/bnb_tp_patch_rank0.json:1)

### Memory results (`memory(GiB)`)

LoRA baseline:
- step 1: `7.60`
- step 2: `7.60`
- step 3: `7.60`
- step 4: `7.62`

Patched QLoRA:
- step 1: `7.18`
- step 2: `7.18`
- step 3: `7.18`
- step 4: `7.18`

Delta:
- absolute reduction: `0.42 - 0.44 GiB` per step
- relative reduction: about `5.5%`

### Quantization evidence in the successful QLoRA probe

The QLoRA run's rank-0 effective census shows:
- `bitsandbytes.nn.modules.Linear4bit: 72`
- `maestro_megatron_bnb.BnbTERowParallelLinear: 72`
- `swift.megatron.tuners.lora.LoraParallelLinear: 144`
- parameter dtypes include substantial `torch.uint8` (`240,648,192` params)

while the LoRA baseline shows only bf16 TE/LoRA modules and `torch.bfloat16` params.

### Interpretation

This controlled probe confirms the intended direction:
- repo-default Megatron path still does not natively honor BNB quantization metadata (`model_info.quant_method` remains `None`)
- but the experimental runtime TP patch now provides real quantized TP modules in training
- and under identical 4-step settings it yields clear memory savings versus LoRA baseline

## 32k Context Verification (2 GPU, TP=2, SP=true) (April 19, 2026)

To validate the same behavior at longer context, I ran the same 4-step controlled pair at `max_length=32768`.

Runs:
- LoRA baseline:
  - [`outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_32k_v1/v0-20260419-174147/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/lora_2gpu_4steps_probe_patchcmp_32k_v1/v0-20260419-174147/logging.jsonl:1)
- Patched QLoRA:
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/v0-20260419-175158/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/v0-20260419-175158/logging.jsonl:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/effective_quantization_rank0.json:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_4steps_probe_patchcmp_32k_v3/bnb_tp_patch_rank0.json:1)

Memory (`memory(GiB)`):

LoRA 32k:
- step 1: `12.86`
- step 2: `12.86`
- step 3: `12.86`
- step 4: `12.86`

Patched QLoRA 32k:
- step 1: `12.72`
- step 2: `12.72`
- step 3: `12.72`
- step 4: `12.72`

Delta at 32k:
- absolute reduction: `0.14 GiB`
- relative reduction: about `1.1%`

Interpretation:
- The patched QLoRA path still shows real quantization evidence at 32k (`Linear4bit`, `BnbTERowParallelLinear`, `torch.uint8` params).
- Savings are present but substantially smaller than the 8k probe, consistent with activation-dominated memory at long context.

## Launcher Stability Fix for Orphaned Workers (April 19, 2026)

During 32k probes, repeated runs showed orphaned Megatron/torch distributed workers remaining resident at 0% util with GPU memory still allocated.

Fix implemented in:
- [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py)

What changed:
- launch Megatron in a dedicated process group (`start_new_session=True`)
- add explicit run-completion watchdog by reading `output_dir/v*/logging.jsonl` for completion markers (`last_model_checkpoint` / `best_model_checkpoint`)
- when completion markers appear, terminate the process group to avoid stale workers
- always run final process-group cleanup in `finally` with TERM then KILL fallback

Validation:
- post-fix smoke and 32k patched QLoRA runs exit with no compute processes left in `nvidia-smi`.

## Launcher Contract Update (April 19, 2026, latest)

The local Megatron launcher now enforces a stronger BNB contract:
- file: [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py)

Behavior:
- `--quant_method bnb` now auto-enables the repo-local experimental TP BNB patch path (no extra flag required).
- BNB runs require Swift audit enabled and force strict effective-quantization checks.
- BNB runs now fail unless all are true:
  - TP replacement occurred (`replaced_total > 0`)
  - live module census includes `Linear4bit` / `BnbTE*`
  - dtype census includes non-zero `torch.uint8` parameter count
- rank-0 summary artifact:
  - `bnb_effective_summary_rank0.json`

Validation run (auto-bnb without explicit patch flag):
- [`outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/bnb_effective_summary_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/bnb_effective_summary_rank0.json:1)
- [`outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/bnb_tp_patch_rank0.json:1)
- [`outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/bnb_autoflag_smoke_1step_v1/effective_quantization_rank0.json:1)

Result:
- `ok: true` in `bnb_effective_summary_rank0.json`
- `replaced_total: 72`
- live `Linear4bit` modules and non-zero `torch.uint8` parameters observed
- no post-run compute-worker leftovers.

## 30B MoE Follow-up (Why Savings Still Look Weak)

Run:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_autopatch_len17408_ep4_gbs1_v1`

Observed:
- bnb patch was active (`replaced_total: 48`) and `Linear4bit` modules were present.
- but only a small parameter fraction became `uint8`:
  - `torch.uint8: 50,331,648`
  - `torch.bfloat16: 8,809,753,200`
  - uint8 ratio ≈ `0.0057` (0.57%).
- run reached step 3 and then OOMed.
- practical symptom: VRAM can still look near-maxed during "QLoRA" runs on this path, because >99% of base params remain bf16 in this model topology.

Key reason:
- this 30B model is MoE-heavy and still contains many unpatched grouped expert linears:
  - `TEColumnParallelGroupedLinear`, `TERowParallelGroupedLinear`, `TEGroupedLinear`
- current runtime patch replaces TP row/column linears, but not grouped expert linears.
- so only a small slice of total base weights gets quantized, which is insufficient for large VRAM wins.

Status update (April 19, 2026, later):
- an experimental grouped replacement path has now been added in
  [`tools/swift_megatron_audit/maestro_megatron_bnb.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/maestro_megatron_bnb.py),
  including replacement counters:
  - `replaced_grouped`
  - `replaced_row_grouped`
  - `replaced_column_grouped`
- 30B grouped coverage is not yet confirmed in a successful train-step artifact; the first 30B smoke rerun failed early during TE grouped-layer allocation (init-time OOM) before the new coverage could be validated end-to-end.

Follow-up reruns for grouped validation (April 19, 2026, latest):
- Rerun A:
  - run: `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v2_allocseg`
  - settings: same 30B TP=4/SP=true smoke, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - outcome: failed during model construction on `TEColumnParallelGroupedLinear` allocation (`torch.OutOfMemoryError`, 20 MiB alloc failure near full 24 GiB usage).
- Rerun B:
  - run: `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v3_allocseg_len4096`
  - settings: same as rerun A, but `max_length=4096`
  - outcome: same init-time `TEColumnParallelGroupedLinear` OOM.

Artifacts status for both reruns:
- only `v*/args.json` exists under each run directory.
- no `bnb_tp_patch_rank*.json` or `effective_quantization_rank*.json` were emitted, confirming failure happened before the patch/audit post-construction checkpoints.

Interpretation:
- current blocker is model-init VRAM headroom in this 4x24GB setup for 30B MoE TE grouped layers, not just runtime train-step memory.
- grouped replacement code is present, but validation on 30B requires an init path that reaches post-load patch/audit phase.

## 30B Grouped Patch Breakthrough (April 19, 2026, latest)

I continued with the next mitigation passes and reached a successful 30B EP=4 run with grouped patching active.

Attempt chain:
- `v4_ep4` switched to `--expert_model_parallel_size 4` (fixing prior accidental `EP=1`), which got past init-time OOM and emitted BNB patch stats.
- next failure was LoRA integration on grouped wrappers:
  - `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`
  - fixed by adding grouped wrapper metadata (`in_features` / `out_features`) in
    [`tools/swift_megatron_audit/maestro_megatron_bnb.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/maestro_megatron_bnb.py).
- next failure was LoRA target resolution trying to descend into grouped wrapper internals:
  - `AttributeError: LoraParallelLinear has no attribute quant_linears`
  - fixed by tightening `find_all_linears` filtering in
    [`tools/swift_megatron_audit/sitecustomize.py`](/home/nate/Documents/maestro-llm/tools/swift_megatron_audit/sitecustomize.py)
    to exclude `quant_linears` paths (`quant_linears.*`, `.quant_linears.`, etc.).

Successful run:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2`

Key artifacts:
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/bnb_tp_patch_rank0.json:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/effective_quantization_rank0.json:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/bnb_effective_summary_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/bnb_effective_summary_rank0.json:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/v0-20260419-184510/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_smoke_1step_v8_ep4_fixnames2/v0-20260419-184510/logging.jsonl:1)

Observed grouped coverage:
- `replaced_total: 144`
  - `replaced_row_parallel: 48`
  - `replaced_row_grouped: 48`
  - `replaced_column_grouped: 48`
- module census includes grouped wrappers:
  - `maestro_megatron_bnb.BnbTEColumnParallelGroupedLinear: 48`
  - `maestro_megatron_bnb.BnbTERowParallelGroupedLinear: 48`
- dtype census:
  - `torch.uint8: 3,674,210,304`
  - `torch.bfloat16: 1,561,995,888`
  - uint8 ratio ≈ `0.7017`
- BNB summary:
  - `ok: true`
  - `meets_uint8_ratio: true`

Train-step status:
- 1-step train + eval + checkpoint completed.
- step metric reported `memory(GiB): 18.43` at `max_length=4096`.

Interpretation update:
- grouped MoE patching is now demonstrably active on 30B in this repo path.
- the old "QLoRA not active" conclusion no longer applies to this patched EP=4 path.
- however, train-step memory still includes activations and non-quantized state; static pre-sequence VRAM may remain high even when quantized storage is present.

## Proof Run: 30B, >17k Context, 4 Steps (April 19, 2026, latest)

Requested proof:
- run Qwen3-Omni-30B with Megatron + sequence parallelism + QLoRA path
- train for 4 steps
- at sequence length above 17k

Run executed:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1`
- key settings:
  - `max_length=17920` (strictly greater than 17000)
  - `train_iters=4`
  - `tensor_model_parallel_size=4`
  - `expert_model_parallel_size=4`
  - `sequence_parallel=true`
  - `quant_method=bnb`, `quant_bits=4`

Evidence:
- run completed with `checkpoint-4` and final metrics:
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/v0-20260419-185212/checkpoint-4/adapter_model.safetensors`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/v0-20260419-185212/checkpoint-4/adapter_model.safetensors)
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/v0-20260419-185212/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/v0-20260419-185212/logging.jsonl:1)
- log entries include:
  - train entry with `iteration: "1/4"` and `memory(GiB): 18.43`
  - eval entry with `iteration: "4/4"` and `memory(GiB): 18.43`
  - summary with `last_model_checkpoint` and `best_model_checkpoint` both pointing to `checkpoint-4`
- quantization/patch evidence is present in same run:
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/bnb_tp_patch_rank0.json:1)
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/effective_quantization_rank0.json:1)
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/bnb_effective_summary_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len17920_v1/bnb_effective_summary_rank0.json:1)

Notes:
- this is a direct counterexample to the prior "17k is hard max" conclusion for this environment/path.
- observed memory in this proof run still sits around `18.43 GiB`, indicating meaningful runtime memory beyond just quantized base-weight storage.

## Follow-up: 24k Context Attempt (April 19, 2026, latest)

Run:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1`
- settings:
  - `max_length=24576`
  - `train_iters=4`
  - same proven topology (`tp=4`, `ep=4`, `sequence_parallel=true`, BNB grouped patch path)

Outcome:
- run completed successfully with `checkpoint-4`.
- logs show:
  - `iteration: "1/4"` at `memory(GiB): 18.43`
  - eval entry at `iteration: "4/4"` and final summary with `last_model_checkpoint`/`best_model_checkpoint`.
- dataset summary in the same run shows actual packed token lengths centered above 22k:
  - `train_dataset`: mean `22861.67`, min `21832`, max `23389`
  - this confirms the run was genuinely operating above the old ~17k regime.

Artifacts:
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/v0-20260419-185751/logging.jsonl:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/v0-20260419-185751/logging.jsonl:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/v0-20260419-185751/checkpoint-4/adapter_model.safetensors`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/v0-20260419-185751/checkpoint-4/adapter_model.safetensors)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/bnb_tp_patch_rank0.json:1)
- [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len24576_v1/effective_quantization_rank0.json:1)

## Large LoRA @ 32k Context (April 19, 2026, latest)

Run:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1`
- settings:
  - `max_length=32768`
  - `lora_rank=128`, `lora_alpha=256`
  - `train_iters=4`
  - `tp=4`, `ep=4`, `sequence_parallel=true`, BNB grouped patch enabled

Result:
- run failed on first optimizer step with CUDA OOM during gradient copy to fp32 main grads:
  - `megatron/core/optimizer/distrib_optimizer.py` -> `shard_main_param.grad = shard_model_grad.float()`
  - allocation failure: `Tried to allocate 2.00 MiB` with GPUs essentially full.
- this indicates large LoRA rank (128) pushes memory over the limit at 32k in this topology.

Evidence:
- console log:
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1.console.log:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1.console.log:1)
- run artifacts still confirm grouped patch activation before failure:
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1/bnb_tp_patch_rank0.json:1)
  - [`outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1/effective_quantization_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank128_v1/effective_quantization_rank0.json:1)

Launcher correctness fix made during this run:
- completion detection previously keyed on presence of `last_model_checkpoint`/`best_model_checkpoint` text, which could appear with `null` values in failing runs.
- updated [`scripts/train_qwen25_omni_lora_megatron.py`](/home/nate/Documents/maestro-llm/scripts/train_qwen25_omni_lora_megatron.py) to parse JSON tail lines and only treat completion as true when checkpoint fields are non-empty paths.

Control run (small LoRA) at the same 32k setting:
- run: `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank8_v1`
- config: `lora_rank=8`, `lora_alpha=32` (all else same: `max_length=32768`, `tp=4`, `ep=4`, `sequence_parallel=true`)
- outcome: succeeded for 4 steps with `checkpoint-4` written.
- logged memory:
  - `iteration 1/4`: `memory(GiB)=18.43`
  - eval at `iteration 4/4`: `memory(GiB)=18.43`
- quantization artifacts remained healthy (`replaced_total=144`, `ok=true`, uint8 ratio ~0.70).

Interpretation:
- at 32k in this environment, the large-rank failure was capacity from LoRA rank expansion, not a regression in the grouped QLoRA patch path.

Additional rank/context sweep (April 19, 2026, latest):

1) 32k @ medium rank (`r=64`, `alpha=128`)
- run: `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len32768_rank64_v1`
- outcome: failed on step 1 with CUDA OOM.
- observed step metric before failure:
  - `iteration: 1/4`, `memory(GiB): 22.66`
- failure traces in console log include OOM allocations in the ~286–312 MiB range near full 24GB capacity.
- quantization artifacts still emitted (`bnb_tp_patch_rank*.json`, `effective_quantization_rank*.json`, `bnb_effective_summary_rank0.json`), confirming patch path was active.

2) 48k @ small rank (`r=8`, `alpha=32`)
- run: `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len49152_rank8_v1`
- outcome: completed 4 steps and wrote `checkpoint-4`.
- key logged metrics:
  - `iteration: 1/4`, `memory(GiB): 18.43`
  - eval at `iteration: 4/4`, `memory(GiB): 18.43`
- train token stats for this run:
  - mean `34292.5`, min `20865`, max `47720` (packed samples reaching well into the 48k regime)
- quantization artifacts healthy (`replaced_total=144`, summary `ok=true`).

Gated next step decision:
- requested rule was: only try 48k @ medium if both prior runs exit cleanly.
- since 32k @ medium failed, 48k @ medium was not auto-launched under that rule.

## 48k @ Rank 24 (April 20, 2026, latest)

Run:
- `outputs/qwen25_omni_lora_megatron_probe/omni30b_bnb_groupedpatch_4steps_len49152_rank24_v1`
- settings:
  - `max_length=49152`
  - `lora_rank=24`, `lora_alpha=48`
  - `train_iters=4`
  - `tp=4`, `ep=4`, `sequence_parallel=true`, BNB grouped patch path

Outcome:
- completed successfully with `checkpoint-4`.
- log metrics:
  - `iteration 1/4`: `memory(GiB)=21.01`
  - eval at `iteration 4/4`: `memory(GiB)=21.24`
- train token stats:
  - mean `34292.5`, min `20865`, max `47720` (same 48k-regime packed data behavior)

Warnings observed:
- repeated autograd stream-mismatch `UserWarning` from `torch.autograd.graph`.
- standard distributed `barrier()` device-context warning from `torch.distributed.c10d_logger`.
- no `CUDA out of memory` / allocator OOM warnings in this run.
- host RAM/swap can still spike substantially during this 48k run (high temporary pressure), but in the successful `rank24` case the job completed and host memory later returned to low steady-state levels.

Actionable guard added:
- launcher now passes `bnb_min_uint8_ratio` (default `0.01`) into audit.
- bnb run fails if effective uint8 ratio is below threshold, preventing expensive "QLoRA" runs with negligible practical quantization.

## Stability Incident: Host RAM Process Leak (April 19, 2026)

Symptom:
- host RAM and swap kept rising even when GPU utilization was near idle.
- process table showed hundreds of `pip list` Python processes in one process group, consuming tens of GiB RSS.

Observed chain:
- leaked workers were rooted under a Torch Inductor compile worker process.
- command pattern repeated recursively: `pip list | grep habana-torch-plugin`.
- source command comes from bitsandbytes Gaudi backend detection (`bitsandbytes/backends/utils.py:get_gaudi_sw_version`).

Why recursion happened in this repo:
- our Megatron audit `sitecustomize.py` is injected via `PYTHONPATH`.
- for BNB runs, `sitecustomize` installed the TP patch in every Python subprocess.
- when a subprocess ran `pip list`, `sitecustomize` imported BNB patch helpers, which imported bitsandbytes, which spawned another `pip list`, recursively.

Fix applied:
- `tools/swift_megatron_audit/sitecustomize.py` now skips BNB TP patch installation for pip entrypoints (`pip`, `pip3`, `python-pip`) using `sys.argv[0]` guard.
- quick validation: with BNB audit env vars set, `python ~/.local/bin/pip list` now exits normally with no recursive spawn.

Operational recovery:
- leaked process group was terminated; GPU compute workers were clear afterwards.

## Stability Follow-up: Orphan Reap Hardening (April 19, 2026, later)

Symptom:
- a 2-GPU QLoRA run completed train/eval/checkpoint, but launcher still failed post-run because two detached `swift/cli/_megatron/sft.py` workers stayed alive with GPU memory.

Fix applied:
- `scripts/train_qwen25_omni_lora_megatron.py` now actively reaps survivor PIDs/process-groups before declaring failure:
  - added `_reap_orphan_workers(run_name)` with TERM grace period and KILL fallback.
  - main path now calls `_reap_orphan_workers` instead of check-only behavior.

Validation:
- run: `outputs/swift_quant_trainstep_probes/qlora_2gpu_1step_orphanfix_v1`
- launcher exited `0`.
- no remaining `nvidia-smi --query-compute-apps` entries after completion.
- quantization artifacts still present and valid:
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_1step_orphanfix_v1/bnb_effective_summary_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_1step_orphanfix_v1/bnb_effective_summary_rank0.json:1)
  - [`outputs/swift_quant_trainstep_probes/qlora_2gpu_1step_orphanfix_v1/bnb_tp_patch_rank0.json:1`](/home/nate/Documents/maestro-llm/outputs/swift_quant_trainstep_probes/qlora_2gpu_1step_orphanfix_v1/bnb_tp_patch_rank0.json:1)
