# MS-Swift Training Notes

[← Back to README](../README.md)

## Audio Token Budget

- ~94.5 audio tokens/second (Qwen2.5-Omni, empirically measured)
- ~6,000 tokens/minute
- 30s clip = ~2,835 audio tokens
- At `max_length=8192`: fits roughly 2 × 30s clips + text overhead

## MS-Swift Packing

- Requires `--attn_impl flash_attn` when packing is enabled (hard error otherwise)
- `packing_length` defaults to `max_length` — set `max_length ≥ 2× avg sample tokens` for packing to combine samples
- Samples exceeding `max_length` are **dropped** (not truncated) when packing is enabled
- 3B fits on 1×24GB at `max_length≤5120`; 7B needs 4-GPU DeepSpeed ZeRO-3
- Confirmed 2x step reduction: 16 samples → 8 steps/epoch at `max_length=5120`

DeepSpeed config: `configs/deepspeed_zero3.json` — includes both `optimizer` and `scheduler` blocks to avoid HF/DeepSpeed LR scheduler group mismatches.

## Megatron Sequence-Parallel Findings (April 8, 2026)

All runs below used 4x24GB GPUs with LoRA (`rank=8`, `alpha=32`, `target_modules=all-linear`), `tensor_model_parallel_size=4`, `sequence_parallel=true`, `micro_batch_size=1`, packing enabled, and flash attention.

`gbs` = `global_batch_size`.

### Qwen2.5-Omni-3B

- 1-step probes (`gbs=4`) validated long context beyond 32k.
- Verified passes at `32768`, `36864`, and `40960`.
- `45056` was interrupted (SIGTERM), so the true upper bound was not finalized in that sweep.
- Reference logs: `outputs/qwen25_omni_lora_megatron_probe/probe_20260408_171119/`.

### Qwen2.5-Omni-7B

- `32768` succeeded for both:
  - 1 step (`train_iters=1`), and
  - 4 steps (`train_iters=4`) with checkpoint-4 written.
- Observed training memory at 32k was ~`10.6 GiB` per GPU in these runs.
- Artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_single/len32768_localhf/v0-20260408-173018/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_single/len32768_localhf_4steps/v0-20260408-174044/`

#### LoRA Rank Capacity Sweep @ 32k (April 19, 2026)

Viability criterion used here: **must complete 4 training steps** (`train_iters=4`) and write `checkpoint-4`.

- Confirmed passes at:
  - `rank=256` (`memory(GiB)=13.87`)
  - `rank=320` (`memory(GiB)=14.62`)
  - `rank=512` (`memory(GiB)=17.1`)
  - `rank=768` (`memory(GiB)=21.14`)
- Confirmed failures at:
  - `rank=896` OOM after step 1 (`memory(GiB)=20.36` at step 1; later forward matmul OOM)
  - `rank=1024` OOM before completing step 1 (optimizer state allocation OOM)
- Current bound on this 4x24GB setup (`tp=4`, `sp=true`, `gbs=4`, packing on):  
  **max verified viable LoRA rank at 32k is `768`; first verified failing rank is `896`.**
- Artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_rank_sweep_32k_4steps_20260419_000007/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_rank_sweep_32k_4steps_fast_20260419_001943/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_rank_sweep_32k_4steps_mid_20260419_004211/`

### Qwen3-Omni-30B-A3B-Instruct (MoE)

- Important: on 4 GPUs, this model required `--expert_model_parallel_size 4` for stable Megatron execution.
- With `gbs=4`, 1-step search:
  - `16384` pass
  - `18432` pass
  - `18688` OOM
- Separate `gbs=4`, 4-step attempts at `16384` and `18432` did not complete to checkpoint in this session (runs ended early), so stability at `gbs=4` was not confirmed.
- With `gbs=1`, 4-step stability search:
  - `16896` pass
  - `17152` pass
  - `17280` pass
  - `17408` OOM
  - `17344` interrupted before completion
- Current best verified 4-step stable context for 30B (`gbs=1`): **`17280`**.
- Additional 4-step checks (same setup) that failed under memory pressure:
  - `20000` OOM (failed after step 1)
  - `19456` OOM (failed after step 1)
  - `18432` failed at/after step 2 (rank crash under pressure)
- Reference logs:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_seqfind_ep4/20260408_175533/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_maxctx_gbs1_4steps/20260408_194108/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_single/`

### QLoRA Note (Megatron Backend Behavior)

- We tested `--quant_method bnb --quant_bits 4 --bnb_4bit_quant_type nf4 --bnb_4bit_use_double_quant true` on the same 30B Megatron setup.
- In an apples-to-apples A/B at `max_length=16000`, `train_iters=1`:
  - LoRA run: `memory(GiB)=21.21`
  - "QLoRA" flag run: `memory(GiB)=21.23`
- The live model structure still showed Megatron/TransformerEngine linear layers (`TE*` with `LoraParallelLinear`), not bitsandbytes `Linear4bit` modules.
- Practical takeaway for this repo right now: on Megatron Omni path, treat `bnb` quant flags as offering no reliable VRAM savings unless backend support is explicitly confirmed.
- A/B artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/lora_len16000_1step/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/`

#### Repo-local patched path (April 19, 2026)

- This repo now includes an experimental Megatron runtime patch path for `bnb` requests:
  - launcher: `scripts/train_qwen25_omni_lora_megatron.py`
  - patch/audit: `tools/swift_megatron_audit/`
- Current launcher contract for `--quant_method bnb`:
  - auto-enables experimental TP BNB replacement
  - enforces strict effective-quantization audit
  - fails if patch replacement / `Linear4bit` module census / `torch.uint8` params are missing
  - enforces a minimum effective uint8 parameter ratio (default `0.01`) so low-coverage runs fail fast
- Artifacts written per run include:
  - `effective_quantization_rank*.json`
  - `bnb_tp_patch_rank*.json`
  - `bnb_effective_summary_rank0.json`
- The launcher also reaps process groups and performs post-run orphan checks to avoid stale 0%-util workers retaining VRAM.
- Current 30B status: an experimental grouped MoE replacement path (`TE*GroupedLinear`) has been added, but end-to-end 30B validation of grouped coverage is still pending; until confirmed, expect quantization coverage to potentially remain too small for large VRAM savings.

### TP/CP Topology Probe (TP=2, CP=2)

- We also tested `tensor_model_parallel_size=2` + `context_parallel_size=2` on the same 30B setup.
- Outcome in this environment was unstable:
  - `max_length=20000`, `train_iters=1` failed before a completed train-step metric with
    `torch.distributed.DistBackendError` and `Failed to CUDA calloc async 4 bytes` (rank 2).
  - `max_length=16000`, `train_iters=1` reached `iteration 1/1` (`memory(GiB)=21.84`) but then failed during distributed collectives with
    `Failed to CUDA calloc async 40 bytes`.
- Practical takeaway for now: this TP/CP topology is not yet a reliable path to longer context on this box without deeper NCCL/memory tuning.
- Why this can happen even though CP should help long-context in theory:
  - `CP=2` introduces additional distributed collectives; failures occurred inside those collectives (`broadcast`/`all_reduce`), not in forward math.
  - `TP=2` changes per-rank tensor shard shapes vs `TP=4`; some temporary/comm buffers can become less favorable.
  - With MoE (`expert_model_parallel_size=4`) layered on top, communicator complexity is higher, and this stack appears fragile in this topology on this machine.
- Artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_tp_cp_experiments/`
