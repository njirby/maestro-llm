# Megatron LoRA Checkpoint Crash: Diagnosis and Fix

Date: 2026-06-21

## Problem

Training Qwen2.5-Omni-7B with LoRA (rank 320, TP=4) on 4x RTX 3090 (24GB each),
saving optimizer state in checkpoints caused the machine to hard-crash during
checkpoint writes. The machine rebooted with an AA LED motherboard code, no
OOM-killer messages in the kernel journal.

System: 30 GiB RAM, 71 GiB swap, Threadripper PRO 7965WX, 2x 1200W PSUs.

## Timeline of Experiments

| Version | Config | Result |
|---------|--------|--------|
| v6 | Default DCP save (model + optimizer shards) | NCCL deadlock on all-gather with TP=4 + LoRA |
| v7 | `no_save_optim=True` (skip optimizer) | 8 steps, both checkpoints saved OK |
| v8 | Monkeypatch `model=[]` in `save_mcore_checkpoint` (keep optimizer in DCP) | Hard crash x2 during checkpoint-4 write |
| **v9** | **Two-phase save: Phase 1 = v7 path, Phase 2 = `torch.save()` per-rank** | **8 steps, both checkpoints saved OK** |
| **v10** | **Resume from v9 checkpoint-4, train to step 12** | **Resume works, loss continuous, saves OK** |

## Root Cause Analysis

### What Megatron's DCP does during sharded save

`dist_checkpointing.save()` runs three stages that are each dangerous under
memory pressure on a 30GB RAM machine:

1. **NCCL collectives** for DCP plan coordination (`gather_object` / `reduce_scatter`
   on WORLD group)
2. **`preload_tensors()`**: `tensor.to("cpu", non_blocking=True)` for all optimizer
   tensors -- pinned memory allocation
3. **`write_preloaded_data_multiproc()`**: forks child processes via
   `mp.get_context("fork")` for parallel disk write

### Forensic evidence

Both v8 crash directories contained `common.pt` (16K) but NO `.distcp` files and
NO safetensors. The `common.pt` is written by `save_preprocess()` +
`common_strategy.save_common()` *before* the sharded tensor write. This proves
the crash happened during the DCP sharded tensor write stage -- specifically in
`FullyParallelSaveStrategyWrapper.save()` -> `TorchDistSaveShardedStrategy.save()`
-> `execute_sync()`.

### Why optimizer state size wasn't the issue

The optimizer state is only ~822MB/rank (~3.3GB total) for 274M LoRA parameters --
the DDP wrapper only tracks `requires_grad=True` params. Raw data size wasn't
enough for OOM. The crash vector was the DCP machinery itself: pinned memory
allocation + process forking under GPU memory pressure.

## Solution: Bypass DCP Entirely

Implemented in `tools/swift_megatron_audit/sitecustomize.py`, activated by
`MAESTRO_SWIFT_CHECKPOINT_LORA_FIX=1` environment variable.

### Save path (two-phase)

**Phase 1** -- Temporarily set `no_save_optim=True` and `no_save_rng=True`, call
the original `save_checkpoint()`. This produces:
- `adapter_model.safetensors` (LoRA weights via bridge)
- `iter_N/common.pt` (tiny metadata, no ShardedTensors)
- Same as the proven v7 path -- no NCCL collectives, no DCP ShardedTensors

**Phase 2** -- Save optimizer + RNG + scheduler state per-rank via `torch.save()`:
- `_get_rng_state()` (unwrapped from `ShardedObject`)
- `opt.state_dict()` for each chained optimizer (metadata: step, param_groups)
- `opt.optimizer.state_dict()` for each inner Adam (parameter state, moved to CPU)
- `opt_param_scheduler.state_dict()` for LR scheduler
- All saved to `checkpoint-N/maestro_extra_rank{rank}.pt`

No DCP, no NCCL, no pinned memory, no fork.

### Load path (resume)

Activated by `MAESTRO_RESUME_FROM` environment variable (set by `--resume-from`
flag in the training script).

1. Load adapter weights via `bridge.load_weights()` (safetensors, no DCP)
2. Sync model params into optimizer buffers via `optimizer.reload_model_params()`
3. Load inner Adam state from `maestro_extra_rank{rank}.pt`, move to GPU,
   call `optimizer.load_state_dict()`
4. Restore LR scheduler, RNG state, iteration counter

### Checkpoint structure

```
checkpoint-N/
  iter_N/common.pt                  # 16K metadata
  adapter_model.safetensors         # ~1.6GB LoRA weights
  adapter_config.json
  additional_config.json
  args.json
  latest_checkpointed_iteration.txt
  maestro_extra_rank0.pt            # ~2.1GB optimizer + RNG per-rank
  maestro_extra_rank1.pt
  maestro_extra_rank2.pt
  maestro_extra_rank3.pt
```

### Instrumented logging

Every save/load phase emits `MAESTRO_CKPT [tag] gpu_mem=...MB t=...` to stdout
for crash diagnosis. Tags: `save_start`, `phase1_done`, `phase2_cpu_done`,
`phase2_saved`, `save_complete`, `load_resume_start`, `load_adapter_done`,
`load_inner_opt_done`, `load_scheduler_done`, `load_rng_done`,
`load_iteration=N`, `load_resume_complete`.

## Verification

### v9 save test (8 steps, save at 4 and 8)

- All 10 MAESTRO_CKPT markers present for both checkpoints
- GPU memory flat at 11,395MB throughout
- No crash, no OOM
- Phase timing: Phase 1 ~18-26s, CPU copy ~3-4s, torch.save ~19-20s

### v10 resume test (checkpoint-4 -> train to step 12)

- Iteration correctly restored at 5
- LR exact match: 4.039e-05 (v9 step 5 = 4.039e-05)
- Loss not spiking: 0.559 at step 5 (well below initial 0.835)
- Eval loss at step 8: 0.348 (v9 was 0.374 -- continuing to improve)
- Checkpoint-12 saved successfully with same structure

## Limitations

- Resume requires same TP/PP layout (no resharding). This is fine for our
  single-machine 4x3090 setup.
- `maestro_extra_rank*.pt` files are ~2.1GB each (~8.4GB total). Adequate disk
  space needed.
- The `ShardedObject` wrapper from `_get_rng_state()` is unwrapped to plain
  list before saving. Legacy files with the wrapper are handled on load.

## Files Changed

| File | Change |
|------|--------|
| `tools/swift_megatron_audit/sitecustomize.py` | Two-phase save + bridge-based load, bypassing DCP |
| `scripts/train_qwen25_omni_lora_megatron.py` | Added `--resume-from` flag |

## GPU Power Note

After a hard crash/reboot, GPU persistence mode and power limits reset. Run:
```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 300
```
The 300W limit (vs 350W default) reduces power spikes during checkpoint saves.
