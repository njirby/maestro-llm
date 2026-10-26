#!/bin/bash
# Self-driving search-POC pipeline: smoke gate -> resume gate -> big run.
# Survives Claude Code session restarts (nohup'd, state via files).
# Status log: outputs/search_lora/pipeline_status.log
cd /home/nate/Documents/maestro-llm || exit 1
S=outputs/search_lora/pipeline_status.log
MODEL=/home/nate/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00
log() { echo "$(date '+%F %T') $*" >> "$S"; }

log "pipeline start; waiting on smoke (pid 4039667)"
# Gate 1: smoke completes with checkpoint-6
for i in $(seq 1 240); do
  ls outputs/search_lora/training/search_smoke/v*/checkpoint-6 >/dev/null 2>&1 && break
  kill -0 4039667 2>/dev/null || { sleep 30; break; }
  sleep 30
done
CKPT6=$(ls -d outputs/search_lora/training/search_smoke/v*/checkpoint-6 2>/dev/null | head -1)
if [ -z "$CKPT6" ]; then log "FAIL: smoke never produced checkpoint-6; aborting"; exit 1; fi
log "smoke checkpoint-6 present: $CKPT6"

# Gate 2: adapter sanity
.venv/bin/python -c "
from safetensors import safe_open
with safe_open('$CKPT6/adapter_model.safetensors','pt') as f:
    ks = list(f.keys())
audio = sum(1 for k in ks if 'audio' in k)
assert audio > 100, f'audio adapters missing: {audio}'
print('adapter ok:', len(ks), 'tensors,', audio, 'audio')" >> "$S" 2>&1 || { log "FAIL: adapter sanity"; exit 1; }

# Gate 3: mini resume test (lazy dataloader resume semantics)
CKPT3=$(ls -d outputs/search_lora/training/search_smoke/v*/checkpoint-3 2>/dev/null | head -1)
.venv/bin/python scripts/train_qwen25_omni_lora_megatron.py --model "$MODEL" \
  --train-jsonl outputs/search_lora/train_smoke500.jsonl --val-jsonl outputs/search_lora/eval_slice_40.jsonl \
  --output-root outputs/search_lora/training --run-name search_smoke_resume \
  --max-length 32768 --lora-rank 128 --lora-alpha 256 --lr 5e-5 \
  --micro-batch-size 1 --global-batch-size 4 --tensor-model-parallel-size 4 \
  --packing false --dataset-num-proc 8 --freeze-vit false --freeze-aligner false \
  --train-iters 6 --save-steps 3 --eval-steps 3 --save-optim --save-rng \
  --resume-from "$CKPT3" --report-to wandb --wandb-project maestro-sft \
  --lazy_tokenize true > outputs/search_lora/smoke_resume.log 2>&1
if [ $? -ne 0 ]; then log "FAIL: resume test rc=$?; aborting (see smoke_resume.log)"; exit 1; fi
log "resume test passed"

# Launch: big run — full train set, 1 epoch horizon, save every 60 iters (~2h)
nohup .venv/bin/python scripts/train_qwen25_omni_lora_megatron.py --model "$MODEL" \
  --train-jsonl outputs/search_lora/train.jsonl --val-jsonl outputs/search_lora/eval_slice_40.jsonl \
  --output-root outputs/search_lora/training --run-name search_r128_tower_v1 \
  --max-length 32768 --lora-rank 128 --lora-alpha 256 --lr 5e-5 \
  --micro-batch-size 1 --global-batch-size 4 --tensor-model-parallel-size 4 \
  --packing false --dataset-num-proc 8 --freeze-vit false --freeze-aligner false \
  --num-train-epochs 1 --save-steps 60 --eval-steps 60 --eval-iters 4 \
  --save-total-limit 12 --save-optim --save-rng \
  --report-to wandb --wandb-project maestro-sft \
  --lazy_tokenize true > outputs/search_lora/search_v1.log 2>&1 &
log "BIG RUN LAUNCHED pid $! (search_r128_tower_v1; save every 60 iters ~2h)"
