#!/usr/bin/env bash
# ==========================================================================
# Qwen2.5-Omni-7B LoRA SFT Training
# ==========================================================================
#
# Trains on 71K agent rollouts (main/search/judge/transcription) with
# Megatron-SWIFT backend, 4x 3090 GPUs, TP=4.
#
# Proven config: r=320 @ 49K context passes the rank sweep probe
# (omni7b_rank_sweep_len49152_4steps_20260426_092711).
#
# Prerequisites:
#   - 4x NVIDIA 3090 (24GB each)
#   - outputs/sft_32k/prepared/train.jsonl and val.jsonl exist
#   - Run in tmux: tmux new -s train7b 'bash scripts/run_7b_agent_sft.sh'
#
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# ==========================================================================
# Hyperparameters
# ==========================================================================

MODEL="Qwen/Qwen2.5-Omni-7B"
TRAIN_JSONL="outputs/sft_32k/prepared/train.jsonl"
VAL_JSONL="outputs/sft_32k/prepared/val.jsonl"
OUTPUT_ROOT="outputs/sft_32k/training"
RUN_NAME="7b_r320_49k_full"

MAX_LENGTH=49152
LORA_RANK=320
LORA_ALPHA=640              # 2x rank
LR=1e-4
MIN_LR=1e-5
WARMUP_FRACTION=0.05
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4         # = 4 grad accumulation steps (DP=1, TP=4)
TP=4
SEQUENCE_PARALLEL="true"
PACKING="true"
SAVE_STEPS=200
EVAL_STEPS=200
NUM_TRAIN_EPOCHS=1

# ==========================================================================
# Validate inputs
# ==========================================================================

for f in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing dataset file: $f" >&2
        exit 1
    fi
done

echo "=== Qwen2.5-Omni-7B LoRA SFT ==="
echo "  Model: ${MODEL}"
echo "  Train: $(wc -l < "${TRAIN_JSONL}") samples"
echo "  Val:   $(wc -l < "${VAL_JSONL}") samples"
echo "  LoRA:  r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Context: ${MAX_LENGTH}"
echo "  GBS: ${GLOBAL_BATCH_SIZE}, MBS: ${MICRO_BATCH_SIZE}"
echo "  Epochs: ${NUM_TRAIN_EPOCHS}"
echo ""

# ==========================================================================
# Save hyperparams
# ==========================================================================

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/${RUN_NAME}_hyperparams.json" << PARAMS_EOF
{
  "model": "${MODEL}",
  "max_length": ${MAX_LENGTH},
  "lora_rank": ${LORA_RANK},
  "lora_alpha": ${LORA_ALPHA},
  "lr": ${LR},
  "min_lr": ${MIN_LR},
  "lr_warmup_fraction": ${WARMUP_FRACTION},
  "micro_batch_size": ${MICRO_BATCH_SIZE},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "tp": ${TP},
  "sequence_parallel": "${SEQUENCE_PARALLEL}",
  "packing": "${PACKING}",
  "save_steps": ${SAVE_STEPS},
  "eval_steps": ${EVAL_STEPS},
  "num_train_epochs": ${NUM_TRAIN_EPOCHS},
  "train_samples": $(wc -l < "${TRAIN_JSONL}"),
  "val_samples": $(wc -l < "${VAL_JSONL}"),
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "started_at": "$(date -Iseconds)"
}
PARAMS_EOF

# ==========================================================================
# Launch training
# ==========================================================================

python scripts/train_qwen25_omni_lora_megatron.py \
    --model "${MODEL}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --val-jsonl "${VAL_JSONL}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --max-length ${MAX_LENGTH} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-warmup-fraction ${WARMUP_FRACTION} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --tensor-model-parallel-size ${TP} \
    --sequence-parallel ${SEQUENCE_PARALLEL} \
    --packing ${PACKING} \
    --save-steps ${SAVE_STEPS} \
    --eval-steps ${EVAL_STEPS} \
    --num-train-epochs ${NUM_TRAIN_EPOCHS} \
    --save-optim \
    --save-rng

# ==========================================================================
# Done
# ==========================================================================

python3 -c "
import json
with open('${OUTPUT_ROOT}/${RUN_NAME}_hyperparams.json') as f:
    h = json.load(f)
h['completed_at'] = '$(date -Iseconds)'
with open('${OUTPUT_ROOT}/${RUN_NAME}_hyperparams.json', 'w') as f:
    json.dump(h, f, indent=2)
"

echo ""
echo "=== Training complete ==="
echo "  Output: ${OUTPUT_ROOT}/${RUN_NAME}"
