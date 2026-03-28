#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

# Ensure venv/bin is in PATH so subprocesses (DeepSpeed, ninja) can find tools.
export PATH="$ROOT_DIR/.venv/bin:$PATH"

SWIFT_BIN="${SWIFT_BIN:-$ROOT_DIR/.venv/bin/swift}"
if [[ ! -x "$SWIFT_BIN" ]]; then
  SWIFT_BIN="swift"
fi

HF_MODEL_LOCAL="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
if [[ -d "$HF_MODEL_LOCAL" ]]; then
  MODEL_PATH_DEFAULT="$HF_MODEL_LOCAL"
else
  MODEL_PATH_DEFAULT="Qwen/Qwen2.5-Omni-7B"
fi

MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/prepared/omni_lua_sft_full_90_10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
RUN_NAME="${RUN_NAME:-full_$(date +%Y%m%d_%H%M%S)_len8k_nopack_val256_ckpt128_keep2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/qwen25_omni_lora}"

echo "Launching full Qwen2.5-Omni LoRA training"
echo "  model:         $MODEL_PATH"
echo "  dataset_dir:   $DATASET_DIR"
echo "  nproc:         $NPROC_PER_NODE"
echo "  dl_workers:    $DATALOADER_NUM_WORKERS"
echo "  run_name:      $RUN_NAME"
echo "  output_root:   $OUTPUT_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/scripts/train_qwen25_omni_lora.py" \
  --profile full \
  --swift-bin "$SWIFT_BIN" \
  --model "$MODEL_PATH" \
  --dataset-dir "$DATASET_DIR" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --run-name "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --max_length 8000 \
  --packing false \
  --padding_free false \
  --gradient_accumulation_steps 1 \
  --eval_steps 256 \
  --save_steps 256 \
  --save_total_limit 2 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --logging_steps 10 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --attn_impl flash_attention_2 \
  "$@"
