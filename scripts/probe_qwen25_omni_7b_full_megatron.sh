#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

HF_MODEL_LOCAL="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
if [[ -d "$HF_MODEL_LOCAL" ]]; then
  MODEL_PATH_DEFAULT="$HF_MODEL_LOCAL"
else
  MODEL_PATH_DEFAULT="Qwen/Qwen2.5-Omni-7B"
fi

MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/prepared/omni_lua_sft_probe_repeat256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/qwen25_omni_full_megatron_probe}"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
MAX_LENGTHS="${MAX_LENGTHS:-4096,5120,6144,7168,8192,9216,10240,11264,12288,13312,14336,15360,16384}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-true}"

echo "Launching Qwen2.5-Omni-7B Megatron max-seq probe (FULL fine-tune, no LoRA)"
echo "  model:            $MODEL_PATH"
echo "  dataset_dir:      $DATASET_DIR"
echo "  output_root:      $OUTPUT_ROOT"
echo "  train_iters:      $TRAIN_ITERS"
echo "  max_lengths:      $MAX_LENGTHS"
echo "  global_batch:     $GLOBAL_BATCH_SIZE"
echo "  micro_batch:      $MICRO_BATCH_SIZE"
echo "  tensor_parallel:  $TENSOR_MODEL_PARALLEL_SIZE"
echo "  context_parallel: $CONTEXT_PARALLEL_SIZE"

STOP_FLAG="--stop-on-fail"
if [[ "$STOP_ON_FAIL" != "true" ]]; then
  STOP_FLAG="--no-stop-on-fail"
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/probe_qwen25_omni_megatron_max_seq.py" \
  --model "$MODEL_PATH" \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --tuner-type full \
  --train-iters "$TRAIN_ITERS" \
  --max-lengths "$MAX_LENGTHS" \
  "$STOP_FLAG" \
  --extra-arg="--global-batch-size" \
  --extra-arg="$GLOBAL_BATCH_SIZE" \
  --extra-arg="--micro-batch-size" \
  --extra-arg="$MICRO_BATCH_SIZE" \
  --extra-arg="--tensor-model-parallel-size" \
  --extra-arg="$TENSOR_MODEL_PARALLEL_SIZE" \
  --extra-arg="--context-parallel-size" \
  --extra-arg="$CONTEXT_PARALLEL_SIZE" \
  --extra-arg="--sequence-parallel" \
  --extra-arg="true" \
  --extra-arg="--lr" \
  --extra-arg="1e-5" \
  "$@"
