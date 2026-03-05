#!/usr/bin/env bash
# Full-dataset 2 GPU ZeRO-3 launcher for Qwen3.5-4B.
# Defaults:
#   - full prepared dataset (no --dataset_limit)
#   - num_train_epochs=5 (override via NUM_TRAIN_EPOCHS or CLI args)
#
# Prerequisite:
#   python scripts/prepare_musiccaps.py
#
# Usage:
#   bash scripts/train_2gpu_zero3_qwen4b_full.sh
#   NUM_TRAIN_EPOCHS=3 bash scripts/train_2gpu_zero3_qwen4b_full.sh
#   bash scripts/train_2gpu_zero3_qwen4b_full.sh --num_train_epochs 2
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

manifest="data/prepared/musiccaps_v1/manifest.jsonl"
if [[ ! -f "$manifest" ]]; then
    echo "Missing prepared manifest at $manifest" >&2
    echo "Run: python scripts/prepare_musiccaps.py" >&2
    exit 1
fi

ok_rows=$(rg -c '"status"\s*:\s*"ok"' "$manifest" || true)
if [[ "${ok_rows:-0}" -lt 1000 ]]; then
    echo "WARNING: prepared manifest has only ${ok_rows:-0} ok rows." >&2
    echo "This may still be a subset; run full prepare for a full training run:" >&2
    echo "  python scripts/prepare_musiccaps.py" >&2
fi

bash scripts/train_2gpu_zero3.sh \
  --config configs/audio_caption_qwen4b_stable.yaml \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}" \
  "$@"
