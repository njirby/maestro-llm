#!/usr/bin/env bash
# Known-good 2 GPU ZeRO-3 launcher for Qwen3.5-4B on 24 GB cards.
# Uses a conservative config validated in this repo:
#   - per_device_train_batch_size=1
#   - gradient_accumulation_steps=1
#   - max_length=256
# Prerequisite:
#   python scripts/prepare_musiccaps.py
#
# Usage:
#   bash scripts/train_2gpu_zero3_qwen4b.sh
#   bash scripts/train_2gpu_zero3_qwen4b.sh --dataset_limit 16 --num_train_epochs 1
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [[ ! -f "data/prepared/musiccaps_v1/manifest.jsonl" ]]; then
    echo "Missing prepared manifest at data/prepared/musiccaps_v1/manifest.jsonl" >&2
    echo "Run: python scripts/prepare_musiccaps.py" >&2
    exit 1
fi

bash scripts/train_2gpu_zero3.sh --config configs/audio_caption_qwen4b_stable.yaml "$@"
