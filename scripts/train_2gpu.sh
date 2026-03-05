#!/usr/bin/env bash
# Train audio_caption on 2 GPUs via Accelerate.
# Projection weights saved after each epoch to outputs/audio_caption/projection_epochN.pt
# Tensorboard logs: outputs/audio_caption/tensorboard/
#
# Usage:
#   bash scripts/train_2gpu.sh
#   bash scripts/train_2gpu.sh --learning_rate 1e-3 --num_train_epochs 3
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/run_experiment.py audio_caption "$@"
