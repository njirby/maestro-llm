#!/usr/bin/env bash
# ZeRO Stage 3 training on 2 GPUs.
# Expects prepared 10s clips via scripts/prepare_musiccaps.py (default config path).
# Projection checkpoints saved each epoch.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Helps reduce CUDA memory fragmentation under ZeRO-3 allgather pressure.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --nproc_per_node 2 scripts/run_experiment.py audio_caption "$@"
