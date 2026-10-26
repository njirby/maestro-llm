#!/usr/bin/env bash
# Run an ear_graft entry point inside the ear-graft container (never the host venv).
#   GPU=0 ear_graft/docker_run.sh -m ear_graft.smoke_test
#   GPU=0 ear_graft/docker_run.sh -m ear_graft.train --run-name pilot --max-steps 50
#   GPU=all NAME=ear-train ear_graft/docker_run.sh -m torch.distributed.run --nproc_per_node 4 -m ear_graft.train --run-name v0
# GPU: a card index, a comma list (0,1) or all. Image: ~/ml/docker/ear-graft. Stop: docker stop $NAME.
# wandb authenticates from ~/.netrc (HOME is mounted); network is host.
set -euo pipefail
IMAGE=${IMAGE:-ear-graft:26.09}
NAME=${NAME:-ear-graft}
GPU=${GPU:-0}
DEV=()
if [ "$GPU" = all ]; then DEV=(--device nvidia.com/gpu=all); else for g in ${GPU//,/ }; do DEV+=(--device "nvidia.com/gpu=$g"); done; fi
ENVS=()
for v in WANDB_PROJECT WANDB_MODE WANDB_ENTITY; do [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}"); done
exec docker run --rm --name "$NAME" \
  "${DEV[@]}" --ipc=host --network host \
  --memory "${MEM:-28g}" --memory-swap "${MEMSWAP:-48g}" \
  --user "$(id -u):$(id -g)" -e HOME=/home/nate -e PYTHONNOUSERSITE=1 \
  -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  "${ENVS[@]}" \
  -v /home/nate:/home/nate -w /home/nate/Documents/maestro-llm \
  "$IMAGE" python "$@"
