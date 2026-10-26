#!/usr/bin/env bash
# Wait for v1 training to exit, then vibe-check its final checkpoint on 4 GPUs.
set -u
cd /home/nate/Documents/maestro-llm
while docker ps --format '{{.Names}}' | grep -qx ear-train; do sleep 30; done
CK=$(ls ear_graft/out/v1-mlp-lora/step_*.pt | grep -v opt | sort -t_ -k3 -n | tail -1)
echo "v1 exited $(date); vibe on $CK"
for g in 0 1 2 3; do
  GPU=$g NAME=vibe$g ear_graft/docker_run.sh -m ear_graft.vibe_check --ckpt "$CK" --shard $g --nshards 4 \
    > "${CK%.pt}.vibe.shard$g.log" 2>&1 &
done
wait
.venv/bin/python -m ear_graft.vibe_check --ckpt "$CK" --report
echo "V1 VIBE DONE $(date)"
