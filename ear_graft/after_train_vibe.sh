#!/usr/bin/env bash
# Stop v0 once step_1400.pt (end of epoch 1) is on disk, vibe-check it, resume training from step 1401, then
# vibe-check the final checkpoint.
set -u
cd /home/nate/Documents/maestro-llm
V0=ear_graft/out/v0; R=ear_graft/out/v0-r1400

vibe() {  # $1 = checkpoint path
  for g in 0 1 2 3; do
    GPU=$g NAME=vibe$g ear_graft/docker_run.sh -m ear_graft.vibe_check --ckpt "$1" --shard $g --nshards 4 \
      > "${1%.pt}.vibe.shard$g.log" 2>&1 &
  done
  wait
  .venv/bin/python -m ear_graft.vibe_check --ckpt "$1" --report
}

until [ -f $V0/step_1400.pt ]; do sleep 20; done
prev=-1; while [ "$(stat -c %s $V0/step_1400.pt)" != "$prev" ]; do prev=$(stat -c %s $V0/step_1400.pt); sleep 10; done
echo "step_1400.pt written ($prev bytes); stopping v0 $(date)"
docker stop -t 30 ear-train
while docker ps --format '{{.Names}}' | grep -qx ear-train; do sleep 5; done

vibe $V0/step_1400.pt
echo "VIBE 1400 DONE $(date)"

GPU=all NAME=ear-train ear_graft/docker_run.sh -m torch.distributed.run --nproc_per_node 4 -m ear_graft.train \
  --run-name v0-r1400 --epochs 2 --resume $V0/step_1400.pt > ear_graft/train_v0_r1400.log 2>&1
echo "RESUMED TRAINING EXITED $? $(date)"

[ -f $R/step_2749.pt ] && vibe $R/step_2749.pt
echo "ALL DONE $(date)"
