#!/usr/bin/env bash
# After v0's chain (resume + final vibe check) is done: smoke-test the v1 setup, then v1 = v0's final checkpoint with
# its linear projector lifted exactly into an MLP + LoRA r16 on the 4B (zero init) + the markers, all trained for one
# epoch on the v1 trainset; then the same vibe check on its final checkpoint.
set -u
cd /home/nate/Documents/maestro-llm
until grep -q "ALL DONE" ear_graft/out/v0/chain.log; do sleep 30; done
V0F=ear_graft/out/v0-r1400/step_2749.pt
[ -f $V0F ] || { echo "no v0 final checkpoint"; exit 1; }

GPU=0 NAME=ear-smoke ear_graft/docker_run.sh -m ear_graft.smoke_v1 --init $V0F > ear_graft/smoke_v1.log 2>&1
grep -q "SMOKE V1 OK" ear_graft/smoke_v1.log || { echo "SMOKE V1 FAILED $(date)"; exit 1; }
echo "smoke v1 ok $(date)"

GPU=all NAME=ear-train ear_graft/docker_run.sh -m torch.distributed.run --nproc_per_node 4 -m ear_graft.train \
  --run-name v1-mlp-lora --projector mlp --lora-r 16 --lr 5e-4 --token-lr 5e-4 --lora-lr 1e-4 --epochs 1 \
  --init-from $V0F --train data/ear_corpus/train_v1.jsonl --val data/ear_corpus/val_v1.jsonl > ear_graft/train_v1.log 2>&1
echo "V1 TRAINING EXITED $? $(date)"
CK=$(ls ear_graft/out/v1-mlp-lora/step_*.pt | grep -v opt | sort -t_ -k3 -n | tail -1)
for g in 0 1 2 3; do
  GPU=$g NAME=vibe$g ear_graft/docker_run.sh -m ear_graft.vibe_check --ckpt "$CK" --shard $g --nshards 4 \
    > "${CK%.pt}.vibe.shard$g.log" 2>&1 &
done
wait
.venv/bin/python -m ear_graft.vibe_check --ckpt "$CK" --report
echo "V1 ALL DONE $(date)"
