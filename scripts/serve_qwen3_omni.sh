#!/bin/bash

set -e

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-4}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional but helps stability on 3090 rigs
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Prevent fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting $MODEL on port $PORT"
echo "tp=$TP_SIZE max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEMORY_UTILIZATION"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code
