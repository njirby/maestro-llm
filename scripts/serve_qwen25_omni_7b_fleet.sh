#!/usr/bin/env bash
# Serve Qwen2.5-Omni-7B as N independent single-GPU vLLM replicas
# (ports 8001..800N). Compare against the TP4 30B-A3B server: smaller model,
# 4x the aggregate batch capacity. Use with the builders' multi-endpoint
# router: --omni-server http://localhost:8001,...,http://localhost:800N
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
N_REPLICAS="${N_REPLICAS:-4}"
BASE_PORT="${BASE_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

pids=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
    port=$((BASE_PORT + i))
    echo "replica $i: GPU $i port $port"
    CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$port" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --kv-cache-dtype fp8 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        > "/tmp/vllm_7b_replica_$i.log" 2>&1 &
    pids+=($!)
done
echo "replicas: ${pids[*]} (logs: /tmp/vllm_7b_replica_N.log)"
wait
