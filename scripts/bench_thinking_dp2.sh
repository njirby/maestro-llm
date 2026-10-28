#!/usr/bin/env bash
# Two 2-card copies of the 27B (:8001 on GPUs 0,1 and :8002 on GPUs 2,3), then the thinking benchmark across both
# (conversations routed stickily, add_thinking_v5.for_server). Compare with bench_thinking_configs.sh's single copy.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROBES_DIR="$PWD/outputs/thinking_capture/hearing_lab/probe_perceptions_pilot20e"
SEQS="${SEQS:-16}"; TAG="${TAG:-dp2_seq$SEQS}"
pid_of() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
for port in 8001 8002; do p=$(pid_of $port); [ -n "$p" ] && { kill "$p"; while kill -0 "$p" 2>/dev/null; do sleep 3; done; }; done
for w in $(pgrep -f "^VLLM::Worker"); do [ "$(ps -o ppid= -p "$w" | tr -d ' ')" = 1 ] && kill -9 "$w"; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 3; done
echo "[$(date +%T)] starting two copies (MAX_SEQS=$SEQS each)"
GPUS=0,1 TP=2 PORT=8001 MAX_SEQS=$SEQS nohup bash ~/ml/serve-qwen38-27b-4card.sh > "outputs/thinking_bench/serve_${TAG}_a.log" 2>&1 &
GPUS=2,3 TP=2 PORT=8002 MAX_SEQS=$SEQS nohup bash ~/ml/serve-qwen38-27b-4card.sh > "outputs/thinking_bench/serve_${TAG}_b.log" 2>&1 &
for port in 8001 8002; do
  for i in $(seq 1 90); do curl -sf -m 3 "http://localhost:$port/v1/models" >/dev/null && break; sleep 10; done
  curl -sf -m 3 "http://localhost:$port/v1/models" >/dev/null || { echo "copy on :$port did not start"; tail -5 "outputs/thinking_bench/serve_${TAG}_"*.log; exit 1; }
done
echo "[$(date +%T)] both up; KV: $(grep -h "GPU KV cache size" outputs/thinking_bench/serve_${TAG}_*.log | grep -o "size: [0-9,]* tokens" | tr '\n' ' ')"
.venv/bin/python scripts/bench_thinking_server.py --run outputs/v5_rollouts/pilot20e --replay outputs/replay/v5_pilot20e \
    --sids s0000 s0001 s0002 --n 64 --workers 48 --tag "$TAG" --url http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1 2>&1 | grep '^{'
for x in a b; do echo "copy $x preemptions: $(grep -ci preempt outputs/thinking_bench/serve_${TAG}_$x.log)"; done
echo "[$(date +%T)] dp2 bench done"
