#!/usr/bin/env bash
# Restart the 27B with each server config and run scripts/bench_thinking_server.py against it (cold prefix cache each
# time, so the configs are compared fairly). Results: outputs/thinking_bench/<tag>.json + this script's log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROBES_DIR="$PWD/outputs/thinking_capture/hearing_lab/probe_perceptions_pilot20e"
port_pid() { ss -ltnp 2>/dev/null | grep ":8001 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
stop() {  # the server on :8001 (by port), plus any orphaned VLLM workers a crashed server left behind (ppid 1)
  p=$(port_pid); [ -n "$p" ] && { kill "$p"; while kill -0 "$p" 2>/dev/null; do sleep 3; done; }
  for w in $(pgrep -f "^VLLM::Worker"); do [ "$(ps -o ppid= -p "$w" | tr -d ' ')" = 1 ] && kill -9 "$w"; done
  for s in $(pgrep -f "[v]llm serve orcarouter/Qwen3.8-27B"); do kill "$s"; done
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 3; done; }
run() {  # tag, then env assignments for the serve script
  tag="$1"; shift
  stop
  echo "[$(date +%T)] $tag: starting server with $*"
  env "$@" nohup bash ~/ml/serve-qwen38-27b-4card.sh > "outputs/thinking_bench/serve_$tag.log" 2>&1 &
  for i in $(seq 1 90); do curl -sf -m 3 http://localhost:8001/v1/models >/dev/null && break; sleep 10; done
  curl -sf -m 3 http://localhost:8001/v1/models >/dev/null || { echo "[$(date +%T)] $tag: SERVER DID NOT START"; tail -5 "outputs/thinking_bench/serve_$tag.log"; return; }
  .venv/bin/python scripts/bench_thinking_server.py --run outputs/v5_rollouts/pilot20e --replay outputs/replay/v5_pilot20e \
      --sids s0000 s0001 s0002 --n 64 --workers 48 --tag "$tag" 2>&1 | grep '^{'
  grep "Avg generation" "outputs/thinking_bench/serve_$tag.log" | tail -20 | grep -o "generation throughput: [0-9.]*" | awk '{s+=$3;n++} END {if(n) printf "   mean generation tok/s over last %d samples: %.0f\n", n, s/n}'
  grep -i "acceptance\|accepted" "outputs/thinking_bench/serve_$tag.log" | tail -1 | cut -c1-200
}
mkdir -p outputs/thinking_bench
run base_shared MAX_SEQS=48 MAX_BATCHED_TOKENS=2048
echo "[$(date +%T)] bench done"
