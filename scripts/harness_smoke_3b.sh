#!/bin/bash
# Tracer-bullet final step: drive a LIVE rollout with the memorized 3B through
# the real harness, against a real REAPER container.
#
# This answers the question the whole exercise was built around: does a model
# trained on our data actually work in the harness we ship?
#
# Sequence:
#   1. merge the LoRA into base weights  — MANDATORY: vLLM cannot apply LoRA to
#      multimodal towers, and this run trained the audio tower, so serving the
#      adapter directly would silently drop those weights.
#   2. serve the merged model (with the tool-calling flags vLLM needs)
#   3. run the harness INSIDE a daw-farm container, pointed at the host server
#   4. dump the resulting conversation for comparison against the memorized rollout
#
# Usage: scripts/harness_smoke_3b.sh <checkpoint_dir> [container] [target_sample]
set -euo pipefail
cd /home/nate/Documents/maestro-llm

CKPT="${1:?usage: harness_smoke_3b.sh <checkpoint_dir> [container] [sample_id]}"
CONTAINER="${2:-daw-farm-reaper-12}"
SAMPLE="${3:-}"
BASE=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
MERGED="outputs/pilot_v4/merged_3b"
HARNESS=/home/nate/Documents/maestro-reaper-plugin
PORT=8000
SCRATCH=/tmp/claude-1000/-home-nate-Documents/80c55f7b-f2e1-418e-bbf2-12d5a0e50473/scratchpad

# --- 0. free the GPUs (training may still hold them) -------------------------
pkill -9 -f "_megatron/sft.py" 2>/dev/null || true
pkill -9 -f "train_qwen25_omni_lora_megatron" 2>/dev/null || true
sleep 8
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 6
echo "GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"

# --- 1. merge ---------------------------------------------------------------
if [ ! -f "${MERGED}/config.json" ]; then
  echo "=== merging adapter into base weights ==="
  rm -rf "${MERGED}"
  .venv/bin/swift export --model "${BASE}" --adapters "${CKPT}" \
    --merge_lora true --output_dir "${MERGED}" 2>&1 | tail -5
fi
MERGED_REAL=$(ls -d "${MERGED}"*/ 2>/dev/null | head -1 || echo "${MERGED}")
echo "merged model: ${MERGED_REAL}"

# --- 2. serve ---------------------------------------------------------------
pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 5
nohup .venv/bin/vllm serve "${MERGED_REAL}" \
  --served-model-name maestro-main --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 4 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "${SCRATCH}/vllm_merged3b.log" 2>&1 &
echo "serving merged model (pid $!) — waiting for readiness"
for i in $(seq 1 90); do
  curl -s -m3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 20
done
curl -s -m5 "http://127.0.0.1:${PORT}/v1/models" | head -c 100; echo

# --- 3. harness inside a fresh container ------------------------------------
echo "=== preparing container ${CONTAINER} ==="
docker restart "${CONTAINER}" >/dev/null
for i in $(seq 1 30); do docker exec "${CONTAINER}" reaper-ready >/dev/null 2>&1 && break; sleep 5; done
docker exec "${CONTAINER}" rm -rf /work/harness /work/live_cwd
docker cp "${HARNESS}" "${CONTAINER}:/work/harness"
docker cp skills "${CONTAINER}:/work/live_cwd_skills" 2>/dev/null || true
docker exec "${CONTAINER}" sh -c "mkdir -p /work/live_cwd && mv /work/live_cwd_skills /work/live_cwd/skills 2>/dev/null || true"
docker exec "${CONTAINER}" pip install -e /work/harness --break-system-packages -q 2>&1 | tail -1 || true

GW=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' "${CONTAINER}")
echo "container -> host gateway: ${GW}"

# stage the target audio at its recorded absolute path
if [ -n "${SAMPLE}" ]; then
  GT=$(.venv/bin/python -c "
import json
for l in open('outputs/ref_run11/manifest.jsonl'):
    e=json.loads(l)
    if e['sample_id']=='${SAMPLE}': print(e['gt_wav']); break")
  if [ -n "${GT}" ]; then
    docker exec -u 0 "${CONTAINER}" mkdir -p "$(dirname "${GT}")"
    docker exec -u 0 "${CONTAINER}" chmod -R 777 "$(dirname "${GT}")"
    docker cp "${GT}" "${CONTAINER}:${GT}"
    echo "staged target: ${GT}"
  fi
fi

echo "=== running live harness rollout ==="
docker exec -e MAESTRO_KEEP_WORKSPACE=1 "${CONTAINER}" \
  python3 /work/harness/scripts/bench_headless.py \
    --llm-url "http://${GW}:${PORT}/v1/chat/completions" \
    --cwd /work/live_cwd --skill-root /work/live_cwd/skills \
    --bash-timeout 300 --verbose 2>&1 | tail -60
