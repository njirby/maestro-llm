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
# Always re-merge: a stale merged dir from an earlier checkpoint is the kind
# of silent mismatch that wastes a whole test round.
echo "=== merging adapter into base weights (${CKPT}) ==="
rm -rf "${MERGED}"
.venv/bin/swift export --model "${BASE}" --adapters "${CKPT}" \
  --merge_lora true --output_dir "${MERGED}" 2>&1 | tail -4
# The merged config inherits the base card's 32k limit; we trained at 65k and
# vLLM refuses to serve beyond the declared value. Make the config state what
# the model actually is.
.venv/bin/python - "${MERGED}" <<'PY'
import json, sys, pathlib, glob
d = sorted(glob.glob(sys.argv[1] + "*/"))[0] if glob.glob(sys.argv[1] + "*/") else sys.argv[1]
p = pathlib.Path(d) / "config.json"
c = json.load(open(p))
def bump(x):
    n = 0
    if isinstance(x, dict):
        if isinstance(x.get("max_position_embeddings"), int):
            x["max_position_embeddings"] = 65536; n += 1
        for v in x.values(): n += bump(v)
    return n
print("  bumped max_position_embeddings fields:", bump(c))
json.dump(c, open(p, "w"), indent=2)
PY
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

# --- 4. live rollouts (one per sample) --------------------------------------
# bench_headless cannot attach audio, so it is useless for a live
# audio-conditioned test; live_rollout_runner seeds the target the way the
# plugin does and runs GREEDY (temperature 0).
docker cp scripts/live_rollout_runner.py "${CONTAINER}:/work/live_rollout_runner.py"
docker cp /home/nate/Documents/maestro-reaper-plugin/src/maestro_plugin/opencode_contract.py \
  "${CONTAINER}:/work/harness/src/maestro_plugin/opencode_contract.py" 2>/dev/null || true

SAMPLES="${SAMPLE:-$(.venv/bin/python -c "
import json
ids=[]
for l in open('outputs/pilot_v4/main_final256_v31.jsonl'):
    r=json.loads(l)
    if len(r['messages']) > 5:   # skip the no-audio edge-case records
        ids.append(r['meta']['sample_id'])
print(' '.join(ids))")}"
echo "=== live rollouts for: ${SAMPLES} ==="

for SID in ${SAMPLES}; do
  GT=$(.venv/bin/python -c "
import json
for l in open('outputs/ref_run11/manifest.jsonl'):
    e=json.loads(l)
    if e['sample_id']=='${SID}': print(e['gt_wav']); break")
  [ -z "${GT}" ] && { echo "  ${SID}: no GT, skipping"; continue; }
  docker exec -u 0 "${CONTAINER}" mkdir -p "$(dirname "${GT}")" 2>/dev/null || true
  docker exec -u 0 "${CONTAINER}" chmod -R 777 "$(dirname "${GT}")" 2>/dev/null || true
  docker cp "${GT}" "${CONTAINER}:${GT}"
  # pristine state per rollout (same hygiene policy as generation)
  docker exec "${CONTAINER}" rm -rf "/work/rollouts/${SID}" /tmp/agents /tmp/search_probes 2>/dev/null || true
  echo "--- ${SID}"
  docker exec -e MAESTRO_KEEP_WORKSPACE=1 "${CONTAINER}" \
    python3 /work/live_rollout_runner.py \
      "http://${GW}:${PORT}/v1/chat/completions" "${GT}" \
      "Recreate this sound in Vital." 2400 "${SID}" 2>&1 | tail -3
  .venv/bin/python scripts/score_live_rollout.py --sample "${SID}" \
    --session "${CONTAINER}" --manifest outputs/ref_run11/manifest.jsonl 2>&1 | tail -12
done
