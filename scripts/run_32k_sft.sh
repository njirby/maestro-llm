#!/usr/bin/env bash
# ==========================================================================
# 32K SFT Production Run
# ==========================================================================
#
# End-to-end pipeline: render targets → CLAP cache → build rollouts → merge
#
# Prerequisites:
#   - Omni server(s) running (local + optional RunPod)
#   - Set OMNI_SERVERS for multi-endpoint: export OMNI_SERVERS=http://localhost:8000,http://runpod:8000
#   - Run in tmux to avoid orphaning workers
#
# Resume: safe to re-run after crash. Targets skip existing WAVs,
#         rollouts skip completed batches + samples via .done sentinels.
#
# Usage:
#   tmux new -s sft32k 'bash scripts/run_32k_sft.sh'
#   OMNI_SERVERS=http://localhost:8000,http://runpod:8000 bash scripts/run_32k_sft.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# ==========================================================================
# Hyperparameters
# ==========================================================================

RUN_DIR="outputs/sft_32k"
SUFFIX="v1"

# --- Target generation ---
N_TARGETS=35000               # generate extra to account for ~5% audibility rejection
ARCHETYPES="bass lead pad keys pluck sequence"
RENDER_JOBS=48                # vita workers (CPU-bound, Threadripper sweet spot)
TARGET_SEED=42

# --- Rollout generation ---
BATCH_SIZE=200                # samples per batch (~7 min/batch at 32 workers)
WORKERS=32                    # concurrent rollout builders (match Omni throughput)
ROLLOUT_SEED=1337

# Builder params
MAX_BATCHES=16                # max subsystem correction batches per sample
NUM_AGENTS=4                  # parallel search agent slices
POOL_TOP_K=48                 # CLAP candidate pool size
CANDIDATES_PER_SLICE=48       # wavetables each search agent evaluates
CANDIDATES_PER_BATCH=8        # candidates per search batch within an agent
MAX_SEARCH_ROUNDS=3           # max rounds of wavetable search

# Diversity knobs
PER_PARAM_MISTAKE_RATE=0.10   # 10% of params get wrong values per batch (Poisson cap 0-6)
TRANSCRIPTION_MISTAKE_RATE=0.15  # 15% of transcriptions get wrong notes on first try
FORCE_RESEARCH_RATE=0.20      # 20% of samples force multi-round wavetable search
NO_AUDIO_RATE=0.05            # 5% of samples start with no audio attached
PARTIAL_INIT_RATE=0.40        # 40% start with 1-4 GT subsystems pre-applied (3-7 batches)
RETRANSCRIBE_RATE=0.15        # 15% re-transcribe mid-conversation after hearing wrong notes
STEER_RATE=0.30               # 30% get 1-2 user steer turns after verdict

# LLM endpoints
: "${OMNI_SERVERS:=http://localhost:8000}"
OMNI_MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
CLAP_DEVICE="cpu"             # GPUs occupied by Omni server

# ==========================================================================
# Save hyperparams for reproducibility
# ==========================================================================

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/hyperparams.json" << PARAMS_EOF
{
  "run_dir": "${RUN_DIR}",
  "suffix": "${SUFFIX}",
  "n_targets": ${N_TARGETS},
  "archetypes": "${ARCHETYPES}",
  "render_jobs": ${RENDER_JOBS},
  "target_seed": ${TARGET_SEED},
  "batch_size": ${BATCH_SIZE},
  "workers": ${WORKERS},
  "rollout_seed": ${ROLLOUT_SEED},
  "max_batches": ${MAX_BATCHES},
  "num_agents": ${NUM_AGENTS},
  "pool_top_k": ${POOL_TOP_K},
  "candidates_per_slice": ${CANDIDATES_PER_SLICE},
  "candidates_per_batch": ${CANDIDATES_PER_BATCH},
  "max_search_rounds": ${MAX_SEARCH_ROUNDS},
  "per_param_mistake_rate": ${PER_PARAM_MISTAKE_RATE},
  "transcription_mistake_rate": ${TRANSCRIPTION_MISTAKE_RATE},
  "force_research_rate": ${FORCE_RESEARCH_RATE},
  "no_audio_rate": ${NO_AUDIO_RATE},
  "partial_init_rate": ${PARTIAL_INIT_RATE},
  "retranscribe_rate": ${RETRANSCRIBE_RATE},
  "steer_rate": ${STEER_RATE},
  "omni_servers": "${OMNI_SERVERS}",
  "omni_model": "${OMNI_MODEL}",
  "clap_device": "${CLAP_DEVICE}",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "started_at": "$(date -Iseconds)"
}
PARAMS_EOF
echo "Hyperparams saved to ${RUN_DIR}/hyperparams.json"

# ==========================================================================
# Step 1: Generate targets
# ==========================================================================

echo ""
echo "=== Step 1: Generate ${N_TARGETS} targets ==="
python scripts/render_sft_targets.py \
    --generate ${N_TARGETS} \
    --archetypes ${ARCHETYPES} \
    --output-dir "${RUN_DIR}" \
    --wavetable-lib data/wavetable_lib.json \
    --midi-catalog outputs/midi_clips/lakh_catalog.jsonl \
    --jobs ${RENDER_JOBS} \
    --seed ${TARGET_SEED}

# ==========================================================================
# Step 2: Pre-compute CLAP embeddings
# ==========================================================================

echo ""
echo "=== Step 2: Pre-compute CLAP embeddings ==="
python scripts/precompute_clap_cache.py \
    --manifest "${RUN_DIR}/manifest.jsonl" \
    --probe-dir outputs/agent_sft/candidate_probes \
    --output "${RUN_DIR}/clap_cache.npz" \
    --device ${CLAP_DEVICE}

# ==========================================================================
# Step 3: Build rollouts
# ==========================================================================

echo ""
echo "=== Step 3: Build rollouts ==="
python scripts/run_sft_production.py \
    --manifest "${RUN_DIR}/manifest.jsonl" \
    --out-dir "${RUN_DIR}/rollouts" \
    --batch-size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --omni-server "${OMNI_SERVERS}" \
    --omni-model "${OMNI_MODEL}" \
    --clap-device ${CLAP_DEVICE} \
    --clap-cache "${RUN_DIR}/clap_cache.npz" \
    --max-batches ${MAX_BATCHES} \
    --num-agents ${NUM_AGENTS} \
    --pool-top-k ${POOL_TOP_K} \
    --candidates-per-slice ${CANDIDATES_PER_SLICE} \
    --candidates-per-batch ${CANDIDATES_PER_BATCH} \
    --max-search-rounds ${MAX_SEARCH_ROUNDS} \
    --per-param-mistake-rate ${PER_PARAM_MISTAKE_RATE} \
    --transcription-mistake-rate ${TRANSCRIPTION_MISTAKE_RATE} \
    --force-research-rate ${FORCE_RESEARCH_RATE} \
    --no-audio-rate ${NO_AUDIO_RATE} \
    --partial-init-rate ${PARTIAL_INIT_RATE} \
    --retranscribe-rate ${RETRANSCRIBE_RATE} \
    --steer-rate ${STEER_RATE} \
    --seed ${ROLLOUT_SEED} \
    --suffix ${SUFFIX} \
    --resume

# ==========================================================================
# Step 4: Merge
# ==========================================================================

echo ""
echo "=== Step 4: Merge ==="
python scripts/run_sft_production.py \
    --merge \
    --out-dir "${RUN_DIR}/rollouts" \
    --suffix ${SUFFIX}

# ==========================================================================
# Done
# ==========================================================================

# Update hyperparams with completion time
python3 -c "
import json
with open('${RUN_DIR}/hyperparams.json') as f:
    h = json.load(f)
h['completed_at'] = '$(date -Iseconds)'
with open('${RUN_DIR}/hyperparams.json', 'w') as f:
    json.dump(h, f, indent=2)
"

echo ""
echo "=== Done ==="
echo "  Rollouts: ${RUN_DIR}/rollouts/main_final_${SUFFIX}.jsonl"
echo "  Hyperparams: ${RUN_DIR}/hyperparams.json"
