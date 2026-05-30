#!/usr/bin/env bash
# Full 16K SFT pipeline: render targets, pre-compute CLAP, build rollouts.
# Omni server must already be running on localhost:8000.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Step 1: Generate 20K targets ==="
python scripts/render_sft_targets.py \
    --generate 20000 \
    --archetypes bass lead pad keys pluck sequence \
    --output-dir outputs/sft_16k \
    --wavetable-lib data/wavetable_lib.json \
    --midi-catalog outputs/midi_clips/lakh_catalog.jsonl \
    --jobs 48 \
    --seed 42

echo ""
echo "=== Step 2: Pre-compute CLAP embeddings ==="
python scripts/precompute_clap_cache.py \
    --manifest outputs/sft_16k/manifest.jsonl \
    --probe-dir outputs/agent_sft/candidate_probes \
    --output outputs/sft_16k/clap_cache.npz \
    --device cpu

echo ""
echo "=== Step 3: Build rollouts ==="
python scripts/run_sft_production.py \
    --manifest outputs/sft_16k/manifest.jsonl \
    --out-dir outputs/sft_16k/rollouts \
    --batch-size 200 \
    --workers 24 \
    --omni-server "${OMNI_SERVERS:-http://localhost:8000}" \
    --clap-device cpu \
    --clap-cache outputs/sft_16k/clap_cache.npz \
    --max-batches 16 \
    --num-agents 4 \
    --per-param-mistake-rate 0.10 \
    --transcription-mistake-rate 0.15 \
    --force-research-rate 0.20 \
    --no-audio-rate 0.05 \
    --partial-init-rate 0.40 \
    --seed 1337 \
    --suffix v1 \
    --resume

echo ""
echo "=== Step 4: Merge ==="
python scripts/run_sft_production.py \
    --merge \
    --out-dir outputs/sft_16k/rollouts \
    --suffix v1

echo ""
echo "=== Done ==="
