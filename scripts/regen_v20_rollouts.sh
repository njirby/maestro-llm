#!/usr/bin/env bash
# Regenerate the v20 8-sample rollout set across all 4 agents, grade against
# a live REAPER session + LLM-as-judge, and produce the HTML spot-check.
#
# Prerequisites:
#   - REAPER running with reapy server (Windows-Vital-via-yabridge)
#   - Qwen3-Omni at http://localhost:8000
#   - Lakh MIDI catalog at outputs/midi_clips/lakh_catalog.jsonl (one-time
#     pre-pass; see scripts/build_midi_clip_catalog.py)
#   - 8-sample manifest at outputs/smoke_test_v18/manifest.jsonl (or
#     regenerate with `python scripts/render_iter_presets.py --generate 8 ...`)
#
# Usage:  bash scripts/regen_v20_rollouts.sh [--skip-grade] [--skip-html]

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MANIFEST=outputs/smoke_test_v18/manifest.jsonl
INDEX_NPY=outputs/wt_retrieval_baseline/wt_index.npz
INDEX_META=outputs/wt_retrieval_baseline/wt_index_meta.json
WT_LIB=data/wavetable_lib.json
OUT=outputs/smoke_v3
OMNI_URL=http://localhost:8000
OMNI_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
SUFFIX=v20

SKIP_GRADE=0
SKIP_HTML=0
for arg in "$@"; do
  [[ "$arg" == "--skip-grade" ]] && SKIP_GRADE=1
  [[ "$arg" == "--skip-html" ]]  && SKIP_HTML=1
done

mkdir -p "$OUT"

echo "==> 1/4  Build search agent (32 records: 4 agents × 8 samples)"
python scripts/build_search_agent_sft_v2.py \
    --manifest "$MANIFEST" \
    --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
    --wavetable-lib "$WT_LIB" \
    --out-jsonl "$OUT/search_final8_$SUFFIX.jsonl" \
    --max-samples 8 --num-agents 4 \
    --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
    --workers 4

echo "==> 2/4  Build judge agent (8 records, ~30% no_match via --force-research-rate)"
python scripts/build_judge_agent_sft_v3.py \
    --manifest "$MANIFEST" \
    --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
    --wavetable-lib "$WT_LIB" \
    --out-jsonl "$OUT/judge_final8_$SUFFIX.jsonl" \
    --max-samples 8 --num-agents 4 \
    --force-research-rate 0.30 \
    --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
    --workers 4

echo "==> 3/4  Build transcription agent (8 records, no Omni call needed)"
python scripts/build_transcription_agent_sft_v3.py \
    --manifest "$MANIFEST" \
    --out-jsonl "$OUT/transcription_final8_$SUFFIX.jsonl" \
    --max-samples 8 --workers 4

echo "==> 4/4  Build main agent (8 records, verdict-driven re-search + mistake injection)"
python scripts/build_main_agent_sft_v3.py \
    --manifest "$MANIFEST" \
    --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
    --wavetable-lib "$WT_LIB" \
    --out-jsonl "$OUT/main_final8_$SUFFIX.jsonl" \
    --max-samples 8 --max-batches 6 --num-agents 4 \
    --mistake-rate 0.20 --transcription-mistake-rate 0.15 \
    --force-research-rate 0.30 \
    --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
    --clap-device cpu --workers 4

if [[ "$SKIP_GRADE" -eq 0 ]]; then
  echo "==> Grade search v20 (LLM judge + audio-grounding sample 1/8)"
  python scripts/grade_agent_sft.py \
      --input "$OUT/search_final8_$SUFFIX.jsonl" \
      --output "$OUT/grades_search_$SUFFIX.jsonl" \
      --llm-judge-server "$OMNI_URL" \
      --audio-grounding-sample-rate 0.125 \
      --workers 16

  echo "==> Grade judge v20 (verdict-aware dimensions)"
  python scripts/grade_agent_sft.py \
      --input "$OUT/judge_final8_$SUFFIX.jsonl" \
      --output "$OUT/grades_judge_$SUFFIX.jsonl" \
      --llm-judge-server "$OMNI_URL" \
      --workers 8

  echo "==> Grade transcription v20"
  python scripts/grade_agent_sft.py \
      --input "$OUT/transcription_final8_$SUFFIX.jsonl" \
      --output "$OUT/grades_transcription_$SUFFIX.jsonl" \
      --workers 8

  echo "==> Grade main v20 (live REAPER + LLM judge — slowest, ~1m30s)"
  python scripts/grade_agent_sft.py \
      --input "$OUT/main_final8_$SUFFIX.jsonl" \
      --output "$OUT/grades_main_$SUFFIX.jsonl" \
      --live-exec-check \
      --llm-judge-server "$OMNI_URL" --llm-judge-model "$OMNI_MODEL" \
      --workers 4
fi

if [[ "$SKIP_HTML" -eq 0 ]]; then
  echo "==> Build HTML spot-check"
  python scripts/build_rollout_spotcheck.py \
      --main          "$OUT/main_final8_$SUFFIX.jsonl" \
      --transcription "$OUT/transcription_final8_$SUFFIX.jsonl" \
      --search        "$OUT/search_final8_$SUFFIX.jsonl" \
      --judge         "$OUT/judge_final8_$SUFFIX.jsonl" \
      --out           "$OUT/rollouts_$SUFFIX.html" \
      --title         "$SUFFIX agent rollouts (verdict-driven, leak-free)"
  echo
  echo "Open: file://$(realpath $OUT/rollouts_$SUFFIX.html)"
fi

echo
echo "==> Done."
echo "Rollouts: $OUT/{main,transcription,search,judge}_final8_$SUFFIX.jsonl"
[[ "$SKIP_GRADE" -eq 0 ]] && echo "Grades:   $OUT/grades_*_$SUFFIX.jsonl"
[[ "$SKIP_HTML"  -eq 0 ]] && echo "HTML:     $OUT/rollouts_$SUFFIX.html"
