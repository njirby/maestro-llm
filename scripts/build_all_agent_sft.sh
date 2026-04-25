#!/usr/bin/env bash
# Regenerate the agent rollout set across all 4 agents (search, judge,
# transcription, main), optionally grade them, and optionally build the
# HTML spot-check.
#
# All params are exposed as flags. Defaults match the v20 production run.
# Pass `--help` to see the full list.
#
# Prerequisites:
#   - Qwen3-Omni at $OMNI_URL (defaults to http://localhost:8000)
#   - REAPER running with reapy server (only needed when grading with
#     --grade and --live-exec, both default ON)
#   - MIDI catalog at $MIDI_CATALOG (one-time pre-pass via
#     scripts/build_midi_clip_catalog.py)
#   - Manifest at $MANIFEST. To generate a fresh one for N samples:
#       python scripts/render_iter_presets.py --generate N \
#           --archetypes bass lead pad keys pluck sequence \
#           --output-dir <dir> --wavetable-lib data/wavetable_lib.json \
#           --midi-catalog <catalog> --jobs 8
#
# Examples:
#   # Default v20-style run (8 samples, full grading + HTML)
#   bash scripts/build_all_agent_sft.sh
#
#   # Build-only, no grading, custom suffix
#   bash scripts/build_all_agent_sft.sh --suffix v21 --no-grade --no-html
#
#   # 64 samples, demo-rate transcription mistakes (50%) for spot-checking
#   bash scripts/build_all_agent_sft.sh --max-samples 64 \
#       --transcription-mistake-rate 0.5 --suffix demo
#
#   # Skip the search and main grades (cheaper smoke pass)
#   bash scripts/build_all_agent_sft.sh --no-grade-search --no-grade-main

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Defaults — match v20 production
# ---------------------------------------------------------------------------

MANIFEST="${MANIFEST:-outputs/smoke_test_v18/manifest.jsonl}"
INDEX_NPY="${INDEX_NPY:-outputs/wt_retrieval_baseline/wt_index.npz}"
INDEX_META="${INDEX_META:-outputs/wt_retrieval_baseline/wt_index_meta.json}"
WT_LIB="${WT_LIB:-data/wavetable_lib.json}"
MIDI_CATALOG="${MIDI_CATALOG:-outputs/midi_clips/lakh_catalog.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/smoke_v3}"

OMNI_URL="${OMNI_URL:-http://localhost:8000}"
OMNI_MODEL="${OMNI_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
SHORTLIST_DIR="${SHORTLIST_DIR:-/tmp/agent_sft/search_shortlists}"
JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-/tmp/agent_sft/judge_outputs}"

# Build-time params (defaults = v20)
SUFFIX="v20"
MAX_SAMPLES=8
NUM_AGENTS=4              # search slices per sample (also judge pool simulation)
MAX_BATCHES=6             # main agent — subsystem batches per record
MISTAKE_RATE=0.20         # main agent — parameter overshoot rate
TRANSCR_MISTAKE_RATE=0.15 # main agent — transcription mistake rate
FORCE_RESEARCH_RATE=0.30  # main + judge — fraction of samples that miss GT in round 1
WORKERS=4
CLAP_DEVICE="cpu"

# Grading params
GRADE_SEARCH=1
GRADE_JUDGE=1
GRADE_TRANSCRIPTION=1
GRADE_MAIN=1
LIVE_EXEC=1               # requires REAPER for main grading
LLM_JUDGE=1               # requires Omni
SEARCH_AUDIO_GROUNDING_RATE=0.125  # 1/8 sample of per-candidate checks for search
GRADE_WORKERS_SEARCH=16
GRADE_WORKERS_JUDGE=8
GRADE_WORKERS_TRANSCRIPTION=8
GRADE_WORKERS_MAIN=4

# Stages
DO_BUILD=1
DO_GRADE=1
DO_HTML=1

# Per-agent build toggles
BUILD_SEARCH=1
BUILD_JUDGE=1
BUILD_TRANSCRIPTION=1
BUILD_MAIN=1

usage() {
  cat <<USAGE
Usage: $0 [options]

Inputs / outputs
  --manifest PATH                 ($MANIFEST)
  --index-npy PATH                ($INDEX_NPY)
  --index-meta PATH               ($INDEX_META)
  --wavetable-lib PATH            ($WT_LIB)
  --midi-catalog PATH             ($MIDI_CATALOG; must already exist)
  --out-dir DIR                   ($OUT_DIR)
  --suffix NAME                   ($SUFFIX) — file naming: <agent>_final<N>_<suffix>.jsonl

Servers
  --omni-server URL               ($OMNI_URL)
  --omni-model NAME               ($OMNI_MODEL)
  --shortlist-dir DIR             ($SHORTLIST_DIR)
  --judge-output-dir DIR          ($JUDGE_OUTPUT_DIR)

Build params
  --max-samples N                 ($MAX_SAMPLES)
  --num-agents N                  ($NUM_AGENTS) — search slices / judge pool sim
  --max-batches N                 ($MAX_BATCHES) — main agent subsystem batches
  --mistake-rate F                ($MISTAKE_RATE) — main agent param overshoot
  --transcription-mistake-rate F  ($TRANSCR_MISTAKE_RATE)
  --force-research-rate F         ($FORCE_RESEARCH_RATE)
  --workers N                     ($WORKERS)
  --clap-device DEV               ($CLAP_DEVICE; e.g. cpu, cuda:0)

Stages
  --build-only                    same as --no-grade --no-html
  --no-build                      skip the build phase
  --no-grade                      skip ALL grading
  --no-html                       skip the spot-check HTML
  --no-grade-search               skip search grading only
  --no-grade-judge                skip judge grading only
  --no-grade-transcription        skip transcription grading only
  --no-grade-main                 skip main grading only
  --no-build-search               skip search build only
  --no-build-judge                skip judge build only
  --no-build-transcription        skip transcription build only
  --no-build-main                 skip main build only
  --no-live-exec                  skip live REAPER bash exec in main grading
  --no-llm-judge                  skip LLM-as-judge axes (structural only)

Grading params
  --search-audio-grounding-rate F ($SEARCH_AUDIO_GROUNDING_RATE)
  --grade-workers-search N        ($GRADE_WORKERS_SEARCH)
  --grade-workers-judge N         ($GRADE_WORKERS_JUDGE)
  --grade-workers-transcription N ($GRADE_WORKERS_TRANSCRIPTION)
  --grade-workers-main N          ($GRADE_WORKERS_MAIN)

  -h, --help                      this help
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)                       MANIFEST="$2"; shift 2 ;;
    --index-npy)                      INDEX_NPY="$2"; shift 2 ;;
    --index-meta)                     INDEX_META="$2"; shift 2 ;;
    --wavetable-lib)                  WT_LIB="$2"; shift 2 ;;
    --midi-catalog)                   MIDI_CATALOG="$2"; shift 2 ;;
    --out-dir)                        OUT_DIR="$2"; shift 2 ;;
    --suffix)                         SUFFIX="$2"; shift 2 ;;
    --omni-server)                    OMNI_URL="$2"; shift 2 ;;
    --omni-model)                     OMNI_MODEL="$2"; shift 2 ;;
    --shortlist-dir)                  SHORTLIST_DIR="$2"; shift 2 ;;
    --judge-output-dir)               JUDGE_OUTPUT_DIR="$2"; shift 2 ;;
    --max-samples)                    MAX_SAMPLES="$2"; shift 2 ;;
    --num-agents)                     NUM_AGENTS="$2"; shift 2 ;;
    --max-batches)                    MAX_BATCHES="$2"; shift 2 ;;
    --mistake-rate)                   MISTAKE_RATE="$2"; shift 2 ;;
    --transcription-mistake-rate)     TRANSCR_MISTAKE_RATE="$2"; shift 2 ;;
    --force-research-rate)            FORCE_RESEARCH_RATE="$2"; shift 2 ;;
    --workers)                        WORKERS="$2"; shift 2 ;;
    --clap-device)                    CLAP_DEVICE="$2"; shift 2 ;;
    --search-audio-grounding-rate)    SEARCH_AUDIO_GROUNDING_RATE="$2"; shift 2 ;;
    --grade-workers-search)           GRADE_WORKERS_SEARCH="$2"; shift 2 ;;
    --grade-workers-judge)            GRADE_WORKERS_JUDGE="$2"; shift 2 ;;
    --grade-workers-transcription)    GRADE_WORKERS_TRANSCRIPTION="$2"; shift 2 ;;
    --grade-workers-main)             GRADE_WORKERS_MAIN="$2"; shift 2 ;;
    --build-only)                     DO_GRADE=0; DO_HTML=0; shift ;;
    --no-build)                       DO_BUILD=0; shift ;;
    --no-grade)                       DO_GRADE=0; shift ;;
    --no-html)                        DO_HTML=0; shift ;;
    --no-grade-search)                GRADE_SEARCH=0; shift ;;
    --no-grade-judge)                 GRADE_JUDGE=0; shift ;;
    --no-grade-transcription)         GRADE_TRANSCRIPTION=0; shift ;;
    --no-grade-main)                  GRADE_MAIN=0; shift ;;
    --no-build-search)                BUILD_SEARCH=0; shift ;;
    --no-build-judge)                 BUILD_JUDGE=0; shift ;;
    --no-build-transcription)         BUILD_TRANSCRIPTION=0; shift ;;
    --no-build-main)                  BUILD_MAIN=0; shift ;;
    --no-live-exec)                   LIVE_EXEC=0; shift ;;
    --no-llm-judge)                   LLM_JUDGE=0; shift ;;
    -h|--help)                        usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

# Activate venv if not already
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Print resolved config
# ---------------------------------------------------------------------------

echo "=== Config ==="
echo "  manifest:                   $MANIFEST"
echo "  midi-catalog:               $MIDI_CATALOG"
echo "  out-dir:                    $OUT_DIR"
echo "  suffix:                     $SUFFIX"
echo "  omni:                       $OMNI_URL ($OMNI_MODEL)"
echo "  max-samples:                $MAX_SAMPLES"
echo "  num-agents:                 $NUM_AGENTS"
echo "  max-batches:                $MAX_BATCHES"
echo "  mistake-rate:               $MISTAKE_RATE"
echo "  transcription-mistake-rate: $TRANSCR_MISTAKE_RATE"
echo "  force-research-rate:        $FORCE_RESEARCH_RATE"
echo "  workers:                    $WORKERS"
echo "  build:                      search=$BUILD_SEARCH judge=$BUILD_JUDGE trans=$BUILD_TRANSCRIPTION main=$BUILD_MAIN"
echo "  grade:                      search=$GRADE_SEARCH judge=$GRADE_JUDGE trans=$GRADE_TRANSCRIPTION main=$GRADE_MAIN"
echo "  live-exec:                  $LIVE_EXEC      llm-judge: $LLM_JUDGE"
echo

OUT_SEARCH="$OUT_DIR/search_final${MAX_SAMPLES}_${SUFFIX}.jsonl"
OUT_JUDGE="$OUT_DIR/judge_final${MAX_SAMPLES}_${SUFFIX}.jsonl"
OUT_TRANSCRIPTION="$OUT_DIR/transcription_final${MAX_SAMPLES}_${SUFFIX}.jsonl"
OUT_MAIN="$OUT_DIR/main_final${MAX_SAMPLES}_${SUFFIX}.jsonl"
GRADE_SEARCH_FILE="$OUT_DIR/grades_search_${SUFFIX}.jsonl"
GRADE_JUDGE_FILE="$OUT_DIR/grades_judge_${SUFFIX}.jsonl"
GRADE_TRANSCRIPTION_FILE="$OUT_DIR/grades_transcription_${SUFFIX}.jsonl"
GRADE_MAIN_FILE="$OUT_DIR/grades_main_${SUFFIX}.jsonl"
HTML_FILE="$OUT_DIR/rollouts_${SUFFIX}.html"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if [[ "$DO_BUILD" -eq 1 ]]; then
  if [[ "$BUILD_SEARCH" -eq 1 ]]; then
    echo "==> Build search agent"
    python scripts/build_search_agent_sft_v2.py \
        --manifest "$MANIFEST" \
        --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
        --wavetable-lib "$WT_LIB" \
        --out-jsonl "$OUT_SEARCH" \
        --shortlist-dir "$SHORTLIST_DIR" \
        --max-samples "$MAX_SAMPLES" --num-agents "$NUM_AGENTS" \
        --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
        --workers "$WORKERS"
  fi

  if [[ "$BUILD_JUDGE" -eq 1 ]]; then
    echo "==> Build judge agent"
    python scripts/build_judge_agent_sft_v3.py \
        --manifest "$MANIFEST" \
        --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
        --wavetable-lib "$WT_LIB" \
        --out-jsonl "$OUT_JUDGE" \
        --judge-output-dir "$JUDGE_OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" --num-agents "$NUM_AGENTS" \
        --force-research-rate "$FORCE_RESEARCH_RATE" \
        --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
        --workers "$WORKERS"
  fi

  if [[ "$BUILD_TRANSCRIPTION" -eq 1 ]]; then
    echo "==> Build transcription agent"
    python scripts/build_transcription_agent_sft_v3.py \
        --manifest "$MANIFEST" \
        --out-jsonl "$OUT_TRANSCRIPTION" \
        --max-samples "$MAX_SAMPLES" --workers "$WORKERS"
  fi

  if [[ "$BUILD_MAIN" -eq 1 ]]; then
    echo "==> Build main agent"
    python scripts/build_main_agent_sft_v3.py \
        --manifest "$MANIFEST" \
        --index-npy "$INDEX_NPY" --index-meta "$INDEX_META" \
        --wavetable-lib "$WT_LIB" \
        --out-jsonl "$OUT_MAIN" \
        --max-samples "$MAX_SAMPLES" --max-batches "$MAX_BATCHES" \
        --num-agents "$NUM_AGENTS" \
        --mistake-rate "$MISTAKE_RATE" \
        --transcription-mistake-rate "$TRANSCR_MISTAKE_RATE" \
        --force-research-rate "$FORCE_RESEARCH_RATE" \
        --omni-server "$OMNI_URL" --omni-model "$OMNI_MODEL" \
        --clap-device "$CLAP_DEVICE" --workers "$WORKERS"
  fi
fi

# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

GRADE_LLM_FLAG=()
if [[ "$LLM_JUDGE" -eq 1 ]]; then
  GRADE_LLM_FLAG=(--llm-judge-server "$OMNI_URL" --llm-judge-model "$OMNI_MODEL")
else
  GRADE_LLM_FLAG=(--no-llm-judge)
fi

if [[ "$DO_GRADE" -eq 1 ]]; then
  if [[ "$GRADE_SEARCH" -eq 1 ]]; then
    echo "==> Grade search"
    python scripts/grade_agent_sft.py \
        --input "$OUT_SEARCH" --output "$GRADE_SEARCH_FILE" \
        "${GRADE_LLM_FLAG[@]}" \
        --audio-grounding-sample-rate "$SEARCH_AUDIO_GROUNDING_RATE" \
        --workers "$GRADE_WORKERS_SEARCH"
  fi

  if [[ "$GRADE_JUDGE" -eq 1 ]]; then
    echo "==> Grade judge"
    python scripts/grade_agent_sft.py \
        --input "$OUT_JUDGE" --output "$GRADE_JUDGE_FILE" \
        "${GRADE_LLM_FLAG[@]}" \
        --workers "$GRADE_WORKERS_JUDGE"
  fi

  if [[ "$GRADE_TRANSCRIPTION" -eq 1 ]]; then
    echo "==> Grade transcription"
    python scripts/grade_agent_sft.py \
        --input "$OUT_TRANSCRIPTION" --output "$GRADE_TRANSCRIPTION_FILE" \
        "${GRADE_LLM_FLAG[@]}" \
        --workers "$GRADE_WORKERS_TRANSCRIPTION"
  fi

  if [[ "$GRADE_MAIN" -eq 1 ]]; then
    echo "==> Grade main"
    LIVE_EXEC_FLAG=()
    [[ "$LIVE_EXEC" -eq 1 ]] && LIVE_EXEC_FLAG=(--live-exec-check)
    python scripts/grade_agent_sft.py \
        --input "$OUT_MAIN" --output "$GRADE_MAIN_FILE" \
        "${LIVE_EXEC_FLAG[@]}" \
        "${GRADE_LLM_FLAG[@]}" \
        --workers "$GRADE_WORKERS_MAIN"
  fi
fi

# ---------------------------------------------------------------------------
# HTML spot-check
# ---------------------------------------------------------------------------

if [[ "$DO_HTML" -eq 1 ]]; then
  echo "==> Build HTML spot-check"
  HTML_ARGS=()
  [[ -f "$OUT_MAIN"          ]] && HTML_ARGS+=(--main          "$OUT_MAIN")
  [[ -f "$OUT_TRANSCRIPTION" ]] && HTML_ARGS+=(--transcription "$OUT_TRANSCRIPTION")
  [[ -f "$OUT_SEARCH"        ]] && HTML_ARGS+=(--search        "$OUT_SEARCH")
  [[ -f "$OUT_JUDGE"         ]] && HTML_ARGS+=(--judge         "$OUT_JUDGE")
  if [[ ${#HTML_ARGS[@]} -gt 0 ]]; then
    python scripts/build_rollout_spotcheck.py \
        "${HTML_ARGS[@]}" \
        --out "$HTML_FILE" \
        --title "$SUFFIX agent rollouts (n=$MAX_SAMPLES)"
    echo
    echo "Open: file://$(realpath "$HTML_FILE")"
  fi
fi

echo
echo "==> Done. Suffix=$SUFFIX  Out=$OUT_DIR"
