#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

AUDIO_DIR="${AUDIO_DIR:-$ROOT_DIR/data/processed/reaper_tuples_lakh/wavs}"
LUA_DIR="${LUA_DIR:-$ROOT_DIR/data/processed/reaper_tuples_lakh/luas}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/data/prompts/omni_lua_user_prompt.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/prepared/omni_lua_sft_full_90_10}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_COUNT="${VAL_COUNT:-256}"
SEED="${SEED:-42}"
SMOKE_REPORT="${SMOKE_REPORT:-$ROOT_DIR/outputs/omni_lua_full_preprocess_smoke.json}"

echo "Building full Omni Lua SFT dataset"
echo "  audio_dir:    $AUDIO_DIR"
echo "  lua_dir:      $LUA_DIR"
echo "  prompt_file:  $PROMPT_FILE"
echo "  out_dir:      $OUT_DIR"
echo "  val_ratio:    $VAL_RATIO"
echo "  val_count:    $VAL_COUNT"
echo "  seed:         $SEED"

"$PYTHON_BIN" "$ROOT_DIR/scripts/build_omni_lua_sft_dataset.py" \
  --audio-dir "$AUDIO_DIR" \
  --lua-dir "$LUA_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --out-dir "$OUT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --val-count "$VAL_COUNT" \
  --seed "$SEED" \
  --require-lua-markers

echo "Running preprocess smoke check"
"$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_test_omni_lua_dataset.py" \
  --dataset-dir "$OUT_DIR" \
  --split train \
  --max-rows 1024 \
  --require-lua-markers \
  --report-out "$SMOKE_REPORT"

echo "Done"
echo "  dataset_dir:  $OUT_DIR"
echo "  stats:        $OUT_DIR/stats.json"
echo "  smoke_report: $SMOKE_REPORT"
