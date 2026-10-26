#!/usr/bin/env bash
# Download the ear-graft audio corpus (~41 GB of HF parquet) into data/ear_corpus/raw/<name>.
# Lossy Wi-Fi: `hf download` resumes partial files, so each source is retried until it completes.
set -uo pipefail
HF=/home/nate/Documents/maestro-llm/.venv/bin/hf
OUT=/home/nate/Documents/maestro-llm/data/ear_corpus/raw
ERR=/home/nate/Documents/maestro-llm/data/ear_corpus/fetch_errors.log
mkdir -p "$OUT"
# Xet stalls on this link (0-byte .incomplete for 13 min, 2026-10-23); plain HTTP with Range resume works.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
# name | repo | include globs. The link does ~650 KB/s, so tier 1 (a slice of every source, ~7.4 GB, ~3 h) comes
# first; tier 2 repeats the full globs (hf skips files already complete).
SOURCES=(
  "esc50|ashraq/esc50|data/*"
  "nsynth|jg583/NSynth|data/test/*"
  "librispeech|openslr/librispeech_asr|all/test.clean/*"
  "fma_small|benjamin-paine/free-music-archive-small|data/train-0000[0-1]-*"
  "audioset_bal|agkphysics/AudioSet|data/bal_train/0[0-3].parquet ontology.json"
  "clotho|CLAPv2/Clotho|data/test/*"
  # tier 2
  "nsynth|jg583/NSynth|data/test/* data/validation/*"
  "esc50|ashraq/esc50|data/*"
  "librispeech|openslr/librispeech_asr|all/test.clean/* all/test.other/*"
  "clotho|CLAPv2/Clotho|data/valid/* data/test/*"
  "fma_small|benjamin-paine/free-music-archive-small|data/*"
  "audioset_bal|agkphysics/AudioSet|data/bal_train/* ontology.json"
)
for s in "${SOURCES[@]}"; do
  IFS='|' read -r name repo globs <<<"$s"
  inc=(); for g in $globs; do inc+=(--include "$g"); done
  for try in $(seq 1 30); do
    echo "[$(date +%T)] $name try $try"
    if "$HF" download "$repo" --repo-type dataset "${inc[@]}" --local-dir "$OUT/$name" --max-workers 4 >/dev/null 2>>"$ERR"; then
      echo "[$(date +%T)] $name OK $(du -sh "$OUT/$name" | cut -f1)"; break
    fi
    sleep 20
  done
done
echo ALL_DONE
