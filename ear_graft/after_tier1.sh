#!/usr/bin/env bash
# Wait for tier-1 downloads (clotho is the last tier-1 source), extract them into the manifest, then run the teacher on
# just those sources (seed 1, own output file). Shares the server with any other running gen job.
set -euo pipefail
cd /home/nate/Documents/maestro-llm
C=data/ear_corpus
until grep -q "clotho OK" $C/fetch.log; do sleep 60; done
SRC=esc50,nsynth,librispeech,fma_small,audioset_bal,clotho
.venv/bin/python -m ear_graft.build_manifest --sources $SRC
exec .venv/bin/python -u -m ear_graft.gen_teacher --sources $SRC --n "${N:-12000}" --pair-frac 0.25 --seed 1 \
  --concurrency 24 --out $C/teacher_dl_s1.jsonl
