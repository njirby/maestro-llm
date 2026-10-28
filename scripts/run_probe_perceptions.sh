#!/usr/bin/env bash
# Probe perceptions for many samples: one build_probe_perceptions.py process (single-threaded numpy, its own render
# container) per sample, PROCS at a time. Samples have different notes, so their render cache keys never collide.
#
#   REPLAY_DIR=... RUN_DIR=... PROBES_DIR=... PROCS=8 RENDER_JOBS=6 scripts/run_probe_perceptions.sh s0000 s0001 ...
set -euo pipefail
cd "$(dirname "$0")/.."
PROCS="${PROCS:-8}"
export RENDER_JOBS="${RENDER_JOBS:-6}"
printf '%s\n' "$@" | xargs -P "$PROCS" -I{} sh -c \
  '.venv/bin/python scripts/build_probe_perceptions.py {} > "${PROBES_DIR}/{}.log" 2>&1 && echo "{} done" || echo "{} FAILED"'
