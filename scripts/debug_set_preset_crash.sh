#!/bin/bash
# Driver: runs each debug_set_preset_crash.py case with a fresh REAPER.
# Emits a one-line summary per case to stdout.

set -u
cd "$(dirname "$0")/.."

CASES=(baseline tiny_tweak tiny_wt small_wt medium_wt huge_wt)
REAPER_BIN=/home/nate/opt/REAPER/REAPER/reaper

start_reaper() {
  pgrep -f "REAPER/reaper" | xargs -r kill 2>/dev/null
  sleep 2
  pgrep -f "REAPER/reaper" | xargs -r kill -9 2>/dev/null
  sleep 1
  "$REAPER_BIN" -nonewinst >/tmp/reaper_diag.log 2>&1 &
  sleep 18
}

probe() {
  source .venv/bin/activate
  python -c "
import reapy
with reapy.inside_reaper():
    _ = len(reapy.Project().tracks)
print('ok')
" 2>/dev/null | tail -1
}

for case in "${CASES[@]}"; do
  echo "=== CASE: $case ==="
  start_reaper
  probe_result="$(probe)"
  if [ "$probe_result" != "ok" ]; then
    echo "REAPER probe failed before test — skipping"
    continue
  fi
  python scripts/debug_set_preset_crash.py "$case" 2>&1 | grep -E '^\{.*\}$' || true
  # Check REAPER still alive after the test
  probe_result="$(probe)"
  if [ "$probe_result" != "ok" ]; then
    echo "REAPER DEAD after $case"
  else
    echo "REAPER ALIVE after $case"
  fi
  echo ""
done
