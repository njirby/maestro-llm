#!/usr/bin/env bash
# Everything after the build for one v5 run: narration (27B edit modes, then Omni first impressions, with the GPU
# swaps), then per rollout, CHAINS at a time: conformance replay -> probe perceptions -> thinking (27B), then merge
# + viewer pages. THINK_URL may list several servers (e.g. two 2-card copies of the 27B on :8001,:8002; each
# conversation sticks to one). Run from the repo root; only touches model servers this script (or its caller) started on
# :8000 / :8001, found by port.
#
#   scripts/run_pilot_pipeline.sh <run name> [build pid to wait for]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; BUILD_PID="${2:-}"
RUN="outputs/v5_rollouts/$NAME"
REPLAY="outputs/replay/v5_$NAME"
export PROBES_DIR="$PWD/outputs/thinking_capture/hearing_lab/probe_perceptions_$NAME"
THINK="$RUN/thinking"
CHAINS="${CHAINS:-3}"
PY=.venv/bin/python
mkdir -p "$PROBES_DIR" "$THINK"
log() { echo "[$(date +%T)] $*"; }

port_pid() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
wait_url() { until curl -sf -m 3 "http://localhost:$1/v1/models" >/dev/null; do sleep 10; done; }
stop_port() {
  local p; p=$(port_pid "$1"); [ -z "$p" ] && return 0
  log "stopping server on :$1 (pid $p)"; kill "$p"
  while kill -0 "$p" 2>/dev/null; do sleep 3; done
}
gpus_free() {   # after every server is stopped (waiting per port deadlocked: :8002 still held GPUs 2,3 after :8001 stopped)
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 3; done; }
start_27b() {   # two TP=2 copies (GPUs 0,1 on :8001, 2,3 on :8002): +42% thinking throughput on this box
  log "starting two 27B copies on :8001 and :8002"
  GPUS=0,1 TP=2 PORT=8001 MAX_SEQS=16 nohup bash ~/ml/serve-qwen38-27b-4card.sh > outputs/serve_qwen38_27b_a.log 2>&1 &
  GPUS=2,3 TP=2 PORT=8002 MAX_SEQS=16 nohup bash ~/ml/serve-qwen38-27b-4card.sh > outputs/serve_qwen38_27b_b.log 2>&1 &
  wait_url 8001; wait_url 8002; log "27B copies up"; }
stop_27b() { stop_port 8001; stop_port 8002
  for w in $(pgrep -f "^VLLM::Worker"); do [ "$(ps -o ppid= -p "$w" | tr -d ' ')" = 1 ] && kill -9 "$w"; done; gpus_free; }
start_omni() { gpus_free; log "starting Omni-Instruct on :8000"; PATH="$PWD/.venv/bin:$PATH" nohup bash scripts/serve_qwen3_omni.sh > outputs/serve_omni_instruct.log 2>&1 & wait_url 8000; log "Omni up"; }

if [ -n "$BUILD_PID" ]; then log "waiting for build pid $BUILD_PID"; while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 30; done; fi
log "build done: $(grep -cE 'validation OK.*GT gate pass' "$RUN.build.log" 2>/dev/null) rollouts pass"

# ---- narration (RESUME=1: already done, skip) -----------------------------------------------------------------------
if [ -z "${RESUME:-}" ]; then
[ -z "$(port_pid 8001)" ] && { stop_port 8000; gpus_free; start_27b; }
log "narration: edit modes (27B)"
$PY scripts/narrate_rollouts_v5.py --run "$RUN" --modes self_edit,target_vs_current --url http://localhost:8001/v1 --arm text --workers 12 > "$RUN.narr_edit.log" 2>&1
$PY scripts/narrate_rollouts_v5.py --run "$RUN" --finalize > "$RUN.finalize1.log" 2>&1
stop_27b; start_omni
log "narration: first impressions (Omni)"
$PY scripts/narrate_rollouts_v5.py --run "$RUN" --modes first_impression --url http://localhost:8000/v1 --workers 4 > "$RUN.narr_fi.log" 2>&1
$PY scripts/narrate_rollouts_v5.py --run "$RUN" --finalize > "$RUN.finalize2.log" 2>&1
log "narration report: $(tr -d '\n ' < "$RUN/narration_report.json" | head -c 400)"
stop_port 8000; gpus_free; start_27b
# narration is final: the oracle render caches (narration ablations, audibility renders) are no longer needed
find "$RUN/oracle" -maxdepth 2 -name render_cache -type d -exec rm -rf {} +
log "oracle render caches removed; run dir now $(du -sh "$RUN" | cut -f1)"
fi
[ -z "$(port_pid 8001)" ] && start_27b
# training records: one per agent type per rollout (searchers stratified)
$PY scripts/select_records_v5.py --run "$RUN" > "$RUN.select.log" 2>&1
log "selection: $(tail -1 "$RUN.select.log")"

# ---- per rollout: replay -> perceptions -> thinking ------------------------------------------------------------------
SIDS=$($PY -c "import json,sys; print(' '.join(json.loads(l)['sample_id'] for l in open('$RUN/index.jsonl') if 'main_id' in json.loads(l)))")
log "chains for: $SIDS"
export RUN REPLAY THINK PY THINK_URL="${THINK_URL:-http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1}"
chain() {
  s="$1"
  # resumable: a finished replay (report.json) and finished perceptions (<sid>.json + done log) are kept
  if [ ! -f "$REPLAY/$s/report.json" ]; then
    $PY -u scripts/replay_in_opencode.py rollout "$RUN" --sample "$s" --out "$REPLAY" --jobs 1 > "$REPLAY.$s.log" 2>&1 || { echo "[$(date +%T)] $s REPLAY FAILED"; return; }
  fi
  if ! grep -q "judge/" "$PROBES_DIR/$s.log" 2>/dev/null; then
    REPLAY_DIR="$PWD/$REPLAY" RUN_DIR="$PWD/$RUN" RENDER_JOBS=8 $PY scripts/build_probe_perceptions.py "$s" > "$PROBES_DIR/$s.log" 2>&1 || { echo "[$(date +%T)] $s PERCEPTIONS FAILED"; return; }
  fi
  $PY scripts/add_thinking_v5.py --mode native --run "$RUN" --replay "$REPLAY" --sids "$s" --out "$THINK" --workers 16 \
      --selection "$RUN/selection.json" \
      --url "${THINK_URL:-http://127.0.0.1:8001/v1}" > "$THINK/$s.log" 2>&1 || { echo "[$(date +%T)] $s THINKING FAILED"; return; }
  # the replay's OpenCode database is scratch once its report and requests are written
  find "$REPLAY/$s" -path "*/oc_root/data/opencode/opencode.db*" -delete
  echo "[$(date +%T)] $s done ($(du -sh "$REPLAY/$s" | cut -f1) replay): replay $(python3 -c "import json;r=json.load(open('$REPLAY/$s/report.json'));print(r['matches'],'/',r['requests'],'diffs',r['diffs'])") ; $(grep '^==' "$THINK/$s.log" | tail -1)"
}
export -f chain
printf '%s\n' $SIDS | xargs -P "$CHAINS" -I{} bash -c 'chain {}'

# ---- merge + pages ------------------------------------------------------------------------------------------------
OUTM="outputs/v5_rollouts/${NAME}_thinking"
$PY scripts/merge_thinking_v5.py --run "$RUN" --thinking "$THINK" --out "$OUTM" > "$OUTM.merge.log" 2>&1
grep '"main_id"' "$RUN/index.jsonl" > "$OUTM/index.jsonl"
for s in $SIDS; do $PY -m maestro.v5.viewer "$OUTM" "$s" -o "$RUN/view/${s}_thinking.html" > /dev/null 2>&1; done
log "merge: $(tail -1 "$OUTM.merge.log")"
rm -rf "$RUN/_sessions"   # the builder's session workdirs; replays used the final-workdir manifests
log "disk: run $(du -sh "$RUN" | cut -f1), replay $(du -sh "$REPLAY" | cut -f1), free $(df -h / | tail -1 | awk '{print $4}')"
log "pipeline done"
