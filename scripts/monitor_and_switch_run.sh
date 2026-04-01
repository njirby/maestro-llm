#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

python scripts/monitor_overfit_and_relaunch.py \
  --run-name Mar05_06-47-08_ml-workstation \
  "$@"
