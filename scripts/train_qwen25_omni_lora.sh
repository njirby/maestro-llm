#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" scripts/train_qwen25_omni_lora.py --profile "${PROFILE}" "$@"
