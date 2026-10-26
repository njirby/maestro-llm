#!/usr/bin/env bash
set -euo pipefail
export OPENCODE_CONFIG=/tmp/opencode/explorer7b/opencode.json
export PATH="/home/nate/Documents/maestro-llm/.venv/bin:$PATH"
cd /tmp/opencode/explorer7b/work
exec opencode run --agent explorer "$@"
