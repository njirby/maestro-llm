#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/3] Upgrading pip/setuptools/wheel..."
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

echo "[2/3] Installing MS-Swift and Omni deps..."
"${PYTHON_BIN}" -m pip install --upgrade \
  ms-swift \
  qwen-omni-utils \
  decord \
  soundfile \
  librosa

echo "[3/3] Verifying install..."
"${PYTHON_BIN}" - <<'PY'
import importlib
import os
from pathlib import Path
import shutil
import sys

missing = []
for name in ["swift", "qwen_omni_utils", "decord", "soundfile", "librosa"]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

swift_bin = shutil.which("swift")
if not swift_bin:
    sibling = Path(sys.executable).parent / "swift"
    if sibling.exists() and os.access(sibling, os.X_OK):
        swift_bin = str(sibling.resolve())
if not swift_bin:
    print("ERROR: swift executable not found in PATH.")
    sys.exit(1)

if missing:
    print("ERROR: missing modules:", ", ".join(missing))
    sys.exit(1)

print("swift:", swift_bin)
print("Python package imports: OK")
PY

echo "Setup complete."
