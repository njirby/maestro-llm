#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "[1/5] Upgrading pip/setuptools/wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[2/5] Installing MS-Swift + Qwen Omni runtime deps..."
"$PYTHON_BIN" -m pip install --upgrade \
  ms-swift \
  qwen-omni-utils \
  decord \
  soundfile \
  librosa

echo "[3/5] Preparing CUDA/NCCL include+lib paths from venv packages..."
readarray -t _CUDA_PATHS < <("$PYTHON_BIN" - <<'PY'
from pathlib import Path
try:
    import nvidia
except Exception:
    raise SystemExit(1)
root = Path(nvidia.__file__).resolve().parent
incs = sorted(str(p) for p in root.glob("*/include") if p.is_dir())
libs = sorted(str(p) for p in root.glob("*/lib") if p.is_dir())
print(":".join(incs))
print(":".join(libs))
PY
)

if [[ "${#_CUDA_PATHS[@]}" -ne 2 ]]; then
  echo "ERROR: failed to resolve venv CUDA include/lib paths from nvidia packages."
  exit 1
fi

CUDA_INCLUDES="${_CUDA_PATHS[0]}"
CUDA_LIBS="${_CUDA_PATHS[1]}"

if [[ -z "$CUDA_INCLUDES" || -z "$CUDA_LIBS" ]]; then
  echo "ERROR: empty CUDA include/lib path list."
  exit 1
fi

export CPATH="$CUDA_INCLUDES:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDES:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_LIBS:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"

echo "[4/5] Installing Megatron-SWIFT dependencies..."
"$PYTHON_BIN" -m pip install --no-build-isolation --no-cache-dir "transformer-engine[pytorch]==2.12.0"
"$PYTHON_BIN" -m pip install --upgrade mcore-bridge "megatron-core==0.16.*"

echo "[5/5] Verifying install..."
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
from pathlib import Path
import shutil
import sys

pkgs = [
    "ms-swift",
    "transformer-engine",
    "mcore-bridge",
    "megatron-core",
    "peft",
]
for p in pkgs:
    try:
        print(f"{p}: {md.version(p)}")
    except Exception as e:
        print(f"ERROR: missing {p}: {e}")
        raise SystemExit(1)

swift_bin = shutil.which("swift") or str(Path(sys.executable).parent / "swift")
mega_bin = shutil.which("megatron") or str(Path(sys.executable).parent / "megatron")
for name, val in [("swift", swift_bin), ("megatron", mega_bin)]:
    if not val or not Path(val).exists():
        print(f"ERROR: {name} executable not found")
        raise SystemExit(1)
    print(f"{name}: {Path(val).resolve()}")

print("LD_LIBRARY_PATH head:", os.environ.get("LD_LIBRARY_PATH", "").split(":")[:5])
print("Megatron setup verification: OK")
PY

echo "Setup complete."
