#!/usr/bin/env bash
set -euo pipefail

# Delete quiet tuple audio files and their matching Lua scripts.
#
# Usage:
#   bash scripts/delete_quiet_tuples.sh [tuples_dir] [workers]
#
# Defaults:
#   tuples_dir = data/processed/reaper_tuples_lakh
#   workers    = 48

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TUPLES_DIR="${1:-data/processed/reaper_tuples_lakh}"
WORKERS="${2:-48}"
QUIET_LIST="$TUPLES_DIR/quiet_deleted_paths.txt"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

echo "Tuple dir: $TUPLES_DIR"
echo "Workers  : $WORKERS"
echo "Quiet list: $QUIET_LIST"
echo

# Step 1: detect + delete quiet audio files (mp3/wav/flac/...)
"$PY" scripts/filter_quiet_reaper_tuples.py \
  --tuples-dir "$TUPLES_DIR" \
  --workers "$WORKERS" \
  --delete \
  --quiet-list-out "$QUIET_LIST"

echo
echo "Deleting matching Lua files..."

# Step 2: delete matching Lua files for each deleted audio file
"$PY" - <<'PY' "$TUPLES_DIR" "$QUIET_LIST"
from __future__ import annotations
import sys
from pathlib import Path

tuples_dir = Path(sys.argv[1])
quiet_list = Path(sys.argv[2])
luas_dir = tuples_dir / "luas"

if not quiet_list.exists():
    print(f"Quiet list not found: {quiet_list}", file=sys.stderr)
    sys.exit(1)

paths = [line.strip() for line in quiet_list.read_text().splitlines() if line.strip()]
total = len(paths)
deleted = 0
missing = 0
failed = 0

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

iterable = paths
if tqdm is not None:
    iterable = tqdm(paths, total=total, desc="Deleting Lua", unit="file", dynamic_ncols=True)

for i, p in enumerate(iterable, start=1):
    stem = Path(p).stem
    lua_path = luas_dir / f"{stem}.lua"
    try:
        lua_path.unlink()
        deleted += 1
    except FileNotFoundError:
        missing += 1
    except Exception:
        failed += 1
    if tqdm is None and (i % 1000 == 0 or i == total):
        print(f"  [{i}/{total}] processed")

print()
print("Lua delete summary:")
print(f"  Deleted: {deleted}")
print(f"  Missing: {missing}")
print(f"  Failed : {failed}")
PY

echo
echo "Done."

