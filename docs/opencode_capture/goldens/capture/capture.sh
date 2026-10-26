#!/usr/bin/env bash
# Re-capture the OpenCode goldens from the real binary against mock_scripted.py.
#
#   docs/opencode_capture/goldens/capture/capture.sh [SANDBOX_DIR]
#   .venv/bin/python docs/opencode_capture/goldens/capture/extract.py SANDBOX_DIR
#
# Isolation (same recipe as opencode/README.md): HOME and XDG dirs point into the
# sandbox, so nothing from ~/.config/opencode, ~/.claude or ~/.agents is read, and
# OPENCODE_CONFIG_DIR points at a staged copy of the repo's opencode/ dir WITHOUT
# plugin/ (the goldens are stock OpenCode behaviour; the plugin is layered on top).
# Set WITH_PLUGIN=1 to stage opencode/plugin too (or PLUGIN_SRC=<dir> instead).
# CONFIG_CONTENT='{...}' is passed as OPENCODE_CONFIG_CONTENT (a last-merged override).
# OFFLINE=1 points every package registry and proxy at a dead port, to prove a
# run needs no network. Offline, glob/grep/skill need ripgrep: OpenCode uses
# `rg` from PATH or $XDG_CACHE_HOME/opencode/bin/rg and otherwise downloads
# ripgrep 15.1.0 there; RG_BIN=<path> seeds that cache.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
SB=${1:-$(mktemp -d)}
PORT=${PORT:-8911}
OC=${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}
rm -rf "$SB"; mkdir -p "$SB"/{home,cfg,work/sub/nested}
SB=$(cd "$SB" && pwd)
WD=$SB/work

if [ -n "${RG_BIN:-}" ]; then mkdir -p "$SB/home/.cache/opencode/bin"; cp "$RG_BIN" "$SB/home/.cache/opencode/bin/rg"; fi
# Same staging as the launcher recipe in opencode/README.md: the config dir
# OpenCode loads (and writes .gitignore/package.json into) is a throwaway dir
# of symlinks into the repo's opencode/ (OC_SRC overrides that source).
SRC=${OC_SRC:-$REPO/opencode}
ln -s "$SRC/opencode.json" "$SRC/agent" "$SB/cfg/"
if [ -f "$SRC/skills/vital/SKILL.md" ]; then ln -s "$SRC/skills" "$SB/cfg/skills"
else echo "WARN: $SRC/skills/vital missing; staging $REPO/skills instead" >&2; ln -s "$REPO/skills" "$SB/cfg/skills"; fi
if [ "${WITH_PLUGIN:-0}" = 1 ]; then ln -s "${PLUGIN_SRC:-$SRC/plugin}" "$SB/cfg/plugin"; fi

for i in $(seq 1 30); do echo "line $i"; done > "$WD/notes.txt"
printf 'one\ntwo\nthree\n' > "$WD/short.txt"
seq 1 3000 > "$WD/long.txt"
python3 -c "print('\n'.join('%04d ' % i + 'z' * 95 for i in range(1000)))" > "$WD/big.txt"
python3 -c "print('y' * 2500)" > "$WD/longline.txt"
: > "$WD/empty.txt"
echo a > "$WD/sub/a.txt"; echo b > "$WD/sub/b.txt"; echo c > "$WD/sub/nested/c.txt"
python3 - "$WD" <<'PY'
import math, struct, sys, wave, zlib
wd = sys.argv[1]
for name, hz in (("target", 220), ("second", 330)):
    with wave.open(f"{wd}/{name}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / 16000))) for i in range(4000)))
open(f"{wd}/noise.raw", "wb").write(bytes(range(256)) * 16)
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
open(f"{wd}/img.png", "wb").write(png)
open(f"{wd}/img2.png", "wb").write(png)
PY

cp -r "$WD" "$SB/work_initial"
python3 "$REPO/docs/opencode_capture/goldens/capture/mock_scripted.py" "$SB/captured.jsonl" "$PORT" "$WD" "$SB/cfg" &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1

# The prompt goes in on stdin: `opencode run "a b"` wraps any positional arg
# containing a space in double quotes before it reaches the model.
printf 'Recreate the sound in target.wav.' > "$SB/prompt.txt"
OFFLINE_ENV=()
if [ "${OFFLINE:-0}" = 1 ]; then
  dead=http://127.0.0.1:9
  OFFLINE_ENV=(HTTP_PROXY=$dead HTTPS_PROXY=$dead http_proxy=$dead https_proxy=$dead NO_PROXY=127.0.0.1
               no_proxy=127.0.0.1 NPM_CONFIG_REGISTRY=$dead/ npm_config_registry=$dead/ BUN_CONFIG_REGISTRY=$dead/)
fi
cd "$WD"
[ -n "${CONFIG_CONTENT:-}" ] && OFFLINE_ENV+=(OPENCODE_CONFIG_CONTENT="$CONFIG_CONTENT")
env -i "${OFFLINE_ENV[@]}" PATH=/usr/local/bin:/usr/bin:/bin TERM=dumb LANG=C.UTF-8 \
  HOME="$SB/home" XDG_CONFIG_HOME="$SB/home/.config" XDG_DATA_HOME="$SB/home/.local/share" \
  XDG_CACHE_HOME="$SB/home/.cache" XDG_STATE_HOME="$SB/home/.local/state" \
  OPENCODE_CONFIG_DIR="$SB/cfg" \
  OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 \
  OPENCODE_DISABLE_CLAUDE_CODE=1 OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
  MAESTRO_BASE_URL="http://127.0.0.1:$PORT/v1" MAESTRO_MODEL=maestro \
  timeout 300 "$OC" run --agent maestro --format json \
  < "$SB/prompt.txt" > "$SB/run.jsonl" 2> "$SB/run.err" || echo "opencode exited $?" >&2
echo "captured $(wc -l < "$SB/captured.jsonl") requests -> $SB/captured.jsonl"
echo "$PORT" > "$SB/port"
