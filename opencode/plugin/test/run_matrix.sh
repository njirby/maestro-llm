#!/bin/bash
# Stock-vs-plugin conformance check for audio-read.ts against a capturing mock OpenAI server.
# Usage: opencode/plugin/test/run_matrix.sh [workdir]   (CPU only; needs ffmpeg, python3, curl)
# Each case runs `opencode run` twice (no plugin / plugin) in an isolated HOME; the mock answers the
# prompt "READ:<path>[,<path>...]" with parallel read tool calls and captures every request body.
# compare.py then checks: tool schemas identical (and `read` == docs/opencode_capture/captured.jsonl),
# non-audio cases byte-identical to stock, audio cases decode to the source bytes.
set -u
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$(mktemp -d)}; mkdir -p $W/cap $W/data/sub
OC=${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}; PORT=${PORT:-8898}; D=$W/data
cd $D
ffmpeg -loglevel error -y -f lavfi -i "sine=f=440:d=1.5" -ar 22050 -ac 2 tone.wav
ffmpeg -loglevel error -y -i tone.wav tone.flac; ffmpeg -loglevel error -y -i tone.wav tone.mp3
ffmpeg -loglevel error -y -i tone.wav -c:a libvorbis tone.ogg
ffmpeg -loglevel error -y -f lavfi -i "color=red:s=8x8" -frames:v 1 dot.png
printf 'hello\nworld\n' > notes.txt; echo x > sub/a.txt; head -c 100 /dev/urandom > blob.bin
python3 - "$W" "$PORT" "$T/../audio-read.ts" <<'PY'
import json, os, sys
W, port, plugin = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
c = {"model": "mock/qwen", "small_model": "mock/qwen", "autoupdate": False, "share": "disabled", "compaction": {"auto": False},
     "provider": {"mock": {"npm": "@ai-sdk/openai-compatible", "name": "Mock",
                           "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "dummy"},
                           "models": {"qwen": {"name": "Mock", "tool_call": True, "attachment": True,
                                               "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
                                               "limit": {"context": 32768, "output": 4096}}}}},
     "agent": {"lean": {"description": "lean", "mode": "all", "model": "mock/qwen", "prompt": "lean test",
                        "permission": {k: "deny" for k in ["edit", "glob", "grep", "write", "todowrite", "webfetch"]}}}}
for name, plug in [("cfg_stock", None), ("cfg_plugin", plugin)]:
    os.makedirs(f"{W}/{name}", exist_ok=True)
    json.dump({**c, **({"plugin": ["file://" + plug]} if plug else {})}, open(f"{W}/{name}/opencode.json", "w"), indent=1)
PY
declare -A C=(
 [txt]="READ:$D/notes.txt" [dir]="READ:$D/sub" [png]="READ:$D/dot.png" [missing_wav]="READ:$D/nope.wav"
 [missing_txt]="READ:$D/nope.txt" [bin]="READ:$D/blob.bin" [wav]="READ:$D/tone.wav" [rel_wav]="READ:tone.wav"
 [mp3]="READ:$D/tone.mp3" [flac]="READ:$D/tone.flac" [ogg]="READ:$D/tone.ogg"
 [par]="READ:$D/tone.wav,$D/dot.png,$D/tone.mp3,$D/notes.txt"
)
for cfg in cfg_stock cfg_plugin; do for k in "${!C[@]}"; do
  out=$W/cap/${cfg}__$k.jsonl; rm -f $out; mkdir -p $W/home_$cfg
  python3 $T/mock_openai.py $out $PORT & MP=$!
  for i in $(seq 50); do curl -s -o /dev/null http://127.0.0.1:$PORT/v1/models && break; sleep 0.1; done
  env -i PATH=/usr/bin:/bin HOME=$W/home_$cfg OPENCODE_CONFIG=$W/$cfg/opencode.json \
    timeout 90 $OC run --agent lean --format json "${C[$k]}" </dev/null > $W/cap/${cfg}__$k.out 2>&1
  kill $MP; wait $MP 2>/dev/null
done; done
python3 $T/compare.py $W
