#!/bin/bash
# Tracer-bullet training: memorize the 8-rollout pilot corpus with Qwen2.5-Omni-3B.
#
# Purpose is NOT capability — it is a pipeline proof: if loss goes to ~0 on the
# regenerated corpus, then the records load, the audio pipeline works in
# training, the context length is handled, and the contract survives
# tokenization. The trained adapter then goes into the harness replay test.
#
# Decisions encoded here:
#   - LoRA (not full finetune): 43 records need far less capacity than a
#     high-rank adapter provides, and it trains faster on 4x3090.
#   - Audio tower + aligner trainable: arm-2 showed the audio path matters,
#     and this corpus is audio-conditioned throughout.
#   - NO train/val split: every record is training data; we want memorization.
#   - max-length is derived from the DATA (see below), never clamped to the
#     model card's 32k max_position_embeddings — RoPE positions are computed,
#     and training at the longer length is what teaches the model to use it.
#
# Usage: scripts/train_pilot_3b.sh <pilot_dir> [max_length]
set -euo pipefail
cd /home/nate/Documents/maestro-llm

PILOT_DIR="${1:-outputs/pilot_v4}"
MODEL_SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
TRAIN_JSONL="${PILOT_DIR}/train_all.jsonl"

# 1. Pool every record kind into one training file (no split).
cat "${PILOT_DIR}"/main_*.jsonl "${PILOT_DIR}"/search_*.jsonl \
    "${PILOT_DIR}"/judge_*.jsonl "${PILOT_DIR}"/transcription_*.jsonl \
    > "${TRAIN_JSONL}"
echo "pooled $(wc -l < "${TRAIN_JSONL}") records -> ${TRAIN_JSONL}"

# 2. Size the context from the data unless overridden.
if [ $# -ge 2 ]; then
  MAXLEN="$2"
else
  MAXLEN=$(.venv/bin/python - "$TRAIN_JSONL" <<'PY'
import json, os, sys, math
import soundfile as sf
from transformers import AutoTokenizer
snap = "/home/nate/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots"
snap = os.path.join(snap, os.listdir(snap)[0])
tok = AutoTokenizer.from_pretrained(snap)
worst = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    text = "".join(str(m.get("content", "")) for m in r["messages"])
    n = len(tok(text)["input_ids"]) + 8 * len(r["messages"])
    for p in r.get("audios", []):
        try:
            info = sf.info(p)
            n += int(info.frames / info.samplerate * 25)
        except Exception:
            n += 250
    worst = max(worst, n)
# round up to the next 8k boundary, with 15% headroom for template overhead
target = int(worst * 1.15)
print(max(32768, int(math.ceil(target / 8192) * 8192)))
PY
)
fi
echo "max-length = ${MAXLEN} (longest record drives this; 3B card says 32768 — deliberately exceeded)"

.venv/bin/python scripts/train_qwen25_omni_lora_megatron.py \
  --model "${MODEL_SNAP}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${TRAIN_JSONL}" \
  --output-root "${PILOT_DIR}/training" \
  --run-name pilot3b_memorize \
  --max-length "${MAXLEN}" \
  --tuner-type lora --lora-rank 128 --lora-alpha 256 \
  --lr 1e-4 --min-lr 1e-5 \
  --micro-batch-size 1 --global-batch-size 4 \
  --tensor-model-parallel-size 4 --packing false --lazy_tokenize true \
  --dataset-num-proc 8 \
  --freeze-vit false --freeze-aligner false \
  --num-train-epochs 20 \
  --save-steps 50 --eval-steps 25 --eval-iters 2 \
  --save-total-limit 6 --save-optim --save-rng \
  --report-to wandb --wandb-project maestro-sft
