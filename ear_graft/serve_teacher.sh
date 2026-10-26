#!/usr/bin/env bash
# Serve Qwen3-Omni-30B-A3B-Thinking (thinker only, text out) on all 4 cards for teacher data generation.
# Serving stays in a venv per box policy; recipe = the proven Omni-30B Instruct boot (vLLM 0.19, FLASH_ATTN, TP4).
# Sampling defaults come from the checkpoint's generation_config.json (T=0.6, top_p=0.95, top_k=20) via vLLM's
# --generation-config auto. Image/video limits must stay >= 1: at 0, vLLM 0.19 skips building the vision tower and leaves
# the thinker's deepstack buffers on the meta device, which crashes profile_run (2026-10-23). The client sends no sampling params. --reasoning-parser splits <think> into reasoning_content.
#   ear_graft/serve_teacher.sh            # foreground; log: data/ear_corpus/teacher_serve.log
set -euo pipefail
M=/home/nate/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Thinking/snapshots/2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b
PORT=${PORT:-8100}
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=FLASH_ATTN HF_HUB_OFFLINE=1 \
  /home/nate/Documents/maestro-llm/.venv/bin/vllm serve "$M" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name omni30b-thinking \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --enable-chunked-prefill --enable-prefix-caching \
  --generation-config auto \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"audio": 2, "image": 1, "video": 1}' \
  --trust-remote-code
