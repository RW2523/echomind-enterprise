#!/bin/bash
# SGLang embedding server (OpenAI-compatible /v1/embeddings API)
# Uses GPU with small mem fraction to coexist with LLM on single-GPU systems.
# On multi-GPU: sglang-embed can use a dedicated GPU.

set -e

MODEL="${SGLANG_EMBED_MODEL:-Alibaba-NLP/gte-Qwen2-1.5B-instruct}"
PORT="${SGLANG_EMBED_PORT:-30001}"

echo "[sglang-embed] Starting SGLang embedding server: $MODEL on port $PORT"
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --is-embedding \
  --mem-fraction-static 0.35 \
  --trust-remote-code \
  --log-level info
