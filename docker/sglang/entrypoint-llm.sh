#!/bin/bash
# SGLang LLM server for chat/completions (OpenAI-compatible API)
# Uses GPU for inference on DGX Spark

set -e

MODEL="${SGLANG_LLM_MODEL:-nvidia/Qwen3-8B-FP8}"
PORT="${SGLANG_LLM_PORT:-30000}"
TP="${SGLANG_TP:-1}"

echo "[sglang-llm] Starting SGLang LLM server: $MODEL on port $PORT"
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --log-level info
