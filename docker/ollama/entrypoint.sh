#!/bin/sh
# Start Ollama, wait for API. Use pre-pulled models from image if volume is empty.

set -e

OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-qwen2.5:7b-instruct}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

# Use volume for persistence at runtime (override build-time OLLAMA_MODELS).
export OLLAMA_MODELS=/root/.ollama

# On first run, copy pre-pulled models from image into volume so no pull at runtime.
if [ -d /opt/ollama-models ] && [ -z "$(ls -A /root/.ollama 2>/dev/null)" ]; then
  echo "[ollama-setup] Copying pre-pulled models from image to volume..."
  cp -a /opt/ollama-models/. /root/.ollama/
fi

echo "[ollama-setup] Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

echo "[ollama-setup] Waiting for API to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "[ollama-setup] Pre-pulling LLM model: $OLLAMA_LLM_MODEL"
ollama pull "$OLLAMA_LLM_MODEL"

echo "[ollama-setup] Pre-pulling embed model: $OLLAMA_EMBED_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"

echo "[ollama-setup] Models ready. Keeping Ollama running."
echo "[ollama-setup] Startup check (ollama ps) - loaded models / GPU usage:"
ollama ps || true

wait $OLLAMA_PID
