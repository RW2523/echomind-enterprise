#!/bin/sh
# Start Ollama, wait for API, pre-pull models so runtime doesn't load.
# Models: LLM (chat) + embed (RAG). Backend/voice depend on these.

set -e

OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-qwen2.5:7b-instruct}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

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
wait $OLLAMA_PID
