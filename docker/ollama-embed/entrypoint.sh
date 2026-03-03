#!/bin/sh
# Ollama embed-only: nomic-embed-text for RAG. Runs on CPU, no GPU needed.
# SGLang handles LLM; Ollama handles embeddings only.

set -e

OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

echo "[ollama-embed] Starting Ollama server (embed-only)..."
ollama serve &
OLLAMA_PID=$!

echo "[ollama-embed] Waiting for API..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "[ollama-embed] Pulling embed model: $OLLAMA_EMBED_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"

echo "[ollama-embed] Warming embed model..."
if command -v curl >/dev/null 2>&1; then
  curl -s -X POST http://127.0.0.1:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$OLLAMA_EMBED_MODEL\",\"prompt\":\".\"}" >/dev/null || true
fi

echo "[ollama-embed] Ready. Embeddings on port 11434."
wait $OLLAMA_PID
