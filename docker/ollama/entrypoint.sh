#!/bin/sh
# Start Ollama and ensure required models are available.
# OFFLINE MODE: When OLLAMA_OFFLINE=1, do NOT pull models; fail clearly if missing.
# PREPARE MODE: When OLLAMA_OFFLINE=0 (or unset), pull models if missing (one-time online prep).

set -e

OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
OLLAMA_OFFLINE="${OLLAMA_OFFLINE:-1}"

echo "[ollama] Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

echo "[ollama] Waiting for API to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

_has_model() {
  ollama list 2>/dev/null | grep -qF "$1"
}

_ensure_models() {
  local missing=""
  _has_model "$OLLAMA_LLM_MODEL" || missing="${missing}${missing:+ }$OLLAMA_LLM_MODEL"
  _has_model "$OLLAMA_EMBED_MODEL" || missing="${missing}${missing:+ }$OLLAMA_EMBED_MODEL"

  if [ -n "$missing" ]; then
    if [ "$OLLAMA_OFFLINE" = "1" ]; then
      echo "[ollama] ERROR: Offline mode (OLLAMA_OFFLINE=1) but required models are missing: $missing"
      echo "[ollama] Run one-time preparation with: OLLAMA_OFFLINE=0 docker compose up -d ollama"
      echo "[ollama] Wait for health, then stop. After that, start with docker compose up -d"
      kill $OLLAMA_PID 2>/dev/null || true
      exit 1
    fi
    echo "[ollama] Pre-pulling missing models (one-time, requires internet)..."
    _has_model "$OLLAMA_LLM_MODEL" || { echo "[ollama] Pulling LLM: $OLLAMA_LLM_MODEL"; ollama pull "$OLLAMA_LLM_MODEL"; }
    _has_model "$OLLAMA_EMBED_MODEL" || { echo "[ollama] Pulling embed: $OLLAMA_EMBED_MODEL"; ollama pull "$OLLAMA_EMBED_MODEL"; }
  else
    echo "[ollama] Required models already present (no network pull)."
  fi
}

_ensure_models

echo "[ollama] Warming LLM model (load into memory)..."
curl -s -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$OLLAMA_LLM_MODEL\",\"messages\":[]}" >/dev/null || true

echo "[ollama] Warming embed model (load into memory)..."
curl -s -X POST http://127.0.0.1:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$OLLAMA_EMBED_MODEL\",\"prompt\":\".\"}" >/dev/null || true

echo "[ollama] Models ready and warmed. Keeping Ollama running."
wait $OLLAMA_PID
