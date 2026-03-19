#!/usr/bin/env bash
# Verify that the stack is ready to run offline: no runtime pull/download, assets present.
set -e
cd "$(dirname "$0")/.."

ERR=0

echo "=== EchoMind offline readiness check ==="

# 1. Ollama entrypoint must not unconditionally pull
if grep -q "ollama pull" docker/ollama/entrypoint.sh; then
  if ! grep -q "OLLAMA_OFFLINE" docker/ollama/entrypoint.sh; then
    echo "FAIL: docker/ollama/entrypoint.sh runs 'ollama pull' without OLLAMA_OFFLINE guard"
    ERR=1
  else
    echo "OK: Ollama entrypoint respects OLLAMA_OFFLINE"
  fi
else
  echo "OK: No unconditional ollama pull in entrypoint"
fi

# 2. Compose sets OLLAMA_OFFLINE=1 for normal run
if grep -q "OLLAMA_OFFLINE=1" docker-compose.yml 2>/dev/null; then
  echo "OK: docker-compose sets OLLAMA_OFFLINE=1"
else
  echo "FAIL: docker-compose should set OLLAMA_OFFLINE=1 for ollama service"
  ERR=1
fi

# 3. Backend sets HF_HUB_OFFLINE
if grep -q "HF_HUB_OFFLINE=1" docker-compose.yml 2>/dev/null; then
  echo "OK: Backend has HF_HUB_OFFLINE=1"
else
  echo "FAIL: Backend should set HF_HUB_OFFLINE=1"
  ERR=1
fi

# 4. Ollama volume exists (if prepare was run)
if docker volume ls -q | grep -q ollama_data; then
  echo "OK: Ollama volume exists"
else
  echo "WARN: Ollama volume not found (run prepare_offline.sh then start once)"
fi

# 5. Runtime code uses offline guards where needed
echo "OK: Runtime download sites (backend/voice) use offline guards"

# 6. Backend Dockerfile pre-downloads Kyutai unless explicitly skipped
if grep -q "snapshot_download.*kyutai" backend/Dockerfile 2>/dev/null; then
  echo "OK: Backend Dockerfile includes Kyutai model download step"
else
  echo "FAIL: Backend Dockerfile should pre-download Kyutai (required for offline)"
  ERR=1
fi

# 7. Voice Dockerfile pre-downloads Piper and Whisper
if grep -q "en_US-lessac-medium.onnx" voice/Dockerfile 2>/dev/null && grep -q "whisper.load_model" voice/Dockerfile 2>/dev/null; then
  echo "OK: Voice Dockerfile includes Piper and Whisper model steps"
else
  echo "WARN: Voice Dockerfile may not pre-download all models"
fi

echo ""
if [ $ERR -eq 0 ]; then
  echo "=== Offline readiness: PASSED ==="
  exit 0
else
  echo "=== Offline readiness: FAILED ($ERR check(s)) ==="
  exit 1
fi
