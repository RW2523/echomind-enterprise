#!/usr/bin/env bash
# One-time online preparation for fully offline operation.
# Run this once with internet; after that, docker compose up runs without network.
#
# 1. Builds all images (base images, apt, pip, npm, Kyutai, Whisper, Piper).
# 2. Starts Ollama with OLLAMA_OFFLINE=0 to pull LLM + embed models into the volume.
# 3. Stops Ollama so the volume persists.
# 4. Normal runs use OLLAMA_OFFLINE=1 and never pull.
set -e
cd "$(dirname "$0")/.."

echo "=== EchoMind offline preparation (requires internet) ==="

echo "[1/4] Building all images (backend first to avoid BuildKit pipe bug)..."
docker compose build backend
BUILDX_METADATA_PROVENANCE=disabled docker compose build

echo "[2/4] Populating Ollama volume (one-time pull)..."
docker compose -f docker-compose.yml -f docker-compose.prepare.yml up -d ollama

echo "[3/4] Waiting for Ollama to have both models (up to ~5 min)..."
max_wait=300
elapsed=0
while [ $elapsed -lt $max_wait ]; do
  if docker compose exec -T ollama sh -c "ollama list 2>/dev/null | grep -q qwen2.5:7b-instruct-q4_K_M && ollama list 2>/dev/null | grep -q nomic-embed-text"; then
    echo "Ollama models ready."
    break
  fi
  sleep 10
  elapsed=$((elapsed + 10))
  echo "  ... waiting (${elapsed}s)"
done
if [ $elapsed -ge $max_wait ]; then
  echo "ERROR: Ollama did not become ready in time. Check logs: docker compose logs ollama"
  exit 1
fi

echo "[4/4] Stopping Ollama (volume persists with models)..."
docker compose stop ollama

echo ""
echo "=== Preparation complete ==="
echo "Start the stack (fully offline):"
echo "  docker compose up -d"
echo ""
echo "To verify offline readiness: ./scripts/verify_offline_readiness.sh"
