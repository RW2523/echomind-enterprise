#!/usr/bin/env bash
# Import an offline bundle on an air-gapped machine and prepare for docker compose up.
# Usage: ./scripts/import_offline_bundle.sh BUNDLE_DIR [PROJECT_DIR]
#   BUNDLE_DIR: path to the exported bundle (e.g. offline-bundle-YYYYMMDD)
#   PROJECT_DIR: path to the EchoMind project (default: current directory)
# After import, run: cd PROJECT_DIR && docker compose up -d
set -e

BUNDLE_DIR="${1:?Usage: $0 BUNDLE_DIR [PROJECT_DIR]}"
PROJECT_DIR="${2:-.}"
BUNDLE_ROOT="$(cd "$BUNDLE_DIR" && pwd)"
PROJECT_ROOT="$(cd "$PROJECT_DIR" && pwd)"

echo "=== Importing EchoMind offline bundle from $BUNDLE_ROOT ==="

echo "[1/4] Loading Docker images..."
for f in "$BUNDLE_ROOT"/image-backend.tar "$BUNDLE_ROOT"/image-voice.tar "$BUNDLE_ROOT"/image-frontend.tar "$BUNDLE_ROOT"/image-ollama.tar "$BUNDLE_ROOT"/image-trtllm.tar; do
  [ -f "$f" ] || continue
  docker load -i "$f"
  echo "  loaded $(basename "$f")"
done

echo "[2/4] Creating Ollama volume and restoring data..."
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$(basename "$(cd "$PROJECT_ROOT" && pwd)" | tr -cd 'a-zA-Z0-9_-')}"
OLLAMA_VOLUME="${PROJECT_NAME}_ollama_data"
docker volume create "$OLLAMA_VOLUME" 2>/dev/null || true
if [ -f "$BUNDLE_ROOT/ollama_data.tar" ]; then
  docker run --rm -v "$OLLAMA_VOLUME:/data" -v "$BUNDLE_ROOT:/in" alpine sh -c "cd /data && tar xf /in/ollama_data.tar"
  echo "  restored ollama_data volume"
else
  echo "  WARN: ollama_data.tar not in bundle (Ollama will fail until models are present)"
fi

echo "[2b/4] TensorRT-LLM HF cache volume..."
TRTLLM_VOLUME="${PROJECT_NAME}_trtllm_hf_cache"
docker volume create "$TRTLLM_VOLUME" 2>/dev/null || true
if [ -f "$BUNDLE_ROOT/trtllm_hf_cache.tar" ]; then
  docker run --rm -v "$TRTLLM_VOLUME:/data" -v "$BUNDLE_ROOT:/in" alpine sh -c "cd /data && tar xf /in/trtllm_hf_cache.tar"
  echo "  restored trtllm_hf_cache volume"
else
  echo "  WARN: trtllm_hf_cache.tar missing (first online start will download/build; or export after warming trtllm)"
fi

echo "[3/4] Optional: copy voice assets into project..."
if [ -d "$BUNDLE_ROOT/voice-assets" ] && [ "$(ls -A "$BUNDLE_ROOT/voice-assets" 2>/dev/null)" ]; then
  mkdir -p "$PROJECT_ROOT/voice/voices"
  cp -a "$BUNDLE_ROOT/voice-assets"/. "$PROJECT_ROOT/voice/voices/"
  echo "  copied voice-assets to voice/voices"
else
  echo "  (voice-assets empty or missing; image has default Piper voices)"
fi

echo "[4/4] Verifying..."
if docker volume inspect "$OLLAMA_VOLUME" &>/dev/null; then
  echo "  Ollama volume: $OLLAMA_VOLUME"
fi
if docker volume inspect "$TRTLLM_VOLUME" &>/dev/null; then
  echo "  TensorRT-LLM HF cache volume: $TRTLLM_VOLUME"
fi

echo ""
echo "=== Import complete ==="
echo "Start the stack (no internet required):"
echo "  cd $PROJECT_ROOT && docker compose up -d"
echo ""
echo "See OFFLINE_DEPLOYMENT.md for troubleshooting."
