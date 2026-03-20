#!/usr/bin/env bash
# Export a complete offline bundle: images + Ollama volume + manifest.
# Run on a machine that has already run prepare_offline.sh and has the stack built.
# Usage: ./scripts/export_offline_bundle.sh [OUTPUT_DIR]
# Default OUTPUT_DIR: ./offline-bundle-YYYYMMDD
set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-./offline-bundle-$(date +%Y%m%d)}"
mkdir -p "$OUTPUT_DIR"
BUNDLE_ROOT="$(cd "$OUTPUT_DIR" && pwd)"

echo "=== Exporting EchoMind offline bundle to $BUNDLE_ROOT ==="

# Compose project name (directory name by default); volume is ${project}_ollama_data
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$(basename "$(pwd)" | tr -cd 'a-zA-Z0-9_-')}"
OLLAMA_VOLUME="${PROJECT_NAME}_ollama_data"
if ! docker volume inspect "$OLLAMA_VOLUME" &>/dev/null; then
  OLLAMA_VOLUME="ollama_data"
fi
if ! docker volume inspect "$OLLAMA_VOLUME" &>/dev/null; then
  echo "WARN: Ollama volume not found (run prepare_offline.sh first); skipping ollama_data.tar"
  OLLAMA_VOLUME=""
fi

TRTLLM_VOLUME="${PROJECT_NAME}_trtllm_hf_cache"
if ! docker volume inspect "$TRTLLM_VOLUME" &>/dev/null; then
  TRTLLM_VOLUME="trtllm_hf_cache"
fi
if ! docker volume inspect "$TRTLLM_VOLUME" &>/dev/null; then
  echo "WARN: TensorRT-LLM HF cache volume not found; skipping trtllm_hf_cache.tar"
  TRTLLM_VOLUME=""
fi

echo "[1/4] Saving Docker images..."
# Image names: compose uses project_service (e.g. echomind-enterprise-backend); ollama/trtllm use fixed image: tags
for pair in "backend:${PROJECT_NAME}-backend" "voice:${PROJECT_NAME}-voice" "frontend:${PROJECT_NAME}-frontend" "ollama:echomind-ollama:setup" "trtllm:echomind-trtllm:1.2.0rc6"; do
  svc="${pair%%:*}"
  name="${pair#*:}"
  if docker image inspect "$name" &>/dev/null 2>&1; then
    docker save -o "$BUNDLE_ROOT/image-${svc}.tar" "$name"
    echo "  saved image-${svc}.tar ($name)"
  else
    echo "  WARN: image $name not found (run docker compose build first)"
  fi
done

echo "[2/4] Exporting Ollama volume..."
if [ -n "$OLLAMA_VOLUME" ]; then
  docker run --rm -v "$OLLAMA_VOLUME:/data" -v "$BUNDLE_ROOT:/out" alpine tar cf /out/ollama_data.tar -C /data .
  echo "  saved ollama_data.tar"
else
  echo "  skipped (no volume)"
fi

echo "[2b/4] Exporting TensorRT-LLM Hugging Face cache (large)..."
if [ -n "$TRTLLM_VOLUME" ]; then
  docker run --rm -v "$TRTLLM_VOLUME:/data" -v "$BUNDLE_ROOT:/out" alpine tar cf /out/trtllm_hf_cache.tar -C /data .
  echo "  saved trtllm_hf_cache.tar"
else
  echo "  skipped (no volume)"
fi

echo "[3/4] Copying voice assets and docs..."
mkdir -p "$BUNDLE_ROOT/voice-assets"
[ -d "voice/voices" ] && [ "$(ls -A voice/voices 2>/dev/null)" ] && cp -a voice/voices/. "$BUNDLE_ROOT/voice-assets/" && echo "  copied voice/voices"
[ -f OFFLINE_DEPLOYMENT.md ] && cp OFFLINE_DEPLOYMENT.md "$BUNDLE_ROOT/" && echo "  copied OFFLINE_DEPLOYMENT.md"

echo "[4/4] Writing manifest..."
cat > "$BUNDLE_ROOT/MANIFEST.txt" << MANI
EchoMind offline bundle
Exported: $(date -Iseconds)
Contents:
- image-backend.tar, image-voice.tar, image-frontend.tar, image-ollama.tar, image-trtllm.tar: Docker images
- ollama_data.tar: Ollama model store (embed model when using TensorRT-LLM for chat)
- trtllm_hf_cache.tar: TensorRT-LLM Hugging Face cache (weights + engines; large)
- voice-assets/: Piper voice files (optional; images include defaults)
- OFFLINE_DEPLOYMENT.md: Deployment instructions

Import on air-gapped machine:
  ./scripts/import_offline_bundle.sh $BUNDLE_ROOT
  cd <project> && docker compose up -d
MANI

echo ""
echo "=== Export complete ==="
ls -la "$BUNDLE_ROOT"
