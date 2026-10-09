#!/usr/bin/env bash
# The STANDARD edition (not Law) on 127.0.0.1:8130 against a fresh throwaway data folder, for the
# T14 legacy smoke test (S1.12). Same host model servers, read-only; no Law gate. Never production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$ROOT/.law-data-legacy"
PORT="${LEGACY_BACKEND_PORT:-8130}"
rm -rf "$DATA"
umask 077
mkdir -p "$DATA"
unset ECHOMIND_EDITION
export ECHOMIND_DATA_DIR="$DATA"
export AUTH_ENABLED=1
AUTH_SECRET="$(openssl rand -hex 32)"; export AUTH_SECRET
export AUTH_ADMIN_USER=admin
AUTH_ADMIN_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=\n')"; export AUTH_ADMIN_PASSWORD
printf '%s' "$AUTH_ADMIN_PASSWORD" > "$DATA/dev_admin_password"
export CORS_ALLOW_ORIGINS="${LEGACY_CORS_ORIGINS:-http://localhost:5175}"
export LLM_BASE_URL="${LAW_LLM_BASE_URL:-http://localhost:8355/v1}"
export LLM_MODEL="${LAW_LLM_MODEL:-nvidia/Qwen3-30B-A3B-FP4}"
export OLLAMA_EMBED_URL="${LAW_EMBED_URL:-http://localhost:11435/api/embeddings}"
export DOCGEN_IMAGE_BACKEND=none
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
export ECHOMIND_EXIT_ON_FATAL_CUDA=0
echo "EchoMind STANDARD edition (legacy smoke): http://127.0.0.1:${PORT}  data=${DATA}"
cd "$ROOT/backend"
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
