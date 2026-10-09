#!/usr/bin/env bash
# Native EchoMind Law development backend (decision D-002).
# Host Python, CPU only, no live ASR (NeMo is not installed on the host), bound to 127.0.0.1.
# Reuses the host-published LLM (:8355) and embedding (:11435) servers read-only.
# Data lives in .law-data/ (gitignored) and must only ever hold fictional fixtures.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${LAW_DATA_DIR:-$ROOT/.law-data}"
PORT="${LAW_BACKEND_PORT:-8110}"

umask 077
mkdir -p "$DATA"
[[ -s "$DATA/dev_auth_secret" ]] || openssl rand -hex 32 > "$DATA/dev_auth_secret"
[[ -s "$DATA/dev_admin_password" ]] || openssl rand -base64 24 | tr -d '/+=\n' > "$DATA/dev_admin_password"
# D-014: the document master key lives outside the data folder.
SECRETS="${LAW_SECRETS_DIR:-$ROOT/.law-secrets}"
mkdir -p "$SECRETS"
[[ -s "$SECRETS/master.key" ]] || openssl rand -base64 32 > "$SECRETS/master.key"
chmod 600 "$SECRETS/master.key"

python3 - <<'PY' || { echo "reranker model missing from the HF cache. Fetch it once (online):
  python3 -c \"from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\"" >&2; exit 4; }
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
PY

export ECHOMIND_EDITION=law
export ECHOMIND_DATA_DIR="$DATA"
export LEGAL_MASTER_KEY_FILE="$SECRETS/master.key"
export AUTH_ENABLED=1
AUTH_SECRET="$(cat "$DATA/dev_auth_secret")"; export AUTH_SECRET
export AUTH_ADMIN_USER="${LAW_ADMIN_USER:-admin}"
AUTH_ADMIN_PASSWORD="$(cat "$DATA/dev_admin_password")"; export AUTH_ADMIN_PASSWORD
export CORS_ALLOW_ORIGINS="${LAW_CORS_ORIGINS:-http://localhost:5174}"
export LLM_BASE_URL="${LAW_LLM_BASE_URL:-http://localhost:8355/v1}"
export LLM_MODEL="${LAW_LLM_MODEL:-nvidia/Qwen3-30B-A3B-FP4}"
export OLLAMA_EMBED_URL="${LAW_EMBED_URL:-http://localhost:11435/api/embeddings}"
export DOCGEN_IMAGE_BACKEND=none
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
export ECHOMIND_EXIT_ON_FATAL_CUDA=0

if [[ "${LAW_SEED_DEMO:-0}" == "1" ]]; then
  # Fictional demo world for end-to-end tests and design reviews (refuses without the explicit flag).
  (cd "$ROOT/backend" && LEGAL_ALLOW_DEMO_SEED=1 python3 -m app.legal.cli seed-demo)
fi

echo "EchoMind Law dev backend: http://127.0.0.1:${PORT}  data=${DATA}"
echo "  sign in as ${AUTH_ADMIN_USER}; password in ${DATA}/dev_admin_password"
cd "$ROOT/backend"
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" ${LAW_RELOAD:+--reload}
