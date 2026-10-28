#!/usr/bin/env bash
# Inside the public demo's backend container (scripts/law/demo_up.sh): the Law backend on the demo's fictional data, with
# one-click entry as alice. The code is mounted read-only; /data is the demo's own data folder (made by seed-demo, with its
# DEMO_INSTANCE marker); the master key is the demo's, which only ever protects fictional data. No secret is in this file.
set -euo pipefail
export HOME=/tmp/home NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl HF_HOME=/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
export ECHOMIND_EDITION=law ECHOMIND_DATA_DIR=/data LEGAL_MASTER_KEY_FILE=/run/secrets/master.key AUTH_ENABLED=1
AUTH_SECRET="$(cat /data/dev_auth_secret)"; export AUTH_SECRET
export AUTH_ADMIN_USER=admin
AUTH_ADMIN_PASSWORD="$(cat /data/dev_admin_password)"; export AUTH_ADMIN_PASSWORD
# The host's model servers, reached through the bridge gateway (never production's containers by name).
export CORS_ALLOW_ORIGINS="$PUBLIC_URL" LLM_BASE_URL=http://host.docker.internal:8355/v1 LLM_MODEL="${DEMO_LLM_MODEL:-nvidia/Qwen3-30B-A3B-FP4}"
export OLLAMA_EMBED_URL=http://host.docker.internal:11435/api/embeddings DOCGEN_IMAGE_BACKEND=none ECHOMIND_EXIT_ON_FATAL_CUDA=0
# Spoken replies use the host's speech service; demo answers are fictional, but that service is not the Law edition and
# its logs are not redacted: leave VOICE_TTS_URL empty to turn spoken replies off.
export VOICE_TTS_URL="${DEMO_TTS_URL-http://host.docker.internal:8002/speak}"
export ECHOMIND_ASR_DEVICE=cpu CUDA_VISIBLE_DEVICES=""          # the GPU is held by the host's models: speech on the CPU
export LEGAL_PROFILE=demo LEGAL_DEMO_SIGNIN=1 LEGAL_DEMO_SIGNIN_FROM=anywhere LEGAL_DEMO_AUTO="${DEMO_AUTO:-alice}"
export LEGAL_TRUSTED_PROXIES="${DEMO_TRUSTED_PROXY:?}"
export LEGAL_PANEL=1 LEGAL_CONVERSATION=1 LEGAL_SUMMARIES=1 LEGAL_TIME_ENTRIES=1 LEGAL_TRANSLATION=1 LEGAL_REGULATORY=1 LEGAL_INTAKE=1
cd /repo/backend
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8110
