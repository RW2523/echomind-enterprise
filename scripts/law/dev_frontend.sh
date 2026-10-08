#!/usr/bin/env bash
# Native EchoMind Law development frontend (Vite) on :5174, proxying /api to the dev backend.
# Open http://localhost:5174/?vertical=law
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/frontend"
[[ -d node_modules ]] || npm ci
export ECHOMIND_DEV_PORT="${LAW_FRONTEND_PORT:-5174}"
export ECHOMIND_DEV_HOST="${LAW_FRONTEND_HOST:-127.0.0.1}"
export ECHOMIND_DEV_API_TARGET="${LAW_API_TARGET:-http://127.0.0.1:${LAW_BACKEND_PORT:-8110}}"
exec npx vite
