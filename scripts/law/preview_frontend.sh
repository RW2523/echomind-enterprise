#!/usr/bin/env bash
# The EchoMind Law frontend as it ships: a production build (into dist-e2e, never the dist a demo serves) served by
# `vite preview` on :5174, proxying /api to the test backend. Used by frontend/playwright.config.ts (2026-11-12); the dev
# server (scripts/law/dev_frontend.sh) stays for development and E2E_DEV=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/frontend"
[[ -d node_modules ]] || npm ci
export ECHOMIND_DEV_PORT="${LAW_FRONTEND_PORT:-5174}"
export ECHOMIND_DEV_HOST="${LAW_FRONTEND_HOST:-127.0.0.1}"
export ECHOMIND_DEV_API_TARGET="${LAW_API_TARGET:-http://127.0.0.1:${LAW_BACKEND_PORT:-8110}}"
npx vite build --outDir dist-e2e --emptyOutDir >/dev/null
exec npx vite preview --outDir dist-e2e
