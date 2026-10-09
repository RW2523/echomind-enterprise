#!/usr/bin/env bash
# Backend for the Law end-to-end tests: a FRESH data folder (.law-data-e2e) seeded with the
# fictional demo world, then the normal dev backend. Started by frontend/playwright.config.ts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAW_DATA_DIR="$ROOT/.law-data-e2e"
rm -rf "$LAW_DATA_DIR"
export LAW_SEED_DEMO=1
export LAW_CORS_ORIGINS="http://localhost:5174"
exec "$ROOT/scripts/law/dev_backend.sh"
