#!/usr/bin/env bash
# Backend for the Law end-to-end tests: a FRESH data folder (.law-data-e2e) seeded with the
# fictional demo world, then the normal dev backend. Started by frontend/playwright.config.ts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAW_DATA_DIR="$ROOT/.law-data-e2e"
rm -rf "$LAW_DATA_DIR"
export LAW_SEED_DEMO=1
export LAW_DEBUG_STACKS=1           # kill -USR1 <backend pid> dumps all thread stacks into the test log
# S1.11: a fictional document-system folder for the connector journey (fresh every run).
DMS="$ROOT/.law-dms-e2e"
rm -rf "$DMS"
mkdir -p "$DMS/Beta/Lease"
printf 'Notice to quit (fictional demo). The landlord gives notice under clause 7 of the warehouse lease.\n' > "$DMS/Beta/Lease/lease-notice.txt"
printf '{"users": ["bob"]}\n' > "$DMS/Beta/Lease/lease-notice.txt.acl.json"
printf 'A draft with no permissions file; it must not be imported.\n' > "$DMS/Beta/Lease/draft.txt"
export LEGAL_CONNECTOR_LOCAL_ROOTS="$DMS"
export LAW_CORS_ORIGINS="http://localhost:5174"
exec "$ROOT/scripts/law/dev_backend.sh"
