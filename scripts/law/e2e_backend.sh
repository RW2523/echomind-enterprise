#!/usr/bin/env bash
# Backend for the Law end-to-end tests: a FRESH data folder (.law-data-e2e) seeded with the
# fictional demo world, then the normal dev backend. Started by frontend/playwright.config.ts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAW_DATA_DIR="$ROOT/.law-data-e2e"
rm -rf "$LAW_DATA_DIR"
export LAW_SEED_DEMO=1
export LAW_IMPORT_EVALUATIONS=1      # S2.8: the newest gate run, for the Evaluations journey
export LAW_DEBUG_STACKS=1           # kill -USR1 <backend pid> dumps all thread stacks into the test log
# S1.11: a fictional document-system folder for the connector journey (fresh every run).
DMS="$ROOT/.law-dms-e2e"
rm -rf "$DMS"
mkdir -p "$DMS/Beta/Lease"
printf 'Notice to quit (fictional demo). The landlord gives notice under clause 7 of the warehouse lease.\n' > "$DMS/Beta/Lease/lease-notice.txt"
printf '{"users": ["bob"]}\n' > "$DMS/Beta/Lease/lease-notice.txt.acl.json"
printf 'A draft with no permissions file; it must not be imported.\n' > "$DMS/Beta/Lease/draft.txt"
export LEGAL_CONNECTOR_LOCAL_ROOTS="$DMS"
# S2.1: fictional files for the reading journey — a scanned notice (no text layer) and a Word MSA with
# numbering, tracked changes, a comment and a hidden instruction — from the backend test fixtures.
FILES="$ROOT/.law-e2e-files"
rm -rf "$FILES"; mkdir -p "$FILES"
(cd "$ROOT/backend/tests/law" && python3 -c "
from fixtures_s21 import contract_docx, scanned_pdf
open('$FILES/termination-notice.pdf', 'wb').write(scanned_pdf())
open('$FILES/harrowgate-msa.docx', 'wb').write(contract_docx())")
# S4.1: intake and enquirer links are off by default; the intake journey turns them on here.
export LEGAL_INTAKE=1 LEGAL_INTAKE_LINKS=1
export LEGAL_SUMMARIES=1             # S4.2: the summaries journey
export LEGAL_TIME_ENTRIES=1          # S4.3: the time-entries journey
export LEGAL_PANEL=1                 # S4.5: the review-panel journey
export LAW_CORS_ORIGINS="http://localhost:5174"
exec "$ROOT/scripts/law/dev_backend.sh"
