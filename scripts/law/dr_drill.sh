#!/usr/bin/env bash
# T13 disaster-recovery drill on a real EchoMind Law data folder (S1.12; OPS-01).
#
#   scripts/law/dr_drill.sh [DATA_DIR]        # default: .law-data-e2e (left by the browser suite)
#
# Takes an encrypted backup with a throwaway backup key, restores it into a new folder, and checks
# the restored copy: the audit chain and its anchors, the schema, the record counts, and that every
# stored document version still opens with the master key and matches its recorded SHA-256.
# Only works on a native development folder; it never touches Docker volumes or production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$(cd "${1:-$ROOT/.law-data-e2e}" && pwd)"
case "$SRC" in "$ROOT"/.law-data*) ;; *) echo "refusing: $SRC is not a .law-data* folder of this clone" >&2; exit 2;; esac
[[ -s "$SRC/echomind.sqlite" ]] || { echo "no database in $SRC" >&2; exit 2; }

umask 077
WORK="$(mktemp -d -t emlb-drill-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
export ECHOMIND_EDITION=law
export LEGAL_MASTER_KEY_FILE="${LEGAL_MASTER_KEY_FILE:-$ROOT/.law-secrets/master.key}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
cd "$ROOT/backend"

step() { printf '\n== %s\n' "$*"; }
step "backup key (throwaway, mode 600)"
python -m app.legal.backup keygen --out "$WORK/backup.key"
step "encrypted backup of $SRC"
ECHOMIND_DATA_DIR="$SRC" python -m app.legal.backup create --out "$WORK/backups/law.emlb" --key-file "$WORK/backup.key"
step "verify the archive (authenticates every chunk)"
python -m app.legal.backup verify --archive "$WORK/backups/law.emlb" --key-file "$WORK/backup.key" >/dev/null && echo "archive authenticates"
step "restore into a new folder"
python -m app.legal.backup restore --archive "$WORK/backups/law.emlb" --key-file "$WORK/backup.key" --target "$WORK/restored"
step "audit chain and anchors (restored copy)"
ECHOMIND_DATA_DIR="$WORK/restored" python -m app.legal.audit verify
step "schema (restored copy)"
ECHOMIND_DATA_DIR="$WORK/restored" python -m app.legal.migrations status
step "every document version opens with the master key and matches its SHA-256"
ECHOMIND_DATA_DIR="$WORK/restored" python - <<'PY'
import hashlib, sqlite3, sys
from app.core.config import settings
from app.legal.documents import blobs
with sqlite3.connect(settings.DB_PATH) as c:
    rows = c.execute("SELECT v.id, v.matter_id, v.wrapped_key, v.sha256 FROM legal_document_versions v "
                     "JOIN legal_documents d ON d.id = v.document_id WHERE d.deleted_at IS NULL").fetchall()
bad = [vid for vid, mid, wk, sha in rows if hashlib.sha256(blobs.load(mid, vid, wk)).hexdigest() != sha]
print(f"{len(rows) - len(bad)} of {len(rows)} versions open and match")
sys.exit(1 if bad else 0)
PY
step "drill passed"
