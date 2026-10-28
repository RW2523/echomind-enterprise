#!/usr/bin/env bash
# Restore the EchoMind Law pilot from a backup archive (commercial-readiness review R10, 2026-10-28):
#   scripts/law/restore_pilot.sh ARCHIVE            # says what it would do, changes nothing
#   scripts/law/restore_pilot.sh ARCHIVE --yes      # does it
# 1. checks the archive with the backup key (LAW_BACKUP_KEY_FILE, default ~/.echomind-law/backup.key) before anything stops;
# 2. stops the Law stack and copies the current data volume, as it is, into echomind-law_prerestore_<time> (kept, so the
#    restore itself can be undone);
# 3. empties the data volume and restores the archive into it (the restore re-verifies every file);
# 4. starts the stack, waits until it is healthy and verifies the audit trail.
# The master key is not in the archive: the stack must start with the same LEGAL_MASTER_KEY_FILE as when the archive was
# made, or its documents cannot be opened. Uses the backend image pinned in .env.law. Never touches production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$(realpath "${1:?usage: $0 ARCHIVE [--yes]}")"
KEY="$(realpath "${LAW_BACKUP_KEY_FILE:-$HOME/.echomind-law/backup.key}")"
ENV_FILE="${LAW_ENV_FILE:-$ROOT/.env.law}"
IMAGE="$(sed -n 's/^LAW_BACKEND_IMAGE=//p' "$ENV_FILE")"
VOL=echomind-law_echomind_law_data
KEEP="echomind-law_prerestore_$(date -u +%Y%m%d-%H%M%S)"
COMPOSE="$ROOT/scripts/law/compose.sh"
log() { echo "$(date -u +%FT%TZ) $*"; }
[[ -r "$ARCHIVE" && -r "$KEY" && -n "$IMAGE" ]] || { echo "need a readable archive, the backup key and LAW_BACKEND_IMAGE in $ENV_FILE" >&2; exit 2; }
helper() { docker run --rm --network none -v "$ARCHIVE:/in/archive.emlb:ro" -v "$KEY:/in/key:ro" "$@"; }

log "checking $ARCHIVE"
helper -e ECHOMIND_EDITION=law "$IMAGE" python -m app.legal.backup verify --archive /in/archive.emlb --key-file /in/key
if [[ "${2:-}" != "--yes" ]]; then
  echo "would stop the stack, keep the current data in volume $KEEP, restore this archive into $VOL and start again."
  echo "run again with --yes to do it."
  exit 0
fi

"$COMPOSE" stop
docker volume create "$KEEP" >/dev/null
docker run --rm --network none -v "$VOL:/old" -v "$KEEP:/keep" "$IMAGE" sh -c 'cp -a /old/. /keep/'
log "current data kept in volume $KEEP"
helper -v "$VOL:/vol" -e ECHOMIND_EDITION=law "$IMAGE" sh -c \
  'find /vol -mindepth 1 -delete && python -m app.legal.backup restore --archive /in/archive.emlb --key-file /in/key --target /vol/restored \
   && cd /vol/restored && for f in * .[!.]*; do [ -e "$f" ] && mv -- "$f" /vol/; done; rmdir /vol/restored'
log "archive restored into $VOL"
"$COMPOSE" up -d backend voice frontend
timeout 240 bash -c 'until [ "$(docker inspect --format "{{.State.Health.Status}}" echomind-law-backend 2>/dev/null)" = healthy ]; do sleep 3; done'
docker exec echomind-law-backend python -m app.legal.audit verify
log "restored and healthy; the previous data stays in volume $KEEP until you remove it (docker volume rm $KEEP)"
