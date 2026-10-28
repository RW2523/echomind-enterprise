#!/usr/bin/env bash
# Update the EchoMind Law pilot stack to this checkout's commit (D-060):
#   scripts/law/deploy_pilot.sh
# Builds the images first and changes nothing if that fails. Then (commercial-readiness review R3/R4/F5, 2026-10-28):
#   1. preflight: the data must be on an encrypted disk (check_disk_encryption.sh) unless LAW_ALLOW_UNENCRYPTED_DISK=1,
#      the owner's recorded acceptance;
#   2. a verified backup of the running pilot (backup_pilot.sh), and the database's migration head, before anything stops;
#      LAW_DEPLOY_WITHOUT_BACKUP=1 skips it only when no pilot is running yet;
#   3. stops the Law stack (so its memory guard can pass), pins the new tags in .env.law and starts it;
#   4. the egress lock must hold (egress_lock.sh --check) on the running pilot before, and on the new one after; if it does
#      not afterwards, the stack is stopped (fail closed). LAW_ALLOW_OPEN_EGRESS=1 records the owner's acceptance.
# If it is not healthy within four minutes: migrations the new version applied are reverted with the new image (it has
# their down steps), the previous tags are pinned again and started. If the revert is refused (it would delete records
# kept since), nothing is restored automatically: the script stops with the backup to restore (RUNBOOK §3).
# Touches only the echomind-law project, never production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV_FILE="${LAW_ENV_FILE:-$ROOT/.env.law}"
COMPOSE="$ROOT/scripts/law/compose.sh"
C=echomind-law-backend
log() { echo "$(date -u +%FT%TZ) $*"; }

if ! "$ROOT/scripts/law/check_disk_encryption.sh"; then
  [[ "${LAW_ALLOW_UNENCRYPTED_DISK:-0}" == 1 ]] || { echo "refusing: the data is not on an encrypted disk (LAW_ALLOW_UNENCRYPTED_DISK=1 records the owner's acceptance)" >&2; exit 5; }
  log "WARNING: deploying onto an unencrypted disk (accepted by LAW_ALLOW_UNENCRYPTED_DISK=1)"
fi

"$ROOT/scripts/law/build_images.sh"
NEW="$(git rev-parse --short HEAD)"
for s in backend voice frontend; do docker image inspect "echomind-law-$s:$NEW" >/dev/null; done
OLD="$(sed -n 's/^LAW_BACKEND_IMAGE=echomind-law-backend://p' "$ENV_FILE")"

running() { docker inspect --format '{{.State.Running}}' "$C" 2>/dev/null | grep -q true; }
head_of() { docker exec "$C" python -m app.legal.migrations status | awk '$NF=="applied"{v=$1} END{print v+0}'; }
OLD_HEAD=""
egress_ok() {
  "$ROOT/scripts/law/egress_lock.sh" --check && return 0
  [[ "${LAW_ALLOW_OPEN_EGRESS:-0}" == 1 ]] && { log "WARNING: egress is open (accepted by LAW_ALLOW_OPEN_EGRESS=1)"; return 0; }
  echo "the egress lock does not hold (sudo scripts/law/egress_lock.sh); LAW_ALLOW_OPEN_EGRESS=1 records the owner's acceptance" >&2
  return 1
}
if running; then
  egress_ok || exit 7                                       # checked on the running pilot, before anything changes
  OLD_HEAD="$(head_of)"
  "$ROOT/scripts/law/backup_pilot.sh"
  BACKUP="$(ls -1t "${LAW_BACKUP_DIR:-$HOME/echomind-law-backups}"/law-pilot-*.emlb | head -1)"
  log "before deploy: migration head $OLD_HEAD, backup $BACKUP"
elif [[ "${LAW_DEPLOY_WITHOUT_BACKUP:-0}" != 1 ]]; then
  echo "refusing: the pilot is not running, so no backup can be taken first (LAW_DEPLOY_WITHOUT_BACKUP=1 for a first install)" >&2
  exit 6
fi

pin() {
  sed -i -e "s/^LAW_BACKEND_IMAGE=.*/LAW_BACKEND_IMAGE=echomind-law-backend:$1/" \
         -e "s/^LAW_VOICE_IMAGE=.*/LAW_VOICE_IMAGE=echomind-law-voice:$1/" \
         -e "s/^LAW_FRONTEND_IMAGE=.*/LAW_FRONTEND_IMAGE=echomind-law-frontend:$1/" "$ENV_FILE"
}
healthy() {
  timeout 240 bash -c 'h() { docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{end}}" "$1" 2>/dev/null; }
    until [ "$(h echomind-law-backend)" = healthy ] && [ "$(h echomind-law-voice)" = healthy ]; do sleep 3; done'
}
rollback() {
  echo "the new stack did not come up healthy; going back to $OLD" >&2
  if [[ -n "$OLD_HEAD" ]] && running && [[ "$(head_of)" -gt "$OLD_HEAD" ]]; then
    # The old image refuses a database newer than its code: revert what the new version applied, with its own down steps.
    if ! docker exec "$C" python -m app.legal.migrations down --to "$OLD_HEAD"; then
      echo "the migrations cannot be reverted without deleting records kept since; restore $BACKUP (RUNBOOK §3) before starting $OLD" >&2
      "$COMPOSE" stop || true
      exit 1
    fi
  fi
  pin "$OLD"
  "$COMPOSE" up -d backend voice frontend && healthy || true
  exit 1
}

"$COMPOSE" stop
pin "$NEW"
if "$COMPOSE" up -d backend voice frontend && healthy; then
  if ! egress_ok; then                                      # fail closed: the version is fine, the network is not
    "$COMPOSE" stop
    echo "stopped: $NEW is installed but the egress lock does not hold; lock it, then start the stack" >&2
    exit 7
  fi
  echo "the pilot now runs $NEW (it ran ${OLD:-nothing pinned})"
else
  rollback
fi
