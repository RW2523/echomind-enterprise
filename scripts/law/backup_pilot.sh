#!/usr/bin/env bash
# Back up the EchoMind Law pilot (D-067): an encrypted archive of its data folder and database, made inside the
# backend container with `python -m app.legal.backup`, verified, and kept on the host outside the data volume.
#   scripts/law/backup_pilot.sh                 # what the daily cron job runs
#   scripts/law/backup_pilot.sh --restore-test  # also restores the new archive into a scratch folder and checks it
# Settings (environment): LAW_BACKUP_KEY_FILE (default ~/.echomind-law/backup.key, made once with
#   docker exec echomind-law-backend python -m app.legal.backup keygen --out /tmp/k  and copied out, or this
#   script's --keygen), LAW_BACKUP_DIR (default ~/echomind-law-backups), LAW_BACKUP_KEEP (newest archives kept, 14).
# Keep the backup key, the archives and the master key (.law-secrets/pilot-master.key) apart, and a copy of
# both keys off this host: an archive cannot be read without its backup key, and its documents not without the
# master key. Touches only the echomind-law-backend container, never production. Never use scripts/backup_data.sh
# (that is production's).
# Commercial-readiness review R2/R7 (2026-10-28): the audit trail's hash chain is verified first, and a broken chain fails
# the backup (an archive of a tampered trail is not a good backup); LAW_BACKUP_OFFHOST_DIR (a mounted disk or share that
# is not this host's) receives a verified copy; every run appends one line to $DIR/status.log and a failure exits non-zero,
# so cron mails it (MAILTO) and the health check can read the last line.
set -euo pipefail
C=echomind-law-backend
KEY="${LAW_BACKUP_KEY_FILE:-$HOME/.echomind-law/backup.key}"
DIR="${LAW_BACKUP_DIR:-$HOME/echomind-law-backups}"
KEEP="${LAW_BACKUP_KEEP:-14}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
TMP="/tmp/law-backup-$STAMP"
log() { echo "$(date -u +%FT%TZ) $*"; }

if [[ "${1:-}" == "--keygen" ]]; then
  [[ -e "$KEY" ]] && { echo "refusing: $KEY exists (a new key cannot read old archives)" >&2; exit 2; }
  mkdir -p -m 700 "$(dirname "$KEY")"
  docker exec "$C" sh -c "umask 077; python -m app.legal.backup keygen --out $TMP.key >/dev/null"
  (umask 077; docker exec "$C" cat "$TMP.key" > "$KEY")
  docker exec "$C" rm -f "$TMP.key"
  chmod 600 "$KEY"
  log "backup key written to $KEY (mode 600); keep a copy off this host"
  exit 0
fi
[[ -r "$KEY" ]] || { echo "no backup key at $KEY (run: $0 --keygen)" >&2; exit 2; }
docker inspect --format '{{.State.Running}}' "$C" 2>/dev/null | grep -q true || { echo "$C is not running" >&2; exit 3; }
mkdir -p -m 700 "$DIR"
STATUS="$DIR/status.log"
trap 'rc=$?; [[ $rc -ne 0 ]] && echo "$(date -u +%FT%TZ) FAILED (exit $rc)" >> "$STATUS"' ERR
if ! AUDIT="$(docker exec "$C" python -m app.legal.audit verify 2>&1)"; then
  echo "$(date -u +%FT%TZ) FAILED audit chain broken" >> "$STATUS"
  echo "the audit trail's hash chain does not verify; not backing up over it (incident: RUNBOOK §8): $AUDIT" >&2
  exit 5
fi

docker exec "$C" sh -c "umask 077; mkdir $TMP"
trap 'docker exec "$C" rm -rf "$TMP" >/dev/null 2>&1 || true' EXIT
docker exec -i "$C" sh -c "umask 077; cat > $TMP/key" < "$KEY"
docker exec "$C" python -m app.legal.backup create --out "$TMP/pilot.emlb" --key-file "$TMP/key" >/dev/null
CHECK="$(docker exec "$C" python -m app.legal.backup verify --archive "$TMP/pilot.emlb" --key-file "$TMP/key")"
OUT="$DIR/law-pilot-$STAMP.emlb"
(umask 077; docker cp -q "$C:$TMP/pilot.emlb" "$OUT")
chmod 600 "$OUT"
IN_SUM="$(docker exec "$C" sha256sum "$TMP/pilot.emlb" | cut -d' ' -f1)"
[[ "$(sha256sum "$OUT" | cut -d' ' -f1)" == "$IN_SUM" ]] || { rm -f "$OUT"; echo "the copy does not match the archive" >&2; exit 4; }
FILES="$(python3 -c 'import json,sys; m=json.load(sys.stdin); print(m["files"], "files,", m["audit"]["events"], "audit events,", json.dumps(m["counts"]))' <<<"$CHECK")"
log "backup $OUT ($(du -h "$OUT" | cut -f1), sha256 ${IN_SUM:0:16}…) verified: $FILES"

if [[ "${1:-}" == "--restore-test" ]]; then
  RESULT="$(docker exec "$C" python -m app.legal.backup restore --archive "$TMP/pilot.emlb" --key-file "$TMP/key" --target "$TMP/restored")"
  python3 -c 'import json,sys; r=json.load(sys.stdin); print("restore test: the restored database matches the archive:", r["files"], "files,", r["audit"]["events"], "audit events,", json.dumps(r["counts"]))' <<<"$RESULT"
fi

if [[ -n "${LAW_BACKUP_OFFHOST_DIR:-}" ]]; then
  [[ -d "$LAW_BACKUP_OFFHOST_DIR" && -w "$LAW_BACKUP_OFFHOST_DIR" ]] || { echo "LAW_BACKUP_OFFHOST_DIR is not a writable folder" >&2; exit 6; }
  (umask 077; cp -- "$OUT" "$LAW_BACKUP_OFFHOST_DIR/")
  [[ "$(sha256sum "$LAW_BACKUP_OFFHOST_DIR/$(basename "$OUT")" | cut -d' ' -f1)" == "$IN_SUM" ]] || { echo "the off-host copy does not match" >&2; exit 4; }
  log "off-host copy in $LAW_BACKUP_OFFHOST_DIR verified"
fi
echo "$(date -u +%FT%TZ) OK $OUT sha256 $IN_SUM" >> "$STATUS"

ls -1t "$DIR"/law-pilot-*.emlb 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do rm -f -- "$old"; log "removed old backup $old"; done
