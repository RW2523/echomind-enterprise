#!/usr/bin/env bash
# Restore the echomind_data volume from a backup produced by scripts/backup_data.sh.
#
# ISO 9001:2015 8.5.4 — see docs/qms/procedures/SOP-11_Customer_Property_and_Data_Handling.md
#
#   ./scripts/restore_data.sh backups/echomind_data_20260922-101500.tar.gz
#   VOLUME=echomind_data_restoretest ./scripts/restore_data.sh <archive>   # rehearsal
#
# THIS OVERWRITES THE TARGET VOLUME. It refuses to run against a live stack.
set -euo pipefail

ARCHIVE="${1:-}"
# The volume is compose-prefixed (e.g. echomind-enterprise_echomind_data), and the prefix depends on
# the directory the stack was brought up from. Ask the running backend what it actually mounts at
# /data; fall back to the compose project name; finally to a bare name.
detect_volume() {
  local v
  v=$(docker inspect echomind-backend \
        --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' 2>/dev/null || true)
  if [ -n "$v" ]; then printf '%s' "$v"; return; fi
  local proj
  proj=$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")
  for cand in "${proj}_echomind_data" "echomind_data"; do
    if docker volume inspect "$cand" >/dev/null 2>&1; then printf '%s' "$cand"; return; fi
  done
  printf 'echomind_data'
}

VOLUME="${VOLUME:-$(detect_volume)}"

log() { printf '  %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "$ARCHIVE" ] || die "usage: $0 <archive.tar.gz>   (set VOLUME=... to restore elsewhere)"
[ -f "$ARCHIVE" ] || die "archive not found: $ARCHIVE"
command -v docker >/dev/null || die "docker not found"

echo "EchoMind data restore"
log "archive : $ARCHIVE"
log "volume  : $VOLUME"

# Verify the checksum before trusting the archive.
if [ -f "${ARCHIVE}.sha256" ]; then
  log "verifying checksum…"
  ( cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$ARCHIVE").sha256" >/dev/null ) \
    && log "  checksum OK ✓" || die "CHECKSUM MISMATCH — archive is corrupt, do not restore"
else
  log "  WARNING: no .sha256 alongside the archive; integrity unverified"
fi
tar -tzf "$ARCHIVE" >/dev/null 2>&1 || die "archive is not a readable tar.gz"

# Refuse to clobber a volume that a running container is using.
INUSE=$(docker ps --format '{{.Names}}' --filter "volume=${VOLUME}" | tr '\n' ' ')
if [ -n "${INUSE// /}" ]; then
  die "volume '$VOLUME' is in use by: ${INUSE}
       Stop the stack first:  docker compose down
       (or restore into a scratch volume:  VOLUME=echomind_data_restoretest $0 $ARCHIVE )"
fi

if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
  echo
  echo "  Volume '$VOLUME' already exists. Restoring REPLACES its entire contents."
  printf "  Type the volume name to confirm: "
  read -r CONFIRM
  [ "$CONFIRM" = "$VOLUME" ] || die "not confirmed — nothing changed"
  log "clearing existing contents…"
  docker run --rm -v "${VOLUME}:/data" alpine:3 sh -c 'rm -rf /data/..?* /data/.[!.]* /data/* 2>/dev/null || true'
else
  log "creating volume…"
  docker volume create "$VOLUME" >/dev/null
fi

log "extracting…"
docker run --rm \
  -v "${VOLUME}:/data" \
  -v "$(cd "$(dirname "$ARCHIVE")" && pwd):/backup:ro" \
  alpine:3 \
  tar -xzf "/backup/$(basename "$ARCHIVE")" -C /data \
  || die "extraction failed"

log "contents now in '$VOLUME':"
docker run --rm -v "${VOLUME}:/data:ro" alpine:3 sh -c 'ls -la /data | head -20' | sed 's/^/    /'

echo
echo "Restore complete. Start the stack with:  docker compose up -d"
echo "Then verify: the document list, a knowledge-chat query with citations, and a saved transcript."
