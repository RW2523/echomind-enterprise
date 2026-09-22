#!/usr/bin/env bash
# Back up the echomind_data volume — the ONLY volume holding customer data.
#
# The SQLite database, the FAISS indexes, uploaded source files, boardroom audio and the generated
# auth secret all live in echomind_data. scripts/export_offline_bundle.sh deliberately does NOT
# cover it (that script ships reproducible model caches), so without this script a disk failure
# loses every customer document, transcript and chat.
#
# ISO 9001:2015 8.5.3 / 8.5.4 — see docs/qms/procedures/SOP-11_Customer_Property_and_Data_Handling.md
# Closes gap G-03 / risk R-02 / objective QO-4.
#
#   ./scripts/backup_data.sh                      # backup to ./backups
#   ./scripts/backup_data.sh /mnt/nas/echomind    # backup to a chosen directory
#   BACKUP_KEEP=14 ./scripts/backup_data.sh       # keep 14 generations (default 7)
#
# A backup you have never restored is not a backup. Run scripts/restore_data.sh against a
# throwaway volume at least every six months and record the result in REG-08 / the management
# review. See "Verifying" at the foot of this file.
set -euo pipefail

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
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/backups}"
KEEP="${BACKUP_KEEP:-7}"
STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="echomind_data_${STAMP}.tar.gz"

log() { printf '  %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found"
docker volume inspect "$VOLUME" >/dev/null 2>&1 || die "volume '$VOLUME' does not exist"
mkdir -p "$DEST" || die "cannot create $DEST"

echo "EchoMind data backup"
log "volume      : $VOLUME"
log "destination : $DEST/$ARCHIVE"

# SQLite must be quiesced or the copy can be torn. The backend holds the DB open, so we ask SQLite
# itself for a consistent snapshot first; if the backend is not running we skip straight to the
# volume copy, which is then trivially consistent.
if docker ps --format '{{.Names}}' | grep -qx echomind-backend; then
  log "backend is running — taking a consistent SQLite snapshot first"
  docker exec echomind-backend python - <<'PY' || die "SQLite snapshot failed"
import sqlite3, os, sys
src = "/data/echomind.sqlite"
dst = "/data/echomind.sqlite.backup"
if not os.path.exists(src):
    print("  no database file yet — nothing to snapshot"); sys.exit(0)
con = sqlite3.connect(src)
bck = sqlite3.connect(dst)
with bck:
    con.backup(bck)          # online backup API: consistent while the app keeps writing
bck.close(); con.close()
print(f"  snapshot written: {dst} ({os.path.getsize(dst)/1048576:.1f} MB)")
PY
else
  log "backend not running — copying the volume directly"
fi

# Exclude the reproducible model caches. /data/hf_cache (VibeVoice, ~16 GB) and
# /data/docgen_models (SDXL-Turbo, ~6.5 GB) are re-downloadable via scripts/prepare_offline.sh, so
# backing them up would turn a ~0.5 GB customer-data backup into a 23 GB one and nobody would run
# it nightly. Customer data — the database, the indexes, uploads, boardroom audio, generated
# documents and the auth secret — is everything else.
EXCLUDES=(--exclude=./hf_cache --exclude=./docgen_models)
log "archiving customer data (excluding reproducible model caches)…"
docker run --rm \
  -v "${VOLUME}:/data:ro" \
  -v "${DEST}:/backup" \
  alpine:3 \
  tar -czf "/backup/${ARCHIVE}" -C /data "${EXCLUDES[@]}" . \
  || die "archive failed"

SIZE=$(du -h "${DEST}/${ARCHIVE}" | cut -f1)
log "written     : ${DEST}/${ARCHIVE} (${SIZE})"

# Checksum so a silently corrupted archive is detectable at restore time.
( cd "$DEST" && sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256" )
log "checksum    : ${ARCHIVE}.sha256"

# Prove the archive is readable and non-trivial before we call it a backup.
log "verifying archive integrity…"
# List once into a file. Piping `tar | grep -q` under `set -o pipefail` makes tar die of SIGPIPE
# when grep exits on the first match, which fails the pipeline and false-negatives the check.
LISTING="$(mktemp)"; trap 'rm -f "$LISTING"' EXIT
tar -tzf "${DEST}/${ARCHIVE}" > "$LISTING" || die "archive is not readable"
ENTRIES=$(wc -l < "$LISTING")
[ "$ENTRIES" -gt 0 ] || die "archive contains no entries"
if grep -q "echomind\.sqlite$" "$LISTING"; then
  log "  contains echomind.sqlite ✓"
else
  log "  WARNING: no echomind.sqlite in archive (empty deployment?)"
fi
log "  ${ENTRIES} entries, archive reads cleanly ✓"

# Retention
if [ "$KEEP" -gt 0 ]; then
  mapfile -t OLD < <(ls -1t "${DEST}"/echomind_data_*.tar.gz 2>/dev/null | tail -n +"$((KEEP+1))")
  if [ "${#OLD[@]}" -gt 0 ]; then
    log "pruning $(( ${#OLD[@]} )) backup(s) beyond the most recent ${KEEP}"
    for f in "${OLD[@]}"; do rm -f "$f" "${f}.sha256"; done
  fi
fi

# Tidy the in-volume snapshot so it is not carried into the NEXT backup as well.
docker ps --format '{{.Names}}' | grep -qx echomind-backend \
  && docker exec echomind-backend rm -f /data/echomind.sqlite.backup 2>/dev/null || true

echo
echo "Backup complete."
echo
echo "This archive contains CUSTOMER DATA. Store it with at least the protection the deployment"
echo "itself has: off the same disk, access-controlled, and encrypted at rest if your customer"
echo "agreement requires it (the archive itself is not encrypted — see SOP-11 §6)."
echo
echo "NOTE: the model caches (/data/hf_cache, /data/docgen_models) are NOT in this archive."
echo "      After a restore, repopulate them with:  ./scripts/prepare_offline.sh"
echo
echo "Restore with:  ./scripts/restore_data.sh ${DEST}/${ARCHIVE}"

# ── Verifying (do this at least every six months; record it) ─────────────────────────────────
#   1. ./scripts/backup_data.sh
#   2. docker volume create echomind_data_restoretest
#   3. VOLUME=echomind_data_restoretest ./scripts/restore_data.sh backups/<archive>
#   4. Inspect: docker run --rm -v echomind_data_restoretest:/d alpine ls -la /d
#   5. docker volume rm echomind_data_restoretest
#   6. Record the date and outcome against objective QO-4.
