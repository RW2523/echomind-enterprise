#!/usr/bin/env bash
# Is the EchoMind Law data on an encrypted disk? (commercial-readiness review F5, 2026-10-28)
#   scripts/law/check_disk_encryption.sh [path]      # default: the Law data volume's folder on this host
# The original files and backups are encrypted by EchoMind itself; the database (extracted document text, search index,
# research history) relies on full-disk encryption (LUKS/dm-crypt), which only the host can provide. Prints the device
# chain and exits 0 when a crypt layer is found, 1 when not. Needs no root (lsblk, findmnt). Never changes anything.
set -euo pipefail
TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  TARGET="$(docker volume inspect echomind-law_echomind_law_data --format '{{.Mountpoint}}' 2>/dev/null || true)"
  [[ -n "$TARGET" ]] || TARGET="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
fi
# The volume's folder may not be readable without root; its parent filesystem is what matters.
while [[ ! -e "$TARGET" && "$TARGET" != / ]]; do TARGET="$(dirname "$TARGET")"; done
SOURCE="$(findmnt -no SOURCE --target "$TARGET" 2>/dev/null | head -1)"
[[ -n "$SOURCE" ]] || { echo "cannot find the filesystem of $TARGET" >&2; exit 2; }
CHAIN="$(lsblk -s -no NAME,TYPE "$SOURCE" 2>/dev/null | awk '{$1=$1; print}' | tr '\n' ' ')"
echo "data path: $TARGET"
echo "device:    $SOURCE  (chain: ${CHAIN:-unknown})"
if lsblk -s -no TYPE "$SOURCE" 2>/dev/null | grep -qx crypt; then
  echo "full-disk encryption: yes"
else
  echo "full-disk encryption: NO — document text in the database is stored unencrypted on this disk"
  exit 1
fi
