#!/usr/bin/env bash
# Put the public demo back the way it was (owner's request 2026-11-07: rehearse the webinar, then wipe the rehearsal).
#
#   scripts/law/demo_reset.sh snapshot    # after a deploy: keep the demo's data as it is now
#   scripts/law/demo_reset.sh restore     # put that data back (chats, cases, updates, uploads made since are gone)
#   scripts/law/demo_reset.sh status
#
# The demo API is stopped for the copy (about 15 s, so the database is consistent) and started again with the same
# public address (and any extra origins); the tunnel, voice and web containers are not touched. The snapshot
# lives outside the repository (fictional data, but never committed): DEMO_SNAPSHOT_DIR, default ~/.echomind-legal-demo-snapshot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DEMO_DATA_DIR:-$ROOT/.law-data}"
SNAP="${DEMO_SNAPSHOT_DIR:-$HOME/.echomind-legal-demo-snapshot}"
API="${DEMO_PREFIX:-echomind-legal-demo}-api"

api_env() {
  docker inspect "$API" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n "s/^$1=//p" | head -1
}
public_url() { api_env PUBLIC_URL; }

restart_api() {
  local url="$1"
  DEMO_PUBLIC_URL="$url" DEMO_EXTRA_ORIGINS="$EXTRA" "$ROOT/scripts/law/demo_up.sh" api >/dev/null
  for _ in $(seq 1 60); do
    [[ "$(docker inspect -f '{{.State.Health.Status}}' "$API" 2>/dev/null)" == healthy ]] && { echo "demo API healthy at $url"; return 0; }
    sleep 2
  done
  echo "the demo API did not come back healthy; see: docker logs $API" >&2
  return 1
}

case "${1:-}" in
  snapshot|restore)
    URL="$(public_url)" EXTRA="$(api_env DEMO_EXTRA_ORIGINS)"
    [[ -n "$URL" ]] || { echo "no running $API to read its public address from" >&2; exit 2; }
    if [[ "$1" == restore && ! -f "$SNAP/echomind.sqlite" ]]; then echo "no snapshot in $SNAP; take one first" >&2; exit 2; fi
    docker stop "$API" >/dev/null
    if [[ "$1" == snapshot ]]; then
      mkdir -p "$SNAP" && chmod 700 "$SNAP"
      rsync -a --delete "$DATA/" "$SNAP/"
      date -u +%Y-%m-%dT%H:%M:%SZ > "$SNAP/.taken_at"
      echo "snapshot of $DATA kept in $SNAP"
    else
      rsync -a --delete --exclude .taken_at "$SNAP/" "$DATA/"
      echo "demo data restored from the snapshot of $(cat "$SNAP/.taken_at" 2>/dev/null || echo 'an unknown time')"
    fi
    restart_api "$URL"
    # a snapshot taken before 2026-11-12 has no display names: set them (only where empty); migrations run on start
    [[ "$1" == restore ]] && docker exec -e ECHOMIND_EDITION=law -e ECHOMIND_DATA_DIR=/data -e HOME=/tmp -w /repo/backend "$API" \
      python -m app.legal.cli demo-names 2>/dev/null | tail -1 || true
    ;;
  status)
    [[ -f "$SNAP/.taken_at" ]] && echo "snapshot taken $(cat "$SNAP/.taken_at") in $SNAP ($(du -sh "$SNAP" | cut -f1))" || echo "no snapshot yet"
    docker ps --filter "name=${DEMO_PREFIX:-echomind-legal-demo}-" --format '{{.Names}}\t{{.Status}}'
    ;;
  *) echo "usage: $0 snapshot|restore|status" >&2; exit 2 ;;
esac
