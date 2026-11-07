#!/usr/bin/env bash
# Daily regulatory updates for the EchoMind Law pilot (S4.4, D-068): fetch on the host, import into the pilot.
#   scripts/law/regulatory_daily.sh          # what the daily cron job runs
# The host fetches new UK legislation (legislation.gov.uk, OGL) and US rules (Federal Register, public domain) into
# LAW_FEEDS_DIR (default ~/echomind-law-feeds; bundles older than 60 days are removed), then pipes the bundle into
# the backend container, which checks every field again and matches updates to open matters. The containers never
# reach the internet. Needs LAW_REGULATORY=1, LAW_NETWORK_MODE=restricted and both feed hosts on LAW_EGRESS_ALLOWLIST in
# .env.law. Touches only echomind-law-backend, never production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C=echomind-law-backend
DIR="${LAW_FEEDS_DIR:-$HOME/echomind-law-feeds}"
OUT="$DIR/regulatory-$(date -u +%Y%m%d-%H%M%S).json"
log() { echo "$(date -u +%FT%TZ) $*"; }

# Network mode (2026-11-07 review point 7): the host fetches only in "restricted" mode with both feeds' hosts allowed. A
# disconnected deployment takes a bundle fetched elsewhere: docker exec -i echomind-law-backend python -m app.legal.regulatory import < bundle.json
setting() { [[ -f "$ROOT/.env.law" ]] && sed -n "s/^$1=//p" "$ROOT/.env.law" | tail -1 || true; }
MODE="${LAW_NETWORK_MODE:-$(setting LAW_NETWORK_MODE)}"
ALLOW="${LAW_EGRESS_ALLOWLIST:-$(setting LAW_EGRESS_ALLOWLIST)}"
if [[ "${MODE:-disconnected}" != restricted ]]; then
  log "network mode is ${MODE:-disconnected}: nothing fetched; import a carried-in bundle instead"; exit 4
fi
for host in www.legislation.gov.uk www.federalregister.gov; do
  [[ ",${ALLOW// /}," == *",$host,"* ]] || { log "$host is not on LAW_EGRESS_ALLOWLIST: nothing fetched"; exit 4; }
done
mkdir -p -m 700 "$DIR"
log "fetch: $(python3 "$ROOT/scripts/law/fetch_regulatory.py" --out "$OUT")"
docker inspect --format '{{.State.Running}}' "$C" 2>/dev/null | grep -q true || { log "$C is not running; the bundle is kept for the next run"; exit 3; }
log "import: $(docker exec -i "$C" python -m app.legal.regulatory import < "$OUT")"
find "$DIR" -name 'regulatory-*.json' -mtime +60 -delete
