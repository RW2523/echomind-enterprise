#!/usr/bin/env bash
# docker compose for the isolated EchoMind Law stack (decision D-002). Never touches production:
# own project name (echomind-law), own containers, ports and volume; see docker-compose.law.yml.
#
#   scripts/law/compose.sh config                      # render + validate, starts nothing
#   scripts/law/compose.sh up -d backend voice frontend
#   scripts/law/compose.sh logs -f backend
#   scripts/law/compose.sh down                        # keeps the echomind-law data volume
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${LAW_ENV_FILE:-$ROOT/.env.law}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE: copy .env.law.example to .env.law and fill it in" >&2
  exit 2
fi

case "${1:-}" in
  up|start|restart|run|create)
    # The DGX Spark's memory is unified (CPU+GPU) and shared with production.
    avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    min_gb="${LAW_MIN_FREE_GB:-12}"
    if (( avail_gb < min_gb )); then
      echo "refusing to start: ${avail_gb} GB available < ${min_gb} GB (production shares this host)" >&2
      exit 3
    fi
    ;;
esac

exec docker compose -p echomind-law \
  -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.law.yml" \
  --env-file "$ENV_FILE" "$@"
