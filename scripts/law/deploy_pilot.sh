#!/usr/bin/env bash
# Update the EchoMind Law pilot stack to this checkout's commit (D-060):
#   scripts/law/deploy_pilot.sh
# Builds the images first and changes nothing if that fails. Then stops the Law stack (so its memory guard can
# pass), pins the new tags in .env.law and starts it. If it is not healthy within four minutes, the previous tags
# are pinned again and started. Touches only the echomind-law project, never production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV_FILE="${LAW_ENV_FILE:-$ROOT/.env.law}"
COMPOSE="$ROOT/scripts/law/compose.sh"

"$ROOT/scripts/law/build_images.sh"
NEW="$(git rev-parse --short HEAD)"
for s in backend voice frontend; do docker image inspect "echomind-law-$s:$NEW" >/dev/null; done
OLD="$(sed -n 's/^LAW_BACKEND_IMAGE=echomind-law-backend://p' "$ENV_FILE")"

pin() {
  sed -i -e "s/^LAW_BACKEND_IMAGE=.*/LAW_BACKEND_IMAGE=echomind-law-backend:$1/" \
         -e "s/^LAW_VOICE_IMAGE=.*/LAW_VOICE_IMAGE=echomind-law-voice:$1/" \
         -e "s/^LAW_FRONTEND_IMAGE=.*/LAW_FRONTEND_IMAGE=echomind-law-frontend:$1/" "$ENV_FILE"
}
healthy() {
  timeout 240 bash -c 'h() { docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{end}}" "$1" 2>/dev/null; }
    until [ "$(h echomind-law-backend)" = healthy ] && [ "$(h echomind-law-voice)" = healthy ]; do sleep 3; done'
}

"$COMPOSE" stop
pin "$NEW"
if "$COMPOSE" up -d backend voice frontend && healthy; then
  echo "the pilot now runs $NEW (it ran ${OLD:-nothing pinned})"
else
  echo "the new stack did not come up healthy; going back to $OLD" >&2
  pin "$OLD"
  "$COMPOSE" up -d backend voice frontend && healthy || true
  exit 1
fi
