#!/usr/bin/env bash
# The EchoMind Legal public demo, apart from production (commercial-readiness review R1, 2026-10-28). Fictional data only.
#   DEMO_PUBLIC_URL=https://… scripts/law/demo_up.sh api     # (re)create the backend container
#   scripts/law/demo_up.sh web                                # (re)create the nginx serving DEMO_DIST and proxying /api
#   scripts/law/demo_up.sh status
# Backend: the Law backend image, its own bridge network (no host network), code read-only, a read-only root with /tmp on its own
# volume (the speech model unpacks there), no new privileges, every capability dropped, 12 GB memory and 512 processes at most, a health check on its own
# port. It reaches the host's model servers through the bridge gateway and publishes only 127.0.0.1:8110.
# Data: DEMO_DATA_DIR (default .law-data, made by seed-demo; the backend refuses demo sign-in without its DEMO_INSTANCE
# marker) and DEMO_MASTER_KEY (default .law-secrets/master.key, the development key that only protects fictional data).
# The public address is a Cloudflare tunnel to 127.0.0.1:3125, started separately (its quick-tunnel URL changes when it
# restarts). Never touches production's containers, volumes or keys.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${DEMO_BACKEND_IMAGE:-echomind-law-backend:latest}"
DATA="$(realpath "${DEMO_DATA_DIR:-$ROOT/.law-data}")"
KEY="$(realpath "${DEMO_MASTER_KEY:-$ROOT/.law-secrets/master.key}")"
HF="${DEMO_HF_CACHE:-$HOME/.cache/huggingface}"
DIST="${DEMO_DIST:-$ROOT/frontend/dist}"
API=echomind-legal-demo-api WEB=echomind-legal-demo-web

case "${1:-}" in
  api)
    [[ -f "$DATA/legal/DEMO_INSTANCE" ]] || { echo "refusing: $DATA was not made by the demo seed (no legal/DEMO_INSTANCE)" >&2; exit 2; }
    : "${DEMO_PUBLIC_URL:?set DEMO_PUBLIC_URL to the public address of the demo}"
    GW="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
    docker rm -f "$API" >/dev/null 2>&1 || true
    docker volume create echomind-legal-demo-tmp >/dev/null
    docker run --rm -v echomind-legal-demo-tmp:/t alpine:3 chown 1000:1000 /t 2>/dev/null \
      || docker run --rm --user 0 --entrypoint chown -v echomind-legal-demo-tmp:/t "$IMAGE" 1000:1000 /t
    docker run -d --name "$API" --restart unless-stopped --user 1000:1000 \
      --add-host host.docker.internal:host-gateway -p 127.0.0.1:8110:8110 \
      --read-only -v echomind-legal-demo-tmp:/tmp --security-opt no-new-privileges:true --cap-drop ALL \
      --memory 12g --memory-swap 12g --pids-limit 512 \
      -e PUBLIC_URL="$DEMO_PUBLIC_URL" -e DEMO_TRUSTED_PROXY="$GW" ${DEMO_TTS_URL+-e DEMO_TTS_URL="$DEMO_TTS_URL"} \
      -v "$ROOT/backend:/repo/backend:ro" -v "$DATA:/data" -v "$KEY:/run/secrets/master.key:ro" -v "$HF:/hf:ro" \
      -v "$ROOT/scripts/law/demo/launch-api.sh:/run/demo/launch-api.sh:ro" \
      --health-cmd "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8110/health', timeout=5)\"" \
      --health-interval 30s --health-timeout 8s --health-start-period 180s --health-retries 3 \
      --entrypoint bash "$IMAGE" /run/demo/launch-api.sh >/dev/null
    echo "started $API (bridge network, 127.0.0.1:8110); trusted proxy $GW"
    ;;
  web)
    docker rm -f "$WEB" >/dev/null 2>&1 || true
    docker run -d --name "$WEB" --restart unless-stopped --network host --read-only --tmpfs /var/cache/nginx --tmpfs /var/run \
      --security-opt no-new-privileges:true \
      -v "$ROOT/scripts/law/demo/nginx.conf:/etc/nginx/conf.d/default.conf:ro" -v "$DIST:/usr/share/nginx/html:ro" \
      nginx:1.27-alpine >/dev/null
    echo "started $WEB on 127.0.0.1:3125 serving $DIST"
    ;;
  status)
    docker ps --filter "name=echomind-legal-demo" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  *) echo "usage: $0 api|web|status" >&2; exit 2 ;;
esac
