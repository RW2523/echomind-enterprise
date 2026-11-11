#!/usr/bin/env bash
# The EchoMind Legal public demo, apart from production (commercial-readiness review R1, 2026-10-28). Fictional data only.
#   DEMO_PUBLIC_URL=https://… scripts/law/demo_up.sh api     # (re)create the backend container
#   DEMO_PUBLIC_URL=https://… scripts/law/demo_up.sh voice   # (re)create the voice container (speech to speech, read-aloud)
#   scripts/law/demo_up.sh web                                # (re)create the nginx serving DEMO_DIST and proxying /api, /voice/ws
#   scripts/law/demo_up.sh tunnel                             # (re)create the named Cloudflare tunnel (scripts/law/demo_tunnel.sh)
#   scripts/law/demo_up.sh status
# DEMO_EXTRA_ORIGINS (comma-separated) lets the API and voice also accept another public address, e.g. an old one while
# people still have it.
# DEMO_PREFIX / DEMO_API_PORT / DEMO_VOICE_PORT run a private copy beside the demo (another data folder; never the same names).
# DEMO_ENV_FILE (a private copy only, never the public demo): extra backend settings for an evaluation, one NAME=value per line
# (2026-11-12: LEGAL_STATED_TRACE, LEGAL_RESEARCH_STATED_CHECK=shadow…).
# Backend: the Law backend image, its own bridge network (no host network), code read-only, a read-only root with /tmp on its own
# volume (the speech model unpacks there), no new privileges, every capability dropped, 12 GB memory and 2048 processes at most, a health check on its own
# port. Each CPU model call (reranker, speech, embeddings) uses at most DEMO_CPU_THREADS threads (2026-11-07 load test: with
# four people at once, every call's own thread team took the backend past 512 processes, "libgomp: Thread creation failed",
# and it restarted). It reaches the host's model servers through the bridge gateway and publishes only 127.0.0.1:8110.
# Data: DEMO_DATA_DIR (default .law-data, made by seed-demo; the backend refuses demo sign-in without its DEMO_INSTANCE
# marker) and DEMO_MASTER_KEY (default .law-secrets/master.key, the development key that only protects fictional data).
# Voice (2026-11-02): the Law voice service (voice/app, image echomind-law-voice) on the CPU, code read-only, on the demo's
# own network with the backend. It accepts only the Law-gated /ws (a signed-in session, an allowed origin, a matter and a
# conversation; every spoken question is a chat turn the backend authorises) and publishes only 127.0.0.1:DEMO_VOICE_PORT;
# nginx forwards /voice/ws to it and nothing else. The backend's read-aloud (/speak) uses it on the private network, so the
# demo no longer calls production's voice service.
# The public address is a named Cloudflare tunnel to 127.0.0.1:3125 (since 2026-11-07: `tunnel`, set up once with
# scripts/law/demo_tunnel.sh; a restart keeps the address). Never touches production's containers, volumes or keys.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${DEMO_BACKEND_IMAGE:-echomind-law-backend:latest}"
DATA="$(realpath "${DEMO_DATA_DIR:-$ROOT/.law-data}")"
KEY="$(realpath "${DEMO_MASTER_KEY:-$ROOT/.law-secrets/master.key}")"
HF="${DEMO_HF_CACHE:-$HOME/.cache/huggingface}"
DIST="${DEMO_DIST:-$ROOT/frontend/dist}"
PREFIX="${DEMO_PREFIX:-echomind-legal-demo}"
API="$PREFIX-api" WEB="$PREFIX-web" VOICE="$PREFIX-voice" NET="$PREFIX-net"
API_PORT="${DEMO_API_PORT:-8110}" VOICE_PORT="${DEMO_VOICE_PORT:-8112}"
VOICE_IMAGE="${DEMO_VOICE_IMAGE:-echomind-law-voice:latest}"
net() { docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null; }

case "${1:-}" in
  api)
    [[ -f "$DATA/legal/DEMO_INSTANCE" ]] || { echo "refusing: $DATA was not made by the demo seed (no legal/DEMO_INSTANCE)" >&2; exit 2; }
    [[ -z "${DEMO_ENV_FILE:-}" || "$PREFIX" != echomind-legal-demo ]] || { echo "refusing: DEMO_ENV_FILE is for a private copy only" >&2; exit 2; }
    : "${DEMO_PUBLIC_URL:?set DEMO_PUBLIC_URL to the public address of the demo}"
    net
    GW="$(docker network inspect "$NET" --format '{{(index .IPAM.Config 0).Gateway}}')"
    docker rm -f "$API" >/dev/null 2>&1 || true
    docker volume create "$PREFIX-tmp" >/dev/null
    docker run --rm -v "$PREFIX-tmp":/t alpine:3 chown 1000:1000 /t 2>/dev/null \
      || docker run --rm --user 0 --entrypoint chown -v "$PREFIX-tmp":/t "$IMAGE" 1000:1000 /t
    docker run -d --name "$API" --restart unless-stopped --user 1000:1000 --network "$NET" \
      --add-host host.docker.internal:host-gateway -p "127.0.0.1:$API_PORT:8110" \
      --read-only -v "$PREFIX-tmp":/tmp --security-opt no-new-privileges:true --cap-drop ALL \
      --memory 12g --memory-swap 12g --pids-limit "${DEMO_PIDS_LIMIT:-2048}" \
      -e OMP_NUM_THREADS="${DEMO_CPU_THREADS:-8}" -e MKL_NUM_THREADS="${DEMO_CPU_THREADS:-8}" \
      ${DEMO_ENV_FILE:+--env-file "$DEMO_ENV_FILE"} \
      -e PUBLIC_URL="$DEMO_PUBLIC_URL" -e DEMO_EXTRA_ORIGINS="${DEMO_EXTRA_ORIGINS:-}" -e DEMO_TRUSTED_PROXY="$GW" -e DEMO_TTS_URL="${DEMO_TTS_URL-http://$VOICE:8000/speak}" \
      -v "$ROOT/backend:/repo/backend:ro" -v "$DATA:/data" -v "$KEY:/run/secrets/master.key:ro" -v "$HF:/hf:ro" \
      -v "$ROOT/scripts/law/demo/launch-api.sh:/run/demo/launch-api.sh:ro" \
      --health-cmd "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8110/health', timeout=5)\"" \
      --health-interval 30s --health-timeout 8s --health-start-period 180s --health-retries 3 \
      --entrypoint bash "$IMAGE" /run/demo/launch-api.sh >/dev/null
    echo "started $API (network $NET, 127.0.0.1:$API_PORT); trusted proxy $GW"
    ;;
  voice)
    : "${DEMO_PUBLIC_URL:?set DEMO_PUBLIC_URL to the public address of the demo}"
    net
    docker rm -f "$VOICE" >/dev/null 2>&1 || true
    docker volume create "$PREFIX-voice-tmp" >/dev/null
    docker run -d --name "$VOICE" --restart unless-stopped --network "$NET" --add-host host.docker.internal:host-gateway \
      -p "127.0.0.1:$VOICE_PORT:8000" --read-only -v "$PREFIX-voice-tmp":/tmp --security-opt no-new-privileges:true --cap-drop ALL \
      --memory "${DEMO_VOICE_MEM:-10g}" --memory-swap "${DEMO_VOICE_MEM:-10g}" --pids-limit "${DEMO_PIDS_LIMIT:-2048}" --oom-score-adj 800 \
      -e OMP_NUM_THREADS="${DEMO_CPU_THREADS:-8}" -e MKL_NUM_THREADS="${DEMO_CPU_THREADS:-8}" \
      -v "$ROOT/voice/app:/workspace/app:ro" -v "$ROOT/voice/voices:/voices:ro" \
      -e HF_HOME=/root/.cache/huggingface \
      -e ECHOMIND_EDITION=law -e VOICE_AUTH_ENABLED=1 -e VOICE_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e VOICE_ALLOWED_ORIGINS="$DEMO_PUBLIC_URL${DEMO_EXTRA_ORIGINS:+,$DEMO_EXTRA_ORIGINS}" \
      -e BACKEND_CHAT_URL="http://$API:8110" -e LLM_URL=http://host.docker.internal:8355/v1/chat/completions \
      -e ECHOMIND_ASR_DEVICE=cpu -e ECHOMIND_ASR_REQUIRE_CUDA=0 -e CUDA_VISIBLE_DEVICES= -e VOICE_PIPER_USE_CUDA=0 \
      -e VOICE_USE_PARAKEET="${DEMO_VOICE_PARAKEET:-1}" -e VOICE_PARAKEET_DEVICE=cpu -e VOICE_PARAKEET_CUDA_GRAPHS=0 \
      -e VOICE_NEMOTRON_STARTUP_LOAD=1 -e WAKE_WORD_FILE=/tmp/wake_word.json -e HOME=/tmp/home -e NUMBA_CACHE_DIR=/tmp/numba \
      -e MPLCONFIGDIR=/tmp/mpl \
      --health-cmd "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)\"" \
      --health-interval 30s --health-timeout 8s --health-start-period 240s --health-retries 3 \
      "$VOICE_IMAGE" >/dev/null
    echo "started $VOICE (network $NET, 127.0.0.1:$VOICE_PORT; answers through $API)"
    ;;
  web)
    docker rm -f "$WEB" >/dev/null 2>&1 || true
    docker run -d --name "$WEB" --restart unless-stopped --network host --read-only --tmpfs /var/cache/nginx --tmpfs /var/run \
      --security-opt no-new-privileges:true \
      -v "$ROOT/scripts/law/demo/nginx.conf:/etc/nginx/conf.d/default.conf:ro" -v "$DIST:/usr/share/nginx/html:ro" \
      nginx:1.27-alpine >/dev/null
    echo "started $WEB on 127.0.0.1:3125 serving $DIST"
    ;;
  tunnel)
    # The named Cloudflare tunnel made by scripts/law/demo_tunnel.sh setup (its config and credentials, read-only). Restarting
    # it keeps the public address; it reaches only the demo's nginx on the host.
    TDIR="${DEMO_TUNNEL_DIR:-$HOME/.echomind-legal-demo-tunnel}"
    [[ -f "$TDIR/config.yml" && -f "$TDIR/credentials.json" ]] || { echo "no tunnel config in $TDIR: run scripts/law/demo_tunnel.sh setup <hostname>" >&2; exit 2; }
    docker rm -f "$PREFIX-tunnel" >/dev/null 2>&1 || true
    docker run -d --name "$PREFIX-tunnel" --restart unless-stopped --network host --user "$(id -u):$(id -g)" \
      --read-only --security-opt no-new-privileges:true --cap-drop ALL --memory 256m --pids-limit 128 \
      -v "$TDIR:/etc/cloudflared:ro" "${DEMO_TUNNEL_IMAGE:-cloudflare/cloudflared:latest}" \
      tunnel --config /etc/cloudflared/config.yml run >/dev/null
    echo "started $PREFIX-tunnel ($(sed -n 's/^  - hostname: //p' "$TDIR/config.yml") -> 127.0.0.1:3125)"
    ;;
  status)
    docker ps --filter "name=$PREFIX-" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  *) echo "usage: $0 api|voice|web|tunnel|status" >&2; exit 2 ;;
esac
