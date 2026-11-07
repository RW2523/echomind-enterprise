#!/usr/bin/env bash
# A stable public address for the EchoMind Legal demo: a named Cloudflare tunnel on the owner's zone (owner's request
# 2026-11-07), instead of the quick tunnel whose random trycloudflare.com address changed on every restart.
#
#   cloudflared tunnel login                                   # once, by the owner: authorises the zone, writes ~/.cloudflared/cert.pem
#   scripts/law/demo_tunnel.sh setup legal.echomind-ajace.com  # create the tunnel, its DNS record and its config (idempotent)
#   scripts/law/demo_up.sh tunnel                              # run it (container echomind-legal-demo-tunnel)
#   scripts/law/demo_tunnel.sh status
#
# The tunnel has one route, the hostname to the demo's nginx on 127.0.0.1:3125, and answers 404 to anything else. Its
# credentials and config live outside the repository in DEMO_TUNNEL_DIR (default ~/.echomind-legal-demo-tunnel, mode 700);
# they are never committed or printed. An existing DNS record for the hostname is never overwritten. Production's tunnel
# (echomind-ajace.com, its own token in production's .env) is not touched.
set -euo pipefail
NAME="${DEMO_TUNNEL_NAME:-echomind-legal-demo}"
DIR="${DEMO_TUNNEL_DIR:-$HOME/.echomind-legal-demo-tunnel}"
ORIGIN="${DEMO_TUNNEL_ORIGIN:-http://127.0.0.1:3125}"
CF="$(command -v cloudflared || echo "$HOME/bin/cloudflared")"
[[ -x "$CF" ]] || { echo "cloudflared not found (looked on PATH and in ~/bin)" >&2; exit 2; }

tunnel_id() { "$CF" tunnel list --name "$NAME" --output json 2>/dev/null | python3 -c 'import json,sys; t=json.load(sys.stdin); print(t[0]["id"] if t else "")'; }

case "${1:-}" in
  setup)
    HOST="${2:?usage: $0 setup <hostname>}"
    [[ -f "$HOME/.cloudflared/cert.pem" ]] || { echo "no ~/.cloudflared/cert.pem: the owner runs 'cloudflared tunnel login' first" >&2; exit 2; }
    mkdir -p "$DIR" && chmod 700 "$DIR"
    ID="$(tunnel_id)"
    if [[ -z "$ID" ]]; then
      "$CF" tunnel create --credentials-file "$DIR/credentials.json" "$NAME" >/dev/null
      ID="$(tunnel_id)"
      echo "created tunnel $NAME ($ID)"
    elif [[ ! -f "$DIR/credentials.json" ]]; then
      echo "tunnel $NAME exists but $DIR/credentials.json is missing; restore it, or delete the tunnel and run setup again" >&2; exit 2
    else
      echo "tunnel $NAME exists ($ID)"
    fi
    chmod 600 "$DIR/credentials.json"
    # Refuses (and says so) if the hostname already has a record that is not this tunnel's.
    "$CF" tunnel route dns "$NAME" "$HOST"
    cat > "$DIR/config.yml" <<EOF
# Written by scripts/law/demo_tunnel.sh: the EchoMind Legal demo only (fictional data).
tunnel: $ID
credentials-file: /etc/cloudflared/credentials.json
no-autoupdate: true
ingress:
  - hostname: $HOST
    service: $ORIGIN
  - service: http_status:404
EOF
    chmod 600 "$DIR/config.yml"
    echo "config written to $DIR/config.yml ($HOST -> $ORIGIN); start it with: scripts/law/demo_up.sh tunnel"
    ;;
  status)
    ID="$(tunnel_id)"
    [[ -n "$ID" ]] || { echo "no tunnel named $NAME"; exit 1; }
    "$CF" tunnel info "$NAME"
    [[ -f "$DIR/config.yml" ]] && sed -n 's/^  - hostname: /hostname: /p' "$DIR/config.yml"
    ;;
  *) echo "usage: $0 setup <hostname> | status" >&2; exit 2 ;;
esac
