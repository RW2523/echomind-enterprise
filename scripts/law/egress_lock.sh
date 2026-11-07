#!/usr/bin/env bash
# T11 enforced (D-071): stop the EchoMind Law pilot's containers reaching anything but each other and this host.
#   sudo scripts/law/egress_lock.sh             # add the firewall rule (once), then test it
#   sudo scripts/law/egress_lock.sh --check     # test only
#   sudo scripts/law/egress_lock.sh --remove    # take the rule away
#   scripts/law/egress_lock.sh --dry-run        # show the rule; needs no root
#   scripts/law/egress_lock.sh --print-service  # show the boot service that puts the rule back after a reboot
#   sudo scripts/law/egress_lock.sh --install-service   # install and start it (2026-11-07 review point 8)
# One rule in Docker's DOCKER-USER chain rejects traffic forwarded from the Law network (echomind-law_default) to any
# other address. Traffic to this host (the model servers, through host.docker.internal) is not forwarded, so it still
# works; production's containers are on their own network and untouched (and the Law containers can no longer reach
# them). The test: the backend must reach the answer model and must fail to open a connection to the internet.
# A reboot clears the rule: --install-service adds a systemd unit that puts it back at every boot, once the Law
# containers' network exists (its address range is read again each time). Never touches production.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
UNIT=/etc/systemd/system/echomind-law-egress-lock.service
service() {
  cat <<EOF
[Unit]
Description=EchoMind Law egress lock (D-071): the Law containers reach only each other and this host
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
# the Law network exists once its containers are created; wait for it (up to five minutes)
ExecStartPre=/bin/sh -c 'for i in \$(seq 1 60); do docker inspect echomind-law-backend >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1'
ExecStart=$SELF --rule-only

[Install]
WantedBy=multi-user.target
EOF
}
case "${1:-}" in
  --print-service) service; exit 0 ;;
  --install-service)
    [[ $EUID -eq 0 ]] || { echo "run with sudo (it installs a systemd unit)" >&2; exit 2; }
    service > "$UNIT" && systemctl daemon-reload && systemctl enable --now "$(basename "$UNIT")"
    echo "installed $UNIT; the rule is put back at every boot"; exit 0 ;;
esac
C=echomind-law-backend
NET="$(docker inspect "$C" --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' | awk '{print $1}')"
[[ "$NET" == echomind-law_* ]] || { echo "unexpected network for $C: '$NET'" >&2; exit 2; }
SUBNET="$(docker network inspect "$NET" --format '{{(index .IPAM.Config 0).Subnet}}')"
RULE=(DOCKER-USER -s "$SUBNET" ! -d "$SUBNET" -m comment --comment "echomind-law egress lock (D-071)" -j REJECT --reject-with icmp-port-unreachable)

check() {
  local model internet
  model="$(docker exec "$C" python -c 'import os, urllib.request
base = os.environ["LLM_BASE_URL"].rstrip("/")
urllib.request.urlopen(base + "/models", timeout=10).read(1); print("reached")' 2>/dev/null || echo "failed")"
  internet="$(docker exec "$C" python -c 'import socket
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5).close(); print("reached")
except OSError as e:
    print("blocked (" + type(e).__name__ + ")")' 2>/dev/null)"
  echo "answer model on this host: $model"
  echo "internet (1.1.1.1:443):    $internet"
  [[ "$model" == reached && "$internet" == blocked* ]] && echo "T11 enforced: PASS" || { echo "T11 enforced: FAIL"; return 1; }
}

case "${1:-}" in
  --dry-run) echo "iptables -I ${RULE[*]}"; exit 0 ;;
  --check) check ;;
  --remove) iptables -D "${RULE[@]}" && echo "rule removed" ;;
  --rule-only)                                          # the boot service: the rule, without the model test
    [[ $EUID -eq 0 ]] || { echo "run with sudo (iptables needs root)" >&2; exit 2; }
    iptables -C "${RULE[@]}" 2>/dev/null || iptables -I "${RULE[@]}"
    echo "rule in place for $NET ($SUBNET)" ;;
  "")
    [[ $EUID -eq 0 ]] || { echo "run with sudo (iptables needs root)" >&2; exit 2; }
    iptables -C "${RULE[@]}" 2>/dev/null || iptables -I "${RULE[@]}"
    echo "rule in place for $NET ($SUBNET)"
    check ;;
  *) echo "usage: $0 [--dry-run|--check|--remove|--print-service|--install-service]" >&2; exit 2 ;;
esac
