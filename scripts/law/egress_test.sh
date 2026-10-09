#!/usr/bin/env bash
# T11 network egress check, capture mode (S1.12): run the full browser suite with every outbound
# attempt recorded inside the backend (Python audit hook, LAW_EGRESS_LOG) and the backend's sockets
# watched from outside (scripts/law/egress_watch.py), then fail on any destination that is not this
# machine. The model servers are on localhost, so an air-gapped run expects zero.
#
# Enforced mode — the backend in a network namespace that can reach only the model servers — needs
# root on this host (unprivileged user namespaces are restricted by AppArmor) and waits for the
# owner's approval.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/.law-egress/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
export LAW_EGRESS_LOG="$OUT/backend-hook.jsonl"
export LAW_UVICORN_LOOP=asyncio                      # uvloop would connect in C, out of the recorder's sight
cd "$ROOT/frontend"
npx playwright test > "$OUT/playwright.log" 2>&1 &
RUN=$!
PID=""
for _ in $(seq 1 240); do
  PID=$(ss -ltnp 2>/dev/null | awk '$4 == "127.0.0.1:8110"' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$PID" ] && break
  sleep 0.5
done
[ -n "$PID" ] || { echo "backend did not start"; kill $RUN; exit 2; }
python3 "$ROOT/scripts/law/egress_watch.py" --pid "$PID" --out "$OUT/backend-watch.jsonl" &
WATCH=$!
wait $RUN
SUITE=$?
kill $WATCH 2>/dev/null; wait $WATCH 2>/dev/null
python3 - "$OUT" "$SUITE" <<'PY'
import collections, json, os, sys
out, suite = sys.argv[1], int(sys.argv[2])
hook = [json.loads(l) for l in open(os.path.join(out, "backend-hook.jsonl"))] if os.path.exists(os.path.join(out, "backend-hook.jsonl")) else []
watch = [json.loads(l) for l in open(os.path.join(out, "backend-watch.jsonl"))]
remote_hook = [e for e in hook if not e["local"]]
remote_watch = [e for e in watch if "remote" in e]
done = next((e for e in watch if e.get("done")), {})
by_dest = collections.Counter(f'{e["event"]} {e["host"]}:{e["port"]}' for e in hook)
summary = {"suite_exit": suite, "hook_events": len(hook), "hook_non_local": len(remote_hook),
           "watch_looks": done.get("looks"), "watch_non_local": len(remote_watch),
           "destinations": dict(by_dest.most_common(20)), "evidence": out}
print(json.dumps(summary, indent=1))
for e in remote_hook[:20] + remote_watch[:20]:
    print("NON-LOCAL:", json.dumps(e))
sys.exit(0 if suite == 0 and not remote_hook and not remote_watch and hook else 1)
PY
