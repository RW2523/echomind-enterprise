#!/usr/bin/env bash
# Phase 1 gate stability check (S1.12, D-030): the full browser suite N times in a row.
#
#   scripts/law/stability_runs.sh 10
#
# Stops at the first run with a failure. On a failure it dumps the backend's Python stacks into
# that run's log (LAW_DEBUG_STACKS, set by e2e_backend.sh) before the run ends. Logs are kept in
# .law-stability/<time>/run_<n>.log; the last line printed is the verdict.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="${1:-10}"
OUT="$ROOT/.law-stability/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
cd "$ROOT/frontend"
for i in $(seq 1 "$N"); do
  L="$OUT/run_$i.log"
  timeout 1500 npx playwright test > "$L" 2>&1 &
  RUN=$!
  dumped=0
  while kill -0 $RUN 2>/dev/null; do
    if [ $dumped = 0 ] && grep -q "✘" "$L"; then
      sleep 20
      PID=$(ss -ltnp 2>/dev/null | awk '$4 == "127.0.0.1:8110"' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
      [ -n "$PID" ] && kill -USR1 "$PID"
      sleep 2
      echo "STACKS DUMPED pid=$PID" >> "$L"
      dumped=1
    fi
    sleep 3
  done
  wait $RUN
  RC=$?
  echo "run $i: exit=$RC $(grep -E '^\s+[0-9]+ (passed|failed|flaky)' "$L" | tr -s ' ' | tr '\n' ' ')"
  if [ $RC != 0 ] || [ $dumped = 1 ]; then
    echo "VERDICT: run $i failed; logs in $OUT"
    exit 1
  fi
done
echo "VERDICT: $N of $N runs passed; logs in $OUT"
