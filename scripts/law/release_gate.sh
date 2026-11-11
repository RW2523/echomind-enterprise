#!/usr/bin/env bash
# EchoMind Legal's release gate (2026-11-12, evaluation fix list item 16): every check in one run, one PASS/FAIL table at the
# end, and a non-zero exit if anything failed. Fictional data only; nothing here touches the public demo unless --demo.
#
#   scripts/law/release_gate.sh [--quick] [--copy http://127.0.0.1:8133 --origin http://127.0.0.1:8134 --container echomind-legal-next-api] [--demo]
#
#   backend      the Law backend suite in the pinned image, no network, the whole repository mounted (tests/law)
#   phrasings    the chat's keyword rules (tests/law/test_law_intent_phrasings.py), its own line
#   vitest       the frontend unit tests;  tsc / tsc-counsel  both type-check configurations
#   browser      Playwright on spare ports (8141 / 5184) against a production build; --quick runs the ten journeys
#                that cover sign-in, access, the chat, research, approvals and the audit trail
#   workflow     scripts/law/workflow_check.py against a PRIVATE copy (--copy, --container): it writes records
#   runsheet     scripts/law/demo_runsheet.py against that copy
#   demo         scripts/law/demo_health.py against the public demo (--demo only: it asks one question there)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${GATE_BACKEND_IMAGE:-echomind-law-backend:latest}"
QUICK=0 DEMO=0 COPY="" CONTAINER="" ORIGIN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --demo) DEMO=1 ;;
    --copy) COPY="$2"; shift ;;
    --container) CONTAINER="$2"; shift ;;
    --origin) ORIGIN="$2"; shift ;;          # the copy's public address (its DEMO_PUBLIC_URL), for the run-sheet's writes
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
LOGS="$(mktemp -d "${TMPDIR:-/tmp}/law-gate.XXXXXX")"
declare -a NAMES=() RESULTS=() TIMES=()
step() {                                   # step <name> <command…>: run it, log it, keep PASS/FAIL and the time
  local name="$1"; shift
  local t0=$SECONDS
  echo "== $name"
  if "$@" >"$LOGS/$name.log" 2>&1; then RESULTS+=("PASS"); else RESULTS+=("FAIL"); fi
  NAMES+=("$name"); TIMES+=("$((SECONDS - t0)) s")
  tail -3 "$LOGS/$name.log"
}
backend() {
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
    -v "$ROOT:/repo" -v "$HOME/.cache/huggingface:/hf:ro" -e HF_HOME=/hf -w /repo/backend --entrypoint python3 "$IMAGE" \
    -m pytest -q -p no:cacheprovider -o addopts="" "$@"
}
QUICK_JOURNEYS='alice sees only her matters|signing out ends the session|a refused authenticator code|a wall approved by|documents: imported into M-101|research: an answer from M-101|exports: redactions reviewed|reviews: a research answer|EchoMind Legal chat: a potential inconsistency|the audit trail shows the emergency access'

step backend backend tests/law -n 4
step phrasings backend tests/law/test_law_intent_phrasings.py
step vitest bash -c "cd '$ROOT/frontend' && npx vitest run"
step tsc bash -c "cd '$ROOT/frontend' && npx tsc --noEmit -p ."
step tsc-counsel bash -c "cd '$ROOT/frontend' && npx tsc --noEmit -p tsconfig.counsel.json"
if [[ $QUICK == 1 ]]; then
  step browser bash -c "cd '$ROOT/frontend' && E2E_API_PORT=8141 E2E_WEB_PORT=5184 npx playwright test --grep '$QUICK_JOURNEYS'"
else
  step browser bash -c "cd '$ROOT/frontend' && E2E_API_PORT=8141 E2E_WEB_PORT=5184 npx playwright test"
fi
if [[ -n "$COPY" ]]; then
  [[ -n "$CONTAINER" ]] && step workflow env LAW_CHECK_BASE="$COPY" LAW_CHECK_CONTAINER="$CONTAINER" python3 "$ROOT/scripts/law/workflow_check.py"
  step runsheet python3 -I "$ROOT/scripts/law/demo_runsheet.py" "$COPY" "${ORIGIN:-$COPY}"
fi
[[ $DEMO == 1 ]] && step demo python3 -I "$ROOT/scripts/law/demo_health.py"

echo
printf '%-12s %-6s %s\n' "Check" "Result" "Time"
bad=0
for i in "${!NAMES[@]}"; do
  printf '%-12s %-6s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"
  [[ "${RESULTS[$i]}" == PASS ]] || bad=1
done
echo "logs: $LOGS"
[[ $bad == 0 ]] && echo "release gate: PASS" || echo "release gate: FAIL"
exit $bad
