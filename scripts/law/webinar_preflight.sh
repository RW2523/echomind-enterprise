#!/usr/bin/env bash
# Pre-flight for a live EchoMind Legal demo or webinar (2026-11-16). Run it 15–30 minutes before going live, from the
# clone the demo runs from:
#   scripts/law/webinar_preflight.sh [public URL]
# 1. The machine: GPU temperature, free memory, no other GPU work besides the EchoMind model servers.
# 2. Warm-up: one short answer from the model, one embedding, one demo chat question (loads the search and the reranker
#    in the demo API), and the voice service (voice_warmup.py when present), so the first question on stage is fast.
# 3. scripts/law/demo_health.py (containers, model, voice, public page, sign-in, case status, one cited answer, disk).
# Reads nothing secret and prints no credentials; the warm-up chat is deleted afterwards.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
URL="${1:-$(docker inspect echomind-legal-demo-api --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^PUBLIC_URL=//p')}"
fail=0
say() { printf '%-5s %-14s %s\n' "$1" "$2" "$3"; [[ "$1" == FAIL ]] && fail=1; return 0; }

# 1. The machine
temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)"
if [[ -n "$temp" && "$temp" -lt 75 ]]; then say PASS "GPU" "${temp} °C"; else say FAIL "GPU" "${temp:-unknown} °C (let it cool below 75 °C; stop other GPU work)"; fi
avail="$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
if [[ "$avail" -ge 20 ]]; then say PASS "memory" "${avail} GB available"; else say FAIL "memory" "${avail} GB available (want 20+; stop other containers)"; fi
others="$(docker ps --format '{{.Names}}' | grep -vE '^echomind-' | tr '\n' ' ')"
if [[ -z "$others" ]]; then say PASS "other work" "only EchoMind containers running"; else say FAIL "other work" "also running: $others"; fi

# 2. Warm-up
t=$(date +%s)
if curl -sf -m 120 http://127.0.0.1:8355/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"nvidia/Qwen3-30B-A3B-FP4","messages":[{"role":"user","content":"Reply with the word ready."}],"max_tokens":8}' >/dev/null; then
  say PASS "model warm" "$(( $(date +%s) - t )) s"
else say FAIL "model warm" "no answer from :8355"; fi
t=$(date +%s)
if curl -sf -m 60 http://127.0.0.1:11435/api/embeddings -d '{"model":"nomic-embed-text","prompt":"search_query: warm up"}' >/dev/null; then
  say PASS "embeddings" "$(( $(date +%s) - t )) s"
else say FAIL "embeddings" "no answer from :11435"; fi
if [[ -f "$ROOT/scripts/law/voice_warmup.py" ]]; then
  if python3 -I "$ROOT/scripts/law/voice_warmup.py" >/dev/null 2>&1; then say PASS "voice warm" "speech model loaded"; else say FAIL "voice warm" "voice_warmup.py failed"; fi
else
  if curl -sf -m 10 http://127.0.0.1:8112/health >/dev/null; then say PASS "voice" "answers (no warm-up script; the first spoken question may wait)"; else say FAIL "voice" "no answer from :8112"; fi
fi
BASE="$URL"
FIRST="$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'][0]['id'])" "$ROOT/eval/fixtures/webinar/runsheet.json" 2>/dev/null)"
if [[ -n "$FIRST" ]] && python3 -I "$ROOT/scripts/law/demo_runsheet.py" "$BASE" ${URL:+"$URL"} --only "$FIRST" >/dev/null 2>&1; then
  say PASS "chat warm" "first run-sheet step answered"
else say FAIL "chat warm" "first run-sheet step did not pass on $BASE"; fi

# 3. Health
echo "----- demo health"
python3 -I "$ROOT/scripts/law/demo_health.py" ${URL:+"$URL"} || fail=1
echo "-----"
[[ $fail == 0 ]] && echo "PRE-FLIGHT: GO" || echo "PRE-FLIGHT: NOT READY (see FAIL lines)"
exit $fail
