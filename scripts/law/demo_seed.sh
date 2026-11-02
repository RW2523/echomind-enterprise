#!/usr/bin/env bash
# A fresh data folder for the EchoMind Legal public demo (owner's request 2026-11-02: demo data with which every feature
# can be used end to end). Fictional data only (eval/fixtures/law/PROVENANCE.md); never a firm's installation.
#   scripts/law/demo_seed.sh <new data folder>
# Then, with a backend on that folder (scripts/law/demo_up.sh api with DEMO_DATA_DIR=<folder>):
#   python3 scripts/law/demo_load.py --base http://127.0.0.1:<port>/api --data-dir <folder>     # every matter's documents
#   python3 scripts/law/demo_enrich.py --base http://127.0.0.1:<port>/api --data-dir <folder>   # the work done on them
# This script: the sign-in secrets the demo backend reads, the demo world (seed-demo: people, matters, templates, a
# playbook, an ended client call with its transcript and Silent Assist alerts), the newest evaluation reports, and the
# saved regulatory bundle with its dates moved so the newest update is today. It refuses a folder that already has data.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${1:?usage: $0 <new data folder>}"
KEY="$(realpath "${DEMO_MASTER_KEY:-$ROOT/.law-secrets/master.key}")"
[[ -e "$DATA/echomind.sqlite" ]] && { echo "refusing: $DATA already has data" >&2; exit 2; }
umask 077
mkdir -p "$DATA"
DATA="$(realpath "$DATA")"
[[ -s "$DATA/dev_auth_secret" ]] || openssl rand -hex 32 > "$DATA/dev_auth_secret"
[[ -s "$DATA/dev_admin_password" ]] || openssl rand -base64 24 | tr -d '/+=\n' > "$DATA/dev_admin_password"
export ECHOMIND_EDITION=law ECHOMIND_DATA_DIR="$DATA" LEGAL_MASTER_KEY_FILE="$KEY" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LEGAL_REGULATORY=1
cd "$ROOT/backend"
LEGAL_ALLOW_DEMO_SEED=1 python3 -m app.legal.cli seed-demo >/dev/null
GATE="$(ls "$ROOT"/eval/law/reports/*-gate-phase2.json 2>/dev/null | tail -1)"
if [[ -n "$GATE" ]]; then
  mapfile -t REPORTS < <(python3 -c "import json,sys; print('\n'.join('$ROOT/eval/law/reports/'+r['report'] for r in json.load(open(sys.argv[1]))['runs'] if r['report']))" "$GATE")
  python3 -m app.legal.evaluations import "${REPORTS[@]}" "$GATE" >/dev/null
fi
python3 - "$ROOT/eval/fixtures/law/regulatory/bundle-20261012.json" <<'PY' | python3 -m app.legal.regulatory import >/dev/null
import datetime as dt, json, sys
b = json.load(open(sys.argv[1]))
newest = max(dt.date.fromisoformat(i["published"]) for i in b["items"])
shift = dt.date.today() - newest
for i in b["items"]:
    i["published"] = (dt.date.fromisoformat(i["published"]) + shift).isoformat()
print(json.dumps(b))
PY
echo "demo data folder ready: $DATA"
