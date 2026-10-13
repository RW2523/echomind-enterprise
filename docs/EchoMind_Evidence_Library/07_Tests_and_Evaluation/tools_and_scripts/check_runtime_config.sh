#!/usr/bin/env bash
# Compare the security-relevant runtime settings in .env with the recorded baseline
# (SOP-06 §10.1; REG-04 NC-2026-009 — login was switched off in .env with no record).
# Only non-secret settings are tracked. Exit 0 = matches, 1 = drift, 2 = cannot check.
#
#   ./scripts/check_runtime_config.sh            # check ./.env
#   ./scripts/check_runtime_config.sh other.env  # check another file
set -uo pipefail
cd "$(dirname "$0")/.."
BASELINE=docs/qms/records/runtime/runtime-config-baseline.env
ENV_FILE=${1:-.env}
[ -f "$BASELINE" ] || { echo "baseline missing: $BASELINE"; exit 2; }
[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE to check"; exit 2; }

drift=0
while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  key=${line%%=*}
  want=${line#*=}
  if grep -qE "^${key}=" "$ENV_FILE"; then
    have=$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d= -f2-)
  else
    have="<unset>"
  fi
  if [ "$have" != "$want" ]; then
    echo "DRIFT  ${key}: recorded '${want}', now '${have}'"
    drift=1
  fi
done < "$BASELINE"

if [ "$drift" -eq 0 ]; then
  echo "OK — runtime settings match ${BASELINE}"
  exit 0
fi
echo
echo "Record each change in docs/qms/registers/REG-07 (a DC entry with the reason),"
echo "then update ${BASELINE} in the same commit."
exit 1
