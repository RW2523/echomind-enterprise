#!/usr/bin/env bash
# Phase 1, 2 and 3 gates: the consolidated security suite (S1.12, S2.8, S3.x), one line per spec test.
#
#   scripts/law/security_suite.sh            # exit 0 only if every group passes
#
# Release-blocking groups (THREAT_MODEL §4): T01, T02, T03, T04, T08, T09, T11, audit chain. The
# browser journeys (frontend/e2e) and the live probes (egress_test.sh, injection_probe.py,
# dr_drill.sh, legacy_smoke.py) are run separately and recorded in TEST_RESULTS.md.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/backend"
export ECHOMIND_DATA_DIR="$(mktemp -d -t emlaw-sec-XXXXXX)"
trap 'rm -rf "$ECHOMIND_DATA_DIR"' EXIT
L=tests/law
declare -a SUITES=(
  "T01 matter isolation, walls, canaries|$L/test_law_policy.py $L/test_law_matters_api.py $L/test_law_walls.py $L/test_law_retrieval.py $L/test_law_scoping.py"
  "T02 revocation ends access, streams and downloads|$L/test_law_scoping.py $L/test_law_retrieval.py $L/test_law_meetings.py $L/test_law_matters_api.py $L/test_law_auth_api.py $L/test_law_route_table.py $L/test_law_documents.py -k t02_or_revok_or_revoc_or_losing_access_or_loses_rights_or_former"
  "T03 consent before capture|$L/test_law_meetings.py"
  "T04 legal holds on every deletion path|$L/test_law_records.py $L/test_law_dr.py -k t04_or_hold_or_held"
  "T08 prompt injection contained|$L/test_law_injection.py"
  "T09 export redaction residue|$L/test_law_exports.py"
  "T11 egress recorder (live run: egress_test.sh)|$L/test_law_egress.py"
  "T13 upgrade, rollback, encrypted restore|$L/test_law_dr.py"
  "T15 source permissions intersected, fail closed|$L/test_law_connectors.py"
  "Audit chain append-only and verifiable|$L/test_law_audit.py"
  "T05/T06 signed versions, verified evidence (S2.2)|$L/test_law_versions.py $L/test_law_evidence.py"
  "Compare and conflicts: visible documents only (S2.3)|$L/test_law_compare.py $L/test_law_conflicts.py"
  "Transcripts: verbatim record, consent kept (S2.4)|$L/test_law_transcripts.py"
  "Silent Assist: matter-only sources, no leaks (S2.5)|$L/test_law_assist.py"
  "AI drafts approved by a second lawyer, hash-bound (S2.6-7)|$L/test_law_minutes.py $L/test_law_reviews.py"
  "Evaluation runs recorded, audit readers only (S2.8)|$L/test_law_evaluations.py"
  "Drafting: nothing invented, approved templates only (S3.1)|$L/test_law_drafting.py"
  "Playbook checks advisory, figures decide (S3.2)|$L/test_law_playbooks.py"
  "T10 Word tracked changes verified before release (S3.3)|$L/test_law_redlines.py"
  "T07 citations: targets only for those who can see them, authorities unverified (S3.4)|$L/test_law_citations.py"
  "Definitions and references: nothing changed without approval (S3.5)|$L/test_law_definitions.py"
  "Obligations: proposed until confirmed, no date guessed, model checked (S3.6)|$L/test_law_obligations.py"
  "Due diligence: preliminary until reviewed, per-viewer visibility (S3.7)|$L/test_law_diligence.py"
  "Governance: both documents quoted and hidden with either (S3.8)|$L/test_law_governance.py"
  "Intake: off by default, screening minimised, consent, links (S4.1)|$L/test_law_intake.py $L/test_law_route_table.py"
  "Summaries: quoted, cited by position, approved by a second person (S4.2)|$L/test_law_summaries.py"
  "Time entries: own work only, approved by the person, CSV without formulas (S4.3)|$L/test_law_time.py"
  "Review panel: separate answers, checked quotes, stop for facts, partner approval (S4.5)|$L/test_law_panel.py"
  "Conversation: matter members only, lengths in the audit; voice routes refused (S5.1)|$L/test_law_conversation.py $L/test_law_pilot_config.py"
  "Regulatory updates: untrusted feed checked, matters only for those who see them (S4.4)|$L/test_law_regulatory.py"
  "Translation: labelled, figures checked, AI-addressed text withheld, access (D-070)|$L/test_law_translation.py"
  "SIEM forwarder: off by default, content-free CEF, nothing skipped (D-069)|$L/test_law_audit_forward.py"
  "Sign-in throttle per real client address; the proxy header believed only from the proxy (D-075)|$L/test_law_throttle.py"
  "Single sign-on: PKCE, browser-bound state, signed tokens, existing accounts only (D-076)|$L/test_law_oidc.py $L/test_law_route_table.py"
)
FAILED=0
printf '%-52s %s\n' "Spec test" "Result"
for g in "${SUITES[@]}"; do
  name="${g%%|*}"; args="${g#*|}"
  # "-k a_or_b" in the table means pytest -k "a or b"
  if [[ "$args" == *" -k "* ]]; then
    files="${args%% -k *}"; expr="${args##* -k }"; expr="${expr//_or_/ or }"
    out=$(python3 -m pytest -q -p no:cacheprovider $files -k "$expr" 2>&1 | tail -1)
  else
    out=$(python3 -m pytest -q -p no:cacheprovider $args 2>&1 | tail -1)
  fi
  [[ "$out" == *failed* || "$out" == *error* || "$out" != *passed* ]] && FAILED=1
  printf '%-52s %s\n' "$name" "$out"
done
exit $FAILED
