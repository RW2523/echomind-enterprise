# EchoMind Legal — implementation status

Statuses: **Not started**, **In progress**, **Tested** (executed tests pass, evidence given), **Blocked** (needs a person,
root or a decision). "Tested (Law)" = delivered earlier on `feature/echomind-law` with its own evidence
(docs/echomind-law/TEST_RESULTS.md) and re-run on this branch. Branch: `feature/echomind-legal-chat`.

## Baseline on this branch before any change (2026-10-26)

| Suite | Command | Result |
|---|---|---|
| Backend, all | pinned image `echomind-law-backend:latest`, `--network none`, `python -m pytest tests -q` | **1402 passed, 3 skipped, 1 failed** (12 min 17 s). The failure, `test_t09_…reads_back_in_libreoffice…`, is `FileNotFoundError: 'soffice'`: LibreOffice is not in the pinned image (CI installs it). |
| Frontend type-check | `npx tsc --noEmit -p .` | clean |
| Frontend unit | `npx vitest run` | **184 passed** (37 files) |

## Blueprint MVP (docs/echomind-legal/11) against this branch

| Ticket | Item | Status | Evidence |
|---|---|---|---|
| LEGAL-001 | Repo audit, current architecture, gap matrix | **Tested** | CURRENT_ARCHITECTURE.md, GAP_ANALYSIS.md (commands and outputs quoted) |
| LEGAL-002 | Firm/matter/membership schema | **Tested (Law)** | migrations 0002–0006; `test_law_matters_api.py` |
| LEGAL-003 | RBAC, ethical walls, deny by default, 404 for hidden | **Tested (Law)** | `test_law_policy.py` (incl. new `findings.review`), `test_law_walls.py`, `test_law_scoping.py` |
| LEGAL-004 | Source ingestion: hash, versions, OCR, anchors | **Tested (Law)** | `test_law_documents.py`, `test_law_extract.py`, `test_law_structure.py` |
| LEGAL-005 | Citation resolver: opaque ids, re-verified spans, tamper refusal | **Tested** | `test_law_counsel.py`: opening returns exact characters; foreign/forged ids 404; changed text 409; revoked member 404 |
| LEGAL-006 | Search filtered by authorization before scoring | **Tested (Law)** | `AuthorizedScope` + per-matter partitions; `test_law_retrieval.py`; chat cross-matter test (`test_another_matter_never_answers…`) |
| LEGAL-007 | Answer verifier + abstention | **Tested (Law)**, reused | research `_answer`; chat tests for verified/abstained states |
| LEGAL-008 | Chat shell (Quiet Counsel) | **In progress** | — |
| LEGAL-009 | Trusted UI component contract (server) | **Tested** | `counsel/schemas.py`; tests reject unknown types, extra fields, >3 components, >3 items, foreign paths, out-of-range highlights |
| LEGAL-009 | Trusted UI component registry (client) | **In progress** | — |
| LEGAL-010 | Cards, source panel, actions with CAS + idempotency | Backend **Tested**, UI **In progress** | `POST /api/legal/counsel/actions`: stale version 409 with current state, replay by key, forged component/action refused, role 403, closed matter 409 |
| LEGAL-011 | Chronology | Not started | (obligations give lawyer-confirmed dates: Tested (Law)) |
| LEGAL-012 | Discrepancy resolution | Backend **Tested**, UI **In progress** | `legal_discrepancies` + reviews (written once, trigger), audited `discrepancy.review`, hidden when a quoted document is not readable |
| LEGAL-013 | Draft editor | **Tested (Law)** in its own pages; chat points there | `test_law_drafting.py`; chat `draft` intent offers the real Drafting page only to roles that may draft |
| LEGAL-014 | Export approval | **Tested (Law)** | `test_law_exports.py` (T09) |
| LEGAL-015 | Backup and restore | **Tested (Law)** | `app/legal/backup`, `test_law_dr.py`, DR drill (Law run 21) |
| LEGAL-016 | Security / prompt-injection suite | **Tested (Law)** + new chat cases | `test_law_injection.py`; chat: injected notes withheld from the model, no extra words, no other matter |
| LEGAL-017 | Offline acceptance | Backend tests run with `--network none`; enforcement **Blocked (owner, root)** | `sudo scripts/law/egress_lock.sh` (D-071) |
| LEGAL-018 | Advanced agents (Council, GraphRAG, MCP) | Not started (Review panel exists, off: Tested (Law)) | — |
| — | Purge covers the chat | **Tested** | `test_purging_a_document_removes_what_the_chat_kept_of_it`; purge verification counts the chat's tables |
| — | Conflict finder recognises delivery dates | **Tested** | `compare/facts.py` anchor `delivery`; `test_law_compare.py` still green |
| — | Content-Security-Policy | Not started | `frontend/nginx.law.conf` has none |

## Test runs on this branch

| Run | What | Result |
|---|---|---|
| C1 | `test_law_counsel.py` + policy, route table, audit, records, compare, migrations, schema, settings | **701 passed** (1 min 49 s, `--network none`) |
