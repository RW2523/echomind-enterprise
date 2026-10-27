# EchoMind Legal — architecture gap analysis and reuse matrix

Blueprint (docs/echomind-legal/01–14) against the code that exists on `feature/echomind-law` (the Law line) and `main`.
Evidence is a path, a test file or a recorded run. **Reuse** = use as is; **Refactor** = keep, change the surface;
**New** = does not exist. Status of the work itself is tracked in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## 1. Summary

The Law line already delivers most of the blueprint's **P0 backend**: identity and sessions, firm/matter roles with
deny-by-default and ethical walls, per-matter encrypted storage and search partitions, quote-verified answers with
abstention, comparison and conflict scanning, firm-template drafting with second-lawyer approval, policy-gated export,
consent-gated recording, a hash-chained audit trail and encrypted backup/restore — about 1,400 backend tests and 184
frontend unit tests.

What it does **not** have is the blueprint's product shape:

1. **Chat-first experience ("Quiet Counsel").** Today there are 50 routes behind a sidebar of ~25 entries; Research is a form
   page and the design system deliberately avoids conversation. The blueprint wants one chat with matter context, four
   destinations (Chat, Matters, Drafts, Activity), answers first, at most three prioritized items, details on demand in a
   right-hand panel.
2. **Trusted dynamic UI protocol.** No server-validated component envelope, no component registry, no `data_ref`
   resolution, no `/ui/actions` contract with version checks and idempotency.
3. **Persistent conversations.** Research turns are saved one by one (`legal_research_turns`); there is no conversation that
   groups turns, carries follow-ups and can be reopened from History.
4. **Human resolution of discrepancies.** The conflict scan produces findings (`legal_comparisons.result_json`) but nothing
   lets an attorney mark one explained / confirmed / dismissed with a reason (FR06, AC05, Journey D).
5. **Opaque citation references.** Citations travel to the browser as full passages with offsets; there is no stable,
   owner-bound citation id that the server re-verifies against the stored version when the source is opened (AC02,
   "tampered IDs fail").
6. **Deployment hardening.** The Law nginx sends no Content-Security-Policy; egress is recorded and tested but enforced only
   after the owner runs `sudo scripts/law/egress_lock.sh` (D-033/D-071).

## 2. Reuse / refactor / new matrix

| Blueprint area (doc) | Existing implementation (evidence) | Decision | Work |
|---|---|---|---|
| Local accounts, sessions, MFA, CSRF, lockout (10) | `app/legal/identity/*`; `test_law_auth_api.py`, `test_law_identity_units.py` | **Reuse** | — |
| Firm/matter model, memberships (06) | `matters/`, migrations 0002–0006; `test_law_matters_api.py` | **Reuse** | — |
| Policy: deny by default, walls, break-glass, 404 for hidden (06) | `app/legal/policy.py`; `test_law_policy.py`, `test_law_walls.py`, `test_law_scoping.py` | **Reuse** | Add one matter action `findings.review` |
| Ingestion: hash, versions, OCR, page/clause anchors (05, FR03) | `documents/ingest.py`, `extract.py`, `structure.py`, `blobs.py` (AES-256-GCM); `test_law_documents.py`, `test_law_extract.py` | **Reuse** | — |
| Per-matter prefiltered hybrid retrieval (06, 07) | `retrieval/scope.py` (`AuthorizedScope`), `partition_index.py`, `search.py`; `test_law_retrieval.py` (T01/T02/T06) | **Reuse** | — |
| Quote verification, abstention (07) | `research/evidence.py`, `research/routes.py` (`_answer`); golden 61/64, held-out 28/29 (TEST_RESULTS run 41) | **Reuse** | Call `_answer` from the chat orchestrator, unchanged |
| Prompt-injection firewall (10) | `research/prompt.py` (`addressed_to_ai`, `withhold`); `test_law_injection.py`, `scripts/law/injection_probe.py` | **Reuse** | Add chat-level injection tests |
| Conflict candidates + model check (07 anomalies) | `compare/facts.py`, `compare/scan.py`; `test_law_compare.py` | **Reuse** | Persist findings with review state |
| Discrepancy resolution workflow (FR06, AC05) | none | **New** | `legal_findings`, review action, audit |
| Conversations / History (06 `conversations`, `messages`) | none (single turns only) | **New** | `legal_conversations`, `legal_conversation_turns` |
| Dynamic component envelope + registry (03) | none | **New** | Pydantic schemas (server) + typed validators and registry (client) |
| `POST /ui/actions` with CAS + idempotency (03, 09) | none (per-feature routes with `version` fields) | **New** | `/api/legal/counsel/actions` |
| Opaque, re-verified citation ids (06 citation validation, AC02) | evidence items carry raw offsets | **New** | `legal_counsel_citations` + resolver that re-matches the quote |
| "What matters today?" briefing (04 Journey B) | Reviews queue (`reviews/queue.py`), documents list | **New (composed)** | Briefing from real records: new documents, open findings, reviews waiting |
| Chat shell, matter selector, privacy and recording status (02) | `law/app/Shell.tsx` (glass sidebar), `ui/MatterSwitcher.tsx` | **Refactor** | New Quiet Counsel shell; existing pages stay reachable inside it |
| Source viewer with exact span (02, 04 Journey C) | `pages/DocumentReader.tsx` (full page) | **Refactor** | Right-hand Source panel in chat; "Open full document" links to the reader |
| Drafts destination (04) | Reviews queue (`/reviews`), per-matter drafting pages | **Refactor** | Drafts page over the existing reviews queue |
| Activity destination (04) | Audit trail (governance roles), matter activity (leads) | **Refactor** | Activity page: the person's own recent events |
| Draft editor + approval + DOCX/PDF export (05, AC06) | `drafting/`, `reviews/`, `exports/`; `test_law_drafting.py`, `test_law_exports.py` | **Reuse** | Reached from chat via its existing pages (chat editor later) |
| Recording with consent (FR08, AC07) | `meetings/capture.py`, consent timeline; `test_law_meetings.py` (T03) | **Reuse** | Header shows the real browser capture state |
| Audit (FR09) | `audit/record.py`, `audit/chain.py`; every Law route audited (`test_law_audit.py`) | **Reuse** | New routes use `@audited` |
| Backup + restore (08, AC09) | `app/legal/backup` (AES-256-GCM, manifest), `scripts/law/dr_drill.sh`; `test_law_dr.py` | **Reuse** | — |
| Offline / WAN deny (FR10, AC08) | `app/legal/egress.py` recorder + `egress_test.sh`; enforcement `egress_lock.sh` needs root | **Reuse; Blocked (owner)** | Run the backend tests with `--network none` |
| CSP, no remote assets (08) | fonts bundled (`@fontsource`); **no CSP header** in `nginx.law.conf` | **New** | Add a strict CSP |
| Bounded job queue (05, 08) | `jobs/worker.py` (one thread, restart-safe), `LEGAL_CONFLICT_CONCURRENCY` | **Reuse** | — |
| Timeline / chronology (P1) | obligations (dates from documents, lawyer-confirmed) | Later | — |
| GraphRAG, Council, MCP (P2) | Review panel (`panel/`, four perspectives, off by default) | Later | Not in this slice |

## 3. Target design for the new parts

### 3.1 Backend: `app/legal/counsel/`

| File | Responsibility |
|---|---|
| `schemas.py` | The component envelope v1.0 and each component's props as strict Pydantic models (`extra="forbid"`, length and count limits). Anything the model writes is plain text inside a validated field — never HTML, URLs, CSS or handlers. |
| `intent.py` | Deterministic intent rules: `ask_fact`, `compare`, `inspect_case`, `draft`, `other_matters`, `deliberate`. No model call, fully unit-tested. |
| `store.py` | Conversations, turns, persisted components (with `version` for CAS), action idempotency keys. |
| `citations.py` | Opaque citation ids bound to owner + matter + version + offsets + quote hash; the resolver re-authorizes, re-reads the version text and re-matches the quote (404 when hidden or tampered, 409 when the version changed). |
| `findings.py` | Findings from the conflict scan, de-duplicated by fingerprint, with states `potential → in_review → explained / confirmed / dismissed`. Only a person changes the state. |
| `briefing.py` | "What matters today?" from records: new documents since the last briefing, open findings, reviews waiting. At most three items. |
| `orchestrator.py` | One turn: authorize → classify → run the existing pipeline (research `_answer`, conflict scan) → validate components → persist → stream events. |
| `routes.py` | `/api/legal/counsel/*`, all `@audited`. |

Events (NDJSON on a POST stream, like research — cookie + CSRF, no EventSource): `turn.started`, `turn.intent`,
`answer.delta`, `citation.verified`, `task.progress`, `ui.final`, then `turn.completed` / `turn.abstained` / `turn.failed` /
`scope_revoked`. The blueprint lists SSE; NDJSON over `fetch` keeps the existing CSRF and same-origin cookie model and the
same event names.

### 3.2 Frontend: `frontend/law/counsel/`

Quiet Counsel tokens (02), a shell with the four destinations, header matter selector, "Private" and real capture state,
centre chat column, right-hand panel for sources and comparisons, and a registry that renders only known, validated
component types (unknown → inert fallback). New code is type-checked under `strict` (separate `tsconfig.counsel.json`).

## 4. Risks

| Risk | Mitigation |
|---|---|
| Changing the shell breaks the existing browser journeys | Matter tools stay reachable as a tab strip inside matter pages; Vitest navigation tests updated with the change; Playwright suite re-run where the model is available |
| ~11 GB free memory on a shared host | No new model or service; the chat reuses TensorRT-LLM and Ollama; the conflict scan stays bounded by `LEGAL_CONFLICT_CONCURRENCY` and reuses unresolved findings instead of rescanning |
| Repository is public | No real names, firm data or secrets in fixtures; demo content labelled `DEMO / SYNTHETIC` |
| The stopped pilot | Not restarted by this work; restarting is the owner's decision (`scripts/law/deploy_pilot.sh`) |
