# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); next slice branch `law/s2.3-compare` |
| Phase | Phase 1 complete (gate accepted 2026-10-09). Phase 2: S2.1 and S2.2 merged into `feature/echomind-law`; S2.3 next |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations | `0001`–`0016` (Law edition only) |
| Production impact | None. Production stack untouched; only read-only `docker ps`/`inspect`/`curl` probes were run. |

## Session log

### Session 1 — 2026-10-08
- Cloned the repo. Four parallel subsystem audits plus manual verification of high-impact findings.
- Baseline tests: backend 50/50, voice 131/131, frontend `tsc` clean. Fresh-DB defects DB-1 and DB-2 confirmed.
- Discovered that production shares this host and has the same compose project name. Isolation decision D-002 proposed.
- Wrote CODEBASE_AUDIT, FEATURE_MATRIX, THREAT_MODEL, DESIGN_SYSTEM (+ `design/prototype.html`), IMPLEMENTATION_PLAN, DECISIONS, LICENSE_REGISTER, TEST_RESULTS and this file.

### Session 1 (continued) — S1.0 Law foundations
Commits on `law/s1.0-foundations` (base `feature/echomind-law@5039c13`):
- `8e26751` fix(db): fresh installs get chunks page columns and boardroom meta columns (DB-1, DB-2)
- `03a8e13` feat(law): edition flag with startup posture guard (D-003)
- `c55a458` feat(law): versioned schema migrations with checksums and down scripts (D-004)
- `fdbaeba` fix(config): declare 15 settings the code read but env could not override (R-13)
- `d5d188a` feat(law): voice posture guard, eval switches off, spoken content out of logs (D-003, D-016)
- `9a52939` build(frontend): Vitest test runner; Tailwind now scans vertical pages (R-12)
- `0d56398` ops(law): isolated compose stack and native dev scripts (D-002)
- `8954c08` fix(law): redact free-text log arguments in the Law edition (T-11)
- `56478af` test(law): committed fictional law fixtures, fixture loader, law eval baseline
- then: ci: frontend unit tests; docs: S1.0 results

**As built vs. plan (S1.0):**
- `schema_migrations` is owned by the runner; `0001_foundation` creates only `organizations`.
- `asr_container.sh` became `scripts/law/compose.sh`, a wrapper for the isolated compose stack that refuses `up` below 12 GB free.
- Content-free logging is a root-handler redaction filter rather than per-call-site edits. A live log scan proved the per-call-site approach had missed lines.
- Dev ports moved because other stacks on this host hold them: 8100 is `cio-vllm-1`, and 3100 is `cio-loki-1`. The dev backend now uses 8110 and the compose frontend 3110/3553.
- No separate `backend-law` CI job: the existing backend and voice jobs run `pytest tests`, which includes the Law tests. The frontend job gains `vitest run`.

**Shared fixes worth cherry-picking to `main` for production:**
- `8e26751` (fresh-install DB defects)
- `fdbaeba` (R-13 env knobs)
- the Tailwind glob part of `9a52939`

These change legacy behaviour only by fixing bugs: vertical pages become fully styled, and env overrides start working.

**Production impact this session:** none. Only these touched shared resources:
- read-only reuse of the host-published LLM (:8355) and embeddings (:11435), for about 14 questions and 5 chunk embeddings;
- one public model download into the host HF cache: `cross-encoder/ms-marco-MiniLM-L-6-v2`, the same model production uses.

### Session 1 (continued) — S1.1 Identity and sessions
- 2026-10-08: owner approved merging S1.0 → merged `2a91cf2`. No cherry-pick to `main` and no compose `up` (not approved).
- Commits on `law/s1.1-identity`:
  - `4a88f02` identity building blocks
  - `1be9500` backend wiring
  - `c2ac0cd` voice
  - `b052424` frontend
  - `acf6904` dev tooling
- **Delivered:**
  - Server-side sessions: hashed tokens, 30 min idle / 10 h absolute, rotation after MFA.
  - TOTP MFA with recovery codes.
  - Lockout (MFA guesses count) and the password policy.
  - Uniform sign-in failures.
  - `__Host-` cookies and CSRF.
  - Deny-by-default guard (the client-error sink now needs a session).
  - Session list and revoke; password change signs out elsewhere.
  - Admin disable, enable and revoke.
  - Law-safe admin bootstrap (never logged).
  - WebSocket session, Origin check and 60 s revocation re-check (transcribe and voice).
  - Voice on-behalf-of tokens, **fixing the audit's voice→RAG 401**.
  - Frontend MFA sign-in, CSRF fetch wrapper and session-expiry return to sign-in.
  - Dev tooling with MFA.
- **As built vs. plan (S1.1):**
  - **OIDC not built.** It is BLOCKED (P-011), and verifying the IdP's RS256 tokens needs a crypto library that is not yet approved. The plan's "stub IdP" tests were skipped rather than shipping a fake.
  - **Revoke-on-role-change** is not wired: there is no role-change API until S1.2. Disable, delete, password change and admin password reset do revoke sessions.
  - **MFA secret** is stored base32 in `users.mfa_secret`. Encryption at rest follows D-014 (LUKS on the host, plus app-level encryption once `cryptography` is approved).
  - The voice service in the Law edition no longer receives `AUTH_SECRET` (least privilege). Its posture requires `VOICE_ALLOWED_ORIGINS` instead of a shared secret.
  - Admin routes still check the legacy `role == "admin"`. The S1.2 policy engine replaces this with firm roles.
- **Found while building:** backend app shutdown hangs because the CUDA-fault watchdog's thread never finishes. This is pre-existing and unrelated to Law. It was raised as a separate task (chip "Fix backend shutdown hang from CUDA watchdog"), and tests avoid running the app lifespan meanwhile.
- **Residual risks to address later:**
  - Anyone holding a new user's initial password can enrol their own authenticator at first sign-in. Planned mitigation (S1.3 admin UX): an admin-issued one-time enrolment link and a forced password change on first sign-in.
  - Throttling is per account (lockout); there is no per-IP sign-in throttle. `RATE_LIMIT_PER_MIN` exists but is off.

### Session 1 (continued) — S1.2 Matters, memberships, policy engine
- 2026-10-08: owner approved merging S1.1 → merged `0caea7e`.
- Commit `3b8574f` on `law/s1.2-matters`.
- **Delivered:**
  - Migration `0003_matters`.
  - `legal/policy.py`: deny-by-default `decide()`/`authorize()`, with hidden matters answering exactly like missing ones.
  - `legal/matters/` repo and routes, `legal/identity/routes.py` for firm roles, `legal/orgs.py` for the default organisation.
  - Principals now carry `org_id`; in the Law edition, admin routes use the `firm_admin` firm role.
  - **Completes S1.1's revoke-on-role-change.**
- **As built vs. plan (S1.2):**
  - **Matter-level changes:** membership changes bump `matters.acl_epoch` but do not revoke sessions. Every request re-evaluates the policy, so removal is immediate for HTTP. Open streams and sockets will compare the epoch from S1.6 on.
  - **Firm-role changes:** removing a firm role revokes all of that person's sessions; adding one does not.
  - **Single organisation:** created on first boot (`LEGAL_ORG_NAME`). Everything is org-scoped, but there is no API to create more organisations; it is not needed for a single-firm pilot.
  - **Who can see a client:** its creator, or anyone through a visible matter. Firm-wide client lists (conflict checks) are deliberately not offered; that is OPT-01.
  - **Audit events** for matter, membership and role changes are not written yet. They come with the hash-chained audit trail in **S1.5**, which instruments these routes.
  - **Ethical walls and break-glass** are not in the engine yet; they come in S1.4. No UI yet; that is S1.3.

### Session 1 (continued) — S1.3 Law workspace shell and design system
- 2026-10-08: owner approved merging S1.2 → merged `ee67615`, and approved the S1.3 packages.
- Commits on `law/s1.3-workspace`:
  - `da43cc3` D-021 backend
  - `c564d90` workspace and design system
  - `af501bc` visual-review fixes
  - `6f8da72` demo seed, E2E and screenshots
- **Delivered:**
  - **Onboarding hardening (closes D-021):** enrolment links, forced password change, two-step reset.
  - **Edition switch:** `/api/auth/config` reports the edition, and the frontend loads the Law workspace as a lazy chunk before any legacy hook mounts.
  - **"Paper & Glass" design system** (`frontend/law/theme/law.css`).
  - **Routing and data:** react-router routes, session gate, matter context from the URL (hidden ≡ missing), matter-keyed caches dropped on leave and on session expiry, navigation guard.
  - **Pages:** Sign-in, Change password, Home, Matters, Matter overview, Members, Security, Users & roles, Not available, `/__design`.
  - **Demo seed CLI** and the Playwright E2E.
- **As built vs. plan (S1.3):**
  - **react-router 7, not 8.** Version 8 needs Node 22, while CI and the Docker image use Node 20.
  - **No Tailwind preset.** Design tokens are applied through scoped CSS classes (`lw-*`); Tailwind is used only for a few layout utilities.
  - **Display preferences** (theme, transparency) are kept in browser storage as a per-viewer convenience, not on the server as DESIGN_SYSTEM §4.2 suggested. A server-side preference can come later.
  - **No command palette (⌘K) yet.**
  - **Matter navigation shows only built sections** (Overview, Members). Documents, Research and the rest appear when their slices land, so nothing is shown that doesn't work.
  - **The marketing `LawExperience` page is not routed** in the Law edition (it was an unauthenticated demo page); the file remains for the legacy `law` vertical.
  - **E2E is not in CI yet.** It needs the backend's Python dependencies plus a Chromium download in the CI job; this is planned with OPS-02.
- **Design approval (🔒 S1.3 acceptance):** screenshots are in `docs/echomind-law/design/screens/` and were sent to the owner for review.

### Session 1 (continued) — S1.4 Ethical walls and break-glass
- S1.3 merged (`10e3f6f`) after design approval. The owner's standing approval to build and merge in order is recorded in DECISIONS.md.
- Commits:
  - `3915f1e` backend: migration `0005_walls`, policy steps for walls and break-glass in the single query, `/api/legal/walls*`, `/api/legal/break-glass*`
  - `3dbb6f5` screens, a11y fix, seed and E2E
- **As built:**
  - Managing partners are the walls authority and may see the firm's matter directory, but only on the walls screen (D-022).
  - Break-glass is requested by matter number with an identical response whether or not the number exists; it is resolved when a different managing partner approves.
  - The banner does not yet say actions are recorded: read-access audit lands in S1.5, and the copy will change then.

### Session 1 (continued) — S1.5 Audit trail
- S1.4 merged (`0b0fabd`).
- Commits:
  - `3fcf8a3` backend: migration `0006_audit`, `legal/audit` (chain, record, routes, CLI), every Law route instrumented
  - `8080275` screens: Audit trail and Matter activity; recorded matter open; banner copy
- **As built:**
  - Hash rule, anchors, the three recording paths, the content policy, fail-closed behaviour and its one known limit (the event is written after the change commits) are in D-008.
  - Entering a matter is now a recorded read (D-023).
  - Migration numbers shift by one from the plan: documents become `0007`.
- **Not built:** the syslog/CEF forwarder. It needs network approval.

### Session 1 (continued) — S1.6 Matter documents and partitioned retrieval
- S1.5 merged (`fd7cd96`).
- Commits:
  - `1d97e00` backend: migration `0007_documents`, `legal/documents`, `legal/retrieval`, `legal/jobs`, master-key posture
  - `873981a` Documents screen
- **As built:**
  - D-005 (option a), D-012 and D-014 (uploaded documents) were implemented as part of the approved plan; details are in DECISIONS.md.
  - **Deviation:** Law has its own tables and partition module instead of refactoring the legacy index, so legacy behaviour is untouched.
  - Search scores sentence windows; the browser test with real models found why (see TEST_RESULTS run 7).
- **Owner, please note:**
  - D-005, D-012 and D-014 were PROPOSED. I treated the standing approval to build the slices in order as approval of the plan's recommendations for this slice, as with D-006, D-007 and D-013 in S1.1.
  - D-012 is the security trade-off: personal data in a matter's documents is searchable by that matter's members.
  - Say if any of the three should change.
- **Not built (later slices):** OCR for scans (S2.1), the PDF viewer (S2.2), labels and export redaction (S1.9–S1.10), the firm library partition, ClamAV.

### Session 1 (continued) — S1.7 Matter-scoping (main security milestone)
- S1.6 merged (`a36719c`).
- **As built (D-025):**
  - The Law edition serves only the Law API. Every legacy route gives 404 and live capture is closed (4404) until consent capture in S1.8.
  - Research is matter-scoped and streams with live scope checks and explicit abstention.
  - Voice answers only from the matter's research stream, through an on-behalf-of token.
  - New Research screen.
  - Migration `0008_research`.
  - The plan's `ownership` migration is not needed: the legacy tables are no longer reachable in Law.
- **Owner, please note:** D-025 changes the plan's approach. The plan said to patch each legacy module; this closes them in Law instead. Legacy editions are unchanged.

### Session 2 — S1.8 Consent and capture control
- S1.7 merged (`9210cb3`).
- **As built (D-026):**
  - Meetings, participants and append-only consent and capture timelines (migration `0009_meetings`).
  - A new consent-gated Law capture socket, with lock-ordered withdrawal.
  - Audio is not stored, only SHA-256 receipts.
  - Pluggable transcription, unavailable on the dev host.
  - Meetings screens: Prepare, Consent, Record.
- **Counsel:** the consent statement, methods, 12-hour freshness and all-party rule are P-001 safe defaults.

### Session 2 (continued) — S1.9 Labels, legal holds, deletion, retention
- The owner approved the S1.5–S1.8 decisions (see Approvals).
- **As built (D-027):**
  - advisory privilege labels with a marker detector;
  - matter-level legal holds with two-person approval and release;
  - one deletion service: soft delete, report, second-person approval, purge with a hold re-check, verification;
  - retention that only ever dry-runs or creates requests (migration `0010_records`).
- Screens:
  - labels and deletion in the document inspector;
  - "waiting for deletion approval" with restore;
  - hold notices;
  - transcript deletion;
  - Admin → Records (deletion requests, legal holds, retention).
- **Counsel:** label set (P-004), hold workflow (P-003) and retention schedules (P-002) are safe defaults.

### Session 2 (continued) — S1.10 Export gateway and redaction
- **As built (D-028):**
  - A matter's documents and transcripts leave as files only through the gateway: request (with the export right), redaction plan, approval by a second lawyer, render, verify, watermark, then a 24-hour encrypted release (migration `0011_exports`).
  - PDF redaction removes the text. Word files are rebuilt clean.
  - Purging a document purges its exports.
- Screens:
  - Export… on documents and transcripts;
  - Exports list;
  - the export page with page previews, the redaction plan, approval and the released file's checks.
- **Counsel:** who may export, the redaction defaults and the watermark wording are P-006 safe defaults.

### Session 2 (continued) — S1.11 DMS connector framework
- **As built (D-029):**
  - The connector contract and sync engine, with a local-folder connector; settings sealed with the master key.
  - Fail closed: a permissions sidecar is required; source permissions are intersected with membership at every read; documents of a stale or disabled connector are hidden; removal at the source queues an approved purge, unless held.
  - Migration `0012_connectors`, which rebuilds the deletions table to allow a `connector` origin.
- Screens: Admin → Connectors; linked folders and source badges on Documents.
- **Blocked:** a real DMS provider needs the firm's choice of provider, credentials and network approval (P-010). Nothing pretends otherwise.
- **Backend stall found and fixed (D-030):**
  - The full browser suite stalled the backend in 3 of 5 runs, with every thread blocked in SQLite connect/close.
  - `get_conn` now keeps one idle WAL connection open, and the suite then passed 5 runs in a row (60/60).
  - Native stacks were not taken (no ptrace permission), so the exact SQLite lock is not proven. `LAW_DEBUG_STACKS=1` stack dumps stay on for E2E.

### Session 2 (continued) — S1.12 Phase 1 gate
- **Gate report:** `docs/echomind-law/releases/phase1-gate.md`, awaiting the owner's sign-off.
- **T08 (D-031):**
  - The real model obeyed a planted "system override" in 10 of 18 answers. It could only write text, never act.
  - Now AI-directed text is withheld from the model (the person sees it marked), the rule is repeated after the sources, uncited sentences are dimmed, and think tags are stripped.
  - Result: 0 of 36 obeyed. A lowercase-sentence bypass in the splitter was found and fixed.
- **T13 (D-032):**
  - `app.legal.backup`: chunked AES-256-GCM under a separate backup key; the restore authenticates first.
  - `migrate_down` refuses to drop the audit trail, held records or source permissions.
  - `dr_drill.sh` passed on real browser-suite data.
  - The legacy backup scripts find production's container on this host: never run them from this clone.
- **T11 (D-033), capture mode:**
  - Every attempt went to this machine; 0 elsewhere.
  - uvloop hides connections from the recorder, so the egress run uses the asyncio loop.
  - Enforced mode needs root (🔒).
- **T14:**
  - Standard-edition API smoke 10/10 and browser smoke 9/9 with the real models; the voice service boots.
  - Found a pre-existing defect, also on `main`: Knowledge Chat cannot send right after sign-in until a reload. It is marked as an expected failure and offered as a separate task.
- **Also:**
  - T02 download-after-removal test.
  - The E2E sign-in helper decides from the server's reply (a stale alert race caused one failed run).
  - `security_suite.sh` and `stability_runs.sh`.
  - FEATURE_MATRIX SEC-02 corrected (it was stale).

### Session 3 — S2.1 Legal ingestion quality (RAG-01)
- **Audit:**
  - PDFs were read from the text layer only (a scan failed with `no_text`).
  - Word files lost automatic numbering, tables, headers, footers and footnotes, and tracked insertions were silently dropped.
  - Only search chunks were stored, and status was per document.
- **S2.1a (D-034):**
  - `documents/extract.py` stores one canonical text per version (`legal_version_text`) with page, line and block maps.
  - Marks (`legal_marks`) record hidden text left out, AI-directed text, tracked changes, comments and low-confidence OCR words.
  - OCR runs with the local Tesseract on pages without text (D-010).
  - Law passages are exact slices (`documents/chunking.py`); the legacy chunker is not used.
  - Progress is written while a version is read. Purges remove all of it. Migration 0013.
- **S2.1b (D-035):**
  - Clauses come from Word numbering or drafting markers; defined terms are found where they are defined. Migration 0014.
  - Citations carry a focus sentence with its clause, page and line, and research sources pass the clause to the model.
  - A single line break is a visual wrap when splitting document sentences.
- **S2.1c:**
  - The reader endpoint and page, and per-version status (migration 0015).
  - The inspector shows "What reading found", progress and a Read button.
  - Research sources show the clause and line and open the document at the cited sentence.
- **S2.1d:** a browser journey (scan, Word, citation → reader) and screenshots 21–23.

### Session 3 (continued) — S2.2 Legal research (RAG-01)
- **Audit:**
  - Research cited whole passages, with no check that the model's words were in them.
  - Every version of a document was searchable, so a draft could answer for a signed contract.
  - Abstention was a single exact sentence, and the refusal floor (0) was set for the stand-in reranker, not the real one.
- **S2.2a (D-036):**
  - Only the current version and the latest signed version are searchable.
  - Answers come from signed versions by default (`versions: signed|current`).
  - A cited clause that reads differently in the other version is reported as a conflict, with both texts.
- **S2.2b (D-037):**
  - The model quotes its source for each statement (`[n: "…"]`), and every quote is verified against that passage.
  - The answer streams one checked paragraph at a time.
  - Evidence carries document, version, status, clause, page, line and exact characters.
  - Statuses are answered, partial, unverified or abstained (with a reason). Migration 0016.
  - Purges blank saved quotes. R-4, R-6, R-7 and R-8 are now tested in Law.
- **S2.2c (UI):**
  - A "Signed versions / Latest drafts" switch.
  - Flagged sentences: failed quotes and uncited sentences.
  - Conflict alerts with links to the other version.
  - An Evidence region listing each quote and its verification, with a "Not signed" pill and a link into the reader.
  - "Copy with citations".
  - The import dialog sets the status of every new version.
- **S2.2d (D-038):**
  - Section-bounded passages (900 characters).
  - Reranker floor calibrated to -10.5.
  - Abstention in other words, narrowly matched, plus a weak-evidence rule.
  - `eval/law/`: a fictional corpus of 9 documents, including a signed and draft MSA and a matter the user cannot reach; golden (70) and held-out (32) sets; `run_law_eval.py` and committed reports.
- **S2.2e:**
  - Browser journeys updated for the Evidence region.
  - New journey 15: a signed MSA and a later draft, the conflict alert, then "Latest drafts" (screenshot 24).

## Approvals

- 2026-10-08: **APPROVED** D-001, D-002, D-003, D-004 and D-019, committing the plan set, and the S1.0 dev dependencies (`vitest`, `@testing-library/react`, `jsdom`).
- 2026-10-08: each slice S1.0–S1.3 approved and merged in turn; then a **standing approval** to build and merge slices in order, as recorded at the top of DECISIONS.md.
- 2026-10-09: **APPROVED** after the S1.5–S1.8 report: D-005, D-012, D-014 (as implemented), D-023, D-025 and D-026. S1.9 started.
- 2026-10-09: **APPROVED** the Phase 1 gate (S1.12, `releases/phase1-gate.md`): merge S1.12 and start S2.1. This does not cover the root-run enforced T11 or the first `compose.sh up`.
- Still 🔒:
  - the first `scripts/law/compose.sh up` on the shared host;
  - any push or PR to GitHub;
  - anything touching production;
  - network integrations (DMS, IdP, licensed sources, syslog forwarding);
  - counsel items P-001…P-015;
  - destructive migrations on non-development data;
  - the manual Microsoft Word check for tracked changes (S3.3).

## Feature checklist (one row per backlog ID)

| ID | Status | Slice | Notes |
|---|---|---|---|
| SEC-01 | PARTIAL (local accounts complete) | S1.1, S1.3 | Local accounts: sessions, TOTP MFA (enrolment by admin link), forced password change, lockout, CSRF, deny-by-default, WS gating, voice OBO, revoke-on-role-change; browser sign-in tested end to end. OIDC BLOCKED (P-011) |
| SEC-02 | DONE | S1.2–S1.5 | Clients, matters, memberships, policy engine with non-disclosure, firm roles, ethical walls with two-person approval and exceptions, break-glass; UI; T01 in backend and browser, walls and break-glass in the browser; every action and refusal audited (S1.5) |
| SEC-03 | DONE | S1.6–S1.7 | Matter documents in separate encrypted storage and per-matter partitions; a scope capability re-checked per search and while streaming; every legacy module closed in Law (D-025); research and voice matter-scoped; canary suites v1 (search) and v2 (research and voice); T01, T02 and T06 pass, including streams and the model |
| SEC-04 | DONE (forwarder awaits network approval) | S1.0, S1.5 | Content-free logs (S1.0). Hash-chained append-only trail with daily anchors, verifier CLI and endpoint, firm trail for governance roles, matter activity for leads, verifiable JSONL and CSV export; every Law route audited, enforced by a test. Off-host syslog/CEF forwarder not built (network approval) |
| SEC-05 | DONE (mechanism) | S1.8 | All-party, explicit, per-participant consent before capture; withdrawal stops capture at once with a lock-ordered guarantee; pause and resume recorded; append-only consent and capture timelines; audio receipts; T03 in backend and browser. Wording and rules: P-001 (counsel) |
| SEC-06 | DONE | S1.9–S1.10 | Advisory labels (S1.9); export gateway (S1.10): plan, second-lawyer approval with a privilege acknowledgement, real PDF redaction, clean Word rebuild, verification of every output, watermark, 24 h encrypted release; T09 in backend and browser. Policy values: P-006 (counsel) |
| SEC-07 | DONE (mechanism) | S1.9 | Holds with two-person approval and release; one deletion service (soft delete, report, approval, purge with a hold re-check, verification, restore); retention dry runs only, and requests at most; T04 in backend and browser. Schedules: P-002/P-003 (counsel) |
| SEC-08 | PARTIAL (framework DONE) | S1.11 | Connector framework, local-folder connector, fail-closed permissions and freshness, sync and removal through the approved purge; T15 in backend and browser. Real provider BLOCKED (P-010) |
| RAG-01 | DONE for the pilot (lawyer review at S2.8) | S1.6–S1.7, S2.1–S2.2 | S2.1: OCR, Word structure, tracked changes and comments, hidden-text flags, clauses, defined terms, exact offsets, per-version status, reader. S2.2: answers from signed versions by default (switch to drafts), version conflicts, every quote verified, statuses with abstention reasons, paragraph streaming, Evidence and "Copy with citations"; golden 70 and held-out 32 evaluation (D-036–D-038). Firm-library scope deferred to S3.1–S3.2 |
| RAG-02 | MISSING | S2.3 | |
| LIVE-01 | PARTIAL (pre-existing) | S2.4 | |
| LIVE-02 | PARTIAL | S1.7, S2.5 | S1.7: in Law, voice needs a matter and answers only from its research stream (on-behalf-of token). The Law voice UI arrives in S2.5 |
| MEET-01 | PARTIAL | S1.8, S2.6–S2.7 | S1.8: matter meetings with participants, consent and consent-gated capture. Legal-mode Boardroom (before, during, after, minutes): S2.6–S2.7 |
| DOC-01 | PARTIAL (pre-existing) | S3.1 | |
| DOC-02 | MISSING | S3.2–S3.3 | Word tracked changes need a spike and a manual Word check |
| DOC-03 | MISSING | S3.4 | External authority stays UNVERIFIED (P-009) |
| DOC-04 | MISSING | S3.5 | |
| DOC-05 | MISSING | S3.6 | |
| DOC-06 | MISSING | S3.7 | |
| DOC-07 | MISSING | S3.8 | |
| OPT-01 | MISSING | S4.1 | Needs business approval |
| OPT-02 | PARTIAL (pre-existing) | S4.2 | MT model BLOCKED |
| OPT-03 | MISSING | S4.3 | Needs business approval |
| OPT-04 | BLOCKED | S4.4 | Licensed source plus network approval |
| OPT-05 | MISSING | S4.5 | Needs business approval |
| OPS-01 | PARTIAL | S1.0, S1.12 + OPS | S1.0: versioned migrations with down scripts + CLI; isolated compose stack. S1.12: encrypted backup and restore (`app.legal.backup`), rollback guard for audit, holds and source permissions, DR drill on real data (D-032). Runbooks and LUKS (host root) remain |
| OPS-02 | PARTIAL | S1.0, gates | S1.0: Law pytest suite, Vitest, committed law fixtures, law eval baseline 6/7. S1.12: `security_suite.sh`, `injection_probe.py`, `egress_test.sh`, `dr_drill.sh`, `legacy_smoke.py` + legacy browser smoke, `stability_runs.sh`. Load test (T12) and CI remain |

## Known risks right now

- **Production shares this host.** Never run `docker compose` in this clone except through `scripts/law/compose.sh` (`-p echomind-law`).
- Memory headroom is about 14 GB. ASR containers need a memory check first.
- Audit anchors sit on the same disk as the database until an off-host forwarder is approved.
- No per-IP sign-in throttle. Failed sign-ins are recorded and per-account lockout works.
- E2E is not in CI. It runs locally with Playwright (`cd frontend && npx playwright test`); the research test uses the host's model server.
- The compose overlay reuses the 1.4.1 backend image, which may not contain `cryptography`. Check or rebuild at the first approved `compose.sh up`.
- Back up the master key separately from the data: without it the stored documents cannot be opened.
- **Browser-suite stall: fixed but not proven (D-030).** Before the fix, the backend stalled inside SQLite in 3 of 5 full runs. Since the fix, 15 consecutive full runs are clean (5 in S1.11, 10 in the S1.12 gate). The exact lock is unproven without native stacks. Stack dumps stay on for browser runs (`kill -USR1 <uvicorn pid>`). If it ever stalls again, the next step is native stacks, which need the owner's permission to attach a debugger.
- **Injection residuals (D-031):**
  - A novel phrasing can pass the deny-list.
  - An injected claim that the model cites to the injected passage is not dimmed; that passage shows its warning.
  - Voice cannot show the marks.
- **Egress is observed, not enforced (D-033)** until the root-run namespace test is approved.
- **Never run `scripts/backup_data.sh` or `scripts/restore_data.sh` from this clone.** They find their volume through the `echomind-backend` container, which on this host is production. Use `python -m app.legal.backup`.
- **Pre-existing on `main`:** standard-edition Knowledge Chat cannot send right after sign-in until a reload. There is an expected-failure test in `frontend/e2e-legacy`, and a separate task was suggested.
- **Research abstention is the weakest measure (D-038).** When the documents do not answer, the model often says so and then adds related, supported context. That counts as an answer: abstention accuracy is 4/6 on the golden set and 1/3 held-out. The reviewer still sees the "not stated" opening and the evidence. Re-measured with a lawyer at the S2.8 gate.
- In Law there is no Boardroom or document studio until their matter-scoped replacements ship (S2.6, S3.x); live transcription needs the speech model, which is not on the dev host (the capture path is tested with a stand-in). This is intended (D-025, D-026).

## NEXT EXACT ACTION

Start **S2.3 Compare and contradictions (RAG-02)** on `law/s2.3-compare`, per IMPLEMENTATION_PLAN §7. S2.2 is merged into `feature/echomind-law`.
1. **S2.3a Two versions, clause by clause:**
   - Units are the blocks reading found (no headers or footers), each with the clause it starts and the clause it sits in.
   - Alignment: same wording in order (renumbering ignored); shared wording in order; moved clauses; and an optional meaning pass with embeddings.
   - A word diff with exact offsets, and labels: number, obligation, negation, formatting only.
   - Saved per person (migration 0017). Access is the reader's (matter plus source permissions), and a purge removes comparisons that quote the document.
2. **S2.3b Explanations:** the model explains each change, quoting both wordings; every quote is verified (D-037); text addressed to an AI is withheld (T08); the scope is re-checked while streaming.
3. **S2.3c Conflicts across documents:**
   - Facts: amounts, periods, percentages, dates, governing law, company names and references by date.
   - Candidates come only from related documents. The model checks each one (CONFLICT / NO CONFLICT) with verified quotes.
   - Signed versions by default; only documents the person can see.
4. **S2.3d Compare UI:** a side-by-side view with insertions and deletions, change labels and explanations, and an "Across documents" scan with findings linking into the reader. Vitest with axe, and browser journey 16.
5. **S2.3e Evaluation:** a seeded contract pair (renumbered, moved, added, removed, changed) and a planted-conflict order form; report alignment precision and recall, and scan precision and recall on the real model.
6. Write D-039, TEST_RESULTS run 16 and the docs; merge.
