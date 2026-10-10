# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); next slice branch `law/s3.9-gate` |
| Phase | Phases 1 and 2 complete (gates accepted 2026-10-09 and 2026-10-10). Phase 3: S3.1–S3.8 merged (S3.3's manual Word check 🔒); S3.9 (the Phase 3 gate) next |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations | `0001`–`0030` (Law edition only) |
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

### Session 3 (continued) — S2.3 Compare and contradictions (RAG-02)
- **Audit:** there was no compare feature, only a comparison block in prompt assembly. S2.2 reported conflicts only between versions of one cited clause.
- **S2.3a — two versions (D-039):**
  - `legal/compare/align.py` aligns units by wording (renumbering ignored), then by shared wording, moves and meaning (embeddings).
  - Each pair has an exact word diff, labelled number, obligation, negation or formatting.
  - Saved per person (migration 0017); access is the reader's; purges remove comparisons that quote the document.
- **S2.3b — explanations:**
  - The model explains each change, quoting both wordings, and the quotes are verified.
  - Text addressed to an AI is withheld. A test caught an inserted note reaching the model through the change hint, and that is fixed.
  - Outcome recorded with counts. A test caught the outcome event failing (missing `matter_id`), and that is fixed.
- **S2.3c — conflicts across documents:**
  - Facts with their exact characters, and parties by cue.
  - Each statement paired with its closest counterpart, and only between documents with the same parties.
  - A legal form, an agreement date and the governing law are reported as found; the model checks the rest in context.
  - Signed versions by default; only documents the person can see.
  - Quote verification now folds Markdown markup and quote style (all research benefits).
- **S2.3d — UI:**
  - A Compare page with "Two versions" (side by side, insertions and deletions, labels, explanations) and "Across documents" (findings with both quotes, rule or model, set-aside checks).
  - Saved comparisons can be reopened; the deletion report counts them.
  - Browser journey 16, screenshots 25–26.
- **S2.3e — evaluation:**
  - `eval/law/run_compare_eval.py` with a golden (tuning) set and a held-out set.
  - The tuning history is reported in D-039, including the first, weaker run.

### Session 3 (continued) — S2.4 Legal transcription (LIVE-01)
- **Audit:**
  - Law capture runs the Nemotron model in-process and does not use the legacy normaliser.
  - Segments were ~0.5 s model chunks with no speaker, on the model's clock (pauses vanished).
  - A transcription error lost text silently.
  - Transcripts were not research sources.
  - Found during the slice: the text pending at a normal stop was discarded.
- **S2.4a — the record (D-017 adopted, D-040):**
  - Immutable segments of whole sentences (migration 0018 trigger).
  - Roster speakers marked live; the model is flushed and restarted at each turn and runs under its process lock.
  - Wall-clock times per segment.
  - Gap and degraded marks, shown to the person and in the transcript.
  - The final text is kept on a normal stop, never after a withdrawal.
- **S2.4b — corrections:**
  - Append-only revisions (who, when, why, source); the text shown is derived.
  - Matter-vocabulary suggestions (participants, companies, defined terms) with provenance, applied only when accepted.
  - Rollback guard.
- **S2.4c — research and records:**
  - "Add to research" makes the corrected transcript a matter document (R-7), with new versions on re-finalising and "corrected since" marked.
  - Deleting a transcript hides, restores and purges that copy too, and the verification covers it.
  - Exports use the corrected text with speakers.
- **S2.4d — UI:**
  - "Speaking now" from the roster (Alt+1…9).
  - Warnings when transcription falls behind or leaves a gap.
  - The transcript with times, speakers, gaps, corrections and their history.
  - A correction dialog (reason required), vocabulary suggestions, and "Add to research".
  - The demo seed has an ended meeting with a transcript; browser journey 17, screenshot 27.
- **S2.4e — accuracy:**
  - `meetings/accuracy.py`: WER verbatim and normalised, entities, negations, speakers.
  - `eval/law/transcription/`: four fictional scripted conversations and a harness (script, file and capture modes).
  - **The real-audio WER is not measured yet:** it needs the speech model, which is not installed here, and loading it on the shared host would risk production (D-040).

### Session 3 (continued) — S2.5 Silent Assist legal packs (LIVE-02)
- **Audit:**
  - The legacy Silent Assistant (closed in Law) had 2 legal regex rules and a whole-knowledge-base fallback (T-03).
  - It had no confirm or dismiss, no telemetry and no tests.
  - Law capture had no assist layer.
- **S2.5a — packs and engine (D-015 adopted, D-041):**
  - `legal/assist/`: three packs by meeting kind.
  - Signed terms and authority from the matter's facts, with the model checking subjects (agreements only, two closest counterparts).
  - Narrow rules for commitment, disclosure, action, quorum and conflict.
  - Evidence-relative wording with exact quotes.
  - Thresholds marked unapproved (P-008).
- **S2.5b — live and records:**
  - Checks run in the background after each stored segment; alerts are stored (migration 0019) and sent.
  - Confirm, or dismiss with a reason, decided once; telemetry.
  - The pack is chosen per meeting.
  - Purges remove alerts with the transcript and remove a purged document's quotes.
- **S2.5c — UI:**
  - The Silent Assist panel with AlertCards (said, source, method, confidence against the threshold, open passage).
  - Confirm and Dismiss with a reason, and the pack choice in Prepare.
  - The demo seed raises two alerts over its meeting; browser journey 18, screenshot 28.
- **S2.5d — evaluation:**
  - `eval/law/assist/` golden and held-out scripted calls through the real capture path and model.
  - The tuning history and a held-out harness correction are reported in D-041.

### Session 3 (continued) — S2.6 Boardroom legal mode (MEET-01)
- **Audit:** the legacy Boardroom report (closed in Law) had four gaps:
  - it read only the first 6,000 characters of the transcript (`boardroom/service.py:806`);
  - it searched the whole knowledge base (`:792`);
  - it dropped timestamps in its checks (`:863`);
  - it verified nothing and had no approval.
  The Law meeting had no agenda and no minutes.
- **S2.6a — agenda and board pack (migration 0020):**
  - Agenda items with their board papers, which must be documents the person can see.
  - The item is marked while recording; each segment keeps its item, and the change is a capture mark.
  - Research limited to an item's papers (`documents` on the research query), with "Ask the whole matter instead".
- **S2.6b — minutes (D-042):** drafted from the whole transcript as corrected.
  - Line-numbered batches, with times, speakers and items taken from the record.
  - The answer is JSON lines with a tolerant parse.
  - Every quote is verified, and items without one are shown "Not verified".
  - Kinds and owners are checked against the record, and first-person commitments are added by rule.
  - Merging is deterministic, and text addressed to an AI is withheld.
- **S2.6c — review (foundation for S2.7):**
  - `legal_ai_artifacts`: fixed versions with a content hash.
  - `legal_review_approvals`: append-only, refused by a rollback.
  - A person's edit is a new version.
  - Approval comes from a second lawyer, bound to the hash; "Request changes" needs a reason.
  - Only approved minutes are filed, as a matter document carrying the approval.
  - A purge takes drafts with the transcript but keeps filed minutes; a hold blocks it.
- **S2.6d — UI:**
  - The Minutes tab: draft, items with their words, flags, the editor, "Review these minutes", and the review decisions.
  - The Agenda editor in Prepare and the agenda item radio group while recording.
  - The board-pack notice in Research.
  - Demo seed: nakamura is a reviewer on M-101. Journey 19, screenshot 29.
- **S2.6e — reader:** journey 19 found that a just-filed document opened in the reader before it was read, and showed "read before the reader existed".
  - The route now says "This version is still being read." (409).
  - The page waits and opens the document when reading finishes.
- **Evaluation:** `eval/law/minutes/` golden and held-out scripted meetings through the real capture path and model. The tuning history is in D-042.
- **As built vs. plan:**
  - The approve route is `POST …/minutes/{aid}/review` with `decision: approved | changes_requested`, so a request for changes is recorded too.
  - Artifact statuses are `draft | approved | superseded | withdrawn`. There is no `in_review`: a draft is reviewable until a newer version supersedes it.
  - Prompt-template version is not yet recorded on artifacts (model and transcript hash are): S2.7.

### Session 3 (continued) — S2.7 Review and approval framework
- **Audit:**
  - Research answers had no reviewer, and approvals lived in five separate places with nothing listing what waits.
  - **A real defect:** the export gateway rendered the version current at approval, not the one its redactions were planned on.
- **S2.7a — artifacts made generic (migration 0021, D-043):**
  - A `research_note` kind beside minutes, with generic read, review and file routes.
  - A newer approval supersedes the earlier one, and the model and prompt version are recorded.
  - A note is shown only to people who can see every document it quotes.
- **S2.7b — hash binding:**
  - An export records its source hash, and approval refuses a changed source (exports from before 0021 are asked for again).
  - Filings are recorded with a copy of the approval. An export of a filed document carries it, and a changed copy is refused.
- **S2.7c — the queue:**
  - `GET /api/legal/reviews` covers AI drafts, exports, deletions, holds, walls and emergency access: waiting for you, and yours with others.
  - It is non-disclosing.
  - The "Reviews" page has a count in the navigation.
- **S2.7d — UI:**
  - `ReviewBanner`, `ReviewPill`, `ReviewDecisions` and `ApprovalDialog` (a confirmation that shows the hash and needs "I have read this version and checked its sources").
  - The research note page and "Send for review" on answers.
  - Minutes moved to the shared components; the meeting opens on its minutes from the queue (`?tab=minutes`).
  - The export page shows a carried approval or a source-changed refusal.
  - Journey 20, screenshots 30–31.
- **Records:** a document purge removes unfiled notes quoting it, and filed notes stay; a rollback refuses to delete notes.
- **As built vs. plan:**
  - Artifact statuses stay `draft | approved | superseded | withdrawn`; "sent back" is the latest decision, not a status.
  - The queue read is not audited, because the navigation polls it; decisions are.
  - Retention policy approvals are not in the queue (they stay on Records).

### Session 3 (continued) — S2.8 Phase 2 gate
- **Evaluation records (D-044):**
  - Migration 0022 `legal_evaluation_runs` (suite, set, commit, models, dataset hash, summary, report hash; never updated).
  - `python -m app.legal.evaluations import`, `GET /api/legal/evaluations` for audit readers, and an "Evaluations" admin page.
  - The E2E backend imports the newest gate run (`LAW_IMPORT_EVALUATIONS`).
- **Gate run:** `eval/law/run_gate.py`, every suite once on `e0f6415`, 10 of 10 ok in about 8 minutes. Results and comparisons are in `releases/phase2-gate.md` §2.
- **T12:** `scripts/law/load_test.py` (research, search and a consented capture socket at once; p50/p95, failures, database busy, memory). A 30-second smoke ran; the measured run needs a window (🔒).
- **Lawyer review:** the rubric with proposed bars (`releases/phase2-rubric.md`); a 67-item packet from this run's held-out outputs (`eval/law/lawyer_review/`); the scorer. The session is 🔒.
- **Security suite:** six Phase 2 groups; all 16 groups pass.
- **Fixes during the gate:**
  - the evaluation scripts no longer mark a commit "+changes" for new report files;
  - the Evaluations page headlines gate and compare runs.
- **Gate report:** `releases/phase2-gate.md`.

### Session 3 (continued) — S3.1 Legal drafting (DOC-01)
- **Audit:** the legacy Document Studio cannot be adapted. It invents "plausible specifics", bans placeholders, reads the global index without matter scope or permissions, cuts its sources to 6,000 characters, keeps no provenance or approval, and has no Word output.
- **S3.1a — firm templates (migration 0023, D-045):**
  - The markup (headings, `{{fields}}`, `[[write: …]]`) with declared field sources and validation.
  - Fixed versions; firm Word styles attachable while a version is a draft.
  - Approval by a managing partner who is not the author, bound to the hash.
  - Six presets; `templates.manage` / `templates.approve`.
- **S3.1b — drafting engine:**
  - Fields from the record, checked quotes or a lawyer: a value must be in its quote, and figures must be figures.
  - Written passages cite their sources. Clause-number citations are matched by their quote, and any uncited sentence or unsupported figure becomes a gap.
  - Passages are widened to the whole matched passage; AI-addressed text is withheld.
- **S3.1c — review and output:**
  - `legal_draft` artifacts with edits as new versions; approval refused while gaps remain.
  - Filed as Word in the firm's styles with the approval in the footer; PDF and Word previews marked "DRAFT — NOT APPROVED".
  - Purge of quoted documents; the Reviews queue path.
- **S3.1d — UI:**
  - The Templates page (presets, editor with field matching, firm styles, approval confirmation, retire).
  - Matter Drafting (list, new draft from approved templates and chosen documents).
  - The draft page (gaps highlighted, every field with its origin and quote, fill in or correct, review, file as Word).
  - Journey 22, screenshots 33–34.
- **Defect found in S2.6 code:** the model client's prose tidy-up cut the tail off JSON answers, which dropped the last minuted item of every minutes batch. Fixed with `llm.complete` (`chat_raw`) and regression tests; minutes re-measured, unchanged on the sets.
- **Evaluation:** `eval/law/drafting/` golden (4) and held-out (3) through the real endpoint and model. The tuning history and all four held-out runs are in D-045.
- **As built vs. plan:**
  - Templates use a markup rather than uploaded Word files; a firm's Word file supplies the styles only. Turning a firm's existing precedent document into a template is S3.2's library import.
  - PDF output is reportlab.

### Session 3 (continued) — S3.2 Clause library and playbooks (DOC-02a)
- **Audit:** no firm positions existed anywhere. The check reuses S2.1 clauses with exact ranges, the S2.3 fact reader, single-document search and checked quotes.
- **S3.2a — library (migration 0024, D-046):**
  - Playbooks with positions by topic: standard, fallbacks with notes, guidance, owner.
  - Fixed versions, approved by a managing partner who is not the author.
  - Precedent proposals read with the S2.1 reader.
- **S3.2b — check:**
  - Candidate clauses by words and search; the model's class with a checked quote.
  - Figures, laws and combined positions decided by rule; unclear and not found.
  - Lawyers agree or disagree with each line. Checks are visible only with their document and purged with it.
- **S3.2c — UI:** the Playbooks page (positions, fallbacks, approval confirmation, new versions, read a precedent) and a document's "Check against a playbook" page, linked from the reader. Journey 23, screenshots 35–36.
- **Evaluation:** `eval/law/playbook/` golden and held-out through the real endpoint and model. The demo journey found the "greater of" case, now in the tuning set and decided by rule (D-046).

### Session 3 (continued) — S3.3 Redlines and Word tracked changes (DOC-02b)
- **Audit:**
  - No tracked-changes writer existed. The S2.1 reader already reads tracked changes; S1.10 exports strip them by design.
  - Comments are written as raw OOXML with lxml (no python-docx upgrade needed).
- **S3.3a — the D-011 spike (`redlines/word.py`, D-047):**
  - Clauses found across paragraphs, ignoring Word's generated numbers, hidden runs and footnote marks.
  - Word-level `w:ins`/`w:del` with author and date; whole paragraphs inserted or deleted with their marks; hidden runs and bookmarks kept; comments with reasons.
  - **T10 `verify`** before every release (accept all = revision, reject all = original); LibreOffice keeps the changes.
- **S3.3b — redlines (migration 0025, which also ends per-kind table rebuilds):**
  - Each line outside the playbook revised to the approved standard or a chosen fallback, on the clause's own text only.
  - The model may only fit defined terms, and is checked.
  - Edits as versions, second-lawyer approval, filed as the document's next version with tracked changes; the exports rule scoped to that version.
- **S3.3c — UI:** "Revise to the playbook" with a fallback choice per line; the redline page (struck and inserted words, reasons, the Word download, edit, approve, file). Journey 24, screenshot 37.
- **Robustness evaluation:** 487/487 revisions placed and verified across 14 documents, LibreOffice 12/12.
- **🔒 Pending:** the manual check in Microsoft Word by a person (D-011, P-013). The product says "Not yet verified by a person in Microsoft Word" until then.

### Session 3 (continued) — S3.4 Citation and quote checker (DOC-03, T07)
- **Audit:** D-037 checked the model's own quotes; nothing read a document for what it cites. S2.1 clauses (with schedules, articles, § and sub-clauses) and the S3.3 word diff were reused.
- **Rules only, no model (D-048):**
  - **References** (clause, section, §, paragraph, article, schedule, exhibit, annex, appendix; lists; "of the <agreement>"; defined short names; "paragraph 2 of Schedule 3") resolve to the exact clause of this document or of a matter document the person can open. Otherwise they are flagged **missing** (the clause does not exist) or **not found** (no such document for this person).
  - **Quotations** (paired in order, UK single quotes included) are compared word for word: **accurate** or **altered**, with struck and inserted words, or **not found**.
  - **External authorities** (US Code, CFR, US cases, UK neutral citations, reports and Acts, EU instruments, CJEU) are listed and always **UNVERIFIED** (P-009); citations that cannot be real are flagged.
- **Storage (migration 0026):** a check keeps targets as offsets, never their words. Passages and diffs are read per viewer, so they are hidden from anyone who cannot open the target and nothing remains after a purge. Notes are append-only, with a rollback guard.
- **UI:** "Check citations" on the reader. The page lists quotations, references and authorities with problems first, the P-009 notice, diffs, "Open at the citation" and "Open the passage", and notes. Journey 25, screenshot 38.
- **T07 evaluation:**
  - Golden 30/30. The first held-out run scored 20/25, and its defects were fixed (s3.4b); that set became tuning data and a new held-out set was committed blind.
  - Golden 55/55; new held-out 27/28. The miss: court rules (Fed. R. Civ. P., CPR) are not recognised. This is recorded for S3.9.
- **Permission:** `research.query`, as planned. A reviewer can check a memo before approving it. Notes are for lawyers.

### Session 3 (continued) — S3.5 Defined terms and cross-references (DOC-04)
- **Audit:** the S2.1 reader keeps one definition per term; it does not see Markdown "## 4." headings as clauses; and it read "(c)" after "(a)" as roman c. Fixed in the reader (roman lists start at (i), and the roman status is tracked). `documents.numbering` (new) completes implied clauses and Markdown schedules for both checkers.
- **The checker (rules only, D-049):**
  - definitions: duplicates (and whether they differ), unused, undefined (heuristic), another form, used before defined;
  - numbering gaps and repeats at every level, including items, schedules and articles;
  - broken and outdated references.

  Each finding has offsets, the clause it is in, and corrections where they are unambiguous.
- **Corrections:** a lawyer picks them; they become a redline (S3.3) with tracked changes. A second lawyer approves it before it is filed. The Word writer deletes a clause whole when asked. Numbering is never changed.
- **Storage:** migration 0027 (checks, append-only notes, rollback guard); purged with the document.
- **UI:** "Check definitions" on the reader. Grouped findings with where they are, places, corrections to pick and notes; the redline page knows a corrections redline. Journey 26 with a new fictional Markdown supply agreement; screenshot 39.
- **Evaluation:** three blind held-out sets in turn (8/12, 13/16, then 12/12 with the control clean); each became tuning data once it informed a change. Golden G1–G10: 52/55, fixes 8/8, no false findings. The misses are undefined terms the rules do not recognise.
- **Shared change:** the citation checker uses the numbering helper (citations-s3.4c/d); T07 is unchanged at 55/55 and 27/28.

### Session 3 (continued) — S3.6 Obligations and deadlines (DOC-05)
- **Rules first (D-050):**
  - sentences that bind someone and give a time, or a consent right, become proposed obligations;
  - each keeps its kind, party, action, the timing in the document's words, and the exact sentence and clause.
- **Dates only from dates the document fixes:** term ends, non-renewal and break deadlines, dated events, and periods after them. The counting is said and must be confirmed. No date was guessed in any run.
- **Checked model proposals:** only for binding sentences the rules did not read. The timing must be the sentence's words, the party named in it, and the model's dates are never used. AI-addressed text is withheld; the rules stand alone if the model is down.
- **Tracker (migration 0028):**
  - proposed until a lawyer confirms with an owner;
  - edit, done, dismiss and reopen as versions with reasons, in an append-only history; the audit holds no content;
  - timezone (`LEGAL_TIMEZONE`); calendar file off by default (`LEGAL_OBLIGATIONS_ICS`);
  - purged with the document.
- **UI:** the matter's "Obligations" page. A document is read; items are grouped as overdue, dated, or running from an event; each shows its words, how its date was reached and its source; confirm, edit, done, dismiss, history. Journey 27; screenshot 40.
- **Evaluation:** four blind held-out sets in turn (rules: 9/13, 5/10, 5/12; with the checked model: 10/12), none with a guessed date, every control clean. Golden G1–G13 with the model: 56/56, dates 15/15.

### Session 3 (continued) — S3.7 Due diligence (DOC-06)
- **Checklists (D-051):** fixed versions with sections, the words that name each expected document, and the questions to ask. A managing partner other than the author approves the hash; a preset is seeded. Firm page "Checklists" with an item editor and approval.
- **Runs:** a background job (`diligence.run`). Documents are matched by their own title line; the missing list; rules ask change of control, assignment, termination and break, governing law against the expected law, liability, and renewal and expiry. Each issue quotes its sentence.
- **Review:** preliminary until verified or dismissed with a reason; append-only trail; visibility per viewer; purge removes issues and names. Migration 0029. Matter page "Due diligence"; journey 28; screenshot 41.
- **Evaluation:**
  - DOC-06 held-out: 9/11, then 5/6 issues, every missing list right, no extra issues, controls clean.
  - Obligations held-out 5: 4/6.
  - Fixes found by these runs (a term from a literal date; an anchor cut to one word; "end this Agreement"; termination without a notice period) are made. The last three are to be measured on fresh held-out sets at S3.9.

### Session 3 (continued) — S3.8 Governance checks (DOC-07)
- **Checks in the S3.7 run and register (D-052):** share capital (register against approvals and the articles), authority (contract values against the signer's limit), constitution (policies against the articles), and IP chain (founders against assignments). Each finding quotes both documents (migration 0030) and is shown only with both. A preset "Corporate governance review" is seeded.
- **UI:** checklists turn the checks on; the register shows the second quote. Journey 29 with a fictional company (seven fixture files); screenshot 42.
- **Evaluation:** golden 9/9, then held-out 3/6 (no extras, control clean), then s3.8b fixes; golden G1–G6 15/15. s3.8b is measured at S3.9.
- **Also in this slice:** S3.7's doc placeholder filled; demo seed test for both presets.

## Approvals

- 2026-10-08: **APPROVED** D-001, D-002, D-003, D-004 and D-019, committing the plan set, and the S1.0 dev dependencies (`vitest`, `@testing-library/react`, `jsdom`).
- 2026-10-08: each slice S1.0–S1.3 approved and merged in turn; then a **standing approval** to build and merge slices in order, as recorded at the top of DECISIONS.md.
- 2026-10-09: **APPROVED** after the S1.5–S1.8 report: D-005, D-012, D-014 (as implemented), D-023, D-025 and D-026. S1.9 started.
- 2026-10-09: **APPROVED** the Phase 1 gate (S1.12, `releases/phase1-gate.md`): merge S1.12 and start S2.1. This does not cover the root-run enforced T11 or the first `compose.sh up`.
- 2026-10-10: **APPROVED** the Phase 2 gate (`releases/phase2-gate.md`): merge S2.8 and start S3.1. The owner added: "approve everything and finish everything". This is taken as standing approval to build and merge every remaining slice and phase gate in order, including the Phase 4 business approval. Items that need a person (the lawyer review, the manual Word check, counsel's P-decisions) or that act outside this machine or on the shared host (a GitHub push, the first `compose.sh up` next to production, root) are confirmed one by one.
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
| SEC-06 | DONE | S1.9–S1.10 | Advisory labels (S1.9); export gateway (S1.10): plan, second-lawyer approval with a privilege acknowledgement, real PDF redaction, clean Word rebuild, verification of every output, watermark, 24 h encrypted release; T09 in backend and browser. Policy values: P-006 (counsel). S2.7: exports bound to their source hash; filed AI drafts carry their approval into exports (D-043) |
| SEC-07 | DONE (mechanism) | S1.9 | Holds with two-person approval and release; one deletion service (soft delete, report, approval, purge with a hold re-check, verification, restore); retention dry runs only, and requests at most; T04 in backend and browser. Schedules: P-002/P-003 (counsel) |
| SEC-08 | PARTIAL (framework DONE) | S1.11 | Connector framework, local-folder connector, fail-closed permissions and freshness, sync and removal through the approved purge; T15 in backend and browser. Real provider BLOCKED (P-010) |
| RAG-01 | DONE for the pilot (lawyer review at S2.8) | S1.6–S1.7, S2.1–S2.2 | S2.1: OCR, Word structure, tracked changes and comments, hidden-text flags, clauses, defined terms, exact offsets, per-version status, reader. S2.2: answers from signed versions by default (switch to drafts), version conflicts, every quote verified, statuses with abstention reasons, paragraph streaming, Evidence and "Copy with citations"; golden 70 and held-out 32 evaluation (D-036–D-038). Firm-library scope deferred to S3.1–S3.2. S2.7: answers sent for review as research notes, approved by a second lawyer, filed (D-043) |
| RAG-02 | DONE for the pilot (lawyer review at S2.8) | S2.3 | Two versions clause by clause (renumbered, moved, added, removed, exact word diff, labelled changes); explanations quoting both wordings, verified; conflicts across documents between documents with the same parties; golden and held-out evaluation (D-039) |
| LIVE-01 | PARTIAL: built; real-audio accuracy pending the speech stack (🔒) | S2.4 | Verbatim immutable record, roster speakers, wall-clock times, gaps and degradation shown, revisions with provenance, matter vocabulary, finalised transcripts in research and purges, accuracy harness (D-040) |
| LIVE-02 | DONE for the pilot; thresholds await P-008 | S1.7, S2.5 | Three legal packs with evidence-relative alerts, exact quotes, confirm or dismiss with a reason, telemetry, matter-only sources (D-041). The voice service's research stream is unchanged (S1.7) |
| MEET-01 | DONE for the pilot | S1.8, S2.6 | Agenda and board-pack research; agenda marks; minutes from the whole transcript with verified sources, guards and deterministic merging; versioned edits; second-lawyer approval bound to the content hash; filing; purge and hold (D-042). Reviews queue and export hash binding: S2.7. S2.7: Reviews queue and shared review components |
| DOC-01 | DONE for the pilot | S3.1 | Firm templates (versions, approval by a managing partner, Word styles, presets); drafts with sourced fields and cited passages, gaps for anything unsupported; second-lawyer approval; filed as Word (D-045) |
| DOC-02 | PARTIAL: built; manual Word check 🔒 | S3.2–S3.3 | Playbooks (D-046); redlines and Word tracked changes, T10 automated checks and LibreOffice round trips pass (D-047). The manual Microsoft Word check by a person is pending |
| DOC-03 | DONE for the pilot (court rules not yet read) | S3.4 | References resolved to exact clauses or flagged missing; quotations compared word for word with diffs; external authorities listed and always UNVERIFIED (P-009), impossible ones flagged; per-viewer targets; T07 golden 55/55, held-out 27/28 (D-048) |
| DOC-04 | DONE for the pilot (undefined-term detection heuristic) | S3.5 | Definitions, numbering and references with offsets; corrections only through an approved redline; held-out 12/12 on the third blind set (D-049) |
| DOC-05 | DONE for the pilot (reminders not sent) | S3.6 | Rules and checked model proposals with exact words; dates only from the document; confirmed by a lawyer; versioned edits; held-out 10/12 with the model (D-050) |
| DOC-06 | DONE for the pilot | S3.7 | Approved checklists; data-room runs; missing list; quoted issues, preliminary until reviewed; held-out 9/11 and 5/6 issues, no extras (D-051) |
| DOC-07 | DONE for the pilot | S3.8 | Register against approvals and articles, signing limits, policies against articles, IP chain; both documents quoted; advisory (D-052) |
| OPT-01 | MISSING | S4.1 | Needs business approval |
| OPT-02 | PARTIAL (pre-existing) | S4.2 | MT model BLOCKED |
| OPT-03 | MISSING | S4.3 | Needs business approval |
| OPT-04 | BLOCKED | S4.4 | Licensed source plus network approval |
| OPT-05 | MISSING | S4.5 | Needs business approval |
| OPS-01 | PARTIAL | S1.0, S1.12 + OPS | S1.0: versioned migrations with down scripts + CLI; isolated compose stack. S1.12: encrypted backup and restore (`app.legal.backup`), rollback guard for audit, holds and source permissions, DR drill on real data (D-032). Runbooks and LUKS (host root) remain |
| OPS-02 | PARTIAL | S1.0, gates | S1.0: Law pytest suite, Vitest, committed law fixtures, law eval baseline 6/7. S1.12: `security_suite.sh`, `injection_probe.py`, `egress_test.sh`, `dr_drill.sh`, `legacy_smoke.py` + legacy browser smoke, `stability_runs.sh`. Load test (T12) and CI remain. S2.8: evaluation runs (migration 0022, import command, Evaluations page), `run_gate.py`, `load_test.py` (T12 smoke), lawyer review packet and scorer, Phase 2 security groups (D-044) |

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
- **Transcription accuracy is unmeasured on real audio (D-040).** The harness and scoring are built and tested. The measurement needs the Law speech stack (the first `compose.sh up`, 🔒), not an install on the shared host, whose memory headroom is about 13 GB with swap full. Speakers are marked by hand: there is no automatic speaker separation.
- **Minutes are small-sample evaluated (D-042).** Recall 1.0 golden and 6/7 held-out, with no invented specifics; the model chooses an item's kind within the guards, so an open point can be minuted as discussion. A second lawyer reviews every draft, and lawyer scoring is S2.8's.
- In Law the legacy Boardroom and document studio stay closed: matter meetings with minutes replace the Boardroom (S2.6), and drafting arrives in S3.x; live transcription needs the speech model, which is not on the dev host (the capture path is tested with a stand-in). This is intended (D-025, D-026).

## NEXT EXACT ACTION

Start **S3.9 Phase 3 gate** on `law/s3.9-gate`, per IMPLEMENTATION_PLAN §7: a planted-error suite, format round trips, and human sign-off. S3.8 is merged into `feature/echomind-law`.
1. **Fresh blind held-out sets,** written and committed before any run, for every Phase 3 checker changed since its last held-out run:
   - citations: court rules (Fed. R. Civ. P., CPR) not yet read; build that first, then the held-out set;
   - definitions s3.5c and undefined-term recall;
   - obligations s3.6g, with the model;
   - diligence s3.7c;
   - governance s3.8b.

   Report every run, including any that fail.
2. **Planted-error suite (`eval/law/run_gate.py --phase 3`):** one command that runs, in order:
   - the Phase 3 golden and held-out sets (drafting, playbook, redline T10, citations T07, definitions, obligations, diligence, governance);
   - the security suite;
   - the browser suite.

   It records each evaluation run (migration 0022) for the Evaluations page.
3. **Format round trips:**
   - drafting Word and PDF, redlines with tracked changes, filed corrections, and exports, opened by LibreOffice and checked (T10 and T09 already automated);
   - the manual Microsoft Word check stays 🔒.
4. **Human sign-off packet** (like S2.8's `lawyer_review`): fictional outputs of every Phase 3 feature for a lawyer to score, with a rubric. 🔒 It needs a person; prepare it, do not fake it.
5. **`releases/phase3-gate.md`:** what is done, the measured results (every held-out run as it went), what is 🔒, the risks, and the decision asked of the owner.
6. Then D-053, run 30; merge. Then Phase 4 (S4.1, S4.2 without MT, S4.3, S4.5; S4.4 blocked), as approved.
