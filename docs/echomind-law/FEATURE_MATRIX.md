# EchoMind Law — Feature Matrix (all backlog IDs)

> Baseline commit `95782e4`. **Status:** `MISSING` · `PARTIAL` (a reusable piece exists) · `INCOMPATIBLE` (existing behaviour must change) · `BLOCKED` (external input needed).
> Evidence references are explained in [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md). Slices are defined in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Tests T01–T15 come from spec §19. "L-xx" tests are new Law tests defined in the plan.

## Summary

| ID | Feature | Status today | Slice(s) | Blocked by |
|---|---|---|---|---|
| SEC-01 | Mandatory auth, SSO/OIDC, sessions, MFA | PARTIAL | S1.1 | OIDC verification needs the firm IdP (P-011) |
| SEC-02 | Tenant/client/matter CRUD, RBAC+ABAC, ethical walls | DONE (S1.2–S1.5; this row was stale until S1.12) | S1.2, S1.3, S1.4 | — |
| SEC-03 | Matter-aware indexes, ACL before retrieval and jobs | DONE (S1.6 documents and partitions; S1.7 existing modules closed, research and voice matter-scoped) | S1.6, S1.7 | — |
| SEC-04 | Audit, admin reports, SIEM export | DONE (forwarder awaits network approval) | S1.5 | — |
| SEC-05 | Consent, pause, revoke, provenance | DONE (mechanism; P-001 wording and rules await counsel) | S1.8 | Policy text (P-001) |
| SEC-06 | Privilege labels, redaction preview, safe exports | DONE (labels S1.9, export gateway S1.10; P-006 values await counsel). S2.7 (D-043): an export is bound to the text its redactions were planned on and refused if the source changed before approval; a document filed from an approved AI draft carries the approval, and a changed copy is refused | S1.9, S1.10 | — |
| SEC-07 | Retention, legal hold, approved deletion, restore | DONE (mechanism; schedules await counsel) | S1.9 | Schedules (P-002/P-003) |
| SEC-08 | Firm DMS adapter with ACL/version sync | PARTIAL (framework DONE with the local-folder connector; real provider BLOCKED) | S1.11 | **Provider and credentials (P-010)** |
| RAG-01 | Matter-scoped legal chat, clause ingestion, OCR, versions | DONE for the pilot, pending the S2.8 lawyer review. S2.1 ingestion: OCR, Word structure, tracked changes and comments as metadata, hidden-text and injection flags, clauses, defined terms, exact offsets with page/line, per-version status, reader. S2.2 research: evidence contract with verified quotes, signed-version default with a switch, version conflicts, abstention with reasons, paragraph streaming, "Copy with citations", golden 70 and held-out 32 evaluation. Firm-library scope deferred to S3.1–S3.2, when the library gets content (D-038).. S2.7 (D-043): an answer can be sent for review as a fixed research note, approved by a second lawyer for its exact content and filed with the approval | S2.1, S2.2, S2.8 | Lawyer review (S2.8, P-008) |
| RAG-02 | Document comparison and contradiction analysis | DONE for the pilot (S2.3), pending the S2.8 lawyer review. Two versions clause by clause (renumbered, moved, added, removed; exact word diff; labelled changes), explanations quoting both wordings (verified), conflicts across documents (amounts, periods, dates, governing law, company names, agreement references; same parties only; checked in context), golden and held-out evaluation (D-039) | S2.3, S2.8 | Lawyer review (S2.8, P-008) |
| LIVE-01 | Legal transcription: speakers, vocabulary, edits | PARTIAL: built (S2.4); real-audio accuracy not yet measured. Verbatim record (immutable segments of whole sentences), roster speakers marked live, wall-clock times, visible gaps and degradation, corrections as revisions, matter-vocabulary suggestions with provenance, finalised transcripts as research sources, accuracy harness (D-040). The word error rate on real audio needs the Law speech stack (first `compose.sh up`, 🔒) | S2.4, S2.8 | Speech stack start (🔒); voice licence check |
| LIVE-02 | Negotiation / client-call / governance Silent Assist packs | DONE for the pilot (S2.5), thresholds awaiting a lawyer's sign-off (P-008). Three packs: signed terms, authority, commitment, disclosure, action, quorum, conflict. Evidence-relative alerts with exact quotes, confirm or dismiss with a reason, telemetry, matter-only sources, golden and held-out evaluation (D-041) | S2.5, S2.8 | Thresholds sign-off (P-008) |
| MEET-01 | Boardroom legal mode: before, during, after, minutes | DONE for the pilot (S2.6). Agenda items with their board papers and research limited to them; agenda marked while recording; minutes from the **whole** corrected transcript with times, roster speakers and agenda items from the record; every item cites verified quotes; kinds and owners checked against the record; versioned edits; approval by a second lawyer bound to the content hash; only approved minutes filed as a document; purge and hold; golden and held-out evaluation (D-042). The firm-wide Reviews queue and export hash binding are S2.7. S2.7: minutes appear in the firm Reviews queue, and use the shared review components | S2.6, S2.7 | — |
| DOC-01 | Legal templates and precedent drafting, DOCX/PDF | DONE for the pilot (S3.1, D-045). Firm templates in a small markup, versioned and approved by a managing partner (not the author), with firm Word styles; six standard presets; fields from the matter record, checked quotes or a lawyer, never the model alone; written passages citing their sources, any unsupported sentence or figure a gap; approval only with no gap left; filed as Word with the approval; golden and held-out evaluation with no wrong values and no invented figures | S3.1 | Firm templates (sample set used until provided) |
| DOC-02 | Clause library, playbooks, compare, Word redlines | PARTIAL: built and tested (S3.2 D-046, S3.3 D-047); the manual Microsoft Word check is pending (🔒). Playbooks with checks; redlines to the firm's positions (model fits defined terms only, checked); Word files with real tracked changes and comments, checked before release (T10: 487/487 revisions verified, LibreOffice round trips kept) | S3.2, S3.3 | Word manual verification (D-011, P-013) |
| DOC-03 | Quote and citation checker with authority provenance | DONE for the pilot (S3.4, D-048): references resolved to exact clauses or flagged missing; quotations compared word for word (altered words shown); every external authority listed as UNVERIFIED, impossible ones flagged; nothing looked up. T07 golden 55/55, held-out 27/28 (court rules not yet read) | S3.4 | Verifying that an authority exists needs a licensed source (P-009) |
| DOC-04 | Defined terms, numbering, cross-references | MISSING | S3.5 | — |
| DOC-05 | Obligations and deadlines tracker | MISSING | S3.6 | — |
| DOC-06 | Due diligence checklist and issue register | MISSING | S3.7 | — |
| DOC-07 | Governance, cap table / approvals, IP assignment checks | MISSING | S3.8 | — |
| OPT-01 | Client intake assistant with conflict screening | MISSING | S4.1 | Business approval |
| OPT-02 | Transcript summaries and local translation | PARTIAL (summary only) | S4.2 | Local MT model licence |
| OPT-03 | Draft time entries | MISSING | S4.3 | Business approval |
| OPT-04 | Regulatory monitoring via an approved source | MISSING | S4.4 | **Licensed source and network approval** |
| OPT-05 | Four-perspective review panel | MISSING | S4.5 | Business approval, benchmark |
| OPS-01 | Migrations, rollback, encrypted backups, runbooks | PARTIAL (S1.12: encrypted backup and restore, rollback guard, DR drill; runbooks and LUKS remain) | S1.0, S1.12 + OPS track | LUKS on target host (D-014) |
| OPS-02 | Golden, legal, security, load and recovery test harness | PARTIAL (S1.12: security suite, injection probe, egress capture, DR drill, legacy smoke, stability runs. S2.8 (D-044): evaluation runs recorded with commit, models and dataset hash, one gate runner for every suite, the T12 load-test script (smoke run; the measured run needs a window, 🔒), the lawyer review rubric, packet and scorer. CI remains) | S1.0, S1.12, S2.8, S3.9 + OPS track | Load window approval (D-002) |

### Spec section mapping
- F1 → SEC-02. F2 → SEC-01 + SEC-04. F3 → SEC-05 + SEC-06 (labels) + SEC-07. F4 → SEC-06. F5 → SEC-08.
- P1 → RAG-01 (+ RAG-02). P2 → LIVE-01 + LIVE-02. P3 → MEET-01.
- D1 → DOC-01. D2 → DOC-02. D3 → DOC-03. D4 → DOC-04. D5 → DOC-05. D6 → DOC-06 + DOC-07.
- Phase 4 → OPT-01…05.

---

## Feature cards

### SEC-01 — Mandatory authentication, SSO/OIDC, secure sessions, MFA
| Field | Detail |
|---|---|
| Status | **PARTIAL.** Local users (PBKDF2 200k), HS256 JWT cookie, `admin`/`user` roles, logout denylist. Auth is off by default; no MFA, SSO, lockout, `Secure` flag or WS-wide enforcement. |
| Evidence | `backend/app/core/auth.py:24-251`, `routes/auth.py:32,63-109`, `main.py:158-168`, `voice/app/auth_check.py:29-55`; voice→backend sends no credential (`voice/app/session.py:1524-1530`) |
| Modifications | Edition guard (D-003); server-side sessions (D-006); `__Host-` Secure Strict cookie plus CSRF token; TOTP MFA and recovery codes (D-007); lockout and back-off; password policy; disable/enable users; session list and revoke; WS session validation and periodic re-check on every WS (transcribe, voice); on-behalf-of tokens (D-013); OIDC client behind a flag; admin bootstrap no longer logs passwords; `/docs` and `/openapi.json` off in Law |
| Dependencies | S1.0 (migration runner, edition flag) |
| Security risks | Session fixation, CSRF, lockout DoS, MFA bypass via recovery codes, token replay between services |
| Tests | L-AUTH-01..12: anonymous denied on every Law route and WS (generated from the route table); expired, idle and revoked sessions denied; revocation on role change; MFA required; lockout; CSRF; voice on-behalf-of token audience/expiry; startup refusal when posture is insecure |
| Acceptance | Anonymous Law APIs, WebSockets and downloads denied; token expiry and revocation verified; OIDC marked BLOCKED until tested against the firm IdP |

### SEC-02 — Tenant, client, matter CRUD; memberships; RBAC + ABAC; ethical walls
| Field | Detail |
|---|---|
| Status | **MISSING.** Only a per-user `tenant` string exists, used in chat and upload. |
| Evidence | `core/auth.py:147-166`, `chat.py:27-35`, `docs.py:146-150`; no client or matter tables (`core/db.py`) |
| Modifications | Tables `organizations, clients, matters, matter_memberships, ethical_walls, wall_entries, wall_exceptions, break_glass_events, user_roles`; policy engine `backend/app/legal/policy.py` (deny precedence); `/api/legal/clients`, `/matters`, `/memberships`, `/walls`, `/break-glass`; non-disclosing 404s; Law UI (matter list, overview, members, walls admin) |
| Dependencies | S1.0, S1.1 |
| Security risks | IDOR, existence leaks, admin bypass, wall misconfiguration, time-bounded membership errors |
| Tests | Policy unit matrix (role × membership × wall × sensitivity × time); API contract tests; **T01** fixture (2 clients, 4 matters, canaries, revoked membership); timing-shape test |
| Acceptance | Restricted matter existence and contents hidden in lists, histories, search, snippets and artifacts; walls deny even firm admins without break-glass |

### SEC-03 — Matter-aware indexes; ACL before FAISS/BM25 and async jobs
| Field | Detail |
|---|---|
| Status | **INCOMPATIBLE.** Global index with post-filter; several unfiltered paths; whole-KB fallback. |
| Evidence | `rag/index.py:36-68, 872, 1000-1083`; `advanced.py:3006-3009`; `silent_assistant/retrieval.py:48-54`; `chat.py:355-377`; `boardroom/service.py:792`; docgen `kb`/`chat` modes |
| Modifications | `PartitionIndex` (dense + BM25 + meta per partition) refactored from the module globals in `rag/index.py`/`rag/sparse.py`; `AuthorizedScope` capability required by `retrieve_*`; Law ingestion writes to `m:<matter>`/`lib:<org>`; auxiliary global indexes disabled for Law; LRU partition loader; `acl_epoch`; job framework storing the principal and re-checking at output |
| Dependencies | S1.2 policy engine |
| Security risks | Missed code path, cache keyed without scope, job running after revocation |
| Tests | **Canary leakage suite** (unique tokens planted per matter; every retrieval mode: dense, sparse, keyword, rerank, evidence, citations, voice, Silent Assistant, Boardroom, docgen); **T02** revocation; **T06** abstain even when a forbidden matter has the answer; static test that `retrieve_*` cannot be called without a scope in Law |
| Acceptance | Zero canary hits outside scope across all modes; revocation effective on the next request or chunk, and within 60 s for open WS |

### SEC-04 — Auditing, admin reports, SIEM-capable export
| Field | Detail |
|---|---|
| Status | **PARTIAL.** `activity_log` records mutations only; mutable, best-effort; logins appear as anonymous. |
| Evidence | `core/audit.py`, `main.py:175-191`; `GET /api/auth/audit` returns the last 150 |
| Modifications | `audit_events` hash chain with no-update/delete triggers and daily anchors (D-008); typed audit API `audit.record(action, object, matter, outcome, details)` called from the policy engine and services, including **reads**; admin viewer with filters; matter-scoped audit for leads; CSV/JSONL export plus optional syslog/CEF forwarder (off by default; needs network approval); chain verifier CLI; content-free log formatter |
| Dependencies | S1.0, S1.2 |
| Security risks | Audit write failure should fail closed for sensitive actions; PII in audit details; admin tampering |
| Tests | Event completeness matrix (each action emits exactly one event); tamper test (manual SQL update aborts; out-of-band edit detected by the verifier); no planted secret or canary text in audit or logs; retention of audit |
| Acceptance | Prompt metadata, access, export, approvals, connector sync and policy changes are traceable without sensitive spill |

### SEC-05 — Recording/transcription consent, pause, revoke, provenance
| Field | Detail |
|---|---|
| Status | **MISSING.** No consent concept; capture starts on WS `start` (`transcribe/ws.py:685`), Boardroom chunk upload, or voice connect. |
| Evidence | grep `consent` returns nothing; `ws.py:685-755`; `boardroom/service.py:224`; `voice/app/server.py:172-215` |
| Modifications | `meetings, meeting_participants, consent_events (append-only), capture_sessions`; consent policy config (P-001); a server gate in WS `start`, chunk upload and voice handshake; pause/resume/withdraw messages; a recording indicator; consent snapshot hash stored on the transcript; audit |
| Dependencies | S1.2, S1.5 |
| Security risks | Race between consent and start; client-side-only enforcement; replayed consent |
| Tests | **T03**: missing, withdrawn and stale consent refuse start; withdrawal mid-stream stops capture and rejects frames; pause gap recorded; timeline and audit correct |
| Acceptance | Backend refuses capture without consent; timeline and audit correct |

### SEC-06 — Privilege/confidentiality labels, redaction preview, safe offline exports
| Field | Detail |
|---|---|
| Status | **PARTIAL.** Regex PII scanner (advisory, not wired to exports); PDF/PPTX renderers exist; no labels; file route is publicly cacheable. |
| Evidence | `core/export_gateway.py`, `routes/export.py`, `docs.py:217-218`, `docgen/render_pdf.py:808-830` |
| Modifications | Label fields on documents, versions, transcripts and artifacts (advisory, user-editable, audited); privilege detector suggestions; export pipeline preview → redaction plan (offset-based) → approve → render → **verify** → watermark → audit; DOCX/PDF sanitization (metadata, comments, revisions, hidden text); `Cache-Control: no-store`; extend detectors (names via matter party list, matter numbers, account numbers, privilege markers) |
| Dependencies | S1.2, S1.5, S1.9 (labels), S3.1 (DOCX renderer for DOCX exports) |
| Security risks | Residual text under redaction boxes, metadata, cached copies |
| Tests | **T09**: extract text and metadata from the exported PDF/DOCX, assert no redacted string remains (including `docProps`, comments, revisions, alt text); unauthorized export denied; watermark present |
| Acceptance | Human approval; redaction verification; no hidden residual data |

### SEC-07 — Retention, legal hold, approved deletion, restore
| Field | Detail |
|---|---|
| Status | **MISSING.** Every delete is a hard delete; `delete-all` is open to any user; transcript delete orphans analysis rows. |
| Evidence | `docs.py:277` (`delete-all`), `transcribe.py:213`, `boardroom.py:73`, `docgen.py:209` |
| Modifications | `retention_policies, legal_holds, hold_scopes, deletion_requests`; one `DeletionService` used by every delete path (soft-delete → dry-run report → approval → purge → verification, reconciling chunks, partitions, analyses, exports and audio); hold check inside the purge transaction; scheduled retention runner (dry-run until a policy is approved); restore from soft-delete; `delete-all` disabled in Law |
| Dependencies | S1.2, S1.5 |
| Security risks | A bypass path deleting held data; derived artifacts surviving purge; backups retaining purged data (documented) |
| Tests | **T04** (user, timer, retention worker, system cleanup and direct service calls all refuse held items); purge verification (no chunk, index or file residue); dry-run accuracy; restore |
| Acceptance | Held data survives every cleanup path; deletion dry run and recovery tested |

### SEC-08 — Firm-configured DMS adapter
| Field | Detail |
|---|---|
| Status | **MISSING.** No connector of any kind. |
| Modifications | `DmsConnector` interface (list/search, fetch version, changes since, ACL fetch, revoke, health); `integration_connections, connector_acl_snapshots, connector_sync_runs`; a **real** `LocalFolderConnector` (watch folder or manual import, which is the offline default); sync engine (rate limit, retry, fail closed on stale ACL); UI for connection status and failure states |
| Dependencies | S1.6, S1.9 |
| Blocked | **Provider choice and credentials (P-010).** No mock connector will be presented as working. |
| Tests | T15 with LocalFolderConnector (ACL intersection via sidecar ACL file, stale → deny, removal → purge from partition); provider tests once available |
| Acceptance | Real provider pilot only with valid credentials and network approval; otherwise BLOCKED |

### RAG-01 — Legal matter-scoped chat
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S2.1–S2.2), pending lawyer review at S2.8.** Law research no longer uses `rag/advanced.py`. It has its own ingestion (D-034, D-035), versions (D-036), evidence contract (D-037) and calibration (D-038). R-4, R-6, R-7 and R-8 are tested guarantees in Law. Measured (held-out): answers 27/29, abstentions 1/3, 0 leaks, citation precision 0.902, quote verification 0.905, p95 6.19 s. Not built: the firm-library scope, deferred to S3.1–S3.2 (D-038). *Pre-S2 audit:* strong hybrid RAG, reranker, gate and citations, but no OCR, DOCX structure, exact offsets, versions, claim verification or abstention contract. Defects R-1..R-8. |
| Evidence | `rag/advanced.py:1510-1722, 2305-2335, 2689-2769, 3394-3488`; `rag/parse.py:273-275`; `sanitize.py:12-19`; `chunking/pipeline.py:324-332` |
| Modifications | Legal ingestion (S2.1): DOCX numbering, tables and footnotes; OCR with provenance; char and page/line offsets; clause segmentation; defined-terms capture; versions with status. Chat (S2.2): `/api/legal/matters/{id}/research/query`; evidence contract; verbatim quote verification; abstention "Not found in this matter's documents"; signed-only vs all versions; conflict indicator; keep pinpoints; fix off-by-one; short legal terms routed to retrieval; firm-library ∪ matter scope; persona `Lawyer` fixed server-side |
| Dependencies | S1.6 |
| Security risks | Injection, fabricated citations, version confusion |
| Tests | Legal golden set (≥60 items) plus a held-out set (≥30); **T05, T06, T08**; citation precision; abstention accuracy; latency |
| Acceptance | Source/version/page/section citation, abstention on unsupported questions, and signed-vs-draft handling, measured on the held-out set with lawyer review |

### RAG-02 — Document comparison and contradiction analysis
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S2.3), pending lawyer review at S2.8.** `legal/compare/`: alignment by wording and meaning with exact offsets; explanations with verified quotes; a conflict scan across the matter's documents (D-039). Measured on the real model, held-out: alignment 10/10, explanation quotes 1.0 verified, scan 4/5 planted with precision 1.0. *Pre-S2 audit:* a comparison reference block existed in prompt assembly, but there was no compare feature. |
| Modifications | Clause alignment (numbering + embedding similarity) → per-clause diff → LLM explanation with both quotes; cross-document conflict scan (numbers, dates, parties, obligations); compare UI (side by side) |
| Dependencies | S2.1 |
| Tests | Two fixture contracts with seeded differences give cited differences; conflicting sources are visible |
| Acceptance | Two contracts produce cited differences; conflicting sources visible |

### LIVE-01 — Legal transcription
| Field | Detail |
|---|---|
| Status | **PARTIAL: built in S2.4 (D-040); accuracy on real audio pending the speech stack.** Verbatim immutable record, roster speakers, wall-clock times, gaps and degradation shown, revisions with provenance, matter vocabulary, finalised transcripts in research, accuracy harness (`eval/law/transcription/`). *Pre-S2 audit:* Nemotron streaming, two-role manual toggle, lossy normalization (legacy path), ASR-clock timestamps, no edits or provenance. |
| Evidence | `transcribe/ws.py`, `session_state.py:404-523`, `RoleToggle.tsx`, `types.ts:144` |
| Modifications | Verbatim mode (D-017); N-role participants from the meeting roster; wall-clock anchors; matter-vocabulary post-correction with provenance (names, entities from matter documents); `transcript_revisions`; pause/resume; WER harness with legal names, dates, numbers and negations |
| Dependencies | S1.8 |
| Tests | WER and entity accuracy on fixture audio (TTS-generated plus public AMI subset, licence permitting); revision provenance; pause gaps |
| Acceptance | Names, numbers, negations, speaker error and revision provenance benchmarked |

### LIVE-02 — Legal Silent Assist rule packs
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S2.5), thresholds pending P-008.** `legal/assist/`: three packs, evidence-relative alerts quoting the exact words said and the exact passage, model checks only where context matters, confirm or dismiss with a reason, telemetry, matter-only sources with no fallback (D-041). Held-out: recall 6/6, precision 1.0 on a small set. *Pre-S2 audit:* `LEGAL` profile with 2 regex rules, verifier with verbatim-proof rules, whole-KB fallback. |
| Evidence | `silent_assistant/profiles.py:140-167`, `verifier.py:439-491` |
| Modifications | Packs `legal_client_call`, `legal_negotiation`, `legal_governance`: rules for inconsistency with the signed agreement, unauthorized commitments, sensitive disclosure, unresolved actions, quorum/conflict/authority; evidence-relative wording (D-015); conservative thresholds; dismiss/confirm with reason; telemetry (alerts/hour, dismiss rate, latency); remove the whole-KB fallback |
| Dependencies | S1.6, S1.7, S2.4 |
| Tests | Mock-call scripts with planted issues: precision, false alerts per hour, missed critical issues, latency |
| Acceptance | Alerts carry a cited source, threshold and dismissal; false alarms measured |

### MEET-01 — Boardroom legal mode
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S2.6, D-042).** Legacy (closed in Law): chunked recording, diarization, report (first 6k characters only), PDF/PPTX export, with no consent, approval, retention, namespace or timestamps in checks. Law: consent-gated capture (S1.8), the S2.4 record, Silent Assist (S2.5), agenda and board pack, minutes from the full transcript with verified sources, second-lawyer approval bound to the content hash, filing, purge and hold. |
| Evidence | `boardroom/service.py:224, 548-622, 776-886, 792, 806, 863` |
| Modifications | Before: board pack Q&A (matter-scoped RAG), agenda and prior minutes. During: consent gate, pause, optional advisory flags. After: map-reduce over the **full** transcript; draft minutes, resolutions, actions (owner and deadline), unresolved items; source-backed review page; approval by attorney or secretary before export; chunk hashing (chain of custody); retention and hold |
| Dependencies | S1.8, S1.9, S2.5, S2.7 |
| Tests | Fixture board meeting: minutes reference timestamps; approval required before export; hold respected |
| Acceptance | Consent, advisory flags, reviewer approval, holds and retention all enforced |

### DOC-01 — Legal templates and precedent drafting
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S3.1, D-045).** Legacy (closed in Law): `contract_review` and `legal_brief` templates; PDF/PPTX only; "plausible specifics" instruction; no provenance. Law: firm templates with versions, approvals and Word styles; presets; sourced fields; no-invention writer with gaps; review, approval and Word filing. |
| Evidence | `docgen/templates.py:48-59, 238-260`; `docgen/pipeline.py:464-466` |
| Modifications | Firm template registry with versions and approvals; legal presets (engagement letter, NDA, demand letter, internal memo, contract review report, board minutes); field extraction with source provenance; legal writer prompt that **forbids invention** and marks gaps `[[NEEDS INPUT: …]]`; DOCX renderer; review and approval |
| Dependencies | S2.2, S2.7 |
| Tests | Template version recorded; every factual field traced to a source or marked as needing input; DOCX/PDF open and round-trip |
| Acceptance | Approved template version, sourced fields, review, and DOCX/PDF export verified |

### DOC-02 — Clause library, playbooks, comparison, Word redlines
| Field | Detail |
|---|---|
| Status | **PARTIAL.** S3.2 (D-046): clause library and playbook checks built. Legacy: no tracked-changes capability (grep confirms); redlines and Word tracked changes are S3.3. |
| Modifications | `clause_playbooks, clause_versions` (approved standard, fallbacks, owner, jurisdiction); deviation analysis; suggested revision with risk label; HTML redline; **DOCX tracked changes and comments spike** (D-011) |
| Dependencies | S2.1, S3.1 |
| Tests | **T10** (true `w:ins`/`w:del`/comments, accept-all equals target, LibreOffice round trip, manual Word open) |
| Acceptance | Actual DOCX tracked changes and comments if promised; otherwise flagged not delivered |

### DOC-03 — Quote and citation checker
| Field | Detail |
|---|---|
| Status | **DONE for the pilot (S3.4, D-048).**<ul><li>Rules only; no model.</li><li>Internal references are resolved to the exact clause of this or another matter document the person can open, or flagged missing.</li><li>Quotations are compared word for word, and altered words shown.</li><li>External authorities (US Code, CFR, US cases, UK neutral citations, reports and Acts, EU instruments, CJEU) are listed and always UNVERIFIED.</li><li>Citations that cannot be real are flagged: unknown reporter, non-existent circuit, future year, a court before it sat.</li><li>A well-formed invented case cannot be told from a real one without a licensed source (P-009). It is shown UNVERIFIED like every other authority, never as checked.</li><li>Court rules (Fed. R. Civ. P., CPR) are not yet read.</li></ul> |
| Modifications | Citation parser (internal refs: §, clause, exhibit, schedule; external: U.S.C., C.F.R., reporters, neutral citations); quote extractor; resolver to exact passages in the matter; external authority marked **UNVERIFIED** unless a licensed source is configured (P-009); report fabricated, missing, unsupported and altered quotes; no replacement authority generated |
| Tests | **T07** with planted fictional authorities and altered quotes |
| Acceptance | Planted fictional authorities flagged; no unlicensed external lookup claimed |

### DOC-04 — Defined terms and cross-references
| Field | Detail |
|---|---|
| Status | **MISSING.** The DoD-shaped cross-ref graph is not reusable for contracts. |
| Modifications | Definition detector (quoted capitalized terms, "means"); usage index; undefined, unused, duplicate and inconsistent findings; numbering validator; broken cross-references; outdated-version references; offsets and proposed fixes |
| Tests | Planted errors detected with offsets |
| Acceptance | Planted broken references and unused terms detected with offsets |

### DOC-05 — Obligations and deadlines tracker
| Field | Detail |
|---|---|
| Status | **MISSING.** The Silent Assistant extracts action items in calls only. |
| Modifications | Extraction (notice, renewal, termination, consent rights, change-of-control, payment) with evidence spans; `legal_obligations` with status, owner, timezone and `confirmed_by`; never displayed as verified until a human confirms; reminders and ICS export only when enabled |
| Tests | Fixture contracts give expected obligations with evidence spans; edits audited; timezone correct |
| Acceptance | Human-confirmed deadlines, cited triggers, timezone, status and edits |

### DOC-06 — Due diligence
| Field | Detail |
|---|---|
| Status | **MISSING.** |
| Modifications | Checklist templates; matter-scoped data-room scan as a background job; missing-document list; issue register with sources and reviewer status; preliminary findings separated from verified ones |
| Tests | Synthetic data room with seeded issues and missing documents |
| Acceptance | Missing-document and seeded-issue tests with provenance and review trail |

### DOC-07 — Corporate governance checks
| Field | Detail |
|---|---|
| Status | **MISSING.** |
| Modifications | Cap table vs approvals, delegation matrix, policy vs bylaws, IP assignment chain; fixture company; findings advisory only |
| Tests | Fictional company fixtures show the expected discrepancies |
| Acceptance | Discrepancies shown; no automatic legal decision |

### OPT-01 … OPT-05 (Phase 4, each behind its own flag and needing separate business approval)
| ID | Status | Key constraints | Acceptance |
|---|---|---|---|
| OPT-01 Intake | MISSING | Scripted administrative intake only; disclaimer; conflict-name screening with minimization; no recording without consent; lawyer approves replies | Administrative only, reviewed replies, consent |
| OPT-02 Summaries / translation | PARTIAL (Boardroom summary) | Page/line/timecode citations; original preserved; **local MT model licence needed** (BLOCKED until chosen) | Citations, reviewer edits, original preserved |
| OPT-03 Time entries | MISSING | Suggestions from approved activities; user must edit and approve; no billing integration | User verification enforced |
| OPT-04 Regulatory monitoring | MISSING / **BLOCKED** | Needs a licensed source and an approved channel; off in air-gap mode | Off without an approved source |
| OPT-05 Review panel | MISSING | Four roles plus moderator; independent citations; disagreement shown; stops on missing facts; partner approval; background queue; not marketed as independent experts | Disagreement and unknowns shown; audit and performance evidence |

### OPS-01 — Migrations, rollback, encrypted backups, runbooks
| Field | Detail |
|---|---|
| Status | **PARTIAL.** Backup and restore scripts (unencrypted; restore skips the consistent snapshot); no migration versioning; stale offline docs; prepare file defaults to the wrong model. |
| Modifications | Versioned migrations with down scripts (S1.0); encrypted backups (D-014) including the partitions directory and audit anchors; restore uses the consistent snapshot; upgrade and rollback runbook; Law compose (D-002); offline prepare/export covering every model; SSO and connector runbooks |
| Tests | **T13** (upgrade, rollback, encrypted restore on seeded data; holds preserved) |
| Acceptance | Upgrade and rollback of a real seeded installation exercised |

### OPS-02 — Test harness
| Field | Detail |
|---|---|
| Status | **PARTIAL.** 50 backend and 131 voice unit tests, `tsc` in CI, golden eval (not in CI, law corpus missing). |
| Modifications | Law fixtures committed (fictional corpus, canaries, audio scripts); pytest Law suites; Vitest and Playwright; legal golden and held-out sets; security suite; network-denied E2E; load and recovery scripts; release-gate report generator recording model and dataset versions |
| Tests | — |
| Acceptance | Reproducible tests with model and dataset versions and a signed release-gate report |
