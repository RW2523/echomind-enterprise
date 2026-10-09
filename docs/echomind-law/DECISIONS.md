# EchoMind Law — Decision Log

> **2026-10-08 approval.** The owner replied *"approved, create the branch and start S1.0"* to the six-item approval list: D-001, D-002, D-003, D-004, D-019, and committing the plan documents. Those entries are marked APPROVED. All other D-entries stay PROPOSED until their slice, and the plan as a whole is the working baseline.

> **2026-10-08 standing approval.** After approving the S1.3 design and merge, the owner wrote *"keep building one by one and build everything fully and tell me"*. This is read as standing approval to:
> - build the slices in order;
> - merge each completed, fully tested slice into `feature/echomind-law` without asking again;
> - add the dependencies already listed in LICENSE_REGISTER §2.
>
> It does **not** cover: pushing to GitHub or opening PRs; starting the compose stack on the shared host; anything touching production; network integrations (DMS, IdP, licensed sources); or counsel's policy decisions (P-items). Those are still asked for explicitly.

**Status values**
- **PROPOSED**: Claude's engineering recommendation. Not yet approved; it needs the product owner's approval before the slice that depends on it starts.
- **APPROVED**: confirmed by the product owner, with date.
- **NEEDS COUNSEL**: a legal or policy decision. Engineering builds configurable mechanisms with safe defaults and **does not** choose the policy.
- **BLOCKED**: cannot proceed without an external input (a credential, licence or provider choice).

Every entry records the options considered, the recommendation, the consequence and how to reverse it.

---

## A. Engineering decisions

### D-001 Branch and repository strategy — APPROVED 2026-10-08
- **Context:** The spec says to modify the existing app in place on a feature branch and never build a greenfield replacement. The product owner described "a new EchoMind for Law Firm, same base, built differently".
- **Options:** (a) branch `feature/echomind-law` in `RW2523/echomind-enterprise`; (b) a new GitHub repo forked from `main`; (c) a long-lived `law` branch plus per-slice feature branches.
- **Recommendation:** **(c)**, built on (a).
  - Create `feature/echomind-law` from `main@95782e4` as the integration branch.
  - Each slice ships as a short-lived branch `law/<slice-id>-<name>`, opened as a PR into `feature/echomind-law`.
  - Rebase periodically on `main` so production fixes flow in.
  - If the product later needs its own repo, fork from this branch with no rework.
- **Consequence:** Shared fixes (e.g. DB-1/DB-2) can also be cherry-picked to `main` for production.
- **Reversal:** Delete the branch. Nothing touches `main` until the owner chooses to merge.

### D-002 Development runtime on the shared DGX Spark — APPROVED 2026-10-08
- **Context:**
  - Production runs on this same host. Its compose project is named `echomind-enterprise`, which matches this clone's folder name.
  - Containers have fixed names and ports.
  - Memory is about 107/121 GB used.
- **Decision:**
  1. **Never run `docker compose up` from this clone without the Law overrides.** Add `docker-compose.law.yml` and `.env.law`. These set `COMPOSE_PROJECT_NAME=echomind-law` and give every service its own container name (`echomind-law-*`), host ports (frontend 3110/3553; no LLM or embed ports) and data volume (`echomind_law_data`). They omit `trtllm`, `ollama` and `cloudflared`, and join production's network (`echomind-enterprise_default`) as `external` so the Law services can reach `trtllm:8355` and `ollama:11434`. Compose v5.0.2 supports `!reset`/`!override` for ports.
  2. **Day-to-day development runs natively.** The backend runs on the host with uvicorn (host Python already has every dependency except NeMo). It uses `ECHOMIND_DATA_DIR=<clone>/.law-data` (gitignored), `LLM_BASE_URL=http://localhost:8355/v1` and `OLLAMA_EMBED_URL=http://localhost:11435/api/embeddings`. The frontend runs on Vite dev at :5174. ASR features are disabled natively.
  3. **ASR, voice and Boardroom integration tests** run in a container started from the existing `echomind-enterprise-backend:1.4.1` image, with the code bind-mounted. Run them only after checking free memory, and with `DOCGEN_IMAGE_BACKEND=none`.
  4. **Production is never touched:** no restart, no volume access, no writes. Read-only `docker ps`/`inspect` is allowed.
- **Consequence:** Law development shares LLM throughput with production. This is acceptable at development volume. Load tests (T12) must be scheduled outside production hours or run on separate hardware, and need approval.
- **Reversal:** Stop the `echomind-law` project and delete `.law-data`/`echomind_law_data`.

### D-003 Edition flag and secure-by-default Law posture — APPROVED 2026-10-08
- **Decision:**
  - Add `ECHOMIND_EDITION=law` on the backend and `VITE_EDITION=law` on the frontend.
  - When the edition is `law`, the backend **refuses to start** unless every one of these holds:
    - `AUTH_ENABLED=1`
    - `AUTH_SECRET` is set (≥32 bytes)
    - `CORS_ALLOW_ORIGINS` is an explicit list
    - `VOICE_AUTH_ENABLED=1`
    - no `nim` image backend
    - debug and eval taps are off: `ECHOMIND_LOG_FULL_PROMPTS`, `LLM_LOG_PAYLOAD`, `/tmp/dump_payload`, `VOICE_ABLATION_MODE`
  - The `public` (cloudflared) profile is not part of the Law compose file.
  - Other verticals keep their current behaviour when the flag is unset.
- **Reversal:** Unset the flag. This only works for non-law deployments; a Law deployment cannot run with the flag off.

### D-004 Database: SQLite now, Postgres at a defined gate — APPROVED 2026-10-08
- **Context:** The spec warns about SQLite concurrency and integrity. The code is raw `sqlite3` SQL across about 40 call sites.
- **Options:** (a) migrate everything to Postgres now, which is a large cross-cutting rewrite and against the "no wholesale rewrite" rule; (b) keep SQLite (WAL) for Phases 1–2 and add versioned migrations, a repository layer for all *new* legal tables, append-only triggers and integrity checks.
- **Recommendation:** **(b)**, plus a hard **switch gate**: move to Postgres (with Row-Level Security as a second ethical-wall layer) before production if any of these is true:
  - more than 25 concurrent users are expected;
  - T12 shows `SQLITE_BUSY` or p95 write latency above 200 ms;
  - the firm requires DB-native RLS.
- **Migration framework:** a small versioned runner (`backend/app/legal/migrations/NNNN_*.sql`) with a `schema_migrations(version, checksum, applied_at)` table, run at boot. Every migration is additive with a paired `down` script. Alembic is avoided because there is no ORM.

### D-005 Retrieval isolation: per-matter index partitions — IMPLEMENTED (S1.6, 2026-10-08; option (a), part of the approved plan)
- **Context:** One global FAISS/BM25 index with a post-search namespace filter fails the spec ("post-filter alone is insufficient") and has a measured timing side channel.
- **Options:**
  - (a) Per-matter physical partitions: one dense and one sparse index per matter, plus one firm-library partition.
  - (b) A global index with a FAISS `IDSelector` pre-filter. BM25 still needs partitions, and the shared index means shared statistics.
  - (c) A separate process per matter (too heavy).
- **Recommendation:** **(a)**.
  - Each partition (`m:<matter_id>`, `lib:<org_id>`) has its own FAISS index, BM25 corpus and metadata.
  - The policy engine returns the set of partitions the caller may read. The retrieval API **requires** an `AuthorizedScope` capability object, so no code path can search without one.
  - Global auxiliary indexes (section, TOC, glossary, cross-ref) are **disabled** in the Law edition until they are partitioned.
  - Legacy verticals keep the global index unchanged.
- **Consequence:**
  - Small per-matter indexes give exact flat search and cheap deletes and rebuilds.
  - Cross-matter "firm-wide search" exists only as an explicitly authorized union.
  - Memory grows per loaded partition, so an LRU of loaded partitions is needed.
- **As built (S1.6):**
  - **Deviation from the plan, for safety.** The plan said to refactor `rag/index.py` and `rag/sparse.py` behind a `PartitionIndex`. Instead, Law documents have their own tables (`legal_documents`, `legal_document_versions`, `legal_chunks`) and their own partition module (`legal/retrieval/partition_index.py`), reusing only the BM25 tokenizer, the embedding client and the cross-encoder.
    - The legacy global index is untouched, so the legacy golden eval cannot regress.
    - No legacy rebuild can ever read a Law passage.
    - A test checks that the legacy `documents`/`chunks` tables and index files stay free of Law text.
  - **Partitions:**
    - Each partition is `DATA_DIR/legal/partitions/m_<matter_id>/` and holds an exact FAISS `IndexFlatIP`, a BM25 corpus with its own statistics, and a meta file naming the partition.
    - Files that name another partition, or disagree with each other, are never served: the partition is rebuilt from `legal_chunks`, which stores each passage's embedding.
    - An LRU keeps `LEGAL_PARTITION_CACHE` partitions loaded and reloads one rewritten by another process.
  - **Scope:**
    - `AuthorizedScope` can only be minted by `scope.authorize()` after a policy allow; any other construction raises.
    - `search()` re-checks it with the live policy before and after it runs.
    - Every candidate is re-read with the scope's matter id in the WHERE clause.
  - **Ranking:** dense, sparse and keyword channels are fused (RRF) and re-ranked by the cross-encoder in sentence windows of about 600 characters; the best window is what the result quotes.
    - Why windows: the real browser test showed the cross-encoder reads only ~1,000 characters, so a clause deep in a long passage was scored irrelevant.
    - A regression test reproduces that failure.
    - The window floor is `LEGAL_RERANK_MIN_SCORE` (0.0, the legacy citation floor); below it the search abstains.
  - **Not built yet:**
    - the firm-library partition (`lib:<org>`; no feature uses it yet);
    - the section, TOC and glossary auxiliary indexes, which do not exist for Law partitions;
    - the `ownership` migration, which moves to S1.7 where it is used.

### D-006 Sessions: server-side, revocable — IMPLEMENTED (S1.1, 2026-10-08; part of the approved plan)
- **Decision:**
  - In the Law edition, replace stateless JWT cookies with an **opaque session token**. Only its SHA-256 is stored, in a `sessions` table.
  - Cookie settings: `HttpOnly`, `Secure`, `SameSite=Strict`, `__Host-` prefix.
  - Lifetimes: 30-minute idle and 10-hour absolute (configurable).
  - Revocation: per session, per user, and on any role or membership change for that user.
  - WebSockets re-validate the session at least every 60 s and immediately on an ACL-epoch bump.
  - Bearer tokens are allowed only for internal service-to-service calls with a short-lived signed **on-behalf-of** token (see D-013).

### D-007 MFA — IMPLEMENTED (S1.1, 2026-10-08; part of the approved plan)
- **Decision:**
  - TOTP (RFC 6238), implemented with the stdlib (`hmac`/`hashlib`/`base64`). It needs no new dependency and works offline.
  - Users get 10 one-time recovery codes, stored hashed. MFA is required for every Law role when the firm policy says so (default **on**).
  - WebAuthn/passkeys are a later option.
- **SSO/OIDC:** build an OIDC Authorization Code + PKCE client against a standard discovery URL. It can only work if the firm's IdP is reachable on the LAN, so it is **BLOCKED for verification** until the pilot IdP is known (P-011). Local accounts are the approved fallback.

### D-008 Audit trail: hash-chained, append-only — IMPLEMENTED (S1.5, 2026-10-08; part of the approved plan)
- **Decision:**
  - A new `audit_events` table. Each row stores `prev_hash` and `hash = SHA-256(prev_hash ‖ canonical_json(row))`.
  - SQLite triggers abort any `UPDATE` or `DELETE`.
  - Daily checkpoint: write the head hash to `/data/legal/audit_anchor/YYYY-MM-DD.json` and include it in backups.
  - A verifier CLI and an admin endpoint re-compute the chain.
  - **Content policy:** log IDs, actions, outcomes, counts and hashes only. Prompts are logged as `{length, sha256}` by default; full-text prompt logging is a firm policy switch (P-005).
  - Reads (document open, citation resolve, transcript view, export) **are** logged, unlike today.
  - The existing `activity_log` stays as a request-level log.
- **As built (S1.5):**
  - Migration `0006_audit`. The hash rule is `SHA-256(prev_hash + canonical_json(fields))`: `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)` over the 16 stored fields (seq and the two hashes excluded), with 64 zeros before the first event. A JSONL export can be re-verified with this rule alone.
  - Anchors are written to `DATA_DIR/legal/audit_anchor/YYYY-MM-DD.json` (0600): on a day's first event, then at most once a minute. `python -m app.legal.audit anchor` forces one.
  - Verification (`python -m app.legal.audit verify`, `GET /api/legal/audit/verify`) reports these problems: an edited event, a broken link (removed or reordered), a chain rewritten after an anchor, a removed tail.
  - Three recording paths, so every Law action leaves exactly one event:
    - `@audited` on every Law route that changes state, plus matter open and audit reads. A test fails if a state-changing `/api/legal` or `/api/auth` route lacks it.
    - Policy refusals (and any access under break-glass) on other routes.
    - A generic `api.request` event, with method, route template and status, for state-changing requests on the existing modules until S1.7 scopes them.
  - **Content:** `details` keeps strings only under named identifier keys; any other string is stored as `{"chars": n}`. Titles, rationales, reasons and typed usernames of unknown accounts are never stored. Tests plant canaries in each of these.
  - **Fail closed:** if an event cannot be written, the request gets a 500. A read returns no data. The workspace opens a matter only after its recorded read succeeds (D-023).
  - **Known limit:** the event is written after the change commits, in its own transaction. If the write fails, the change stands and is reported as failed (500). The gap is logged (`Audit: could not record <action>`) but is not in the chain. Joining audit and business writes in one transaction is planned with the Postgres move (D-004 gate).
  - **Who sees what:**
    - The firm-wide trail (`audit.read.org`) is open to firm admins, managing partners and records admins.
    - Matter numbers and titles are shown only to roles that already hold the matter directory (managing partners, records admins). Firm admins see an opaque matter reference.
    - A matter's own activity is for its lead.
    - Reading, verifying and exporting the trail are themselves recorded.
  - Anonymous refusals (no session) are not events: anyone could flood the trail with them. Failed sign-ins **are** recorded, so a per-IP sign-in throttle stays on the risk list.
  - The syslog/CEF forwarder is not built: it needs network approval.

### D-009 Frontend foundations for the Law workspace — IMPLEMENTED (S1.3; react-router 7, not 8, because of Node 20)
- **Decision:**
  - **Routing:** add `react-router` (v7, data router) for the Law workspace only. This gives deep links like `/matters/M-101/documents/…`.
  - **State:** React Context for session and the active matter. TanStack Query (`@tanstack/react-query`) for server state with **matter-keyed caches**, purged when the matter changes or the user signs out.
  - **Tests:** Vitest + Testing Library + `jsdom` for unit and interaction tests. Playwright for E2E against a seeded native backend.
  - **Icons:** Lucide (`lucide-react`, ISC) replaces emoji.
  - **Fonts:** self-hosted Inter Variable and Source Serif 4 via `@fontsource-variable/*` (OFL), bundled at build time with no runtime CDN.
  - **Isolation:** Law UI code lives in `frontend/law/`, and its design tokens are scoped under `[data-edition="law"]`, so the other verticals are visually unchanged.
- **Consequence:** New runtime dependencies are reviewed in LICENSE_REGISTER.md before install.

### D-010 OCR engine — PROPOSED
- **Decision:**
  - Use Tesseract 5 (Apache-2.0) via `pytesseract` with an `eng` traineddata pre-cached for offline use.
  - Run OCR per page only when PyMuPDF finds no text layer, or the text layer is below a density threshold.
  - Record provenance on every page and chunk: `ocr=true`, engine and version, mean confidence, low-confidence word spans.
  - The UI labels OCR text "OCR text, verify against image" and shows the page image beside it.
- **Alternative:** a GPU OCR (docTR or PaddleOCR) can be evaluated later if Tesseract accuracy is insufficient on the pilot's scans.

### D-011 DOCX export and true tracked changes — PROPOSED, with a feasibility spike
- **Decision:**
  - Build a DOCX renderer with python-docx for normal documents.
  - Write redlines by emitting raw OOXML `<w:ins>`/`<w:del>` runs with `w:author`/`w:date`, plus comments (`comments.xml`, `w:commentRangeStart/End`, `w:commentReference`). python-docx 1.2 adds a comments API; upgrading the 1.1.2 pin is part of the spike.
- **Verification (T10):**
  1. Parse the XML and assert `w:ins`/`w:del` exist.
  2. Programmatically "accept all" and "reject all", then compare against the target and original text.
  3. Round-trip through LibreOffice headless (`soffice --convert-to`), re-parse, and check the revisions survive.
  4. **Manual open in Microsoft Word by a human before it is called DONE.**
- **Stop condition:** If the spike fails, the HTML side-by-side redline ships and DOCX tracked changes are marked **BLOCKED**, never "done". The timing question (pilot launch or Phase 3) is P-013.

### D-012 Remove global PII redaction for Law ingestion — IMPLEMENTED (S1.6, 2026-10-08; part of the approved plan)
- **Context:** `sanitize.py` rewrites emails, phone numbers and long numbers in **every** document. That destroys contract content, Bates numbers and e-discovery data, and makes citations quote altered text.
- **Decision:**
  - For Law partitions, index the **original** text.
  - Run PII and privilege detection as **labels** (stored findings with offsets) that drive access warnings, export redaction (S1.10) and log-scrubbing.
  - Legacy verticals keep the current behaviour.
- **Risk:** Sensitive identifiers become searchable inside an authorized matter. That is the intended behaviour for legal work and stays inside the matter's ethical wall.
- **As built (S1.6):** `chunk_document(..., redact_pii=False)` for Law. The default is unchanged for legacy.
  - `find_pii()` records non-overlapping findings (email, phone, long number, card, SSN, ID number) as kind plus offsets in `legal_findings`. The value is never stored.
  - The inspector shows the counts. Use by labels and export redaction comes in S1.9–S1.10.

### D-013 Service-to-service identity (voice → backend, jobs) — IMPLEMENTED (S1.1, 2026-10-08; part of the approved plan)
- **Decision:**
  - At the WS handshake, the voice service captures the user's session and obtains a **short-lived (5 min), audience-bound on-behalf-of token** from the backend.
  - It sends that token on every RAG call and refreshes it while the session lives.
  - Background jobs (OCR, indexing, diligence, panel) store the *initiating principal ID*. They **re-check authorization** before writing results and before exposing them.
  - There is no blanket service-account bypass.
- **Fixes:** the current voice-to-backend 401 under auth (CODEBASE_AUDIT §4).

### D-014 Encryption at rest — PARTLY IMPLEMENTED (S1.6: uploaded documents; volume encryption remains a deployment step)
- **Decision (defence in depth):**
  1. **Volume level:** the Law data volume lives on a LUKS-encrypted disk. This is a deployment runbook step that needs root on the target host and is **not** done by the app.
  2. **Application level:** original uploads, Boardroom audio chunks, export outputs and connector credentials are encrypted with AES-256-GCM. Each file gets its own key, wrapped by a master key from `LEGAL_MASTER_KEY_FILE` (mode 0600, outside the data volume). This uses the `cryptography` library (Apache-2.0/BSD) and needs approval to add.
  3. **Backups:** encrypted with `age` or AES-GCM before they leave the host.
- **Not encrypted at the app level:** SQLite and FAISS (covered by LUKS). Re-evaluate with Postgres (D-004).
- **As built (S1.6):** uploaded documents only.
  - Each version gets AES-256-GCM with a random per-file key. The associated data is the version id, so a swapped or edited file fails to open (tested).
  - The per-file key is wrapped by the master key (AES key wrap, RFC 3394).
  - Posture refuses to start the Law edition unless `LEGAL_MASTER_KEY_FILE` exists, is mode 600, holds 32 base64 bytes and lies outside the data folder.
  - Dev scripts create `.law-secrets/master.key` (gitignored). The compose overlay mounts it read-only at `/run/secrets/law_master_key`.
  - `cryptography` is pinned in `backend/requirements.txt`. **The 1.4.1 image the compose overlay reuses may not contain it**: check or rebuild at the first approved `compose.sh up`.
  - Downloads re-verify the SHA-256 recorded at import.
  - Still to come: Boardroom audio, export outputs and connector credentials (S1.8, S1.10, S1.11); key rotation; LUKS (deployment runbook).

### D-015 Language of verdicts in legal mode — PROPOSED
- **Decision:** Never label a spoken statement "true" or "false". The Silent Assistant uses evidence-relative wording:
  - "Matches signed agreement §8.2"
  - "Differs from signed agreement §8.2"
  - "No matching source in this matter"
  - "Needs review"
- Each alert carries its cited quote, confidence, reason and a Dismiss/Confirm action. The default threshold is conservative (P-008 sets the final value with the lawyers).

### D-016 Evaluation backdoors and debug taps — PROPOSED
- **Decision:**
  - Under `ECHOMIND_EDITION=law`, the voice ablation mode (`VOICE_ABLATION_MODE`, `/tmp/echomind_ablation_mode`) and the payload dump taps are **hard-disabled**: the code path is unreachable and a test asserts it.
  - On `main` they stay for the research experiments unless the owner decides to remove them.

### D-017 Transcript fidelity mode — PROPOSED
- **Decision:** Add a `verbatim` transcript mode, the default for the Law edition:
  - no lowercasing or punctuation stripping;
  - no suffix de-duplication or short-token drops;
  - frames are never silently dropped (backpressure, with a visible "capture degraded" warning instead).
- Store the raw ASR text immutably. Corrections go into `transcript_revisions`, recording who changed what, when and why. The display text is derived.
- Paraphrased "polished" text is never indexed as if it were the record.

### D-018 Retire `livetranscript/` from the Law build — PROPOSED
- `livetranscript/` is a standalone legacy app that nothing references, with no auth, `CORS *` and unpinned NeMo. Exclude it from Law images and docs. Deleting it from `main` is the owner's call.

### D-019 Design language: "Liquid Glass, quietly" — APPROVED 2026-10-08
- The direction follows Apple's Liquid Glass guidance:
  - glass only on the navigation and control layer;
  - documents and evidence on solid "paper";
  - no glass-on-glass;
  - tint only primary actions.
- The UI is deliberately un-chatbot-like: no sparkles, avatars, typing dots or "Ask AI" language.
- Full spec in [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md). A visual prototype (`design/prototype.html`) is offered for approval before the build.

---

### D-020 MFA secret storage until encryption at rest lands — RECORDED (S1.1)
- TOTP secrets are stored base32 in `users.mfa_secret`, protected by database access only.
- They become encrypted at rest with D-014: LUKS on the host now, and app-level AES-GCM once the `cryptography` dependency is approved (planned S1.6).
- Until then, a database copy would expose second factors, so backups must stay encrypted (OPS-01).

### D-021 First-sign-in authenticator enrolment — IMPLEMENTED (S1.3, 2026-10-08)
- An account without an enrolled authenticator enrols one at its first sign-in. Whoever holds the initial password can therefore bind their own authenticator.
- **Implemented in S1.3** (`da43cc3`):
  - admin-issued one-time enrolment links that expire (72 h; `LEGAL_ENROLLMENT_LINK_REQUIRED` lets a firm relax this);
  - a forced password change after an administrator sets someone's password;
  - an admin "reset two-step" action, which signs the person out and issues a new link;
  - the first administrator's link is written 0600 next to the data.

### D-022 Managing partners see the matter directory for walls — RECORDED (S1.4)
- To screen someone from a matter they cannot see themselves, a managing partner needs to find it. The walls authority (the `walls.read` firm action, held by managing partners) can therefore list the firm's matters (number, title, client) at `/api/legal/walls/matter-directory`, and only there.
- This exposes no content (documents, transcripts, members) and does not grant access to any matter.
- Attorneys requesting a wall can only protect matters and clients they can already see.
- Break-glass requesters (firm admins) never see the directory: they name the matter by number, and only the approving managing partner sees what it resolves to.

### D-023 Opening a matter is a recorded read — RECORDED (S1.5)
- The workspace resolves a matter from the person's own list, so a hidden matter and a missing one look the same (S1.3). That list alone left no trace of a visit, and a break-glass visit was invisible to the matter's lead; the S1.5 browser test caught this.
- Entering a matter now calls `GET /api/legal/matters/{id}`, which records `matter.read`, with reason `break_glass` under emergency access. Nothing of the matter is shown until that succeeds.
- One event per visit: leaving the matter drops its cache, so coming back is recorded again. Switching tabs inside the matter is not a new visit.
- No new disclosure: the call is made only for a matter already in the person's visible list.

### D-025 The Law edition closes the existing modules; their capabilities return matter-scoped — RECORDED (S1.7)
- **Context:** The plan for S1.7 was to add matter scoping to each existing module: legacy chat, documents, transcripts, the Boardroom, the document studio and export evaluation. None of them knows about matters. Each reads from the shared global index or from unscoped tables, and in the Law edition they were still reachable by any signed-in person. For example, a legacy upload went into the shared index and was answerable for everyone.
- **Decision:**
  - In the Law edition the backend serves only `/api/auth/*` and `/api/legal/*` (plus `/health` and `/api/version`).
  - Every other `/api` route answers 404 to everyone, signed in or not, before any session lookup.
  - The live-transcript socket closes with 4404.
  - The capabilities come back as matter-scoped Law endpoints, each built on an `AuthorizedScope`:
    - research (now);
    - consent-gated capture and meetings (S1.8, S2.4–S2.7);
    - drafting (S3.x);
    - export (S1.10).
- **Why this rather than patching each module:**
  - It is deny-by-default and makes the canary suite hold for every module at once.
  - No legacy code path can reach Law data, and Law content never enters the legacy stores.
  - Legacy editions are unchanged: the gate is Law-only.
- **Consequence:**
  - Until each Law replacement ships, the Law workspace has no live transcription, Boardroom or document studio. The Law UI never offered them.
  - The voice service in the Law edition needs a matter (`/ws?matter=<id>`). Its grounded answers come only from that matter's research stream, through an on-behalf-of token. The legacy `ask-voice` routes are closed.
  - The on-behalf-of token is accepted on one route only: `/api/legal/matters/{id}/research/query`.
- **Research (as built):**
  - Answers come from the matter's partition only (S1.6 search, 6 passages).
  - Sources are fenced as escaped, numbered quotations that the model is told never to treat as instructions.
  - Wording is evidence-relative (D-015).
  - The answer streams as `sources` → `chunk`… → `done`.
  - The scope is re-checked every `LEGAL_STREAM_RECHECK_SEC` (2 s) and at the end. A revocation ends the stream with `scope_revoked` and nothing more is sent.
  - No relevant passage means an explicit abstention without calling the model.
  - Every question and its outcome are audited (the question only as length and SHA-256). History is private to its owner within the matter.

### D-026 Consent-gated capture as built — RECORDED (S1.8)
- **A new Law capture socket.** The socket is `/api/legal/matters/{m}/meetings/{id}/capture`. It does not reuse the legacy live-transcript handler, which writes into the shared knowledge base (D-025).
  - The socket refuses before anything loads:
    - 4403: not a contributor;
    - 4404: the meeting is not visible;
    - 4409: the matter or meeting is closed;
    - 4412: consent is missing, withdrawn or stale;
    - 1008: no session or a foreign origin.
  - It closes with 4401 if the session ends mid-capture.
- **Consent (P-001 safe defaults, awaiting counsel):**
  - all-party, explicit, per participant, before any capture;
  - methods: spoken on the record, in writing, or confirmed by the host;
  - the person recording consent confirms they read the statement, whose SHA-256 is stored;
  - consent older than `LEGAL_CONSENT_MAX_AGE_HOURS` (12) must be asked again;
  - a new or returning participant must consent;
  - **anyone on the matter can record a withdrawal**, even read-only members and on a closed matter.
- **Ordering guarantee:**
  - One lock per meeting. A withdrawal stamps its time and flags running captures inside the lock. Frame acceptance and transcript storage check the flag and stamp their own time inside the same lock.
  - So no accepted audio and no stored text is later than the withdrawal. A database re-check every `LEGAL_CONSENT_RECHECK_SEC` covers captures in another process.
- **Audio:**
  - Not stored by default; audio retention is P-002 and needs counsel.
  - Each ~5 s batch leaves a SHA-256 custody record (size and time).
  - Paused audio is discarded unread.
  - The capture records the hash of the consent state it relied on.
- **Transcription:**
  - Pluggable. The default wraps the live speech-recognition model when it is installed; the dev host has none, so it reports "unavailable" and capture control still works.
  - Segments are stored per matter and meeting; reading them is audited.
- **Audit:** participant names and transcript text are matter content and never appear in the audit trail; tested.

## B. Policy decisions — NEEDS COUNSEL (pilot firm)

Engineering provides configurable mechanisms with **the safest default**, and each item is recorded here until counsel approves it. Nothing in this table is decided by Claude.

| ID | Question | Safe engineering default until decided |
|---|---|---|
| P-001 | **Recording consent:** all-party or one-party per jurisdiction; accepted methods (verbal on record, written, host attestation); consent script text; re-consent on rejoin | **All-party, explicit, per participant, before any capture**; capture refused otherwise; withdrawal stops capture immediately |
| P-002 | **Retention schedule** per artifact type (documents, versions, transcripts, audio, AI drafts, chats, exports, audit) | **No automatic deletion.** Retention runs are dry-run only until a policy is approved. |
| P-003 | **Legal hold** workflow: who may create and release, approvals, notification | Hold creation and release need a records-admin request plus a second approver; held data is never purged by any path |
| P-004 | **Privilege labels** taxonomy and meaning | Advisory labels: `Privileged`, `Work product`, `Confidential`, `Not privileged`, `Unreviewed` (default `Unreviewed`) |
| P-005 | **Prompt and answer logging** policy | Metadata and hashes only; no prompt text in logs |
| P-006 | **Export and disclosure** policy (who may export, redaction requirements, watermark text) | Export needs matter `can_export`, a redaction preview and an explicit approval; watermark "CONFIDENTIAL — <matter> — <user> — <timestamp>" |
| P-007 | **Staff-facing AI notice** wording | Draft notice shipped, marked *awaiting counsel approval* |
| P-008 | Who signs off legal test results, alert thresholds, templates and redaction rules | No legal output is called "validated" without a named reviewer |
| P-009 | Which legal authority libraries may lawfully be used offline | None. All case and statute citations are marked **UNVERIFIED**. |
| P-010 | DMS provider (iManage / NetDocuments / SharePoint / none) | Local import plus the connector interface only (SEC-08 BLOCKED) |
| P-011 | Identity provider and MFA policy | Local accounts plus TOTP |
| P-012 | Concurrent-user and latency targets on real hardware | Measure and publish (T12); no claims before that |
| P-013 | Word tracked changes needed at pilot launch, or Phase 3 | Phase 3; HTML redline earlier |
| P-014 | Practice area and first pilot workflow | Corporate / in-house: client meetings, contract review, board minutes (spec §2) |
| P-015 | Minutes-only mode: when may the transcript be deleted | Never automatically; manual only, after the hold check and policy approval |

## C. Recording / retention decision log (awaiting legal approval)

| Date | Item | Status | Approver |
|---|---|---|---|
| 2026-10-08 | P-001 consent defaults as above | Awaiting counsel | — |
| 2026-10-08 | P-002 / P-003 / P-015 retention and hold defaults as above | Awaiting counsel | — |
