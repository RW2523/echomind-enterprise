# EchoMind Law — Decision Log

> **2026-10-08 approval.** The owner replied *"approved, create the branch and start S1.0"* to the six-item approval list: D-001, D-002, D-003, D-004, D-019, and committing the plan documents. Those entries are marked APPROVED. All other D-entries stay PROPOSED until their slice, and the plan as a whole is the working baseline.

> **2026-10-08 standing approval.** After approving the S1.3 design and merge, the owner wrote *"keep building one by one and build everything fully and tell me"*. This is read as standing approval to:
> - build the slices in order;
> - merge each completed, fully tested slice into `feature/echomind-law` without asking again;
> - add the dependencies already listed in LICENSE_REGISTER §2.
>
> It does **not** cover: pushing to GitHub or opening PRs; starting the compose stack on the shared host; anything touching production; network integrations (DMS, IdP, licensed sources); or counsel's policy decisions (P-items). Those are still asked for explicitly.

> **2026-10-09 approval.** After the S1.5–S1.8 report, which listed D-025's change of approach and the three proposed decisions adopted in S1.6, the owner replied *"approved, merge and start S1.9"*. D-005, D-012, D-014 (as implemented), D-023, D-025 and D-026 are therefore APPROVED. The P-items (consent wording and rules, audio retention) stay with counsel.
>
> **2026-10-09 Phase 1 gate approval.** After the S1.12 gate report (`releases/phase1-gate.md`), the owner replied *"approved, merge S1.12 and start S2.1"*. Phase 1 is accepted. D-027 to D-033 are as recorded (built and tested). This approval does not cover the items still marked 🔒: the root-run enforced T11, the first `compose.sh up`, network integrations, pushes and counsel items.

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

### D-005 Retrieval isolation: per-matter index partitions — APPROVED 2026-10-09 (implemented S1.6, option (a))
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

### D-010 OCR engine — ADOPTED in S2.1a (Tesseract 5 run as a local program, not through pytesseract; see D-034)
- **Decision:**
  - Use Tesseract 5 (Apache-2.0) via `pytesseract` with an `eng` traineddata pre-cached for offline use.
  - Run OCR per page only when PyMuPDF finds no text layer, or the text layer is below a density threshold.
  - Record provenance on every page and chunk: `ocr=true`, engine and version, mean confidence, low-confidence word spans.
  - The UI labels OCR text "OCR text, verify against image" and shows the page image beside it.
- **Alternative:** a GPU OCR (docTR or PaddleOCR) can be evaluated later if Tesseract accuracy is insufficient on the pilot's scans.

### D-011 DOCX export and true tracked changes — PROPOSED, with a feasibility spike (S3.3: automated checks pass, manual Word check pending; see D-047)
- **Decision:**
  - Build a DOCX renderer with python-docx for normal documents.
  - Write redlines by emitting raw OOXML `<w:ins>`/`<w:del>` runs with `w:author`/`w:date`, plus comments (`comments.xml`, `w:commentRangeStart/End`, `w:commentReference`). python-docx 1.2 adds a comments API; upgrading the 1.1.2 pin is part of the spike.
- **Verification (T10):**
  1. Parse the XML and assert `w:ins`/`w:del` exist.
  2. Programmatically "accept all" and "reject all", then compare against the target and original text.
  3. Round-trip through LibreOffice headless (`soffice --convert-to`), re-parse, and check the revisions survive.
  4. **Manual open in Microsoft Word by a human before it is called DONE.**
- **Stop condition:** If the spike fails, the HTML side-by-side redline ships and DOCX tracked changes are marked **BLOCKED**, never "done". The timing question (pilot launch or Phase 3) is P-013.

### D-012 Remove global PII redaction for Law ingestion — APPROVED 2026-10-09 (implemented S1.6)
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

### D-014 Encryption at rest — APPROVED 2026-10-09; PARTLY IMPLEMENTED (S1.6: uploaded documents; volume encryption remains a deployment step)
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

### D-015 Language of verdicts in legal mode — ADOPTED in S2.5 under the standing approval (see D-041)
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

### D-017 Transcript fidelity mode — ADOPTED in S2.4 under the standing approval (see D-040)
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

### D-023 Opening a matter is a recorded read — APPROVED 2026-10-09 (S1.5)
- The workspace resolves a matter from the person's own list, so a hidden matter and a missing one look the same (S1.3). That list alone left no trace of a visit, and a break-glass visit was invisible to the matter's lead; the S1.5 browser test caught this.
- Entering a matter now calls `GET /api/legal/matters/{id}`, which records `matter.read`, with reason `break_glass` under emergency access. Nothing of the matter is shown until that succeeds.
- One event per visit: leaving the matter drops its cache, so coming back is recorded again. Switching tabs inside the matter is not a new visit.
- No new disclosure: the call is made only for a matter already in the person's visible list.

### D-025 The Law edition closes the existing modules; their capabilities return matter-scoped — APPROVED 2026-10-09 (S1.7)
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

### D-026 Consent-gated capture as built — APPROVED 2026-10-09 (S1.8; P-001 values stay with counsel)
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

### D-027 Records as built: labels, holds, deletion, retention — RECORDED (S1.9)
- **Labels (P-004 safe default):**
  - The labels are advisory: Privileged, Work product, Confidential, Not privileged and Unreviewed. They never change who can see something.
  - On documents and meetings, a lead, attorney or reviewer sets the label; a paralegal's choice is recorded as a suggestion.
  - At indexing, a detector suggests a label from explicit markers in the text ("attorney-client privilege", "privileged and confidential", "legally privileged", "without prejudice", "work product"). "Confidential" alone is deliberately not a marker: it appears in most contracts.
  - Every change is kept and audited.
- **Legal holds (P-003 safe default):**
  - A hold covers whole matters (document-level holds can come later).
  - A records administrator requests it; it takes effect once a different records administrator or a managing partner approves. Release takes two people as well.
  - Members see an "on hold" notice, not the hold's details.
- **Matter directory (extends D-022):**
  - Records administrators join managing partners in seeing the firm's matter directory (number, title, client) at `/api/legal/matter-directory`. They need it to place holds and judge deletions.
  - No content is shown and no access is granted.
- **Deletion:**
  - One service (`legal/records/deletion.py`) handles documents and meeting transcripts.
  - A person's request soft-deletes at once: the item is hidden from lists, search, research and downloads, and can be restored. A retention request is only proposed.
  - The dry-run report counts versions, passages, files, index entries, findings, labels and the research answers quoting the item.
  - Approval needs a records administrator or managing partner other than the requester.
  - The purge deletes inside one transaction that re-checks holds; then stored files and index vectors are removed. Research answers keep their wording but lose quotes from the item.
  - Verification re-checks the database, index, files and quotes, and is stored with the request.
  - Consent records, capture timelines and audio receipts are kept when a transcript is deleted.
  - The audit trail is never purged. The deletion record keeps only a hash of the title.
- **Retention (P-002 safe default):**
  - A records administrator drafts a policy ("documents / meeting transcripts, N days after the matter closed"); a managing partner approves it.
  - Runs are dry runs: one is scheduled daily and they can also be run on demand. They count what approved policies select, preview what drafts would, and count held matters as skipped.
  - `LEGAL_RETENTION_APPLY` (off) allows an "apply" run, and even that only creates deletion requests for a person to approve. **Nothing is ever deleted automatically.**
- **Known limits:**
  - Backups keep purged data until they rotate (OPS runbook).
  - Research answers keep their own wording, which may paraphrase a purged document; only quotes are removed.

### D-028 The export gateway as built — RECORDED (S1.10; P-006 safe defaults)
- **The only way out.** In Law, a matter's documents and meeting transcripts leave as files only through `/api/legal/matters/{m}/exports`. The old export and studio routes are closed (D-025). Source downloads stay inside the firm: `no-store`, audited, members only.
- **Flow:**
  - *Request:* `export.request` plus the membership's `can_export`.
  - *Redaction plan:* adjustable by the requester only, and only while a draft.
  - *Submit.*
  - *Approval:* by a lead or attorney other than the requester. Privileged and work-product documents need an explicit acknowledgement.
  - *Then:* render, verify, watermark "CONFIDENTIAL — <matter> — <requester> — <time>".
  - *Release:* stored encrypted for `LEGAL_EXPORT_TTL_HOURS` (24), downloadable by the requester and the approver only.
- **Plan defaults:**
  - Redacted: personal data, IBANs, credentials, card numbers (Luhn-checked) and other matters' numbers, which would reveal that those matters exist.
  - Flagged but kept: this matter's number (it is in the watermark), party names, and privilege markers.
- **Rendering:**
  - PDF to PDF: the planned strings are found on every page and removed with applied redactions. Annotations, form fields, attachments, scripts, links, metadata and XMP are scrubbed.
  - Everything else is rebuilt from the readable content: tracked insertions are accepted; deletions, hidden runs and comments are dropped. It is then written as a new Word file (python-docx's own template stripped of its custom XML and thumbnail) or a new PDF, with a black "[REDACTED]" mark.
- **Verification:**
  - The file is opened again and checked: no planned string in text, metadata or any package part; clean properties; clean structure; a watermark on every page or in the Word header.
  - Any failed check means the export is `failed` and nothing is released.
  - The preview is the same rendering as page images, made on request and never stored.
- **Records:** purging a document or transcript also purges its exports (S1.9 verification gained an `exports` check). Audit events carry counts, the format and outcomes, never the purpose or the plan's strings.
- **Known limits:**
  - Text inside images cannot be found until OCR (S2.1); the plan warns on image-heavy pages.
  - A string split across a PDF line break is not found by search, so verification fails closed and the export is not released.
  - Word output keeps paragraph styles, bold, italic, underline and tables, but not complex layout.

### D-029 Connectors as built: fail-closed permissions and freshness — RECORDED (S1.11)
- **Framework:**
  - A `DmsConnector` contract: list a folder's items with their permissions, fetch bytes.
  - A sync engine that imports through the normal path (type checks, AES-GCM storage, indexing job, audit).
  - The only implementation is a **local-folder connector** (a folder on this server). Its root must lie inside `LEGAL_CONNECTOR_LOCAL_ROOTS`; paths are normalised; symlinks are never followed.
  - Connector settings are sealed with the master key (AES-GCM, AAD = connector id).
  - **No real DMS provider exists or is presented as working:** that needs the firm's choice of provider, credentials and network approval (P-010).
- **Roles:**
  - Firm administrators create, sync, disable and enable connectors, and see only counts (links, documents, last run). They never see which matters use them or what they hold.
  - A matter's lead links a folder to the matter (`matter.update`). Members see the linked folders and each document's source.
- **Permissions (fail closed):**
  - A file needs a sidecar `<file>.acl.json` = `{"users": [...]}`. Without a valid one it is not imported, and an imported document that loses it becomes visible to nobody.
  - The usernames that exist in the firm are stored as the document's permission list. Every read for a person — list, details, versions, file, search, labels, deletion, exports, export approval — requires them to be on that list **and** a live member of the matter (the policy engine).
  - Documents of a disabled connector, or of one without a successful sync within its `stale_after_minutes`, are hidden from everyone until it syncs again.
  - Exports of hidden documents are hidden too.
- **Sync:**
  - Runs on schedule (`sync_minutes`, through the job worker) or on demand.
  - A changed file becomes a new version (people cannot upload versions of connector documents); permissions are replaced each run.
  - A file gone from the source is hidden at once and queued for the S1.9 approved purge (origin `connector`; the deletions table was rebuilt to allow it). A legal hold keeps it, shown as "removed at source".
  - A run that cannot read the source is `failed` and changes nothing.
- **Known limits:**
  - BM25 statistics are per matter, so restricted documents still contribute to term statistics; their passages are filtered out before ranking results are returned.
  - Groups in source permissions are not supported yet; usernames only.

### D-030 One idle SQLite connection held open per process — RECORDED (S1.11)
- **Problem:**
  - The backend opens and closes a SQLite connection for every database call (`get_conn`).
  - In WAL mode, whichever connection closes last checkpoints the WAL, deletes it and tears down its shared-memory index; the next call rebuilds them. Under load this happened on nearly every request.
  - In the S1.11 browser runs the backend stalled in three of five full runs. A Python stack dump (`LAW_DEBUG_STACKS=1`, SIGUSR1) showed every thread blocked inside SQLite: three in `sqlite3.connect` (request handler, job worker, audit writer) and one in `conn.close()`. All of them were in futex waits with no CPU use.
- **Decision:**
  - `get_conn` keeps one idle connection per process open on the current database. It is never used for queries; it only keeps the WAL and its index in place, so closing an ordinary connection no longer tears them down.
  - Automatic checkpoints still run every 1000 pages.
  - The idle connection is closed at process exit, which checkpoints the WAL into the file. It is also closed when `DB_PATH` changes, as it does between tests.
  - Online backups (`scripts/backup_data.sh` uses the SQLite backup API) are unaffected. Nothing in the code needs exclusive access to the file.
- **Not proven:** native stacks were not taken. Attaching a debugger to the running process needs a ptrace permission that this session was not granted, so the exact SQLite lock is not identified. The evidence is the stack dump and repeated clean full runs after the change (TEST_RESULTS run 12).
- **Kept for the S1.12 gate:** `LAW_DEBUG_STACKS=1` (set only by `scripts/law/e2e_backend.sh`) makes SIGUSR1 dump every thread's Python stack to the server log.

### D-031 Prompt-injection defences in research (T08) — RECORDED (S1.12)
- **Finding:** the real answer model (Qwen3-30B) often obeys instructions planted in a source.
  - With a "system override" passage it appended the planted marker in 10 of 18 answers. Once it also stated the planted false claims ("matter M-202 settled for EUR 9 million", "your export of all matters is complete").
  - It could not act on them: the model has no tools, sees only the matter's own passages, and cannot start an export.
- **Structural guarantees (unchanged, now tested end to end with a model that obeys anything):**
  - Retrieval is fixed by the authorized scope before the model runs.
  - The request to the model carries no tools; its output is only text.
  - Nothing it writes changes memberships, walls, labels, holds, deletions or exports.
- **New layers:**
  1. **Repeated rule:** the rule against following source text is repeated after the sources.
  2. **Withholding:** sentences in a source that are addressed to an AI are replaced by a marker in the model's copy. Examples are claimed system messages, "note to any AI assistant", and "end your answer with…". The patterns are narrow, so ordinary legal wording is never withheld; there is a test list of near misses. The person still sees the whole passage, marked "Contains text addressed to an AI assistant. It was withheld from the answer."
  3. **Uncited-sentence marking:** every sentence of an answer that cites no source is listed, and the screen dims it ("1 sentence cites no source — shown dimmed. Check it before relying on it."). Sentence splitting handles legal abbreviations and lowercase sentence starts. A lowercase start was a bypass found while testing.
  4. **Tag stripping:** leaked `<think>` tags are removed from answers.
- **Measured** with `scripts/law/injection_probe.py` (real model, 3 trials per injection and question):

  | Defences | Strongest injection obeyed |
  |---|---|
  | None | 10 of 18 |
  | Repeated rule only | 6 of 9 |
  | Repeated rule plus withholding | 0 of 36 |

  - All three layers left answers cited: 42 of 42, with 21 of 21 correct abstentions.
  - Three held-out injections, written before the patterns and never used to tune them, were obeyed 0 times *with or without* the defences. So how far the patterns generalise is **not shown**.
- **Limits:**
  - The patterns are a deny-list; a novel phrasing can pass. The uncited-sentence marking is the backstop, but it cannot catch an injected claim that the model cites to the injected passage. That passage is then shown with its warning.
  - Voice speaks the answer as it streams, so it cannot show the marks.
  - Hidden-text detection at ingestion stays in S2.1.

### D-032 Encrypted backups, restore and safe rollback (T13) — RECORDED (S1.12; OPS-01)
- **Backups:** `python -m app.legal.backup keygen|create|verify|restore`.
  - **What it holds:** a consistent snapshot (SQLite online backup) plus the data folder (encrypted documents and exports, audit anchors), without the live database files or the model caches.
  - **Sealing:** sealed with AES-256-GCM in 1 MiB chunks under a **backup key** kept apart from the data and from the master key. Each chunk's nonce and associated data bind its position and whether it is the last, so altered, reordered or truncated archives do not open.
  - **Manifest:** lists every file's SHA-256, the schema, the audit head and record counts.
  - **The master key is never in a backup.** A restore needs the archive, the backup key and the master key.
- **Restore safeguards:**
  - The whole archive authenticates before anything is written.
  - The target folder must be new or empty.
  - Only plain files inside the target are accepted, with no traversal, absolute paths, links or devices.
  - Every file and the database's counts, schema and audit head must match the manifest.
- **Auditing:** a backup is recorded as `backup.create` (operator, file count, bytes, manifest hash).
- **Rollback guard:** `migrate_down` refuses, changing nothing, when the rollback would:
  - delete the audit trail (below 0006, if it has events);
  - delete records under an active legal hold (below 0010; T04);
  - make documents restricted by their source's permissions visible to the whole matter (below 0012).

  Such an upgrade is rolled back by restoring the backup taken before it.
- **The legacy scripts are unsafe from this clone:**
  - `scripts/backup_data.sh` and `scripts/restore_data.sh` find their volume through the container named `echomind-backend`, which on this host is **production**.
  - Their archives are not encrypted.
  - They are never run from the Law clone. The Law runbook uses `app.legal.backup` (inside the Law container once compose is approved).

### D-033 Network egress check (T11): capture mode now, enforced mode needs root — RECORDED (S1.12)
- **Capture mode** (`scripts/law/egress_test.sh`) runs the full browser suite with two recorders:
  - a Python audit hook in the backend (`LAW_EGRESS_LOG`) that records every name lookup and connect or send attempt, from any thread, successful or not;
  - an outside watcher (`scripts/law/egress_watch.py`) that maps the backend's socket descriptors to the kernel socket tables every 50 ms, catching native code.
- **Pass rule:** anything not on this machine fails the run.
- **uvloop:** uvloop connects in C, out of the hook's sight; the first run recorded nothing. The egress run therefore uses uvicorn's asyncio loop (`LAW_UVICORN_LOOP=asyncio`), which does not change which destinations the application chooses.
- **Enforced mode** — the backend in a network namespace that can reach only the model servers — needs root. This host restricts unprivileged user namespaces through AppArmor, and bubblewrap cannot configure loopback without privileges. **🔒 Waiting for the owner's approval.**

### D-034 How a legal document is read — RECORDED (S2.1a; RAG-01, T-06)
- **One canonical text per version, stored** (`legal_version_text`):
  - Page, line and block maps go with it.
  - Every search passage is an exact slice of it, so a citation's offsets point at the characters it quotes.
  - Law passages are cut by paragraph from this text (`documents/chunking.py`, about `LEGAL_CHUNK_CHARS` each). The legacy chunker rewrote whitespace and gave short files no offsets, so Law no longer uses it.
- **What a reader sees is what EchoMind reads:**
  - **PDF:**
    - Text that is white (unless on a dark fill or an image), under 2 pt, invisible (unless over an image, as a searchable scan's OCR layer is) or off the page is left out.
    - It is recorded as a `hidden_text` mark with the text, so it can be reviewed.
    - Visual lines are kept, so page and line citations match the page. Running headers and footers are still dropped as before.
  - **Word:**
    - Hidden (`w:vanish`), white or tiny runs are left out and marked.
    - Automatic numbering is resolved as Word shows it ("12.3", "(b)", "Article I"), including numbering that comes through a heading style. Each numbered paragraph also records its full path ("12.3(b)").
    - Tables are read row by row ("cell | cell").
    - Footnotes, endnotes, headers and footers follow the body as their own blocks. References read "[fn 1]", which cannot be mistaken for an answer citation "[1]".
- **Tracked changes are read as accepted.** Insertions are in and deletions out. Each change is recorded with author and date, and a deletion keeps its text at its position.
- **Comments are never part of the text.** They are recorded beside the words they cover, with author, date and comment text.
- **Text addressed to an AI is flagged where it is, at import** (`instructions` marks, the same patterns as D-031). Hidden text is checked too. The version's flags count both.
- **OCR (D-010):**
  - It runs only on pages with almost no text (under `LEGAL_OCR_MIN_CHARS`) that carry an image, one page at a time.
  - It uses the local `tesseract` program with the English model, 300 dpi, a 120 s per-page timeout and at most `LEGAL_OCR_MAX_PAGES` pages.
  - Each OCR page records its mean confidence. Words under `LEGAL_OCR_LOW_CONFIDENCE` are marked for checking.
  - With OCR off, a scan is reported as `needs_ocr` rather than indexed empty.
  - `pytesseract` was not needed.
- **Progress** is written as a version is read (`reading`, `ocr` page n of N, `indexing`, `done` / `failed`).
- **Purges** remove the stored text and marks and verify that they are gone.
- **Known limits:**
  - Text boxes and shapes in Word files are not read yet.
  - Mixed pages (some text plus a scanned region) are not OCR'd.
  - OCR accuracy is measured only on clean synthetic scans so far (character error rate 0.0 at 100–200 dpi). Real scans will be measured on pilot samples.
  - The Docker image needs `tesseract-ocr` and `tesseract-ocr-eng` (at the first approved build).

### D-035 Clauses, defined terms and pinpointed citations — RECORDED (S2.1b; RAG-01)
- **Clauses:**
  - From Word's numbering when a paragraph has it (the full path, e.g. "12.3(b)").
  - Otherwise from drafting markers at a line start:
    - "Article IV", "Part 2", and "Schedule 2", "Exhibit A", "Annex" or "Appendix" (level 0);
    - "Section 5.1", "Clause 4" and "§ 7";
    - a bare "4.1" or "12." followed by a capital, a quote or a bracket;
    - "(a)" and "(i)" under the last numbered clause.
  - A four-digit year, an amount ("12,000") or a date ("1 January") is not a clause.
  - A clause runs until the next clause at its level or higher, so it contains its sub-clauses.
  - Headers, footers and footnotes are never clauses.
  - Stored in `legal_clauses` (migration 0014).
- **Defined terms:** `"Term" means …` (also "shall mean", "has the meaning", "includes", "refers to"), and `Name (the "Term")`, with straight or curly quotes. Stored in `legal_defined_terms` with the definition sentence. Usage checks come in S3.5.
- **Pinpointed citations:**
  - A search quote stays its ~600-character window. Its **focus** is the sentence in it that shares most words with the question.
  - The citation gives the focus offsets and the clause, page and line where the focus is. Research sources pass the clause to the model (`clause="12.3(b)"`).
  - Versions read before S2.1 have no maps and keep their passage pages.
- **Purges** remove clauses and terms, with verification.
- **Limits:**
  - These are heuristics. Unusual numbering ("1-1", "A.1.a") is not recognised yet.
  - The focus is lexical, not semantic.
  - Definitions inside tables are found only when they use the patterns above.

### D-036 Which version answers come from, and conflicts between versions — RECORDED (S2.2a; RAG-01, T05)
- **Searchable versions:** each document's current version, plus its latest signed (`executed`) version when that is older. Other versions leave the index; their text, clauses and marks stay for the reader and for comparison. Ingestion, status changes and restores keep this in line (`documents/versions.py`).
- **Policies** (search and research take `versions`):
  - **`signed` (default):** the latest signed version. A document never signed is used in its current version, and its citations say `signed: false`.
  - **`current`:** the latest drafts.
- **Conflicts:**
  - For each cited clause, the other searchable version of the same document is compared after folding case and spacing. For a signed quote that is the current draft; for a draft quote, the signed version.
  - A different wording is reported with both versions' numbers, statuses and texts.
  - Conflicts between different documents come in S2.3.
- **Signing:** signing a draft (setting it to `executed`) makes it the version answers come from, at once.

### D-037 The evidence contract and quote verification — RECORDED (S2.2b; RAG-01, legacy R-4, R-6, R-7, R-8)
- **Quoted citations:** the model cites every statement with a short exact quote, `[n: "…"]`. Bare `[n]` is still read, as "not quoted".
- **Paragraph streaming:** the answer streams one checked paragraph at a time. Each quote is verified against the passage the model was given:
  - **verbatim:** found as written;
  - **normalized:** found after folding Unicode forms, quotes, dashes, spacing and case, or as the pieces of an ellipsis in order;
  - **failed.**

  Nothing unverified is shown unmarked. `chunk` events still carry each checked paragraph's display text (quotes removed) for the voice service.
- **Claims:** a claim is supported if one of its quotes is found, failed if it quotes and none is found, and unquoted if it only gives numbers.
- **Statuses:**
  - **`answered`:** nothing flagged;
  - **`partial`:** some claims failed, or some sentences cite nothing;
  - **`unverified`:** no supported or unquoted claim;
  - **`abstained`:** the reason is `no_passages` or `model`.

  Failed claims are **flagged, not dropped**, so the reviewer sees what the model said and that its quote is not in the source.
- **Evidence:** each quote with document, version and its status, `signed`, clause, page, line and the exact characters of the version's text. Saved with the turn, with the conflicts and the policy (migration 0016). A purge blanks the document's quotes in saved evidence and conflicts, and its verification checks them.
- **Legacy defects in Law** (they were in `rag/advanced.py`, which Law does not use; these are now tested guarantees):
  - **R-4:** a two-word legal term is researched.
  - **R-6:** pinpoints such as "clause 12.1 (p. 1)" are kept, and evidence numbers match their sources.
  - **R-7:** one answer draws on several documents. Meeting transcripts become research sources when they are finalised in S2.4.
  - **R-8:** the persona is the server's; client-sent persona or system text is ignored.
- **Measured on the real model** (8 questions, the demo MSA): 10 of 11 quotes verbatim. The one failure was an escaped inner quote, `(\"Provider\")`, which the parser now reads (tested).

### D-038 Passages, the relevance floor and when an answer counts as an abstention — RECORDED (S2.2d; RAG-01, T05, T06)
- **Passages follow the document's sections.** Search passages are cut at headings and top-level clauses, up to `LEGAL_CHUNK_CHARS` (900) characters, so a passage no longer mixes the end of one clause with the start of the next. Offsets stay exact, so every citation still maps to the version's text (D-034).
- **Relevance floor:** `LEGAL_RERANK_MIN_SCORE` = **-10.5** for the real cross-encoder (ms-marco-MiniLM-L-6-v2). It was calibrated on the golden set. At the old floor of 0, 22 of the golden questions were refused because their correct passages scored between -10 and 0. The test fixtures use a stand-in reranker with its own scale, so they keep 0.0. The floor only removes passages that are clearly unrelated; whether to answer is still the model's decision, followed by the checks below.
- **When an answer counts as an abstention:**
  - **No passages** above the floor: the model is not called, and the status is `abstained` with reason `no_passages`.
  - **The model abstains and nothing it says is supported:** it replies with the abstention sentence (rule 3), or says it in other words. The status is `abstained` with reason `model`.
    - "In other words" is narrow. The **opening sentence** must say that the documents or sources do not state, mention, specify, contain, provide, address or answer something, and must cite nothing (`prompt.says_not_stated`).
    - "Our work does not include tax advice" is an answer, not an abstention.
  - **The model says it in other words, and all its citations are weak:** the best cited passage scores below `LEGAL_ABSTAIN_WEAK_SCORE` (-6.0). The status is again `abstained`.
  - **Otherwise it is an answer.** A supported answer that also contains the abstention sentence stays an answer, and the checks mark what is flagged. A reply with no supported claim that is not an abstention is `unverified` (D-037).
- **Evaluation** (`eval/law/run_law_eval.py`; the real model `nvidia/Qwen3-30B-A3B-FP4` on the host, with nomic-embed-text and the cross-encoder):
  - **Golden** (70 items, the set used for tuning): answers 62/64, abstentions 4/6, 0 false abstentions, 0 T06 leaks, citation precision 0.912, quote verification 0.979, T05 conflicts 7/7, R-7 multi-source 2/2, p50 2.47 s, p95 5.99 s.
  - **Held-out** (32 items, never used for tuning; run once each with the system frozen): answers 27/29, abstentions 1/3, 1 false abstention, 0 leaks, precision 0.902, verification 0.905, p50 2.69 s, p95 6.19 s.
  - **Held-out correction, reported openly.** The first held-out run reported one "leak". It was a false positive in the harness: the forbidden string "4.2" matched "clause 4.2" in a permitted document. The item's forbidden list was corrected to `["two instalments", "4.2 million"]` and the held-out set was run again with the same system code. No system change was made between the two runs.
- **Known weakness: abstention.** When the documents do not answer, the model often says so and then adds related context it can cite ("The documents do not state who the CEO is. They name Thomas Reyes as General Counsel [1]."). Because that context is supported, the answer is not counted as an abstention (golden 2 of 6 and held-out 2 of 3 abstention items). The reviewer still sees the opening sentence and the evidence. The rule was not tuned further on the held-out set, so as not to overfit. The S2.8 gate re-measures it with a lawyer's scoring rubric.
- **Firm library scope deferred.** The plan allows "matter plus an optional firm library". The `lib:<org>` partition exists (D-005), but nothing puts content in it until templates and playbooks are approved (S3.1–S3.2). Research therefore stays matter-only, and the firm-library option will be added with the first approved library content.
- **Other known limits** (retrieval and model, not the evidence contract):
  - A question that names a party ("Harrowgate's indemnity") can miss a clause written in defined roles ("Provider shall indemnify").
  - The model sometimes leaves out a fact that its cited passage contains. One held-out answer, "who chaired the meeting", opened with "not stated". It then quoted "Karin Lindqvist (Chair)" in its own text with a bare `[1]` citation to a weakly scored passage, so it was counted as an abstention.

### D-039 Comparing versions and finding conflicts across documents — RECORDED (S2.3; RAG-02, T-06, T-07)
- **Two versions, clause by clause** (`compare/align.py`):
  - **Units.** A version is compared as the blocks reading found (paragraphs, headings, list items, table rows, footnotes). Headers and footers are left out because they repeat on every page. Each unit knows the clause it starts and the clause it sits in. A Markdown heading such as "## 7. Warranties" numbers itself.
  - **Alignment in passes:**
    1. Same wording, in order, with the clause number set aside. A clause renumbered by an insertion is "unchanged, renumbered 4.2 → 4.3".
    2. Shared wording, in order, between the stretches matched in pass 1: at least `LEGAL_COMPARE_MATCH_MIN` (0.5), with a 0.1 bonus for the same number.
    3. Moved: leftovers that share at least `LEGAL_COMPARE_MOVE_MIN` (0.8) of their wording.
    4. Same meaning: leftovers whose embeddings are within `LEGAL_COMPARE_MEANING_MIN` (0.85, nomic-embed-text) and that still share some wording. This pass is skipped, and the result says so, when the embedding model is unavailable.

    A pair outside the longest run in order is marked as moved. Results read in the right-hand version's order, with a removed clause shown after where its neighbour went.
  - **Within a pair:** a word diff marks exactly the characters removed and inserted, as offsets into each version's text. Nearby pieces are grouped, so "thirty (30)" → "sixty (60)" is one change.
  - **Labels:** a number changed, an obligation changed (shall, may, must, will …), a negation added or removed, or formatting only (case and quote style; a change in spacing alone is not a change).
  - **Exact by construction:** nothing in a comparison is generated. It is computed from the stored text (S2.1), so it is repeatable.
- **Explanations** (`compare/engine.py`):
  - The model explains one change at a time. It sees source 1 (left) and source 2 (right), fenced, with text addressed to an AI withheld. The changed words given as a hint are withheld and escaped too: a test found that an inserted "note to the AI" reached the model through the hint, and that is fixed.
  - It must quote inside its citations (with an example in the prompt), and every quote is verified as in research (D-037).
  - An added or removed clause has only one source; a bare citation of the missing side is ignored, not counted as failed.
  - The scope is re-checked while streaming. Explanations are saved with the comparison.
- **Conflicts across documents** (`compare/facts.py`, `compare/scan.py`):
  - **Documents:** the ones the person can see, in the version answers come from (signed by default, or the latest drafts).
  - **Facts:** amounts, periods (years become months, weeks become days, business days kept apart), percentages, dates, the governing law, companies with their legal form, and references to another agreement by its date. Each keeps its exact characters.
  - **Parties:** from the document's own words — "between", "To:", "From:", "Signed for", "Accepted for", "on behalf of", a defined role after the name ("Provider"), or the letterhead.
  - **Candidates:**
    - A company named with another legal form is compared between documents that name the same two companies or mention each other's title.
    - Amounts, periods, percentages, dates and the governing law are compared only between documents with the same parties. A law firm's own invoice terms are therefore never set against a supplier's.
    - Each statement is paired with its closest counterpart in the other document: by the eight words around the value (company names excluded), and only if that counterpart's own closest match is this statement. A counterpart with the same value means the documents agree. Dates are compared only when they mark the same kind of event.
    - A reference such as "the Master Services Agreement dated 2 February 2026" is checked against the date the referenced agreement gives itself.
  - **Reported as found:** a company's legal form, an agreement's date and the governing law each document chooses are self-contained statements. They are reported with their exact words, and whether they matter is the lawyer's call. In the tuning runs, the model talked itself out of both "Inc. vs Ltd" and a governing-law difference ("the order form supplements the MSA").
  - **Checked by the model:** amounts, periods, percentages and dates depend on context. The model gets both passages and each document's parties, says first what each passage governs and between whom, then gives CONFLICT or NO CONFLICT, then quotes both.
    - Findings are worded "the documents say different things", never which one is right.
    - Rejected candidates are kept as "checked and not in conflict".
    - If the model fails, a candidate is "unclear", never hidden.
- **Access and records:**
  - Comparisons and scans need `research.query`, read only versions the person may read, and are private to their owner (migration 0017).
  - A saved comparison is shown again only while every document it quotes is still visible to its owner. A scan leaves out findings from a document the owner can no longer see, and says how many are hidden.
  - A purge removes every comparison and scan that quotes the document, whole; the purge report counts them and the verification checks them.
  - Audit records hold counts only.
  - The API is `/api/legal/matters/{id}/compare` (plus `/scan` and `/{cid}/explain`), not the plan's `/contracts/compare`, since any two documents can be compared.
- **Evaluation** (`eval/law/run_compare_eval.py`, real model and embeddings on the host; TEST_RESULTS run 16):
  - **Golden (tuning) set:** the MSA v1 against v2 and against a restructured v3, plus an order form with five planted contradictions.
    - Final: alignment 17/17 changes correct, with nothing extra; explanation quotes verified 0.947, both wordings quoted in 6/6 changed clauses, 1.95 s per change; scan 5/5 planted, precision 1.0, 8 candidates, 3.8 s.
    - The first run was weaker, reported here for the record: scan precision 0.815, with 5 false conflicts from unrelated statements and a law firm's own invoices; explanation verification 0.81. The changes since are the ones described above: local context, mutual counterparts, same parties, facts reported as found, parties and reasoning in the prompt, missing sides, a citation example, and folding markup and quote style.
  - **Held-out set:** a lease draft and a lease side letter, never used to adjust the system and run once: alignment 10/10, explanation quotes verified 1.0 (both wordings 6/6), scan 4/5 planted with precision 1.0 and no false conflicts. The miss is the limit recorded below before the run: "the Lease dated 20 March 2024" is a one-word title.
- **Limits:**
  - A one-word title ("the Lease dated …") is not matched to a document, so that reference is not checked.
  - Facts outside the listed kinds (obligations in words, conditions, parties' roles) are not compared across documents; that is S3.5 and S3.8 work.
  - The meaning pass is calibrated for nomic-embed-text only.

### D-040 Legal transcription: the record, speakers, corrections and accuracy — RECORDED (S2.4; LIVE-01, D-017, R-7)
- **The record is the model's text** (D-017):
  - Law capture never used the legacy normaliser (audit). It stores the speech model's text as produced.
  - Segments now end at a sentence end once they have at least `LEGAL_SEGMENT_MIN_WORDS` (12) words, or at a speaker change, a pause or the end. Before, every ~0.5 s model chunk became a segment.
  - Segments are immutable: a database trigger refuses any change (migration 0018).
- **The last words are kept:** a pre-existing defect discarded the text pending at a normal stop. The final text is now stored unless a withdrawal, a revocation or the meeting's end stopped the capture. A withdrawal recorded since is checked under the same lock as withdrawals, so the S1.8 guarantee still holds: no text is stored after a withdrawal.
- **Speakers from the roster:**
  - The person recording marks who is speaking (`{"type": "speaker"}`; Alt+1…9 in the UI), from the meeting's consented participants. This replaces the legacy two-role toggle.
  - At each change, the model is flushed and restarted, as the legacy toggle did, so words are not carried across speakers. Times continue on the capture's audio clock.
  - Each change is an append-only capture mark.
  - There is no automatic speaker separation: the model cannot do it. Speakers can be corrected afterwards.
- **Wall-clock times:**
  - Every accepted frame is anchored to the time it arrived. A segment's audio times map to wall-clock start and end, so pauses show in the transcript; the audio clock has none.
  - The model now runs under its process lock, as legacy transcription does.
- **Nothing lost silently:**
  - A transcription error records a "gap" mark, tells the person ("degraded"), and appears in the transcript as "Not transcribed".
  - When the model falls more than `LEGAL_CAPTURE_LAG_WARN_SEC` (5 s) behind real time, the capture says so and records it, then records when it has caught up.
  - Frames were already never dropped: a slow model delays reading the socket.
- **Corrections are revisions** (`legal_transcript_revisions`, append-only):
  - Text or speaker, with who, when, why and the source (a person, or the matter's vocabulary with its provenance). What is shown is derived: the latest revision, else the raw text. The history and the text as heard are always shown.
  - Contributors may correct; read-only members may not.
  - A rollback below 0018 is refused while corrections exist.
- **Matter vocabulary:**
  - The meeting's participants, the companies named in documents the person can see (without their legal form) and the documents' defined terms are matched against what was heard (similarity at least 0.8).
  - Only differences in letters or word boundaries are offered ("harrow gate systems" → "Harrowgate Systems", "petra fos" → "Petra Voss"). Case or a possessive alone is not ("the provider's" is not "Provider").
  - A suggestion is applied only when accepted, and then becomes a revision with its document, version and characters.
- **Finalised transcripts are research sources** (R-7):
  - After the meeting, a contributor adds the transcript, as corrected, to research. It becomes a matter document (kind note, status final), read like any other, with a new version each time it is finalised again. It is marked when corrected since.
  - Research, compare, exports, holds and deletion apply as to any document.
  - Requesting deletion of a transcript hides that copy too. Restoring brings it back. Purging removes the segments, the corrections and the copy (files, passages and index entries), and the verification checks all of them.
  - Exports of a transcript use the corrected text with speakers, times and gaps.
- **Accuracy harness:**
  - `app.legal.meetings.accuracy` scores word error rate (verbatim and normalised: case, punctuation, filler; spoken numbers, years, ordinals and amounts in one form), entities (names, amounts, dates, references, counting spoken forms), negations kept, and speaker accuracy by word alignment.
  - `eval/law/transcription/` holds four fictional scripted conversations (client call, negotiation, board meeting, witness interview) and `run_transcription_eval.py`, with three modes:
    - **script:** a self-check, where everything must score perfectly (it does);
    - **file:** scores transcripts produced elsewhere;
    - **capture:** speaks each turn with a local Piper voice and streams it through the real capture socket, marking speakers.
- **Not measured yet, and why:** the real word error rate needs the speech model. On this host NeMo is not installed, PyTorch is CPU-only, and memory headroom is about 13 GB with swap full, beside production model servers. Installing and loading it here would risk production. The capture mode stops and says so; it runs on the Law speech stack once the first `compose.sh up` is approved (🔒). A Piper voice's licence must be checked before its results are published, and AMI stays out until its licence check.

### D-041 Silent Assist legal packs — RECORDED (S2.5; LIVE-02, D-015, P-008, T-03, T-06)
- **Not the legacy assistant:**
  - The legacy Silent Assistant (`silent_assistant/`) stays closed in Law. It fell back to the whole knowledge base (T-03), let the client choose the namespace, and had no confirm or dismiss and no telemetry.
  - The Law packs are new code (`legal/assist/`). They read only the matter's documents that the person recording can see, in their signed versions.
- **Packs:**
  - `legal_client_call` and `legal_negotiation`: signed terms, authority, commitment, disclosure, action.
  - `legal_governance`: quorum, conflict, authority, action.
  - The default follows the meeting's kind (board → governance, negotiation → negotiation, otherwise client call). A contributor can choose another pack or turn alerts off before the meeting.
- **Rules:**
  - **Signed terms:**
    - Amounts, periods, percentages and dates said are paired with their closest counterparts, by the words around the value (S2.3 facts).
    - A statement about "the agreement" is checked against agreements (agreement, contract, lease, deed, order form, amendment, side letter), never against a letter or minutes. In tuning, a law firm's engagement letter was cited as "the signed agreement", which is why this rule exists.
    - The model checks the two closest counterparts in turn: context words alone can tie, as with a fee clause and a cap that mentions "the Fees paid". If either has the same value, nothing differs.
    - A statement that names the agreement and agrees with it is marked as matching.
  - **Authority:** only when a figure is being agreed, offered or approved (whole verb forms), against a limit recorded with authority wording ("authorised … not reduced below USD 400,000").
  - **Commitment and disclosure:** narrow phrases, from the firm's side (and the client, for disclosure), only when a counterparty is a participant.
  - **Action:** "I/we will send, circulate, confirm …".
  - **Quorum:** fewer than half the directors present. No quorum rule is cited, so the alert says to check the articles.
  - **Conflict:** a vote by someone who declared an interest in it, earlier in the meeting or in the matter's minutes.
- **Evidence-relative wording (D-015):**
  - "Differs from signed agreement, cl. 4.1", "Matches …", "Needs review: …"; never "true" or "false".
  - Every alert quotes the exact sentence said, by its characters in the stored segment. A document source is quoted by its exact characters too.
  - Each alert carries its confidence, the threshold in force, and whether it was found by rule or checked in context.
  - A model check counts only if it quotes. Without a verifiable quote it scores 0.6, below the 0.7 threshold, so nothing is raised.
- **Thresholds await a lawyer (P-008):**
  - `LEGAL_ASSIST_MIN_CONFIDENCE` 0.7, the rules' confidences, and `LEGAL_ASSIST_NEAR_MIN` 0.2 are engineering defaults.
  - `LEGAL_ASSIST_THRESHOLDS_APPROVED` stays off until P-008 is signed off, and the UI says so.
- **Live:**
  - Each stored segment is checked in the background, so audio is never held up. Alerts are stored and sent on the capture socket.
  - None are sent after access is revoked or the person signs out. Checks still running at a normal stop may finish (`LEGAL_ASSIST_FINISH_SEC`).
  - Text addressed to an AI is withheld from the model (T08).
- **Decisions and telemetry:**
  - Confirm, or dismiss with a reason (at least three characters). Contributors only, decided once.
  - Telemetry: alerts, critical, per hour (only after five minutes of capture), confirmed, dismissed, dismiss rate, and latency from the words being stored to the alert.
  - The audit holds counts and the decision, never content.
- **Records:**
  - Alerts are matter content (migration 0019). Requesting deletion of a transcript hides them, and its purge removes them, verified.
  - A purged document's quote and name are removed from the alerts that cite it, which keep their decision.
- **Evaluation** (`eval/law/assist/run_assist_eval.py`): scripted calls played through the real capture socket and engine, with the real model checking subjects. There is no audio: speech accuracy is S2.4's.
  - **Golden (tuning) set, 3 calls:** recall 1.0 (8/8 critical), precision 1.0, no false alerts, latency p50 0 s and max 4.9 s. The first run had recall 7/8 and cited the engagement letter. Fixed by agreements-only signed terms and checking the two closest counterparts.
  - **Held-out set, 3 calls** (lease, NDA, board), never used to adjust the packs:
    - First run: recall 5/6, one "false" alert. That was the harness: two consecutive turns by one speaker formed one segment, and it numbered turns by segment, while the alert ("two years" against the NDA's five) was right.
    - With the harness corrected and identical system code: recall 1.0 (6/6), precision 1.0, max latency 2.6 s. Both runs are reported.
  - **These sets are small (six calls).** The numbers show the mechanism works; they are not a measured false-alarm rate. The lawyer-scored measurement is S2.8's.
- **Limits:**
  - Phrase rules miss other wordings: recall is bounded by the phrase lists, by design (conservative).
  - Speakers are as marked by the person recording (S2.4).
  - Spoken amounts in words ("eighteen thousand dollars") are not read as figures. Digits are, with the currency before or after.

### D-042 Boardroom legal mode: agenda, minutes and their approval — RECORDED (S2.6; MEET-01, D-037, T08)
- **Not the legacy Boardroom report.** The legacy report (`boardroom/service.py`) stays closed in Law. The audit found that:
  - it sent the model only the first 6,000 characters of the transcript (`service.py:806`);
  - it searched the whole knowledge base, not the matter (`:792`);
  - it dropped timestamps in its checks (`:863`);
  - it had no verification of what it wrote, and no approval.
  The Law minutes are new code (`legal/meetings/minutes.py`, `legal/artifacts.py`) over the Law transcript (S2.4).
- **Before the meeting:**
  - Agenda items (migration 0020) carry their board papers: matter documents the person can see.
  - "Research these papers" opens research limited to that item's documents. The page says so and offers "Ask the whole matter instead". Research checks the documents against the same visibility rules as any search.
- **During:**
  - The person recording marks the agenda item. Each stored segment keeps the item it falls under, and the change is a capture mark.
  - Consent, pause, gaps and Silent Assist are unchanged (S1.8, S2.4, S2.5).
- **After — minutes from the whole transcript:**
  - Drafting needs `ai.draft`, an ended meeting and a transcript that has not been deleted.
  - Each line of the transcript as corrected carries its segment, wall-clock time, roster speaker and agenda item, taken from the record, never from the model. Lines are sent in batches (`LEGAL_MINUTES_BATCH_CHARS`, 1,800 characters) and never split, so nothing is cut off.
  - The model returns one JSON object per line: a decision, an action (owner and deadline only if said), an unresolved point or a discussion point. Each item cites its lines with a short exact quote. A JSON-lines answer survives a cut-off: in the first tuning run, one pretty-printed object was cut off at 1,200 tokens and gave nothing.
  - **Verified (D-037):** every quote is checked against its line. An item with no verified quote is kept but shown as "Not verified against the transcript".
  - **Checked against the record (`guard`):**
    - An action must cite a line where someone commits, and a decision a line where it is taken; otherwise it is filed as discussion.
    - When the model names no owner, "I will …" makes the speaker of that line the owner. Role labels are stripped from owners.
    - A first-person commitment that no item cites is added by rule, quoting the line exactly.
  - **Merged deterministically:** items go under the agenda item of their first cited line, in time order, and repeats are merged with their sources. There is no second model pass that could invent a summary.
  - Text addressed to an AI is withheld from the model (T08).
- **Review and approval (the foundation of S2.7):**
  - Minutes are an AI artifact (`legal_ai_artifacts`). Each version is fixed: a trigger refuses content changes, and the content is hashed (SHA-256 of canonical JSON).
  - A person's edit (wording, kind, owner, deadline, removal, an added item, with a reason) is a new version made by that person. Every item keeps its sources, and a new version supersedes earlier drafts.
  - Approval needs `artifacts.approve` (lead, attorney or reviewer). It must come from someone other than the version's author, and it must name the content hash the reviewer read. Only the latest draft can be reviewed. "Request changes" needs a reason.
  - Decisions are append-only (`legal_review_approvals`). A rollback below 0020 refuses to delete them unless forced.
- **Filing:** only approved minutes can be filed. Filing makes a matter document, "Minutes — {meeting}", status final. Its text carries the approval (who, when) and the approved content hash. Filing again is a no-op.
- **Records:**
  - Drafts and their decisions are matter content. A transcript purge removes them, verified. Filed minutes are a document and are kept (the deletion report says so).
  - A legal hold blocks the purge (T04).
  - A purged document's link from the artifact is cleared.
- **Evaluation** (`eval/law/minutes/run_minutes_eval.py`): scripted meetings held through the real capture socket, with agenda and speaker marks, and minuted by the real model on this host through the real endpoint. Measured: planted decisions, actions and unresolved points found; owners and deadlines; verified quotes; and invented specifics (numbers or names never said, which must be 0).
  - **Golden (tuning) set, a board meeting, 26 turns, 2 batches.** The first run had 0 items (the cut-off above). Then recall 8/9 with owners 2/3. The guards brought it to recall 1.0, owners 3/3, deadlines 3/3, verified 1.0, invented 0.
  - **Held-out set, a lease committee, 12 turns,** never used to adjust the minutes: recall 6/7, owners 3/3, deadlines 3/3, verified 1.0, invented 0. It missed "Finance has not yet modelled the cost": it was minuted as discussion, not as an open point.
  - The held-out set was run on two system versions with identical results. The second version added "always minute attendance, quorum and interests" (from the golden set) and stripped role labels from owners (seen in both sets' output).
  - **These sets are small.** The numbers show the mechanism works; they are not a measured quality. Lawyer scoring is S2.8's.
- **Limits:**
  - The kind of an item is the model's choice within the guards, so a point can be minuted as discussion when a lawyer would call it open. The reviewer can change the kind.
  - Speakers and agenda items are as marked by the person recording.
  - An owner the model names is kept as given, not checked against the roster. The evaluation counts any name never said as invented, and the reviewer can correct it.

### D-043 Review and approval framework: research notes, one queue, hash binding — RECORDED (S2.7; RAG-01, MEET-01, SEC-06, T-09)
- **Audit:**
  - Research answers carried sources and a status, but no reviewer: the only way to share one was "Copy with citations".
  - Minutes (S2.6) were the only AI draft with an approval.
  - Approvals lived in five places (exports, deletions, holds, walls, emergency access), and none showed what was waiting.
  - **A real defect in the export gateway (S1.10):** redactions were planned on the version current at the request, but approval rendered the version current *at approval*. A new version uploaded in between (or a transcript corrected) would have been released with a plan made for other text, for example an account number added in v2 that nobody reviewed.
- **AI drafts made generic (`legal/artifacts.py`, migration 0021):**
  - Kinds are `minutes` (subject: meeting) and `research_note` (subject: research turn).
  - Shared rules: fixed versions with a content hash, approval by a second person bound to the hash, append-only decisions, and only approved versions filed.
  - When a newer version is approved, the earlier approval is marked superseded and no longer counts.
  - Every draft records the model and the prompt version (`research-s2.2`, `minutes-s2.6`).
- **Research notes:**
  - The person who asked can "Send for review" an answered, partly supported, unverified or abstained answer (not an error or a revoked scope).
  - It is fixed as given: question, answer, status and reason, evidence, the cited passages, conflicts, who asked and when.
  - Sending again opens the same note.
  - A note is shown, reviewed and filed only for people who can see **every** document it quotes (source permissions, S1.11). Anyone else gets the same 404 as for something that does not exist.
- **Generic routes:** `GET/POST /api/legal/matters/{id}/artifacts/{aid}[/review|/file]` serve any kind. Minutes keep their S2.6 routes too.
- **Approving (`ApprovalDialog`):** "Approve …" opens a confirmation that:
  - shows the full content hash it binds;
  - needs "I have read this version and checked its sources" (an attestation, not proof);
  - then approves exactly that hash.
  "Request changes" needs a reason. The author is told another lawyer approves.
- **`ReviewBanner`:** shows where a draft stands (draft, sent back with the reason, approved by whom and when with the hash, replaced), used by minutes and notes.
- **One queue (`GET /api/legal/reviews`, the "Reviews" page with a count in the navigation):**
  - **Waiting for you:**
    - AI drafts you may approve, did not make, whose sources you can all see, and that were not sent back;
    - exports for a second lawyer, of documents you can see;
    - deletions, holds and their release, walls and emergency access, for those who hold those decisions, never their own.
  - **Yours, with others:** your drafts (waiting, or sent back with the reason), exports and deletion requests.
  - Only matters you can open are read, so the queue never discloses another matter.
  - Reading the queue is not written to the audit trail, because the navigation polls it every two minutes (like the research history). Every decision taken from it is.
- **Hash binding:**
  - **Exports:** each export records `source_sha256`, the hash of the source's kind, version and readable text, when requested. Approval re-reads the source and, if anything differs, marks the export failed (`source_changed`) and refuses it: "The document changed after this export was requested… Ask for a new export."
  - Exports requested before 0021 have no hash, so they are refused with "requested before exports were bound to their text".
  - **Filings:** filing records the artifact, its kind and version, the document version, the body's hash, the content hash, and who approved and when (`legal_artifact_filings`, never updated). The approval is copied so the filed document keeps it after the draft is purged.
  - **Exporting a filed document:** the plan carries "Filed from … approved by …". A copy changed after approval (a new version uploaded) is refused: "Have the changes approved as a new version before exporting it."
- **Records:**
  - Purging a document removes every version of every research note that quotes it, with their decisions (the report counts them). A note can be sent again from the answer, whose quotes are removed too.
  - Filed notes are documents of their own and stay. A filing goes only with its document.
  - A rollback below 0021 refuses to delete research notes.
- **Limits:**
  - A note is a snapshot: later versions of its documents do not update it, by design.
  - Minutes still have their own review route alongside the generic one.
  - The queue lists at most what the five existing approval flows hold. Retention policies (approved on the Records page) are not in it.

### D-044 Phase 2 gate: evaluation records, one gate run, load test and lawyer review — RECORDED (S2.8; OPS-02, T12)
- **Evaluation runs are records (migration 0022):**
  - Each report imported by `python -m app.legal.evaluations import` keeps its suite, set, commit, models, the SHA-256 of the dataset file it read, its summary, and the report's name and hash.
  - The items (questions, answers, quotes) stay in the report file.
  - A report is imported once, and runs are never changed (a trigger).
  - Firm audit readers (firm admin, managing partner, records admin) see them at `GET /api/legal/evaluations` and on the "Evaluations" page; nobody else does.
- **One gate run on the code under test (`eval/law/run_gate.py`):**
  - Every suite runs as its own script with its own throwaway data, and the real models on this host.
  - Suites: research, compare, assist and minutes (golden and held-out), the scripted transcription harness, and the injection probe.
  - Held-out sets run once on this version. Nothing was changed after seeing them.
  - Latency figures come from the suite reports.
- **T12 load test (`scripts/law/load_test.py`):**
  - Against a loopback development backend seeded with the demo world: research (two users), search and a consented capture socket streaming real-time audio frames, all at once.
  - It reports p50/p95 per operation, failures, database-busy errors and the backend's memory.
  - The answer model and embeddings are shared with production on this host, so a measured run needs an approved window (`--window-approved`, 🔒). Without it only a 30-second smoke with one user per kind runs, no heavier than the browser tests.
  - Speech recognition under load needs the speech stack (🔒).
- **Lawyer review (the session is 🔒):**
  - A rubric with proposed bars for the owner (`releases/phase2-rubric.md`).
  - A shuffled packet of held-out outputs, labelled only by what each item is.
  - A scorer that reports every measure against its bar, with agreement between two reviewers. A measure below its bar is reported, never hidden.
- **The security suite** gains the Phase 2 groups: versions and evidence, compare, transcripts, Silent Assist, approvals and hash binding, evaluation records.
- **What the gate run showed (on `e0f6415`), against the slice results:**
  - Research golden fell to 60/64 (62/64 at S2.2), and T05 conflicts to 6/7. Held-out rose to 28/29 (27/29).
  - Minutes held-out deadlines fell to 2/3 ("next week" was not captured this time; 3/3 at S2.6).
  - The injection probe had 1 obeyed answer in 63, flagged on screen (0 unflagged; S1.12 had 0 of 36).
  - The system code under these suites did not change between the runs, so these are run-to-run variation of the model. They are reported, not tuned away: the held-out sets were not used to change anything.
- **"+changes" on gate reports:** each suite marked its commit `+changes` because the earlier suites' new report files were untracked; the code was `e0f6415` throughout (checked with `git status`). The evaluation scripts now ignore `eval/law/reports` when checking for changes.

### D-045 Legal drafting from approved firm templates, without invention — RECORDED (S3.1; DOC-01, P-008, T08)
- **Audit:** the legacy Document Studio (closed in Law, D-025) cannot be adapted.
  - Every section writer is told to "write finished, realistic prose with plausible specifics", and placeholders are banned (`docgen/pipeline.py:464-466`).
  - Its templates push for "concrete numbers" and "real dates".
  - Its sources are any document id in the global index, with no matter scope or permission check, cut to 6,000 characters.
  - It keeps no provenance, versions, owners or approvals, and has no Word output.
  Reusable pieces were limited to the ideas of the block schema and the fenced untrusted input. The Law code is new (`legal/drafting/`).
- **Firm templates (migration 0023):**
  - A small markup: `#` headings, paragraphs with `{{fields}}`, and `[[write: an instruction]]` for a passage written from the documents.
  - Each field declares its source: `documents` (with the question to ask them), `matter` (number, title, client), `person`, or `today`. The template is checked: no undeclared or unused field.
  - Versions are fixed (a trigger); firm styles (a Word file) may be attached while a version is a draft, and they are part of its hash.
  - `templates.manage` (managing partner, attorney) writes; `templates.approve` (managing partner, not the author) approves exactly the hash read, retiring the previous version.
  - Only approved versions are drafted from.
  - Six standard presets (engagement letter, mutual NDA, letter before claim, memorandum, contract review report, board minutes) install as drafts for the firm to review (P-008). The demo seed approves them.
- **Fields (`engine.extract`):**
  - Matter fields come from the record, and `today` is the drafting date. Person fields are left for a lawyer.
  - Document fields are asked of the passages the person can see (signed versions first, or only the chosen documents). The model answers with the bare fact and the exact words that state it.
  - The quote must be found in a passage (D-037), and the value must be in the quote: every figure for amounts, dates, periods and percentages, every word for names and text. An amount, date or percentage must contain a figure.
  - Otherwise the field is left `[[NEEDS INPUT]]`, and the model's suggestion is shown with the reason it was not used.
- **Written passages (`engine.write`):**
  - The model writes from the passages found and the confirmed fields only, citing each sentence with an exact quote, and writes `[[NEEDS INPUT: …]]` for anything missing.
  - Quotes are checked and citation marks become the passage's sources. A citation labelled by its clause number is matched to the source that holds its quote.
  - A sentence with no checked quote is marked `[[NEEDS INPUT: check against the documents]]`.
  - A figure no source or field contains becomes `[[NEEDS INPUT: confirm "…" — not in the sources]]`.
  - The model sees each matched passage in full (about 900 characters), not the narrow window search ranks by.
  - Text addressed to an AI is withheld (T08).
- **Review (S2.7):**
  - A draft is an AI artifact, `legal_draft`. A lawyer's change (a field filled or corrected, a written passage rewritten) is a new version marked as theirs; the firm's template wording changes only in a new template version.
  - A second lawyer approves a version only when no gap is left. The draft is shown only to people who can see every document it quotes.
  - Filing makes a Word document in the firm's styles (gaps highlighted, metadata cleared, the footer naming the matter, template version and approval with its hash) and records the filing.
  - A purge of a quoted document removes unfiled drafts.
  - PDF and Word previews say "DRAFT — NOT APPROVED" until approval.
- **Found while building it:**
  - **The shared model client cut JSON answers.** Its prose tidy-up trims a reply that does not end on a sentence back to its last sentence, which cut the tail off every JSON object. It cut drafting fields to `"quote":`, and **the last minuted item of every minutes batch since S2.6**.
  - Structured answers now use `llm.complete` (the reply as written, a new `chat_raw` on the client), with tests that fail without it. Minutes were re-measured: unchanged on these sets (golden recall 1.0, held-out 6/7, nothing invented).
- **Evaluation** (`eval/law/drafting/run_drafting_eval.py`): fictional matters drafted from the presets by the real model.
  - **Golden (tuning, 4 scenarios):** first run 0/9 fields (the JSON cut), then 6/9, 8/9 and 9/9. A contract-review scenario was added to the tuning set for written-passage citations. Final: fields 11/12, absent facts left open 2/2, **wrong values 0, invented figures 0**, unsupported sentences 0, about 9 s per draft.
  - **Held-out (3 scenarios, never used to tune), run four times as defects in the code were fixed, and reported:**
    - Run 1: fields 9/10, absent 2/2, wrong 0, invented 0. It exposed two parser defects: numbers joined across lines, so a stated date counted as invented; and clause-number citations not read.
    - Run 2: 8/10. It showed uncited sentences left in the text (one wrong: "The date of the Lease is 10 October 2026"). That led to the rule that every unsupported sentence is a gap.
    - Run 3: 9/10. It showed duplicate markers, and passages too narrow for the writer, which a new tuning scenario confirmed.
    - **Final run (4):** fields 8/10 (the agreement title twice not stated: cautious, not wrong), absent 2/2, **wrong 0, invented 0**, 4 unsupported sentences, each marked for a lawyer.
  - **Small sets.** The lawyer review measures quality.
- **Limits:**
  - The model sometimes declines a fact a document does state, which leaves a gap for a person: the safe side.
  - Written passages can repeat themselves; the reviewer edits them.
  - Field values are taken as the document words them (for example "14 Quay Street, Gothenburg 411 04, Sweden").
  - Word round trips are checked automatically through LibreOffice. The manual Microsoft Word check belongs to S3.3 (🔒).

### D-046 Clause library and playbook checks — RECORDED (S3.2; DOC-02a, P-008, T08)
- **Audit:** nothing in the legacy code or in Law held a firm position on a clause. Law already had what a check needs:
  - clauses with exact character ranges (S2.1);
  - the fact reader for amounts, periods, percentages and governing law (S2.3);
  - search limited to one document (S2.6);
  - checked quotes (D-037).
- **Playbooks (migration 0024):**
  - A playbook (for example "Commercial services, supplier") holds positions by topic: twelve topics, including liability cap, excluded losses, term, termination, payment, confidentiality, governing law, assignment, indemnity, data protection, force majeure and warranties.
  - Each position has the standard wording, up to five fallbacks (each with a note such as who must agree), guidance and an owner.
  - Versions are fixed (a trigger). `playbooks.manage` (managing partner, attorney) writes; `playbooks.approve` (managing partner, not the author) approves exactly the hash read, retiring the version before it.
  - Lawyers who do not manage playbooks see approved positions only.
- **From the firm's precedents:** a precedent (Word, PDF or text) is read with the S2.1 reader. Its leaf clauses are proposed with the heading above them and the topic their words suggest. Nothing is stored until a lawyer adds a position, which then waits for approval.
- **The check (`playbooks/check.py`), for each approved position:**
  - **Candidates:** the document's leaf clauses that say most about the topic, by their words and by a search of that document alone; four at most, or the matched passages if the document has no clause numbers.
  - **The model** sees the standard, the numbered fallbacks and the candidates. It answers which clause deals with the topic, the exact words that decide it, and standard, fallback n, or outside.
  - **Checks on the answer:**
    - Words not found in the clause give **unclear**, for a lawyer.
    - A fallback the position does not have gives **unclear**.
    - **The figures decide:** a different amount, period, percentage or law from the chosen position makes the line **outside**, by rule. So does a clause that combines positions ("the greater of USD 250,000 and the fees") when the position does not.
    - No candidate, or no clause named, gives **not found**.
  - **Advisory throughout:** nothing is called compliant. Each line has Agree, or Disagree with a reason, for a lead, attorney or reviewer (append-only; the latest counts).
  - A check is shown only to people who can see the document. It goes with its document in a purge. Text addressed to an AI is withheld (T08).
- **Evaluation** (`eval/law/playbook/run_playbook_eval.py`): a fictional playbook approved, and contracts checked through the real endpoint with the real model.
  - **Golden (tuning):**
    - First run 12/12, with no deviation passed in error.
    - The demo journey then showed "the greater of USD 250,000 or the fees" classed standard, so that case joined the tuning set. With only a prompt change, the model called it fallback 1 (passed in error, 14/15).
    - The combination rule followed: **15/15**, deviations caught **5/5**, **passed in error 0**.
  - **Held-out (never tuned on):** run on both versions with the same result: **9/10**, deviations caught **4/4** (one as "unclear": the model named a fallback that does not exist), **passed in error 0**.
  - **Small sets.** The lawyer review measures quality.
- **Limits:**
  - Topics are found by their words and a search, so a clause on two topics can be shown for both.
  - Positions in prose ("reasonable notice") are judged by the model alone; only figures and laws have the rule.
  - Suggested revisions and redlines are S3.3.
- **`playbook-s3.2c` (S3.7):**
  - **The finding:** the full browser run found the model naming "fallback 1" for a position with no fallbacks. The line went to "unclear", although the clause's thirty days matched none of the firm's options.
  - **The change:** when the model names an option the position does not have, the figures decide if they match none of the options (outside, by rule). Otherwise a lawyer still decides.
  - **Golden runs:** 14/15 and then 15/15 on two runs. The one miss was a "not found" answer, a run-to-run variation in the model on a path the change does not touch.
  - **Held-out:** 9/10, deviations 4/4, passed in error 0, as before.

### D-047 Redlines and Word tracked changes: the D-011 spike passes its automated checks — RECORDED (S3.3; DOC-02b, T10, P-013)
- **Audit:**
  - No tracked-changes writer existed. The S2.1 reader already read `w:ins`/`w:del` as accepted and showed them; the S1.10 export deliberately rebuilds Word without revisions.
  - The S2.3 word diff aligns clauses rather than writing them.
  - Comments were written with lxml directly (raw OOXML, as D-011 planned); the python-docx upgrade was not needed.
- **The writer (`redlines/word.py`), as D-011 planned:**
  - A clause is found by its words, across one or more paragraphs, ignoring numbers Word generates ("12.3", "(a)", "Article I"), hidden runs and the reader's footnote marks.
  - Paragraphs are paired in order. Each pair becomes a word-level diff:
    - plain runs, `w:del`/`w:delText` and `w:ins`, each with `w:author` and `w:date`;
    - changes separated only by spaces read as one change;
    - the paragraph's properties are kept, and the runs take its first visible run's formatting.
  - An unpaired original paragraph is deleted whole, including its paragraph mark. A new paragraph is inserted whole, keeping the style but not Word's generated number.
  - Hidden runs and bookmarks in a revised paragraph are kept as they were. A comment with the reason is anchored on the clause, with the comments part, relationship and content type added when missing.
  - A clause that cannot be found is reported, never guessed.
  - Text and PDF documents are first written as a plain Word file, one paragraph per line.
- **T10, checked before every release:** `verify` replays each change on exactly the paragraphs it matched.
  - Accepting all must give the original with each clause revised and nothing else.
  - Rejecting all must give the original, including the file's own earlier tracked changes.
  - Every change must have an author and a date, and the comments part must exist.
  - A file that fails is not sent or filed (409).
  - Tests use the S2.1 Word fixture (automatic numbering, existing tracked changes, a comment, hidden text, a table, a footnote). LibreOffice opens the result and keeps the changes and their author (tested).
- **Redlines (`redlines/service.py`):**
  - From a playbook check (S3.2), each line outside the playbook (or each topic the lawyer names) is revised to the approved standard, or to the fallback the lawyer picks. The clause's own number is kept, and only the clause's own text is revised: never the next heading, clause or a table.
  - **The model only fits the firm's wording to the contract's defined terms.** It sees the firm's wording and the contract's defined terms (S2.1), never the clause's substance. The fit is kept only if every word it adds appears in the clause, and its figures, periods, percentages and laws are the firm's. Otherwise the firm's wording is kept, and the redline says why.
  - With the real model, the first prompt (which showed the whole clause) produced "fits" that copied the contract's substance back ("thirty (30) days", "Delaware"). The check rejected every one. The prompt now shows only the defined terms.
  - A revision identical to the clause is skipped, and "nothing to revise" is said plainly.
  - A redline is an AI artifact (S2.7): edits are new versions (wording, a fallback, a change dropped), and a second lawyer approves.
  - Filing adds the Word file with tracked changes as the document's next version (status draft) and records the filing. The exports rule of S2.7 applies to that version only: later versions of the client's document are the client's.
  - It is visible only with the document and purged with it.
- **Robustness evaluation (`eval/law/redline/run_redline_eval.py`, no model):** every clause of 14 documents (the Markdown corpora and the Word fixture) was revised four ways: a figure changed, words appended, shortened to a sentence, and a paragraph added.
  - First run: 489 revisions, 478 placed and 478 verified, 11 not placed.
  - The 11 were "Article I", hidden text, and a clause running into a table and a footnote. All were refused rather than guessed, then fixed.
  - **Final: 487/487 placed and verified, LibreOffice round trips 12/12 kept the tracked changes.**
- **Not yet done: the manual check in Microsoft Word (🔒, a person).** Until someone opens these files in Word, the product says "Not yet verified by a person in Microsoft Word", and DOC-02 is not called DONE (P-013).
- **Limits:**
  - A revised paragraph takes its first run's formatting: bold or italic words inside it are lost in the revision.
  - Fields and hyperlinks inside a revised paragraph are not kept.
  - A clause inside a table cell is not revised.

### D-048 Citation and quote checker: rules only, nothing looked up — RECORDED (S3.4; DOC-03, T07, P-009)
- **Audit:**
  - The research evidence check (D-037) verified the model's own quotes against retrieved passages. It did not read a document for what *it* cites.
  - The S2.1 reader already stores every clause with exact character ranges, including "Schedule 1", "Article I", "§ 2" and "3.3(b)". The S3.3 word diff already showed changed words.
- **No model.** The check is deterministic, so the same document always gives the same result.
  - Any member who can see the document may run a check (`research.query`, as IMPLEMENTATION_PLAN §5 planned). This includes a reviewer checking a memo before approving it. A first draft used `ai.draft`, which left reviewers out, and was changed before merge.
  - Notes are for a lead, attorney or reviewer. Audit actions: `check.citations`, `check.citations.note`.
- **What is read (`citations/parse.py`):**
  - **References:** clause, section, §, paragraph, article, schedule, exhibit, annex and appendix.
    - It reads lists and ranges ("Clauses 2.1 and 9.3", "clauses 10 to 12": both ends).
    - It reads "of the <named agreement>" and short names the document defines ('the Share Purchase Agreement (the "SPA")').
    - It reads "paragraph 2 of Schedule 3".
    - A clause or schedule's own heading is not a reference to itself.
  - **Quotations:** quotation marks are paired in order: curly ones by shape (single ‘UK’ quotes included, apostrophes excluded), straight ones alternately. A quotation never runs across a blank line.
    - Four words or more are checked, or two when a clause reference introduces them.
    - Definitions ("X" means …) are not quotations.
    - A reference that introduces a quotation is checked through it: the quotation must be in that clause, and the item shows the reference it was cited as.
  - **External authorities:** US Code and CFR sections, US cases with reporters, UK neutral citations, UK law reports (old and new style), UK Acts, EU regulations and directives, and CJEU cases.
- **Resolution (`citations/check.py`):**
  - A reference resolves to the exact clause of this document, or of a matter document the person can open whose title matches the name used. An exact title is preferred, then executed, final, received, draft.
  - A reference with no document named is read in this document. If this document has no such clause, it is read in the agreement named earlier in the paragraph, and the item says so.
  - **Missing:** the document exists but has no such clause ("The Master Services Agreement has no clause 18.3."). The item keeps which document was searched.
  - **Not found:** no document the person can open has that name. A document they cannot see reads exactly like one that does not exist (T06).
  - **Quotations:**
    - A quotation is compared word for word with the closest passage: first in the cited clause, then in the whole document, in the named document, and then in this one (its own words excluded).
    - The closest passage is chosen by the fewest words changed, so a changed word reads as a substitution.
    - Ellipses split a quotation into parts found in order. A bracketed capital ("[t]he") is accurate.
    - **Altered** shows the source's words struck and the quotation's words underlined. A quotation found in a different clause from the one cited says so.
  - **External authorities are always UNVERIFIED (P-009):** they are listed and never looked up (`external_lookup: false`), and a notice on every check says so.
  - **Malformed:** a citation that cannot be real is flagged:
    - an unknown reporter (F.5th, So. 4th);
    - a numbered circuit above the 11th;
    - a year in the future;
    - a UK court citing a year before it sat, or the House of Lords after 2009;
    - a US Code title above 54, or a CFR title above 50.
- **What is stored (migration 0026):**
  - A check keeps where each item points (document, version, offsets), never the target's words.
  - The passage and the changed words are read again each time the check is shown, and only for a person who can open that document. Anyone else sees "not found".
  - A purge of the checked document removes its checks and notes. A purge of a document others' checks point into leaves those items as "The document it pointed to was deleted.", and verification counts any trace.
  - Notes are append-only (a trigger). Rolling back below 0026 is refused while notes exist.
- **Evaluation (T07, `eval/law/citations/run_citations_eval.py`, no model):** fictional documents with planted citations, run through the real endpoint. The expectations are what a careful lawyer would conclude by hand.
  - **Golden (tuning):**
    - Writing it found four gaps: lists, a capitalised "The Lease", old English reports, and future-year EU instruments. A heading read as a reference was also found and fixed.
    - First complete run: **30/30**, no extra items.
  - **Held-out, first run (`citations-s3.4`): 20/25.** All authorities were right and all missing clauses were caught, but there were three defects:
    - a short defined name, `(the "SPA")`, paired every later quotation mark the wrong way;
    - "clause 7.2 of the SPA" was looked up in the memo (a false "missing");
    - "paragraph 1 of Schedule 4" was resolved in the memo.

    These were fixed in `citations-s3.4b`. Because this set had now informed changes, it became tuning data (G5–G7). A **new held-out set was written and committed before s3.4b was run** (commit c8e8abc).
  - **Golden on s3.4b: 55/55**:
    - missing caught 9/9, false missing 0;
    - altered caught 3/3, false alarms 0;
    - malformed caught 5/5, wrongly flagged 0;
    - all authorities unverified, no lookups.
  - **Held-out, second run (s3.4b): 27/28**:
    - references 11/11, quotations 3/3, malformed caught 3/3, wrongly flagged 0, no lookups.
    - The miss is "Fed. R. Civ. P. 81(c)": **court rules (US Federal Rules, English CPR) are not recognised, so they are not listed at all.** This is recorded, not tuned on this set; it is to be fixed with a fresh held-out set at the Phase 3 gate (S3.9).
  - **Small sets.** The lawyer review measures usefulness.
- **Limits:**
  - Court rules are not read (above). Neither are treaties, state statutes, law-review articles or US "Id." and "supra" short forms.
  - A range ("clauses 10 to 12") checks both ends, not the clauses between.
  - A document is matched by its title, not by how it is described ("the engagement letter dated 2 March").
  - An abbreviation is only understood when the document defines it.
  - "Malformed" catches citations that cannot exist, not ones that merely do not: a well-formed invented case is UNVERIFIED, like every other authority, until a person checks it in a licensed source (P-009).
- **Later versions:**
  - `citations-s3.4c` resolves clauses known only by a Markdown heading ("## 12. Termination"; S3.5's `documents.numbering`).
  - `citations-s3.4d` follows that helper's schedule scoping.
  - T07 was unchanged on both: golden 55/55, held-out 27/28.

### D-049 Defined terms and cross-references: rules only, corrections only through a redline — RECORDED (S3.5; DOC-04)
- **Audit:**
  - The S2.1 reader keeps only a term's first definition, so the checker reads definitions again from the text.
  - The reader does not see Markdown headings ("## 4. Fees") as clauses.
  - It read "(c)" after "(a)" as roman c, nesting it under (a).
  - Fixed in the reader: a roman list starts at (i) and counts on, and an item's roman status is tracked, so "(i)" under "(c)" nests.
  - `documents.numbering` (new, used by both checkers) completes the clause list:
    - clause 4 when only 4.1 and 4.2 were seen, with the heading from its heading line;
    - Markdown schedules ("## Schedule 1 Products");
    - a schedule's own "1." never hides clause 1 of the body.
- **No model; any member who can see the document may run a check** (`research.query`; audit `check.definitions`). Notes are for lawyers.
- **What is found (`definitions/check.py`), each with exact offsets and the clause it is in:**
  - **Definitions:**
    - every definition: "X" means / includes / has the meaning; (the "X"); each a "X";
    - a term defined twice, saying whether the meanings differ. A signpost ("has the meaning given in clause 2.2") is not a second definition;
    - a definition never used. Parties' short names and terms used only in another form are not reported;
    - a defined term used in another form: "the fees" for "Fees" (a single word only after "the", "any", "such"…), "confidential information";
    - a term used before it is defined, unless defined in a Definitions or Interpretation clause.
  - **Undefined terms** (heuristic, advisory). A capitalised phrase reads as a term when:
    - it ends with a word defined terms usually end with (Date, Period, Fee, Information…);
    - or it starts with a defined term ("Shareholder Consent");
    - or it is a single word used at least twice after "a", "the" or a number, standing as a noun ("five Directors", "a Director"; not "the Harrowgate platform").

    Company names, documents ("Order Form"), headings, names defined in place and possessives are excluded.
  - **Numbering:**
    - gaps and repeats at every level: clauses, sub-clauses, items "(a)", roman items, schedules and articles;
    - numbering is never changed automatically.
  - **References:**
    - this document's own references that point nowhere ("This document has no clause 12."), one finding per missing clause with every place;
    - references whose named heading does not match ("Clause 4 is headed Confidentiality, not Liability.").

    References to other documents are S3.4's.
- **Corrections:**
  - A finding may offer corrections:
    - the defined form of a term;
    - keeping a reference's number with the right heading, or pointing it to the clause with that heading;
    - deleting an unused definition that is a clause of its own.
  - A lawyer (`ai.draft`) picks at most one per finding. The picks become a **redline artifact** (S3.3): one change per clause, only the words named, with tracked changes and comments. A second lawyer approves it, and only then is it filed as the next version.
  - The Word writer now deletes a clause whole when its revision is empty (T10 verification holds).
  - The redline page says the redline came from the definitions check.
- **Storage (migration 0027):** checks with the document, purged with it; append-only notes; rollback refused while notes exist.
- **Evaluation (`eval/law/definitions/run_definitions_eval.py`, no model):** planted drafting errors and clean controls. Each held-out set was written and committed before its run, and became tuning data once it informed a change.
  - **Golden on the first version:** the first complete run caught 26/26 but gave 7 false findings, including in the control. These were tuned away (plurals, party names, "each a", possessives, the reader's (i) under (c)): **26/26, fixes 4/4, false findings 0, control clean.**
  - **Held-out 1 (s3.5): 8/12 caught.** Missed: undefined "Board", "Directors" and "Shareholder Consent"; Article 5 reported as "Clause 5".
  - **Held-out 2 (s3.5b): 13/16 caught, 5 unexpected, control not clean.** The causes:
    - a schedule's "1." hid the body's Definitions clause;
    - a signpost definition was read as a second definition;
    - "Brightline Data, Inc." was read as a term;
    - a list's second item was shown as "7".

    One unexpected finding was right: the control really does define a term it never uses, and its expectation was corrected when it moved to golden.
  - **Held-out 3 (s3.5c): 12/12 caught, fixes 3/3, control clean.** One unexpected finding, "Clause 5.1 appears twice", is true but redundant: it follows from the planted duplicate clause 5.
  - **Golden on s3.5c (G1–G10, both earlier held-out sets included): 52/55, fixes 8/8, false findings 0, controls clean.** The three misses are undefined terms the rules do not recognise: "Board" (used once), "Background IP" and "Statement of Work".
- **Limits:**
  - Undefined-term detection is a heuristic, recall about 8 in 11 on these sets. Terms not shaped like defined terms are missed.
  - Ranges are not expanded.
  - "Part 2 of Schedule 1" is not read.
  - Definitions in tables are not read.
  - A gap is reported, never renumbered.

### D-050 Obligations and deadlines: rules first, checked model proposals, no date guessed — RECORDED (S3.6; DOC-05)
- **Audit:**
  - Silent Assist (S2.5) and minutes (S2.6) take action items from meetings, not contracts.
  - The S2.3 fact reader gives periods and dates; S2.1 clauses and `documents.numbering` (S3.5) give where each sentence is.
- **Reading a contract (`obligations/extract.py`, rules):** a clause sentence that binds someone ("shall", "must", "may", "is payable", "survives", "unless … gives notice") and gives a time, or a consent right, becomes a proposed obligation. It keeps:
  - its kind: expiry, renewal, termination (break rights included), change of control, consent, payment, notification, post-termination, confidentiality, survival, duty or notice. The kind is read from the main clause, not from its "unless …" condition;
  - the party, as the document names it;
  - the action, and when, both in the document's words;
  - the exact sentence, at its offsets, without a heading line read into it.
- **Dates come only from dates the document fixes,** with the counting said and to be confirmed:
  - a term's end: N months or years from a dated start ("the Effective Date", "the date of this agreement", "from and including"), read as the day before the anniversary;
  - a non-renewal deadline counted back from that end;
  - a break date (the Nth anniversary) less its notice period;
  - a dated event ("shall take place on …", "shall release … on …") or a defined date ("on the Maturity Date");
  - a period after a dated event ("within eighteen (18) months after Completion"; not business days).

  Everything else runs from an event (an invoice, a breach, termination) and has no date until it happens. A date is never guessed.
- **The model (`obligations/model.py`, on by default, `LEGAL_OBLIGATIONS_MODEL`)** proposes obligations only for binding sentences the rules did not read. Each proposal is checked:
  - its timing must be copied from the sentence, or it is dropped;
  - a party the sentence does not name is left blank;
  - its action must be the sentence's words, or the rules' reading is used;
  - **its dates are never used**: the rules compute one from the sentence or there is none.

  AI-addressed text is withheld (T08). If the model cannot be reached, the rules' findings stand and the page says so.
- **The tracker (migration 0028):**
  - Everything is proposed until a lawyer (lead, attorney, reviewer) confirms it, with an owner who is a member of the matter.
  - Edits (date, timezone, party, wording), done, dismiss (with a reason) and reopen are versions; a reason is required to edit or dismiss. The history is append-only, and the audit trail holds no content.
  - Dates are shown at the end of the day in a timezone (`LEGAL_TIMEZONE`, UTC until the firm sets it), and overdue is judged by it.
  - The calendar file is off unless `LEGAL_OBLIGATIONS_ICS` is on, and then holds confirmed, dated obligations only.
  - Obligations are shown only with their document and purged with it.
- **Evaluation (`eval/law/obligations/run_obligations_eval.py`):** fictional contracts with a lawyer's deadlines list and controls with none. Each held-out set was committed before its run and became tuning data after.
  - **Golden, first run (rules):** 10/21 found, dates 1/5. Tuned to 21/21 and 5/5; one expectation was renamed from confidentiality to survival (a naming choice), and recorded.
  - **Held-out 1 (s3.6b): 9/13** (11/13 counting two found under another kind); 0/4 dates. The runner first reported 2 guessed dates; that was its own bug (an empty date counted as a guess), corrected from the saved results without a re-run: **0 guessed**.
  - **Held-out 2 (s3.6c): 5/10,** 0/2 dates, none guessed, control clean.
  - **Held-out 3 (s3.6d): 5/12,** 0/3 dates, none guessed, control clean.

    **The rules alone found about half of what a lawyer lists in unseen wording**, while never guessing a date. This is why the checked model pass was added.
  - **Held-out 4 (s3.6e, rules and the model): 10/12** (11/12 counting one found under another kind), dates 2/3, **none guessed, control clean.** The control was an engagement letter whose "We will advise you" was correctly not listed. The miss: "fifteen (15) years from and including 24 June 2026".
  - **Golden (G1–G13) with the model: 56/56, dates 15/15, none guessed, controls 4/4.** Three extras, all in the demo MSA and arguably trackable: the notice formalities, the monthly fee and monthly invoicing.
- **Limits:**
  - Business days are not counted into dates.
  - Recurring deadlines ("quarterly", "each anniversary", "by the fifth Business Day of each month") have no next date computed.
  - A sentence gives at most one item of each kind.
  - The model's kind can differ from a lawyer's ("notice" for a repair notice duty).
  - Reminders are not sent.
- **Later versions (S3.7):**
  - `s3.6f` reads a term counted from a literal date ("from 1 July 2024", "from and including 24 June 2026"); golden G1–G16 with the model: 67/68, dates 18/18, none guessed.
  - **Held-out 5 (s3.6f): 4/6,** none guessed, control clean. "End this Agreement" was read as notice, and "shall expire two (2) years after the Effective Date" had no date: a bug, the anchor was cut to its first word.
  - `s3.6g` fixes both. It is to be measured on a fresh held-out set at the Phase 3 gate.

### D-051 Due diligence: firm checklists, data-room runs, a preliminary issue register — RECORDED (S3.7; DOC-06)
- **Audit:** the job runner (S1.6) takes new job kinds. The S3.6 obligations reader, the S2.3 governing-law reader and S2.1 clauses give the questions; S3.2's approval-by-hash pattern gives checklists.
- **Checklists:**
  - Items are grouped by section. Each has the phrases that name the documents expected and the questions asked of them: change of control, assignment, termination, governing law, liability, renewal.
  - Versions are fixed (a trigger). A managing partner other than the author approves exactly the version read (its hash), which retires the version before it (`diligence.manage` / `diligence.approve`, P-008).
  - A preset ("Share acquisition — core documents", ten items) is seeded for the demo.
- **A run is a background job (`diligence.run`, migration 0029):**
  - Each item is matched to the documents whose own title or first line names it; a document's body mentioning "the supply agreement" does not make it one. Unanswered items are the missing-document list.
  - Matched documents are asked their item's questions by rules:
    - change of control: every binding sentence about it;
    - assignment: a consent right;
    - termination: convenience, break dates, other rights;
    - governing law, compared with the checklist's expected law;
    - liability: a cap is low, unlimited is high;
    - renewal: the deadline, high if it has passed, medium within six months; and a term ending within a year.

    Each issue quotes its sentence at its offsets and names its clause.
  - The documents read are those the person who started the run can see. What is shown is filtered again for each viewer: issues on a document a viewer cannot open are not shown, and neither is its name.
- **Review:**
  - Every issue is **preliminary** until a lawyer (lead, attorney, reviewer) verifies it (with an optional note) or dismisses it (with a reason); reopening is allowed. Versions are checked.
  - The trail is append-only, and the audit holds no content. Rolling back below 0029 is refused while checklists or decisions exist.
  - A purge of a document removes its issues and trail, and every run forgets its name.
- **Evaluation (`eval/law/diligence/run_diligence_eval.py`, no model):** synthetic data rooms against the preset.
  - **Golden, first run:** missing lists 3/3, issues 18/18, no extras, control clean.
  - **Held-out 1 (s3.7): missing 3/3, issues 9/11, no extras, control clean.** The misses were a term counted from a literal date (fixed in the obligations reader: s3.6f, s3.7b). Golden G1–G6: 29/29.
  - **Held-out 2 (s3.7b): missing 2/2, issues 5/6, no extras, control clean.** The miss: a termination right with no notice period. Fixed in s3.7c, to be measured at the Phase 3 gate.
  - Both held-out sets were committed before their runs.
- **Limits:**
  - Matching is by title words, so a document named oddly is reported missing (the reviewer sees the full document list).
  - Questions are rules only, with no model pass here.
  - No deal-level report export yet; the register is the record.
  - Checklist items cannot ask custom questions.

### D-052 Governance checks: a company's documents read against each other, advisory only — RECORDED (S3.8; DOC-07)
- **Where it lives:** governance checks are part of a due-diligence run (S3.7). A checklist asks for them by name (`share_capital`, `authority`, `constitution`, `ip_chain`), and their findings join the same register: preliminary until a reviewer decides, with the same append-only trail.
  - A firm preset, "Corporate governance review", is seeded and approved for the demo.
  - Journey 29 found the seed dropping a preset's checks; fixed, with a test.
- **No model and no legal decision:** every finding is a discrepancy for a lawyer to judge. Documents are recognised by their own title line (register, articles or bylaws, minutes or written consents, delegation of authority, policies, IP assignments).
- **The checks:**
  - **Share capital:** the register's or cap table's holdings by class against the shares approved in resolutions or consents, and against the classes the articles create. Both "N ordinary shares" and "N shares of common stock" are read, and a nominal value ("of GBP 1 each") is not part of a class.
  - **Authority:** a contract's stated total value against the signing limit, in the delegation of authority, of the role that signed it ("Signed for … by J. Park, Finance Director"). A role the delegation does not list is left to the reviewer.
  - **Constitution:** a policy against the articles on the board quorum, the notice of board meetings (hours and days compared), and the minimum number of directors.
  - **IP chain:** each founder named in the company's documents against the IP or invention assignments.
- **Two documents:** each finding quotes both documents, at their offsets (migration 0030, `related_json`).
  - A finding is shown only to someone who can open both documents, and it is purged with either.
  - Rolling back below 0030 is refused while such findings exist.
- **Evaluation (`eval/law/governance/run_governance_eval.py`):** fictional companies against the preset.
  - **Golden, first run: 9/9,** no extras, control clean.
  - **Held-out 1 (s3.8), in other styles: 3/6, no extras, control clean.** The misses were US stock wording, a signature line cut at "Inc.", and notice in hours.
  - These were fixed in s3.8b, together with two faults the moved set then showed in golden: a "written consent of the board" not taken as an approval, and "ordinary shares of GBP 1 each" taken as a new class (a false finding).
  - **Golden G1–G6: 15/15, no extras, controls clean.** s3.8b is to be measured on a fresh held-out set at the Phase 3 gate.
- **Limits:**
  - Only the phrasings above are read: a quorum "of a majority" or approvals by option plans are not.
  - A contract with no stated total value is not checked.
  - Transfers between holders are not reconciled.
  - Findings are advisory and never change a document.

### D-053 The Phase 3 gate: fresh blind sets, fixes measured on new ones, a review packet — RECORDED (S3.9; OPS-02, T07, T10)
- **Held-out discipline:**
  - Every checker changed since its last held-out run got a fresh blind set, committed before it was run (`99c1342`).
  - When a run led to a fix, the version changed, the set moved to golden, and a new blind set was written and committed before that version was run.
  - Every run is reported in `releases/phase3-gate.md` §2, including the runs that failed.
  - **The gate runner cites a held-out report rather than running it again** when the report was made on the current rules or prompt version and the same held-out file.
- **Gate runner (`eval/law/run_gate.py --phase 3`):** runs the drafting, playbook, T10 redline, citation, definitions, obligations, diligence and governance suites (golden and held-out), then `scripts/law/security_suite.sh` and the Law browser journeys. Their counts are kept as reports, and the evaluation runs are imported for the Evaluations page.
- **Fixes made at the gate (each with a test written red first):**
  - `citations-s3.4f`: court rules keep lettered sub-paragraphs.
  - `citations-s3.4g`: UK Act names with joining words or a bracketed part. A word that opens a sentence (See, Under, The) is not part of the name.
  - `drafting-s3.1b`: a party already named by an earlier field is left for a person, with the value as a suggestion and the reason.
  - `drafting-s3.1c`: a field kind `agreement`. A value naming no document is left for a person. The demand-letter and contract-review presets use it, and the template editor offers it. Templates already installed keep their kinds until a new version.
  - `drafting-s3.1d`, in two parts:
    - `retrieval.search(floor=False)`: drafting no longer applies the research answerer's abstention floor (`LEGAL_RERANK_MIN_SCORE`). Every drafted value is checked against its quote instead. The floor dropped a short document's only passage.
    - With no source at all, a written passage is a gap for the lawyer, and the model is not asked.
  - `drafting-s3.1e`, after the first full gate run gave two wrong golden values on the code that had given none:
    - an agreement's name must be headed by its document word ("the annual licence fee" is not one);
    - a value saying the point is open ("to be agreed", "TBC", "not stated") is left for a person.
  - `drafting-s3.1f`: the matter's client is compared first, whatever the field order, and legal forms (Ltd, AB) do not make another party. A demand letter's recipient had been read as the firm's own client.
    - The drafting runner can now name a scenario's matter client, as a firm's matter would.
  - The obligations evaluation now records the model pass and its version. Before, it said "rules only".
- **Final held-out results:**

  | Tool | Version | Result |
  |---|---|---|
  | Citations | `s3.4g` | 24/24 |
  | Definitions | `s3.5c` | 9/11, fixes 2/2 |
  | Obligations | `s3.6g` with the model | 7/7, none guessed |
  | Diligence | `s3.8b` | issues 5/5 |
  | Governance | `s3.8b` | 4/4 |
  | Drafting | `s3.1f` | 9/10, wrong 0, invented 0 (golden, two runs: 41/42 and 39/42, wrong 0) |
  | Playbooks | `s3.2c` | 9/10, passed in error 0 |

- **Format round trips:** two LibreOffice tests were added:
  - a corrections redline that deletes a clause whole;
  - the released Word export, read back without its redacted text.

  The manual Microsoft Word check stays 🔒.
- **Lawyer review (🔒):**
  - `make_packet.py --phase 3` groups each tool's held-out output with the documents it ran on, with a row per group for what was missed.
  - `score.py --phase 3` scores it against the bars proposed in `releases/phase3-rubric.md`.
  - The scoring needs a person.
- **Model variation:** the same drafting code gave 0 and then 2 wrong values on the tuning set. Each cause found became a rule that leaves the field for a person. Drafted values are shown with their quotes, and the lawyer review measures what remains.
- **Limits:**
  - The sets are small and fictional.
  - The recall of undefined terms is limited to D-049's patterns.
  - A cited drafted sentence can narrow its quote.
  - Treaty articles and state codes are not read as authorities.

### D-054 Intake assistant: scripted, screened against the firm's register, minimised, approved replies — RECORDED (S4.1; OPT-01, P-001, P-006)
- **Audit:**
  - The repository had no receptionist or intake code (the plan's "legacy Smart Receptionist" does not exist here).
  - There was no register of matter parties to screen against; consent existed only for meetings (0009).
  - Firm roles were fixed by a CHECK constraint.
- **Off by default:**
  - `LEGAL_INTAKE` turns intake on; every intake route answers 404 otherwise.
  - Enquirer links need `LEGAL_INTAKE_LINKS` as well. They are the Law edition's only routes without a session: the Law guard lets `/api/intake-link/` through only with both flags on, and answers 404 otherwise.
  - `/api/auth/me` reports `features` so the navigation shows "Enquiries" only when intake is on.
- **Roles and permissions (migration 0031):**
  - **Two new firm roles:** `intake` (front desk) and `conflicts` (the conflicts team). `user_roles` is rebuilt with the new CHECK.
  - **`intake.manage`:** intake staff, paralegals, attorneys and managing partners.
  - **`intake.approve`:** attorneys and managing partners approve replies.
  - **`conflicts.review`:** the conflicts team and managing partners (the walls authority, D-022) see matches and decide.
  - **`intake.purge`:** records administrators.
  - **`parties.manage`:** a matter's lead and attorneys. It stays allowed on closed matters, because their parties still matter for conflicts.
  - **Rollback guard:** rolling back below 0031 is refused while intakes, party records or grants of the new roles exist.
- **The party register:** `legal_matter_parties` records adverse parties, related parties, the client's group and others per matter. Removal needs a reason and keeps the history. Always on: conflict searches read it.
- **The script:**
  - `intake-script-1` has nine fixed questions: name, organisation, email, telephone, kind of matter, other parties, a brief description, a date that matters, how they heard of us. Each has fixed wording and a fixed acknowledgement. At least an email or a telephone number is required.
  - The disclaimer (`intake-disclaimer-1`: not legal advice, no lawyer-client relationship, no confidential information yet, a conflicts check first) is accepted before any question.
  - **No model is called anywhere in intake** (a test fails if one is).
- **Screening:** names given (the enquirer, their organisation, the other parties) are checked against every client (former ones too) and every live party on any matter (closed matters too). Names are folded (case, punctuation, a leading "the", legal forms such as Ltd and GmbH), then:
  - equal names match;
  - the same words in another order match;
  - names of five or more letters at 88% likeness match ("Alfa Holdings" and "Alpha Holdings").
- **Minimisation:**
  - **Front desk and lawyers who are not conflicts officers:** see "Possible conflict: refer to the conflicts team", "No match found by the automatic check. The conflicts team confirms before we act", and then the team's decision.
  - **The conflicts team:** sees each match (name given, name matched, as client or party, the client, the matter number), whatever the matter's sensitivity or walls, and records cleared or conflict with a note. Notes are shown only to them.
  - **The enquirer:** never learns the screening.
- **Consent before recording (P-001 default):**
  - A telephone or in-person intake can attach a call recording only while the caller's consent (verbal or written) is recorded.
  - Recordings are stored encrypted (D-014 blob store, `legal/intake/`).
  - Withdrawing consent stops recording and deletes the recording.
  - Transcribing recordings needs the speech stack (🔒 first `compose.sh up`) and is not built here.
- **Replies:** from four templates: acknowledge; we can help (only once cleared); we cannot help (gives no reason); more information (with the firm's request). Each is approved by an attorney or managing partner who did not write it, for the exact text read (its hash).
  - A reply naming another client or party of the firm (a name the enquirer did not give) is refused.
  - A released reply appears on the enquirer's link page, and staff can copy it. Nothing is emailed: there is no network integration.
- **Conversion:** a cleared intake becomes a client (new or existing) and a matter, with the converting lawyer as lead (`matters.create`). The other parties named go on the new matter's party register.
- **Enquirer links:**
  - A 32-byte random token, of which only the SHA-256 is stored. It is shown once and expires after `LEGAL_INTAKE_LINK_DAYS` (14).
  - Answers are submitted once; the link closes when the intake is closed or purged.
  - Rate-limited to 30 requests per link and 300 per address per 10 minutes.
  - The page shows only the disclaimer, the questions and the approved reply.
- **Tokens in logs:** link and sign-in enrolment tokens are masked in the backend's access log (`uvicorn.access` filter) and in nginx's access log (`echomind_masked` format, no referrer). The intake page sends `Referrer-Policy: no-referrer` and `Cache-Control: no-store`.
  - **Residual:** nginx's error log, whose format is fixed, still prints the request line when the backend is unreachable.
- **Records:** a records administrator purges an intake's personal data: answers, matches, replies, the recording and decision notes. The append-only trail of what was done, by whom and when, stays. The audit keeps flags and counts only.
- **Limits:**
  - Screening is by name only: no addresses, no company numbers, no corporate families beyond what lawyers record as "client's group".
  - Names in other scripts (transliteration) are not matched.
  - The register is only as complete as the matters' party lists.
  - Single-word party names in replies are matched as the firm wrote them.

### D-055 Summaries with page/line and timecode citations; translation blocked — RECORDED (S4.2; OPT-02, T08)
- **Audit:**
  - The only summary was the legacy Boardroom's, which stays closed in Law (D-025).
  - The AI artifact and review framework (S2.6–S2.7) gives versions, second-person approval bound to a hash, the Reviews queue and filing.
  - The S2.1 reader stores page boundaries and line starts.
  - Finalised transcripts carry `[HH:MM:SS]` wall-clock times (S2.4).
- **Off by default:** `LEGAL_SUMMARIES`; every summary route answers 404 otherwise. `/api/auth/me` reports the feature for the navigation.
- **How a summary is made (`summary-s4.2b`):**
  - It runs as a background job on the document's current version, for the person who asked, if they can still see it.
  - The text is cut at blank lines into sections of about 2,400 characters. Text addressed to an AI is withheld first, and each passage is fenced so it cannot close its fence.
  - The model gives up to ten points per section, each with a word-for-word quote. No model sees anything but the section.
- **Citations from positions, never from the model:** each quote is found in the section (verbatim or normalised).
  - **Transcripts:** the wall-clock time of the segment it falls in.
  - **Paged documents:** the page, and the line counted among the page's lines with text (as the reader numbers them).
  - **All documents:** the innermost numbered clause, where there is one.
  - A quote that is not found keeps its point, marked "not verified" with no citation. A figure in a point that its quote lacks is listed for a lawyer.
- **Review:** a summary is an AI artifact of kind `summary`. A reviewer's rewording, reordering or removal is a new version. Edited points keep their quote and citation and are marked as edited.
  - Approval comes from a lead, attorney or reviewer other than the version's author, for its hash.
  - Filing makes a matter document listing each point with its citation and quote, and the approval.
  - **The original document is never changed.** A summary is shown only to people who can see the document.
- **The shared T08 detector (S4.2 finding):** "AI assistant: …" labels and "ignore your instructions" were not withheld. Both now are, for every feature (tests red first). Ordinary wording ("disregard the rules of the site", "Model: XJ-200") still passes.
- **Translation: BLOCKED.** It needs a local translation model and its licence, approved by the owner. Nothing is built, and the page says so. When one is approved, translations will be separate versions; the original is always kept.
- **Evaluation (`eval/law/summaries`, real model), measuring facts covered, citations right, quotes verified and figures unsupported:**
  - **Golden on `s4.2`:** facts 17/19 (a dense one-section contract lost two clauses to a six-point cap), citations 17/17, quotes 21/21, figures 0.
  - **Golden on `s4.2b`** (smaller sections, ten points), two runs: 19/19 and 18/19, citations all right, quotes all verified, figures 0.
  - **Held-out 1 (blind, run once on s4.2b):** facts 15/15 and citations 15/15 (clauses, pages and lines, timecodes), quotes 16/16, figures 0.
- **Limits:**
  - Points usually restate their quote closely, which is accurate but not condensed.
  - Facts spanning sections are summarised separately.
  - The page and line of an OCR page depend on its recognised lines.
  - Word documents have no pages: they are cited by clause and line.

### D-056 Time-entry drafts from the person's own audited work; approved by them; CSV only — RECORDED (S4.3; OPT-03)
- **Audit:**
  - The append-only audit trail (S1.5) already records every person's actions with the matter and the time.
  - No billing integration exists, and none is built (spec: "no billing integration").
- **Off by default:** `LEGAL_TIME_ENTRIES`; every route answers 404 otherwise.
- **Suggestions, by rule (no model):**
  - **What counts:** only the person's own allowed actions that are work on a matter. They are grouped into kinds (document review, legal research, drafting, redlining, playbook review, citation and definitions checking, comparing versions, obligations, due diligence, meetings and minutes, review and approval, summaries, document management, exports). Opening a matter, lists and refused requests do not count.
  - **Sittings:** per matter and per day in the firm's timezone, events under 15 minutes apart are one sitting. A sitting counts its whole minutes from first to last event, plus 6 minutes for the last piece of work.
  - **Totals:** rounded up to 0.1 hour.
  - **Narrative:** the kinds of work with their counts, in the order first done. It never holds titles, quotes or questions.
  - **Matters:** those the person can no longer open are left out.
- **The person's own drafts:** each person sees and changes only their own entries (another person gets 404).
  - They edit (0.1-hour units up to 24 hours, a narrative up to 1,000 characters), approve, discard or reopen, each against the version they saw.
  - Suggesting again follows the activity for untouched drafts and keeps a person's own figures. The suggestion stays visible beside them.
- **Export:** approved entries only, as CSV (`Date, Matter, Client, Hours, Minutes, Narrative, Approved`), `no-store`. A cell that a spreadsheet would run as a formula (`= + - @`) is written as text.
  - Nothing is sent anywhere.
- **Migration 0032:** rolling back is refused while approved or edited entries exist.
- **Found by journey 32:** three actions within 45 seconds were suggested as 0.2 hour, because the seconds were rounded up a whole unit. Spans now count whole minutes (test first).
- **Limits:**
  - Only work done in EchoMind is seen. Telephone calls, email and thinking time are not, so suggestions undercount and are a starting point.
  - The 15-minute gap and the 6-minute allowance are engineering defaults for the firm to confirm (P-012-style policy, with the firm's billing rules).

### D-057 The four-perspective review panel: separate cited answers, a moderator, a stop for missing facts, partner approval — RECORDED (S4.5; OPT-05, T08)
- **Audit:**
  - Research answers (S2.2) already give the evidence contract: numbered sources, `[n: "quote"]` citations, checked quotes, and evidence located by clause and line.
  - The artifact framework (S2.6–S2.7) gives versions, approval bound to a hash and the Reviews queue.
  - Nothing existed for several perspectives.
- **Off by default:** `LEGAL_PANEL`; every route answers 404 otherwise.
- **How it runs (`panel-s4.5c`, a background job):**
  - **Search:** the matter is searched once, as the person who asked, by the research rules (signed versions; the research floor applies).
  - **Four perspectives:** a deal lawyer, risk and compliance, a litigator, and a specialist in a chosen area (ten areas). Each answers in its own model call, with the same sources and without seeing the others.
  - **Rules for each answer:** the research rules apply, plus its angle. Text addressed to an AI is withheld. Every quote is checked, and a citation labelled by clause number is matched to the source holding its quote (s4.5b). Uncited sentences are listed for the partner. The answer ends with the facts it lacks.
  - **The moderator:** reads the four answers only (no sources, citations stripped). It lists issues, each perspective's position, disagreements and missing facts. Positions are capped in length, and a figure the moderator gives that no answer gave is flagged.
- **The missing-facts stop:** the panel stops when there are missing facts and either the moderator says the question cannot be answered without them, or three of the four perspectives lack facts (s4.5c).
  - A stopped panel cannot be approved and does not wait in the Reviews queue.
  - The lawyer answers what they can, and the panel runs again as a new version. Their facts are marked "given by the lawyer (not from the documents)".
- **Partner approval:** the matter's lead or a managing partner who did not ask approves exactly the version read; others get 403.
- **Presentation:** every panel says it is "Four perspectives drafted by the same AI model … not independent experts and not legal advice". The screens never call the perspectives experts.
- **Benchmark (`eval/law/panel`, real model, five calls per panel):** it measures stop decisions, quotes verified, uncited sentences, perspectives answered, matrix figures unsupported, and seconds per panel.
  - **Golden on `s4.5`:** stop 3/3, quotes 20/20. But one matter's perspectives all cited by clause number, so nothing was checked (17 uncited), and a model echoed the instruction as a missing fact.
  - **`s4.5b`:** golden stop 3/3, quotes 32/32, uncited 7, figures 0, 37.8 s.
  - **Held-out 1 (blind, s4.5b): stop 2/3,** quotes 29/29, figures 0, 30.4 s. All four perspectives lacked the date of a warranty notice, but the moderator did not stop.
  - **`s4.5c`:** held-out 1 moved to golden. Golden (6) stop 6/6, quotes 61/61, all 24 perspectives answered, figures 0, 37.7 s.
  - **Held-out 2 (blind, s4.5c): stop 3/3, quotes 32/32,** all answered, figures 0, **35.5 s per panel**.
- **Limits:**
  - **Similar perspectives:** the four perspectives come from one model and are often alike. The disagreement matrix shows only what the answers say.
  - **Repeated facts:** missing facts can repeat in different words, and the lawyer answers any subset.
  - **Search:** a short question that misses the documents' words can find no sources, and all four then abstain.
  - **Time:** about 30 to 40 seconds a panel on this host.
- **At the Phase 4 gate (D-058):**
  - **`s4.5d`:** journey 33 found citations written as a quote followed by its clause, `"quote" [12.2]`, or a bare `[6.1]`, which went unchecked. They are now turned into the numbered form and checked like the rest. Golden on `s4.5d`: stop 5/6 and 6/6, quotes 63/63 and 60/60. **Held-out 2 (blind, `s4.5d`): stop 3/3, quotes 30/30**, 36.5 s a panel, with one clause number flagged as an unsupported figure (advisory).
  - **Continuing without the facts:** in gate runs 1 and 2 each re-run with the lawyer's facts found new missing facts, so the panel never reached a partner. A lawyer who may draft for the matter can now continue without them, giving a reason (1–500 characters). This makes a new version marked `proceeded_without` (the facts left open, who, why, when), which the page shows to the partner; the review is then done and waits for a partner in the Reviews queue. The audit records only the number of facts.

### D-058 The Phase 4 gate: optional features off by default, measured, and blocked items stated — RECORDED (S4.6; OPS-02)
- **Scope:** Phase 4 is optional. The owner's business approval of 2026-10-10 covered building and merging each slice behind its own flag, off by default:
  - **S4.1** intake (`LEGAL_INTAKE`, `LEGAL_INTAKE_LINKS`);
  - **S4.2** summaries (`LEGAL_SUMMARIES`);
  - **S4.3** time entries (`LEGAL_TIME_ENTRIES`);
  - **S4.5** the review panel (`LEGAL_PANEL`).
- **Blocked by design and not built:**
  - **S4.4 regulatory monitoring:** a licensed source and an outbound channel need approval.
  - **Machine translation:** a local model and its licence need approval.
- **The gate runner:** `run_gate.py --phase 4` does four things:
  - runs the summaries and panel golden sets;
  - cites their blind held-out runs, already made on the same versions;
  - runs the security suite and the browser journeys;
  - records the evaluation runs for the Evaluations page.

  Per-document and per-panel times are now among the latencies it gathers.
- **What the gate found (two failed runs, each only on journey 33):**
  - Citations written as `"quote" [12.2]` or a bare `[6.1]` went unchecked (`panel-s4.5d`, D-057).
  - A panel lacking facts could not reach a partner: a lawyer can now continue without them, with a reason recorded and shown (D-057).
- **Final run `20261010-215020` on `2b2fcd7`: 6 steps, 0 failed:**
  - summaries golden 19/19;
  - panel golden stop 6/6, quotes 63/63;
  - held-outs cited: summaries 15/15, panel stop 3/3 and quotes 30/30;
  - security 28 groups and 1063 tests;
  - browser 33/33.

  In that run the panel did not stop, so its facts and continue paths are covered by backend and Vitest tests rather than by the journey.
- **What "done" means here:**
  - Every Phase 4 feature has tests written red first, and a browser journey.
  - The two features that use the model have golden and blind held-out evaluations, and every held-out run is reported, including those that led to fixes.
  - Quality beyond these measures is for the lawyer review (🔒): rubric `releases/phase4-rubric.md`, with a 53-item packet and the scorer.
- **Before the pilot:** the owner chooses which flags the pilot firm turns on. Intake links are the Law edition's only routes without a session, so they keep their own switch.

### D-059 Pilot configuration: every Phase 4 feature on — APPROVED by the owner 2026-10-11 (OPT-01…03, OPT-05)
- **Decision:** the owner asked to "turn on all phase 4 features for the pilot":
  - intake (`LEGAL_INTAKE`), including the enquirer links (`LEGAL_INTAKE_LINKS`);
  - summaries (`LEGAL_SUMMARIES`);
  - time entries (`LEGAL_TIME_ENTRIES`);
  - the review panel (`LEGAL_PANEL`).

  S4.4 and machine translation are not built and stay BLOCKED.
- **How:**
  - **Code:** each feature stays off by default, so a stack without these settings has none of them.
  - **Compose overlay:** `docker-compose.law.yml` now passes each switch from `.env.law` to the backend, off when unset; before this, the pilot stack could not turn them on. The enquirer link lifetime is passed too (14 days unless set).
  - **Pilot template:** `.env.law.example` turns all of them on. A test checks both files, and `compose.sh config` rendered them on with the template and off without it.
- **When it takes effect:** at the first `scripts/law/compose.sh up` of the pilot stack, which is still 🔒. Nothing was started.
- **What turning them on means:**
  - **Enquirer links** are the only Law routes without a sign-in (THREAT_MODEL T-24). The Law stack has no public ingress, so a link works only where the firm's network reaches the server.
  - **Summaries and the panel** use the shared model server.
  - **Intake recordings** need the speech stack.
  - **Counsel's wording and policies** (consent P-001, retention of enquiries P-002, sign-off P-008) and the firm's billing rules are still the safe defaults until counsel and the firm decide.
  - **The lawyer review** of summaries and the panel (Phase 4 packet) has not been held, so their output is reviewed by a second person or a partner before use, as the features already require.

### D-060 The pilot stack: Law images from one commit, capped below production, speech recognition on first use — RECORDED (OPS-01; first `compose.sh up` approved by the owner 2026-10-11)
- **Approval:** "approved, create .env.law and start the pilot stack" (2026-10-11).
- **Found before starting:**
  - The 1.4.1 backend image lacks `cryptography`, so encrypted document storage would fail at startup. It also lacks `tesseract` (OCR) and the search reranker's model; the model cannot be fetched at run time, since the stack is offline.
  - The overlay mounted this checkout's code. A branch switch here would change what the pilot runs, and a restart could apply a development branch's migrations to pilot data.
  - The template's master key was the development backend's key.
- **Images (`scripts/law/build_images.sh`):** thin images on the 1.4.1 ones, built from a clean commit, tagged by commit and pinned in `.env.law`. Compose never builds or pulls them. The build needed the network once (apt and PyPI).
  - **Backend:** adds `cryptography` 46.0.3 (which brings `cffi` 2.1.1), `tesseract` 5.3.4 with English, and the reranker from the host's model cache.
  - **Backend and voice:** the commit's code is baked in.
  - **Frontend:** the commit's build, with its own self-signed certificate.
- **Protecting production:** each container is capped, with no swap: backend 8 GB, voice 2 GB (raised to 3 GB, D-062), frontend 256 MB. Each has `oom_score_adj` 800, so under host-wide pressure the kernel ends Law processes first.
- **The first start crash-looped, and was stopped within about a minute:**
  - Both containers loaded speech models on CPU at start: the backend its Nemotron model, the voice service Parakeet (which the base stack turns on). Each hit its then 4 GB cap and was killed and restarted.
  - Production's containers kept the same start times, with no restarts.
  - **Fix:** `ECHOMIND_ASR_PRELOAD` (default on, so production is unchanged) is off in the pilot, and Parakeet is off in the pilot's voice service, which the Law interface does not use.
  - Measured in a throwaway container: the model adds 5.7 GB on CPU, and the backend's cap allows for it.
- **Running (commit `7022d01`):**
  - **Where:** project `echomind-law`, at `https://<host>:3553` and `http://<host>:3110`, with a self-signed certificate for `localhost`.
  - **Data:** in volume `echomind-law_echomind_law_data`. Secrets are in `.env.law` and `.law-secrets/pilot-master.key` (mode 600, gitignored).
  - **Idle use:** about 0.8 GB for the backend and 1.65 GB for voice. Host memory available fell from 13.3 to about 10.4 GB.
  - **Verified:**
    - Law edition, with sign-in enforced: every API route is 401 without a session, and a made-up enquirer link is 404.
    - The Phase 4 switches are on, MFA is required, and all 33 migrations are applied.
    - The enquirer-link page sends no referrer and is not cached.
    - Production's containers are unchanged.
- **First sign-in:** as `admin`, with the password in `.env.law`, binding an authenticator through the one-time enrolment link in the data volume (`/data/legal/bootstrap_enrollment_token`, 72 hours). Nobody has signed in yet, and the link is unused.
- **Limits:**
  - **Live transcripts:** the first one loads speech recognition on CPU. The backend then holds about 7 GB until restarted, leaving about 4 GB available on the host. Transcription speed on CPU is unmeasured.
  - **Reach:** ports 3110 and 3553 are published on all interfaces, so the firm's network can reach the pilot. There is no public ingress. Browsers warn about the self-signed certificate.
  - **Admin password:** reset to the `.env.law` value at every start while it is set there.
  - **Master key:** `pilot-master.key` must be backed up separately from the data; without it the documents cannot be opened.
  - **Voice:** it idles near its cap. If it grows it is killed alone, and the Law interface is unaffected.

### D-061 The demo world on the pilot, and Silent Assist on real speech — RECORDED (owner's request 2026-10-11; LIVE-02, OPS-02)
- **Request:** "fill in demo details and build a product; make sure all the features are present: chat bot, live transcript + silent assistant, conversation assistant and document studio; make sure everything is there and working".
- **Where each feature is in the Law edition** (the legacy names are closed in Law by D-025, because they were not matter-scoped):
  - the chat bot is **Research** (S2.2): questions on a matter, answered from its documents with checked quotes;
  - Live Transcript is **Meetings** (S1.8, S2.4): consent first, then a verbatim transcript;
  - the silent assistant is **Silent Assist** (S2.5), in the meeting;
  - the document studio is **Drafting** (S3.1, firm templates to Word), with redlines (S3.3) and summaries (S4.2);
  - the conversation assistant is the new **Conversation** (D-062).
- **Demo data on the pilot (fictional only):**
  - `seed-demo` gives 9 people, 3 clients and 4 matters, with templates, a playbook and a checklist;
  - `scripts/law/demo_load.py` imports 13 fixture documents into M-101 and M-114 through the pilot's own API;
  - `scripts/law/demo_signin.py` prints the demo people's sign-in details, for the owner to run (never printed into a log or a chat).
- **`scripts/law/demo_check.py`** checks each feature end to end on the running pilot, with the real model, speech recognition and voice. For the meeting, a fictional client call is spoken by offline Piper voices and streamed after consent.
- **Found with real speech, each fixed with a test written red first (Silent Assist only; document comparison unchanged):**
  1. **Spoken money:** amounts in words ("five hundred thousand dollars") were not read; only digits were. They are now read in transcripts, including across a stray full stop inside the figure.
  2. **Split figures:** the segmenter can split a figure ("is five." | "Hundred thousand dollars"). A figure is now read across the same speaker's previous segment, quoted in the segment it ended in, and never across two speakers.
  3. **Long clauses:** in a long clause the subject is far from the figure (the corpus MSA cap scored 0.1 against a 0.2 floor). For a spoken statement, a clause whose sentence covers every subject of the statement is now also a candidate. The model still decides.
  4. **Scattered full stops:** the speech model scatters full stops ("The liability. Cap under. Clause eight. Is five."), so a figure's context is read with sentence-ending marks blanked. Offsets are kept and quotes are as stored.
  5. **Echoed instructions:** the subject check sometimes copied its instruction back ("Same subject: no, then in a few words …") instead of judging. It is asked once more.
  6. **Plain words:** that first line reads like a template, so Silent Assist now asks it in plain words ("answer yes or no in your own words …"). The document comparison is unchanged.
- **Evaluation:**
  - **Golden:** the demo call and the pilot's own transcript, which informed the fixes, were added. Each golden run, as it came out: 8/9 (after fixes 1–2), then 9/9 and 10/10 critical, precision 1.0, no false alerts.
  - **Held-out:** two calls with spoken and split amounts were written and committed before any run. Each version was run once, and every run gave critical 8/9, precision 0.933.
    - The split figure was caught, but in one segment, because the first turn had fewer than 12 words. The harness scores it on the first turn: one miss and one false alert.
    - "I will ask … to confirm" is not an action: the action rule uses a fixed list of verbs. This is a known limit and was not tuned on.
- **On the pilot (CPU speech recognition, synthetic Piper voices):** the live transcript keeps up with speech (16–17 s of speech each time, never behind; 22–37 s a call end to end, synthesis included). The misstated cap was flagged in 3 of 7 spoken runs across the builds:
  - 2 of 2 on `333539a`;
  - 0 of 1 on `071de48`, where the subject check echoed its instruction (fix 5);
  - 1 of 4 on `0846afe`.

  On `0846afe` the transcripts lost "thousand dollars" in three runs. The owner asked for that to be fixed, and it was investigated as follows:
  - **Streaming does not drop words.** The same audio, streamed live and sent whole, gave the same words for 9 clips. With or without an earlier speaker in the capture, the same audio gave the same words in 6 runs.
  - **The variation came from the check.** Piper varied the audio on every run (three separate syntheses of the same line differed), and the speech model misheard some versions.
  - **The check is now repeatable.** `demo_check.py` speaks with Piper's variation turned off, so a result can be repeated.
  - **A sixth fix came out of the repeatable call.** The amount was read across the split and matched to cl. 8.1, but the model copied its first line back every time. Silent Assist now asks that line in plain words (fix 6), while the document comparison keeps its prompt. On two garbled transcripts the model then judged conflict 6 of 6 times, and on a fee statement no conflict 3 of 3, with no echoes.
  - Golden stayed at critical 10/10 and precision 1.0; held-out on `fc7c08d` gave the same result as before.

  The action alert was raised in every run.

### D-062 The conversation assistant (S5.1), and the voice service's own routes refused — RECORDED (owner's request 2026-10-11; SEC-01)
- **What it is:** a Conversation page on each matter.
  - Press the microphone and ask; press again to send (at most 60 seconds), or type instead.
  - This server transcribes the question with its own speech recognition (the model meetings use, loaded on first use).
  - The research pipeline answers it from the matter's signed documents with checked quotes, exactly as a typed Research question. It is stored in the person's research history.
  - The answer is read aloud by this server's voice, without its citation marks. Reading aloud can be turned off, and the choice is remembered in the browser.
- **Routes (`app/legal/conversation`):**
  - `POST …/conversation/transcribe?sample_rate=16000` takes 16-bit mono PCM and returns `{text, seconds}`.
  - `POST …/conversation/speak` `{text}` returns WAV, with `no-store`.
  - Both need `research.query` on the matter, so a matter is invisible here as anywhere (404).
  - The audio is not kept, and the audit holds the policy and lengths only.
  - Without speech recognition or a voice, the server says so (503) and typing still works.
- **Off unless `LEGAL_CONVERSATION` is on.** It is on in the pilot template; the compose overlay passes `LAW_CONVERSATION`.
- **Security finding, fixed:** the pilot's nginx passed `/voice/*` to the voice service. Its own HTTP routes (`/speak`, `/voices/installed`, its test page) answer **without a session**, so anyone who could reach the pilot could make it synthesise speech. The Law interface never calls them.
  - The Law frontend now uses `frontend/nginx.law.conf`: the standard config with `/voice/` refused (404). A test keeps the two otherwise identical, so production's nginx is untouched.
  - Speech for the Law edition goes through the signed-in backend.
  - Verified on the pilot: all four routes return 404 on both ports.
- **Tests:**
  - backend `test_law_conversation.py` (7, red first) and pilot config (the nginx test, red first);
  - Vitest `conversation.test.tsx` (4, axe clean);
  - browser journey 34 (typed question; this test server has no speech model or voice, and the page says so);
  - security suite group "Conversation";
  - on the pilot, `demo_check.py conversation`: a spoken question is transcribed, answered with the cap, and read aloud.
- **The voice service's cap:** it idles near 1.6 GB, because it preloads a second voice the Law edition does not use. At its 2 GB cap the kernel stopped it while reading a long answer aloud (memory cgroup; production untouched), so the pilot gives it 3 GB. Three spoken questions then passed, with voice steady at 1.7 GB.
- **Updating the pilot:** `scripts/law/deploy_pilot.sh` builds first and changes nothing if that fails, then swaps the images and goes back if they are unhealthy. A hand-chained update once stopped the pilot after the build had refused (uncommitted changes); it was down for about two minutes.
- **Limits:**
  - Each spoken question is answered on its own. Follow-ups like "and what about termination?" are not joined to the previous question.
  - The first question after a restart waits while speech recognition loads.

### D-063 Conversation remembers follow-up questions (S5.1b) — RECORDED (owner's request 2026-10-12)
- **Request:** "make conversation remember follow-up questions".
- **How:**
  - The page sends the conversation so far (at most six turns) with each question.
  - The model first turns a follow-up into one standalone question (`research/followup.py`), using the conversation only to resolve what it refers to.
  - That question is what the documents are searched with, what is answered (from the documents only) and what is kept in the research history.
  - The page shows it: "You asked …, understood from the conversation as: …".
- **Safety:**
  - Earlier answers go to the rewrite without their citation marks, and with text addressed to an AI withheld.
  - A reply that is not a short question (an answer, too long, addressed to an AI), or a failure, falls back to the follow-up as asked.
  - Without a conversation nothing changes.
  - The audit holds the number of turns and whether the question was rewritten.
- **Controls:** "New conversation" forgets it.
- **Tests (red first):** backend 6, Vitest 1. On the pilot, harper's "And was the notice for the April failures given in time?" was understood and answered with citations.

### D-064 Demo data: a public-law reference library, the base corpus, a practice group and six fictional cases — RECORDED (owner's request 2026-10-12)
- **Request:** crawl some content, load the base data set, and create a group of users and synthetic cases. Asked which content to crawl, the owner chose **public-domain law**.
- **The reference library (R-001), real public law:** five texts fetched once from official sources (`scripts/law/fetch_reference.py`), the only network use, with the owner's approval.
  - Sources: the US Code (Federal Arbitration Act, E-SIGN Act; public domain) and UK Acts (Late Payment of Commercial Debts (Interest) Act 1998, Unfair Contract Terms Act 1977, Contracts (Rights of Third Parties) Act 1999; Open Government Licence v3.0).
  - Operative text only. Each document is labelled as real law; the licences are in LICENSE_REGISTER §1b.
  - Every demo lawyer reads it (read-only members).
- **The base data set (M-130):** the fictional evaluation corpus, with the Harrowgate MSA's signed v1 and draft v2 as versions of one document. Its Beta settlement memo is filed on Beta Shipping's M-215, for isolation.
- **The practice group and six fictional cases:**
  - People: harper, sofia, kenji, lena, omar and theo, seeded by `seed-demo-practice`. It needs the demo world, prints no password and is idempotent (tests red first).
  - Cases: M-301 to M-306 across litigation, employment, venture, real estate, technology and data, and regulatory. Each has 4–5 documents written by the local model from a fixed brief, with every figure checked, and disagreements planted on purpose for Compare, Silent Assist and the panel.
  - The files are committed (`eval/fixtures/law/synthetic`), so the demo is the same every time.
- **Found while checking the flows:**
  - US-style parties ("by and among Kestrel Robotics, Inc., … and Aldermoor Ventures LP") were not recognised, so the term sheet was never compared with the purchase agreement.
  - Fixed: "among" now cues parties, a comma may precede the legal form, and LP / L.P. is a form (test red first).
  - Compare golden is unchanged (alignment 4/4 and 13/13, scan 5/5, precision 1.0); held-out, run once, is unchanged (scan 4/5, precision 1.0).
  - On the pilot the scan now finds the option pool (12% against 10%) and the closing date.
- **On the pilot:** 6 people, 7 clients, 8 matters, 29 memberships and 40 documents, all searchable.
- **The flows:**
  - `DEMO_FLOW.md` and a published guide give eleven walkthroughs.
  - `demo_check.py` checks them end to end: research, conversation, follow-up, conflicts, reference, employment, meeting, drafting, summary, panel, time and intake.
- **Before a real firm:** start from a fresh data volume, with the owner's approval. The demo data is fictional, apart from the public law.

### D-065 Liquid Glass, a Home dashboard, the command palette, tools as tiles, starter questions — RECORDED (owner's request 2026-10-12)
- **Request:** "the UI is so boring; a glass Apple liquid UI, the new trend, an advanced UI/UX that looks like production"; and "keep the flow more usable".
- **The material (`theme/liquid.css`, a layer over law.css):**
  - A slow aurora behind every page, so the glass has colour to bend.
  - Clear glass for navigation and chrome, with a lit edge and a light that follows the pointer.
  - Frosted sheets for content, which keep text at full contrast; long reading (the reader, answers, transcripts) stays near-opaque.
  - A liquid pill for the current page, and gradient primary actions.
  - Reduced transparency, reduced motion and higher contrast fall back to solid, still surfaces. DESIGN_SYSTEM §10.
- **Flows made shorter:**
  - **Home** is a dashboard: a greeting, quick actions (each asks which matter), counts, matters as cards, and what is waiting for a decision.
  - **The command palette (⌘K / Ctrl K, or the sidebar's search)** jumps to any matter, its tools or firm pages.
  - Each matter's overview offers its tools as tiles.
  - Empty Research and Conversation pages offer starter questions.
  - One navigation list (`app/nav.ts`) feeds the sidebar, the palette and the tiles, so they cannot disagree.
- **Tests:** Vitest for Home, the palette and the tiles, axe clean (6 new). Every existing test and journey still finds what it looked for: the labels are unchanged.

### D-066 T12 load test on the pilot; live capture no longer waits for speech recognition — RECORDED (owner's approval 2026-10-12: "run now, capped")
- **The run:** 10 virtual users on the pilot (8 researchers on their own matters, 1 searching, 1 recording), 4 minutes per stage, 03:05–03:58 EDT. A watcher probed production's answer model every few seconds and would have stopped the test after 3 slow probes (over 3× its baseline or baseline + 2 s). It never did: production's probe stayed under 0.25 s against a 0.03 s baseline, with no restarts and no memory kills.
- **What it found, and the fixes (tests first):**
  - **Stage 2: capture dropped with "keepalive ping timeout".** The capture socket awaited the speech model for every audio frame, so under load nothing read the socket and the browser's pings went unanswered. Reading is now separate from transcription: frames are accepted at once (consent lock and custody unchanged) and queued, with turns, agenda moves and pauses, in the order received; a worker feeds the model. Audio waiting more than `LEGAL_CAPTURE_BACKLOG_SEC` (120 s) is not transcribed and leaves a "gap" mark at its place in the meeting; its custody record is kept. At a stop, waiting audio is still transcribed for up to `LEGAL_CAPTURE_FINISH_SEC` (300 s), never after a withdrawal. Lag is now the age of the audio the model has just read.
  - **Stage 1: capture timed out on a cold start.** With `ECHOMIND_ASR_PRELOAD=0`, the first recording loads the model (about 20 s here); it loaded on the event loop and held up everyone. It now loads on a worker thread.
- **Result after the fixes (warm model):** capture streamed 4 minutes (2,397 frames, all in custody) with no failures; research answer p50 4.4 s, p95 9.7 s; search p50 0.75 s, p95 2.2 s; no database-busy errors; backend memory 4.7 GB of its 8 GB cap.
- **Still true:** under this load, speech recognition on CPU fell behind real time. It passed the 120 s backlog once (one marked gap), and the stop took 97 s while it caught up. A live transcript for 10 busy users needs the speech model on a GPU or fewer users at once. The capture says so ("degraded") and the transcript marks the gap; nothing is lost silently.
- **Reports:** `eval/law/reports/20261012-03*-load-t12.*` (TEST_RESULTS run 40).

### D-067 Pilot backups — RECORDED (in the pending-items plan the owner approved, 2026-10-12)
- `scripts/law/backup_pilot.sh` makes an encrypted archive inside the backend container with `python -m app.legal.backup`, verifies it, and copies it to `~/echomind-law-backups` on the host, outside the data volume, with its sha256 checked. It keeps the newest 14.
- A user cron job runs it daily at 02:30. `--restore-test` also restores the archive into a scratch folder and compares the counts. The first run restored 12 matters, 51 documents and 4,716 audit events, matching the archive.
- **Keys, kept apart:**
  - the backup key: `~/.echomind-law/backup.key`, mode 600;
  - the master key: `.law-secrets/pilot-master.key`;
  - a second copy of the master key: `~/.echomind-law-master/`.
  - **Both keys still need a copy off this host** (🔒 owner). An archive cannot be read without its backup key, and its documents not without the master key.

### D-068 Regulatory updates from official public sources (S4.4, OPT-04) — RECORDED (owner's approval 2026-10-12: "fetch daily")
- **Sources** (public law only, no licence to buy):
  - UK: new legislation from legislation.gov.uk (Atom), under the Open Government Licence v3.0.
  - US: rules and proposed rules from the Federal Register API, which are US Government works in the public domain.
- **The channel:** the host fetches daily at 06:15 (`scripts/law/regulatory_daily.sh`, cron) and pipes the bundle into the backend. The Law containers make no internet connections (T11; D-071 enforces it). The bundle holds only titles, short summaries, dates, kinds, agencies and official links.
- **Import:** everything is checked again on import, because the bundle is untrusted text:
  - a known source;
  - an https link on that source's own site;
  - a valid date;
  - bounded plain text with control and format characters removed.
  - Licence and jurisdiction come from our own table, never from the bundle. A bad item is skipped and counted; a bad bundle is refused whole.
- **Matching (`regulatory/topics.py`):** an update concerns an open matter when:
  - the jurisdictions agree (England & Wales, Scotland and Northern Ireland are UK; a US state is US);
  - an update made for Scotland, Wales or Northern Ireland, by its kind or title, also needs a matter there; and
  - a term of the matter's practice area appears as a whole word, or its plural.
  - The terms found are kept and shown as the reason. It is a plain rule a lawyer can check, never a model; the term lists need a lawyer's review (P-008).
- **The first real import (2026-10-12) tuned it.** It fetched 48 updates (12 UK, 36 US) and made 26 matches, mostly noise: "commercial" in a fishery quota, "enforce" in Coast Guard safety zones, "freight" in a rail notice, and a Scottish housing act against an England & Wales lease.
  - The terms are now specific legal words and phrases, and the regional rule was added (tests first).
  - `python -m app.legal.regulatory rematch` recomputed the undecided matches; a decided match is kept as decided. **3 matches remain, each plausible:** Venezuela sanctions for the investigations matter; a social-security pensions change for the employment matter; a housing change for the lease matter.
  - A correction slip (an id that is a path) is now kept instead of skipped.
- **Who sees what:** lawyers see each update with only the matters they may see; walls and highly restricted matters hold. On a matter, a lawyer marks a match "reviewed" or "not relevant", which needs a reason. The audit records the decision, never the reason's words.
- **Settings:** off by default (`LEGAL_REGULATORY`) and on for the pilot. A rollback is refused once decisions exist.

### D-069 Syslog/CEF audit forwarder, off by default — RECORDED (in the pending-items plan the owner approved, 2026-10-12)
- **What it sends:** each audit event goes, in order, as one CEF message in RFC 5424 syslog, over UDP or TCP (RFC 6587 octet counting). The message holds what the trail already holds: who, what, outcome, matter id, counts and flags, and the event's hash and sequence number. It never holds matter content.
- **Severity:** CEF 3 for allowed, 6 for refused and 8 for errors. Break-glass grants, and every access made under one, are at least 7.
- **Delivery:** at-least-once across restarts.
  - A cursor (migration 0035) records the last event sent to the destination and moves only after a send succeeds.
  - Over TCP, "sent" means the operating system accepted it: plain syslog has no receipt. A batch cut short by a crash is sent again; a SIEM can drop repeats by `cn1` (seq) and check the `cs5` hashes against the chain.
  - A new or changed destination starts at the end of the trail, unless `LEGAL_AUDIT_SYSLOG_FROM_START` is set.
  - A failed send is retried with back-off, up to 60 s, and logged by type only.
  - `python -m app.legal.audit forward --once` sends what is pending.
- **Settings:** off by default (`LEGAL_AUDIT_SYSLOG` empty; `LAW_AUDIT_SYSLOG` in `.env.law`). Turning it on needs:
  - the firm's SIEM address and network approval (🔒);
  - with the Law network lock (D-071), an exception for that address (a SIEM on this host itself needs none).
- **Built by a helper agent from a written brief,** reviewed, and merged. Its tests were checked against planted bugs (TCP length in characters, the cursor moved before sending), which they caught. 15 tests.

### D-070 Machine translation with the local model (OPT-02) — RECORDED (in the pending-items plan the owner approved, 2026-10-12)
- **No new model and no new licence:** translations use the answer model already registered (LICENSE_REGISTER; its NVIDIA checkpoint terms are still to verify, as for every feature).
- **What it translates:**
  - A document's current version: a few paragraphs at a time, streamed beside the original.
  - A summary's points: each translation shown under its point.
  - Twelve languages.
- **Labelled every time:** "Machine translation by this server's model, not reviewed by anyone. The original prevails: check it before relying on the translation."
- **Checks:**
  - Every figure of the original is looked for in the translation; separators are ignored, so 19,750 = 19.750 = 19 750. A missing figure is flagged.
  - Text addressed to an AI is withheld from the model (T08).
  - A reply with the wrong number of passages is redone one passage at a time, never guessed.
  - Access is re-checked between batches.
- **Nothing is stored.** The audit records counts only.
- **Settings:** off by default (`LEGAL_TRANSLATION`) and on for the pilot. Requests stop at `LEGAL_TRANSLATE_MAX_CHARS` (30,000) and continue from there.
- **Still open:** a lawyer's review of translation quality per language (P-008). Until then, translations are for reading only, never for filing.

### D-071 T11 enforced: a firewall rule for the pilot's network — RECORDED (needs the owner's sudo)
- **The rule:** `scripts/law/egress_lock.sh` adds one rule in Docker's DOCKER-USER chain. It rejects anything forwarded from the Law network to another address.
  - Traffic to this host, where the model servers are, is not forwarded, so it still works.
  - Production's network is untouched, and the Law containers can no longer reach it.
- **The test:** the backend must reach the answer model and must fail to connect to the internet.
- **It needs root, so the owner runs it.** A reboot clears the rule.

### D-072 Transcription measured with synthetic voices; the longer look-ahead by default — RECORDED (in the pending-items plan the owner approved, 2026-10-12)
- **Why a proxy:** real-voice accuracy needs the firm's recordings. This is a proxy measure.
- **How:** `scripts/law/transcription_proxy.py` speaks the four scripted legal conversations with fixed Piper voices. It streams them in real time through the pilot's capture socket, with speakers marked, and scores them with the S2.4 harness.
- **Nothing is lost in capture.** The same audio sent as whole clips gave word-for-word the same text, with or without extra silence. The misses are the speech model's: "Harrowgate" heard as "Harbor Good", "cap" as "cat", "a signed amendment" as "a sign".
- **Look-ahead (`att_context_size` right context; the model supports 0, 1, 6 or 13 frames of 80 ms).** Means of the four conversations:

  | Look-ahead | Normalised word error | Verbatim word error | Entities kept | Negations kept |
  |---|---|---|---|---|
  | 6 frames (560 ms), as the standard edition | 24% | 59% | 29% | 54% |
  | 13 frames (1,120 ms) | 11.5% | 34% | 48% | 83% |

  The Law stack now uses 13 by default (`LAW_ASR_ATT_CONTEXT_RIGHT`, `LAW_ASR_CHUNK_MS`). Live text arrives about half a second later.
- **Still true:**
  - The model sometimes drops the last word or two before a pause, at either look-ahead ("below USD four hundred [thousand]"). At 1,120 ms this hit the demo call's key line: "five hundred thousand dollars" was heard as "five hundred", so the meeting check's cap alert did not fire. It fired at 560 ms. It was kept at 1,120 ms on the 19-turn measure; the demo guide says to say amounts in full.
  - Figures and names are the weak point: entities kept are under half.
  - The transcript is a record to correct (S2.4 revisions), not to rely on unread.
  - Synthetic speech is cleaner than a meeting room, so real accuracy will be lower.
  - On a GPU the speech model would also keep up under load (D-066).
- **Reports:** `eval/law/reports/20261012-0417*` (560 ms) and `20261012-0511*` (1,120 ms).

### D-073 A quote given without a reference is cited where it is found; the operations runbook — RECORDED (owner's approval 2026-10-12: "start the next pending items and finish everything")
- **Found on the pilot:** the reference-library question was answered correctly, quoting the Act word for word ("Section 1(1) states, "…""), but with no `[n]`. So it showed as unverified in 2 of 3 runs.
- **Change (tests first):**
  - In a sentence that cites nothing, a quote of four or more words found in a retrieved source is cited to the first source, in rank order, that contains it.
  - It is verified like any other quote and marked `matched`. The evidence says "Reference added: the answer quoted this without one".
  - A quote found nowhere stays uncited, so its sentence is still marked unsupported.
  - A passage flagged for text addressed to an AI is never matched (T08).
- **Evaluation:** golden, then held-out once, on dbb7a6a.
  - **Golden:** answers 61/64, abstentions 4/6, citation precision 0.90, quote verification 0.99, no "unverified" answers. The last run, 2026-10-10, gave 60/64, 0.93, 1.0 and 2.
  - **Held-out:** 28/29, abstentions 1/3, precision 0.95, quote verification 1.0, no "unverified". The last run gave 28/29, 0/3, 0.95, 0.97 and 3.
  - **No reference was added by matching in either set,** so these differences are the model's day-to-day variance, not this change. The change acts only when the model quotes without a marker, as it did on the pilot.
- **Runbook (OPS-01):** `RUNBOOK.md` covers health checks, updates and rollback, backups and restores, regulatory updates, the network lock, people and access, incidents, and what never to do.
- **Not done here:** generic single sign-on (OIDC, SEC-01). Reading the sign-in code was refused by the session's permission rules, alongside a refused attempt to open the pilot to the internet, so it was left for the owner to decide. OIDC also needs the firm's identity provider (P-011).

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
