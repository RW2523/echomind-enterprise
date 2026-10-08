# EchoMind Law — Implementation Plan (end-to-end)

> **Status:** v1.0 for approval. Baseline `main@95782e4`.
> Every statement about current behaviour is backed by [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md). Every proposal awaiting approval is listed in [DECISIONS.md](DECISIONS.md). Per-feature status is in [FEATURE_MATRIX.md](FEATURE_MATRIX.md), security in [THREAT_MODEL.md](THREAT_MODEL.md), UI in [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md).
> **Nothing below is implemented yet.**

---

## 0. How to read this plan

- **Phases** follow spec §2: 0 Audit → 1 Foundations → 2 Pilot core → 3 Document workflows → 4 Optional. An **OPS track** runs alongside all of them.
- **Slices** (`S<phase>.<n>`) are PR-sized vertical increments. Each one runs the spec's loop: read code → design and contract → failing tests (including negative and access tests) → backend → frontend → targeted and regression tests plus an offline check → update PROGRESS.md and TEST_RESULTS.md → show the changed files.
- **Status vocabulary:** `DONE`, `PARTIAL`, `DEMO ONLY`, `BLOCKED` (spec §15). Nothing is called DONE without passing its acceptance test.
- **Approval gates** (🔒) mark where Claude stops and asks: destructive migration, new external network, credentials or licences, deployment, Word tracked-changes claim, policy conflicts.

## 1. Goals, non-goals, principles

**Goal.** Turn the existing EchoMind into a secure, matter-centric legal workspace for a corporate or in-house pilot, without breaking the other verticals. The pilot covers private client meetings, contract review, drafting from approved precedents, and matter-scoped Q&A with verifiable sources.

**Non-goals.**
- A case-law database or Westlaw/Lexis replacement.
- Autonomous filing or sending.
- Any legal conclusion without lawyer review.
- Claiming that local hosting guarantees privilege.

**Engineering principles.**
1. **Additive and reversible.** New code lives in `backend/app/legal/` and `frontend/law/`. Shared modules get the smallest hooks needed, behind `ECHOMIND_EDITION=law`.
2. **Authorization before everything.** Backend policy first, then retrieval partitioning, then UI.
3. **Evidence first.** Every AI output carries sources, status and a reviewer.
4. **Offline by default.** Any network use is a named, approved exception.
5. **Tests prove claims.** Each backlog ID has an executable acceptance test. Negative tests come first.

## 2. Target architecture

```
                       ┌──────────────────────── frontend (nginx, :3110/:3553 for Law) ────────────────────────┐
 Browser ─TLS──────────▶  React 19 SPA · frontend/law/ (router, MatterContext, "Paper & Glass" design system)    │
                       └──────────────┬───────────────────────────────────────────────┬─────────────────────────┘
                                /api/legal/* + legacy /api/* (law-hardened)        /voice/ws (session + OBO token)
                                      ▼                                               ▼
 ┌────────────────────────────────────────── backend (FastAPI) ──────────────────────┐   ┌───── voice ─────┐
 │ legal/identity   sessions · MFA · CSRF · OBO tokens · OIDC(opt)                   │   │ STT→LLM→TTS     │
 │ legal/policy     authorize(principal, action, resource) → Decision (RBAC+ABAC+walls) │ │ RAG via backend │
 │ legal/audit      hash-chained audit_events · read & write events · verifier       │◀──┤ (OBO token,     │
 │ legal/matters    orgs · clients · matters · memberships · walls · break-glass     │   │  matter scope)  │
 │ legal/documents  import · versions · labels · OCR · legal structure parsing       │   └─────────────────┘
 │ legal/retrieval  PartitionIndex (dense+BM25 per matter / firm library) · AuthorizedScope
 │ legal/research   evidence contract · quote verification · abstention               │
 │ legal/meetings   consent ledger · capture sessions · transcripts(verbatim+revisions) · minutes
 │ legal/records    labels · retention · holds · DeletionService                      │
 │ legal/exports    preview → redaction plan → approve → render → verify → watermark  │
 │ legal/connectors DmsConnector interface · LocalFolderConnector · sync engine       │
 │ legal/jobs       persistent job table + in-process worker (principal captured, re-check on output)
 │ legal/drafting · clauses · citations · definitions · obligations · diligence (Ph3) │
 │ existing modules (chat, transcribe, boardroom, docgen, silent_assistant) → matter-aware hooks
 └───────────┬─────────────────────────────┬──────────────────────────────┬──────────┘
             ▼                             ▼                              ▼
   SQLite (WAL) + versioned        /data/legal/partitions/<pid>/    /data/legal/blobs (AES-GCM files)
   migrations (D-004)              faiss + bm25 + meta (D-005)       audit anchors · exports
             ▲
   trtllm (Qwen3-30B-A3B) · ollama (nomic-embed-text) · cross-encoder (CPU) · Nemotron/Parakeet/VibeVoice · Tesseract
```

### 2.1 Runtime topology

| Environment | How it runs | Data |
|---|---|---|
| **Dev (this DGX Spark, shared with production)** | Native backend via `scripts/law/dev_backend.sh` (uvicorn, port 8110, CPU, no ASR) and native frontend via `scripts/law/dev_frontend.sh` (Vite :5174). Both reuse production's `trtllm:8355` and `ollama:11435` read-only. ASR/voice tests run in container `echomind-law-backend` (image `echomind-enterprise-backend:1.4.1`, code bind-mounted, memory checked first). | `<clone>/.law-data/` (gitignored), fictional data only |
| **Law compose (integration / demo)** | `docker compose -p echomind-law -f docker-compose.yml -f docker-compose.law.yml --env-file .env.law up -d backend voice frontend`. Distinct container names, ports 3110/3553 and volume `echomind_law_data`. Joins `echomind-enterprise_default` as an external network for the LLM and embeddings. No `cloudflared`. 🔒 First run needs approval. | `echomind_law_data` |
| **Firm deployment (pilot)** | Dedicated host: full stack, Law edition, LUKS volume, no public ingress, model servers not published on the host. Runbook in OPS-01. 🔒 | Firm-owned |

## 3. Data model (new tables; SQLite, versioned migrations)

**Conventions:**
- IDs are UUIDv4 text. Every row has `created_at`/`updated_at` in UTC ISO-8601 with milliseconds and `created_by`.
- Optimistic `version` integer on mutable entities.
- Soft-delete via `status`; purge happens only through `DeletionService`.
- Migrations live in `backend/app/legal/migrations/NNNN_name.up.sql` with a `.down.sql`, recorded in `schema_migrations(version, name, checksum, applied_at)`.
- Every migration is **additive**. Destructive migrations are 🔒.

### 3.1 Phase 1 schema (DDL sketch)

```sql
-- 0001 foundation
CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, checksum TEXT, applied_at TEXT);
CREATE TABLE organizations(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT, created_by TEXT);

-- 0002 identity (extends existing users; legacy columns untouched)
ALTER TABLE users ADD COLUMN org_id TEXT;             ALTER TABLE users ADD COLUMN display_name TEXT;
ALTER TABLE users ADD COLUMN email TEXT;              ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'; -- active|disabled
ALTER TABLE users ADD COLUMN mfa_secret_enc TEXT;     ALTER TABLE users ADD COLUMN mfa_enabled INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN failed_logins INTEGER DEFAULT 0; ALTER TABLE users ADD COLUMN locked_until TEXT;
ALTER TABLE users ADD COLUMN password_changed_at TEXT;
CREATE TABLE user_roles(user_id TEXT, role TEXT CHECK(role IN ('firm_admin','managing_partner','attorney','paralegal','reviewer','records_admin','read_only')), granted_by TEXT, granted_at TEXT, PRIMARY KEY(user_id, role));
CREATE TABLE sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT, last_seen_at TEXT, idle_expires_at TEXT, abs_expires_at TEXT, revoked_at TEXT, revoke_reason TEXT, mfa_verified INTEGER, ip TEXT, user_agent TEXT, csrf_hash TEXT);
CREATE TABLE mfa_recovery_codes(user_id TEXT, code_hash TEXT, used_at TEXT, PRIMARY KEY(user_id, code_hash));

-- 0003 matters & policy
CREATE TABLE clients(id TEXT PRIMARY KEY, org_id TEXT NOT NULL, name TEXT NOT NULL, code TEXT, status TEXT DEFAULT 'active', created_at TEXT, created_by TEXT, updated_at TEXT, version INTEGER DEFAULT 1);
CREATE TABLE matters(id TEXT PRIMARY KEY, org_id TEXT NOT NULL, client_id TEXT NOT NULL REFERENCES clients(id), number TEXT NOT NULL, title TEXT NOT NULL,
  practice_area TEXT, jurisdiction TEXT, status TEXT DEFAULT 'open' CHECK(status IN ('open','closed','archived')),
  sensitivity TEXT DEFAULT 'restricted' CHECK(sensitivity IN ('standard','restricted','highly_restricted')),
  retention_policy_id TEXT, acl_epoch INTEGER DEFAULT 1, created_at TEXT, created_by TEXT, updated_at TEXT, version INTEGER DEFAULT 1, UNIQUE(org_id, number));
CREATE TABLE matter_memberships(id TEXT PRIMARY KEY, matter_id TEXT NOT NULL, user_id TEXT NOT NULL,
  role TEXT CHECK(role IN ('lead','attorney','paralegal','reviewer','read_only')), can_export INTEGER DEFAULT 0,
  valid_from TEXT, valid_to TEXT, granted_by TEXT, granted_at TEXT, revoked_by TEXT, revoked_at TEXT, reason TEXT);
CREATE INDEX ix_mm_user ON matter_memberships(user_id); CREATE INDEX ix_mm_matter ON matter_memberships(matter_id);

-- 0004 ethical walls & break-glass
CREATE TABLE ethical_walls(id TEXT PRIMARY KEY, org_id TEXT, name TEXT, rationale TEXT, status TEXT DEFAULT 'pending', requested_by TEXT, approved_by TEXT, approved_at TEXT, created_at TEXT);
CREATE TABLE wall_entries(id TEXT PRIMARY KEY, wall_id TEXT, subject_type TEXT CHECK(subject_type IN ('user')), subject_id TEXT, object_type TEXT CHECK(object_type IN ('matter','client')), object_id TEXT, effect TEXT DEFAULT 'deny' CHECK(effect IN ('deny')));
CREATE TABLE wall_exceptions(id TEXT PRIMARY KEY, wall_id TEXT, user_id TEXT, matter_id TEXT, reason TEXT, requested_by TEXT, approved_by TEXT, approved_at TEXT, expires_at TEXT, revoked_at TEXT);
CREATE TABLE break_glass_grants(id TEXT PRIMARY KEY, user_id TEXT, matter_id TEXT, reason TEXT, requested_at TEXT, approved_by TEXT, approved_at TEXT, expires_at TEXT, ended_at TEXT);

-- 0005 audit (append-only, hash-chained)
CREATE TABLE audit_events(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, ts TEXT NOT NULL, actor_user_id TEXT, actor_roles TEXT, on_behalf_of TEXT,
  session_ref TEXT, ip TEXT, action TEXT NOT NULL, object_type TEXT, object_id TEXT, matter_id TEXT, outcome TEXT CHECK(outcome IN ('allow','deny','error')),
  reason TEXT, details_json TEXT, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT,'audit_events is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT,'audit_events is append-only'); END;

-- 0006 documents & versions (legal layer over existing documents/chunks)
CREATE TABLE legal_documents(id TEXT PRIMARY KEY, org_id TEXT, matter_id TEXT, library INTEGER DEFAULT 0, title TEXT, doc_kind TEXT,
  source TEXT DEFAULT 'local', source_ref TEXT, sensitivity TEXT, privilege_label TEXT DEFAULT 'unreviewed', privilege_set_by TEXT, privilege_set_at TEXT,
  retention_policy_id TEXT, current_version_id TEXT, status TEXT DEFAULT 'active' CHECK(status IN ('active','deleted','purged')), created_at TEXT, created_by TEXT, updated_at TEXT, version INTEGER DEFAULT 1);
CREATE TABLE document_versions(id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_no INTEGER, parent_version_id TEXT,
  status TEXT CHECK(status IN ('draft','final','signed','superseded')), effective_date TEXT, content_sha256 TEXT, byte_size INTEGER, mime TEXT,
  blob_ref TEXT, page_count INTEGER, ocr_pages INTEGER DEFAULT 0, ocr_engine TEXT, ocr_mean_conf REAL, kb_doc_id TEXT, partition_id TEXT,
  index_status TEXT DEFAULT 'pending', created_at TEXT, created_by TEXT);
-- chunks gain provenance (existing table; also fixes DB-1)
ALTER TABLE chunks ADD COLUMN page_start INTEGER; ALTER TABLE chunks ADD COLUMN page_end INTEGER;
ALTER TABLE chunks ADD COLUMN char_start INTEGER; ALTER TABLE chunks ADD COLUMN char_end INTEGER;
ALTER TABLE chunks ADD COLUMN version_id TEXT;    ALTER TABLE chunks ADD COLUMN partition_id TEXT;
CREATE INDEX ix_chunks_doc ON chunks(doc_id); CREATE INDEX ix_chunks_partition ON chunks(partition_id);

-- 0007 ownership & matter on existing artifacts (nullable; legacy rows stay NULL = non-legal)
ALTER TABLE chats ADD COLUMN owner_user_id TEXT; ALTER TABLE chats ADD COLUMN matter_id TEXT;
ALTER TABLE transcripts ADD COLUMN owner_user_id TEXT; ALTER TABLE transcripts ADD COLUMN matter_id TEXT; ALTER TABLE transcripts ADD COLUMN meeting_id TEXT;
ALTER TABLE boardroom_sessions ADD COLUMN owner_user_id TEXT; ALTER TABLE boardroom_sessions ADD COLUMN matter_id TEXT; ALTER TABLE boardroom_sessions ADD COLUMN meeting_id TEXT;
ALTER TABLE docgen_jobs ADD COLUMN owner_user_id TEXT; ALTER TABLE docgen_jobs ADD COLUMN matter_id TEXT;
ALTER TABLE transcript_analysis ADD COLUMN matter_id TEXT; -- likewise assistant_* tables
CREATE TABLE legal_jobs(id TEXT PRIMARY KEY, kind TEXT, matter_id TEXT, principal_id TEXT, status TEXT, progress REAL, error_code TEXT, cancel_requested INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT);

-- 0008 meetings & consent
CREATE TABLE meetings(id TEXT PRIMARY KEY, matter_id TEXT NOT NULL, kind TEXT CHECK(kind IN ('client_call','negotiation','board','internal','other')), title TEXT, scheduled_at TEXT,
  consent_policy_id TEXT, status TEXT DEFAULT 'draft', created_at TEXT, created_by TEXT);
CREATE TABLE meeting_participants(id TEXT PRIMARY KEY, meeting_id TEXT, display_name TEXT, role TEXT, user_id TEXT, external INTEGER, jurisdiction TEXT);
CREATE TABLE consent_policies(id TEXT PRIMARY KEY, org_id TEXT, name TEXT, rule TEXT CHECK(rule IN ('all_party','one_party')), methods_json TEXT, script_text TEXT, version INTEGER, approved_by TEXT, approved_at TEXT);
CREATE TABLE consent_events(id TEXT PRIMARY KEY, meeting_id TEXT, participant_id TEXT, event TEXT CHECK(event IN ('granted','withdrawn')), method TEXT, policy_id TEXT, policy_version INTEGER,
  captured_by TEXT, captured_at TEXT, evidence_ref TEXT, note TEXT);           -- append-only via triggers
CREATE TABLE capture_sessions(id TEXT PRIMARY KEY, meeting_id TEXT, kind TEXT CHECK(kind IN ('live','boardroom','voice')), started_by TEXT, started_at TEXT, ended_at TEXT,
  consent_snapshot_sha256 TEXT, status TEXT, pauses_json TEXT);
CREATE TABLE transcript_revisions(id TEXT PRIMARY KEY, transcript_id TEXT, segment_id TEXT, before_text TEXT, after_text TEXT, edited_by TEXT, edited_at TEXT, reason TEXT);
CREATE TABLE audio_chunk_custody(session_id TEXT, chunk_index INTEGER, sha256 TEXT, byte_size INTEGER, received_at TEXT, uploader_id TEXT, PRIMARY KEY(session_id, chunk_index));

-- 0009 records management
CREATE TABLE retention_policies(id TEXT PRIMARY KEY, org_id TEXT, name TEXT, artifact_type TEXT, retain_days INTEGER, action TEXT CHECK(action IN ('review','delete')), policy_version INTEGER, approved_by TEXT, approved_at TEXT, active INTEGER DEFAULT 0);
CREATE TABLE legal_holds(id TEXT PRIMARY KEY, org_id TEXT, name TEXT, reason TEXT, status TEXT CHECK(status IN ('requested','active','release_requested','released')), requested_by TEXT, approved_by TEXT, created_at TEXT, released_by TEXT, released_at TEXT);
CREATE TABLE hold_scopes(hold_id TEXT, scope_type TEXT CHECK(scope_type IN ('matter','client','document','transcript','meeting','custodian')), scope_id TEXT, PRIMARY KEY(hold_id, scope_type, scope_id));
CREATE TABLE deletion_requests(id TEXT PRIMARY KEY, object_type TEXT, object_id TEXT, matter_id TEXT, requested_by TEXT, reason TEXT, dry_run_json TEXT,
  status TEXT CHECK(status IN ('pending','approved','blocked_by_hold','executed','failed','cancelled')), approved_by TEXT, executed_at TEXT, verification_json TEXT);
CREATE TABLE labels(id TEXT PRIMARY KEY, object_type TEXT, object_id TEXT, kind TEXT CHECK(kind IN ('privilege','confidentiality','pii_finding')), value TEXT, source TEXT CHECK(source IN ('user','detector')), offsets_json TEXT, set_by TEXT, set_at TEXT);

-- 0010 exports
CREATE TABLE export_requests(id TEXT PRIMARY KEY, matter_id TEXT, artifact_type TEXT, artifact_id TEXT, format TEXT, requested_by TEXT, requested_at TEXT,
  findings_json TEXT, redaction_plan_json TEXT, preview_blob TEXT, approved_by TEXT, approved_at TEXT, executed_at TEXT, output_blob TEXT, output_sha256 TEXT,
  verification_json TEXT, watermark TEXT, status TEXT);

-- 0011 connectors
CREATE TABLE integration_connections(id TEXT PRIMARY KEY, org_id TEXT, provider TEXT, display_name TEXT, config_enc TEXT, status TEXT, last_sync_at TEXT, last_error_code TEXT, created_by TEXT, created_at TEXT);
CREATE TABLE connector_acl_snapshots(id TEXT PRIMARY KEY, connection_id TEXT, external_ref TEXT, acl_json TEXT, fetched_at TEXT, sha256 TEXT);
CREATE TABLE connector_sync_runs(id TEXT PRIMARY KEY, connection_id TEXT, started_at TEXT, ended_at TEXT, status TEXT, added INTEGER, updated INTEGER, removed INTEGER, denied INTEGER, error_code TEXT);
```

### 3.2 Later schema (Phases 2–4)

- `ai_artifacts`: kind, matter, sources, model, prompt-template version, content hash, status `draft|in_review|approved|rejected|superseded`.
- `review_approvals`.
- `research_threads` / `research_turns` (evidence JSON). These extend chats/messages with an `evidence_json` column.
- `clause_playbooks`, `clause_versions`.
- `legal_templates` (with versions and approvals).
- `defined_terms`, `citation_checks`.
- `legal_obligations` (trigger, due date, timezone, owner, status, `confirmed_by`, evidence).
- `diligence_checklists`, `diligence_items`, `diligence_issues`.
- `evaluation_runs` (commit, models, datasets, results).
- `intake_sessions`, `time_entry_drafts`, `panel_reviews` (Phase 4).

## 4. Authorization model

### 4.1 Roles

**Firm roles** (`user_roles`) grant org-level actions only. **Firm admin has no matter-content access by default.**

| Action | firm_admin | managing_partner | attorney | paralegal | reviewer | records_admin | read_only |
|---|---|---|---|---|---|---|---|
| users.manage, connectors.manage, system.settings | ✓ | | | | | | |
| clients.create, matters.create | | ✓ | ✓ (becomes lead) | | | | |
| walls.request | | ✓ | ✓ | | | | |
| walls.approve, wall_exceptions.approve (two-person: approver ≠ requester) | | ✓ | | | | | |
| holds.request / holds.approve (≠ requester) | | approve | request | | | request | |
| retention.configure / retention.run (dry-run; execute only with approved policy) | | | | | | ✓ | |
| audit.read (org-wide metadata) | ✓ | ✓ | | | | ✓ | |
| breakglass.request / breakglass.approve (≠ requester) | request | approve | | | | | |

**Matter roles** (`matter_memberships.role`, valid within `[valid_from, valid_to)` and not revoked):

| Action | lead | attorney | paralegal | reviewer | read_only |
|---|---|---|---|---|---|
| matter.read, documents.read, research.query, meetings.read | ✓ | ✓ | ✓ | ✓ | ✓ |
| matter.update, members.manage | ✓ | | | | |
| documents.import, documents.version | ✓ | ✓ | ✓ | | |
| labels.set (privilege / confidentiality) | ✓ | ✓ | suggest | ✓ | |
| meetings.create, consent.capture, capture.start/pause/stop | ✓ | ✓ | ✓ | | |
| ai.draft (minutes, redlines, memos, extracts) | ✓ | ✓ | ✓ | | |
| artifacts.approve (≠ author) | ✓ | ✓ | | ✓ | |
| export.request (requires `can_export`) | ✓ | ✓ | ✓ | | |
| export.approve (≠ requester) | ✓ | ✓ | | | |
| deletion.request | ✓ | ✓ | | | |
| audit.read (this matter) | ✓ | | | | |

### 4.2 Decision algorithm (`legal/policy.py`)

```
authorize(principal, action, resource) -> Decision(allow|deny, reason_code, matter_id, obligations[])
  1  session valid (not revoked, idle/absolute not expired, MFA satisfied if required)   else deny 'unauthenticated'
  2  resolve resource → matter_id server-side (client hints ignored); unknown → deny 'not_found'
  3  ethical wall denies (user × matter|client) without approved, unexpired exception        → deny 'not_found'
  4  org-level action → firm role check                                                      → allow/deny
  5  matter status closed/archived and action mutates content (records actions excepted)     → deny 'matter_closed'
  6  active membership grants action  OR  active break-glass grant for read actions           → continue  (break-glass never crosses a wall)
     else                                                                                    → deny 'not_found'
  7  attribute obligations: export ⇒ can_export ∧ redaction review if privileged/PII;
     delete/purge ⇒ no active hold scope; highly_restricted ⇒ explicit membership (no break-glass)
  8  default deny
  every deny audited; content reads audited (doc open, citation resolve, transcript view, export)
```

- **Non-disclosure:** matter-scoped resources return `404 {"error":"not_found"}` for both deny and missing. Org-level actions return 403.
- **`acl_epoch`:** incremented on any membership, wall, exception, break-glass or matter-status change. Sessions cache decisions keyed by `(user, matter, epoch)`. Streams and WebSockets re-check on each chunk and every ≤60 s.

### 4.3 Enforcement points

The policy engine is called from:
- every `/api/legal/*` route;
- every **legacy** route when `ECHOMIND_EDITION=law` (chat, docs, transcribe, boardroom, docgen, export). These are either rewired to require a matter or disabled with 404 in the Law edition. Examples: `delete-all`, `debug-retrieval` and `data-preview` are disabled.
- every WebSocket (`/api/transcribe/ws`, `/voice/ws`);
- retrieval (`AuthorizedScope`);
- background jobs (on start and before writing or exposing output);
- file serving and exports.

A route-table test asserts that every route registered in the Law app is either in the public allow-list or has a declared policy action. **Any unannotated route fails CI.**

## 5. Retrieval isolation design (D-005)

- **Partitions.** `m:<matter_id>` holds matter documents, transcripts and meeting artifacts. `lib:<org_id>` is the firm library: approved playbooks, templates and precedents, readable per policy. Legacy verticals keep the global index.
- **Storage.** `/data/legal/partitions/<pid>/{dense.faiss, dense_ids.json, sparse.json, meta.json}`.
  - `IndexFlatIP` on normalized vectors (cosine), giving exact search because matters are small.
  - Above 50k vectors in one partition, switch to `IndexIDMap2(IndexFlatIP)` with IVF, keeping deletes via `IndexIDMap2.remove_ids`.
- **Refactor.** Extract the FAISS/BM25 logic from the module-level globals in `rag/index.py` and `rag/sparse.py` into `PartitionIndex(add, remove, search_dense, search_sparse, stats)`. The existing module API delegates to a default global instance, so legacy behaviour is unchanged (proved by existing tests and the golden eval).
- **API.** `retrieve(scope: AuthorizedScope, query, opts)`. The scope is constructed **only** by `policy.scope_for(principal, matter_id, purpose)` and carries partition IDs, `acl_epoch` and an HMAC tag. In the Law edition, retrieval functions assert a valid scope, and a unit test inspects call sites.
- **Pipeline reuse.** Query classification, RRF fusion, cross-encoder rerank, relevance gate and evidence extraction from `rag/advanced.py` are reused on the scoped candidate set.
  - In Law: BookRAG, section, TOC, glossary and cross-ref global indexes are **off**; the whole-KB fallback is **off**; and the transcript time-window shortcut is replaced by a matter-scoped, explicit "Summarize meeting X" action.
- **Caches.** Partition LRU (default 32 loaded). The query-embedding cache holds no content and is fine. Any answer or evidence cache is keyed by `(scope.partitions, epoch, query hash)`.
- **Ingestion jobs.** Upload → blob stored (encrypted) → `legal_jobs` ingest job captures the principal → parse/OCR → chunk with offsets → embed (batched; the Ollama request loop is parallelized ×4) → write to partition → `index_status=ready`. Before marking ready, the job re-checks that the principal still has `documents.import`; otherwise it rolls back.
- **Revocation.** Removing a membership bumps the epoch, so the next request is denied. Open streams terminate with `scope_revoked`, and client caches for that matter are purged.

## 6. API contracts (new routes; all JSON, typed Pydantic models, OpenAPI generated)

**Errors:**
- `401 unauthenticated`
- `403 forbidden` (org-level only)
- `404 not_found` (non-disclosing)
- `409 conflict` (state, hold, version)
- `412 precondition_failed` (consent missing, approval missing)
- `413 too_large`, `415 unsupported_type`
- `422 validation_error`
- `423 locked` (account lockout)
- `429 rate_limited`

All mutating routes need the CSRF header and accept an `Idempotency-Key`. Each route emits the audit action named in the table.

| Method & path | Policy action | Audit action | Slice |
|---|---|---|---|
| POST `/api/auth/login` · `/api/auth/mfa/verify` · `/api/auth/logout` | public / session | `auth.login.*`, `auth.mfa.*`, `auth.logout` | S1.1 |
| POST `/api/auth/mfa/enroll` · `/confirm`; GET/DELETE `/api/auth/sessions[/{id}]`; POST `/api/auth/password` | self | `auth.*` | S1.1 |
| POST `/api/auth/obo-token` (internal: voice service, audience `voice`) | session | `auth.obo.issue` | S1.1 |
| GET/POST `/api/legal/clients`; GET/PATCH `/api/legal/clients/{id}` | clients.* | `client.*` | S1.2 |
| GET/POST `/api/legal/matters`; GET/PATCH `/api/legal/matters/{id}` | matters.create / matter.read / matter.update | `matter.*` | S1.2 |
| GET/POST `/api/legal/matters/{id}/memberships`; PATCH/DELETE `…/{mid}` | members.manage | `membership.*` | S1.2 |
| GET/POST `/api/legal/walls`; POST `/walls/{id}/approve`; POST `/walls/{id}/exceptions`, `/exceptions/{eid}/approve` | walls.* | `wall.*` | S1.4 |
| POST `/api/legal/break-glass`; POST `/break-glass/{id}/approve`; POST `/break-glass/{id}/end` | breakglass.* | `breakglass.*` | S1.4 |
| GET `/api/legal/audit` (org) · GET `/api/legal/matters/{id}/activity`; GET `/api/legal/audit/verify`; GET `/api/legal/audit/export?format=jsonl\|csv` | audit.read | `audit.read`, `audit.export` | S1.5 |
| POST `/api/legal/matters/{id}/documents/import` (multipart; `kind`, `version_of?`, `status`) | documents.import | `document.import` | S1.6 |
| GET `/api/legal/matters/{id}/documents`; GET `…/documents/{docId}`; GET `…/documents/{docId}/versions`; GET `…/versions/{vid}/file` (no-store) | documents.read | `document.read`, `document.file` | S1.6 |
| POST `…/documents/{docId}/reindex`; PATCH `…/documents/{docId}` (title, kind, labels) | documents.version / labels.set | `document.*`, `label.set` | S1.6 / S1.9 |
| POST `/api/legal/matters/{id}/research/query` (stream) → `{answer, evidence[], status, abstention_reason?}` | research.query | `research.query` (metadata only) | S1.7 (scoped legacy) → S2.2 (evidence contract) |
| GET `/api/legal/matters/{id}/research/threads[/{tid}]` | research.query | `research.read` | S1.7 |
| POST `/api/legal/matters/{id}/meetings`; GET `…/meetings[/{mid}]` | meetings.create / read | `meeting.*` | S1.8 |
| POST `…/meetings/{mid}/participants`; POST `…/meetings/{mid}/consents` (grant/withdraw) | consent.capture | `consent.granted\|withdrawn` | S1.8 |
| POST `…/meetings/{mid}/capture/start\|pause\|resume\|stop` → `capture_session_id` (WS then binds to it) | capture.* | `capture.*` | S1.8 |
| PATCH `…/transcripts/{tid}/segments/{sid}` (correction with reason) | meetings.create | `transcript.revise` | S2.4 |
| POST `…/meetings/{mid}/minutes/draft`; POST `…/minutes/{aid}/approve` | ai.draft / artifacts.approve | `minutes.*` | S2.6 |
| GET/POST `/api/legal/holds`; POST `/holds/{id}/approve\|release\|release/approve` | holds.* | `hold.*` | S1.9 |
| GET/POST `/api/legal/retention/policies`; POST `/api/legal/retention/run?dry_run=true` (records_admin; execute only with approved policy) | retention.* | `retention.*` | S1.9 |
| POST `/api/legal/deletions` (request) → dry-run report; POST `/deletions/{id}/approve` | deletion.request / approve | `deletion.*` | S1.9 |
| POST `/api/legal/matters/{id}/exports/preview` → findings, redaction plan, preview; POST `/exports/{eid}/approve`; POST `/exports/{eid}/execute`; GET `/exports/{eid}/file` | export.* | `export.*` | S1.10 |
| GET/POST `/api/legal/connectors`; POST `/connectors/{id}/sync`; GET `/connectors/{id}/runs` | connectors.manage | `connector.*` | S1.11 |
| POST `/api/legal/matters/{id}/contracts/compare` | research.query | `compare.run` | S2.3 |
| POST `…/redlines/draft`; GET `…/redlines/{id}`; GET `…/redlines/{id}/export?format=html\|pdf\|docx` (docx only after T10) | ai.draft / export.* | `redline.*` | S3.3 |
| POST `…/citations/check`; POST `…/definitions/check` | research.query | `check.*` | S3.4 / S3.5 |
| GET/POST `…/obligations`; PATCH `…/obligations/{id}` (confirm/edit) | ai.draft / matter.update | `obligation.*` | S3.6 |
| POST `…/diligence/runs`; GET `…/diligence/runs/{id}`; PATCH `…/diligence/issues/{id}` | ai.draft | `diligence.*` | S3.7 |
| GET `/api/legal/evaluations/reports` | audit.read | `eval.read` | S2.8 |

**WebSocket changes (Law edition):**
- `/api/transcribe/ws` `start` must carry `capture_session_id`. The server verifies session, membership, consent snapshot and capture state **before** loading ASR context. The `namespace` field is ignored.
- New server messages: `capture_paused`, `capture_resumed`, `consent_withdrawn` (followed by stop), `scope_revoked`.
- New client messages: `pause`/`resume` already exist and are now consent-aware.
- `/voice/ws` requires `matter_id` in `set_context`. The voice service gets an OBO token and calls the matter research endpoint. A bare `set_context` no longer resets scope.

## 7. Phase plan and slices

Effort is in **engineering-days, Claude-assisted with one human reviewer**, as ranges. The assumptions are in §10.

### Phase 0 — Audit (this deliverable) ✅ awaiting review
- **Done:** repository and runtime audit, feature matrix, threat model draft, design system and wireframes, decision log, license register, baseline unit tests (TEST_RESULTS.md).
- **Remaining for the exit gate:**
  1. Owner review and approval of DECISIONS D-001…D-019. 🔒
  2. Golden-eval baseline on this commit. It needs the law corpus. Recreate the two DEMO documents as **committed fictional fixtures** in `eval/fixtures/law/`, and run against the **Law dev backend**, not production (S1.0).
  3. Counsel contact for the P-items.
- **Exit gate:** gap matrix reviewed; no destructive changes made.

### Phase 1 — Foundations (SEC-01…08, OPS-01/02 foundations)

#### S1.0 Law foundations (PR-1) — est. 3–4 d
- **Goal:** a safe place to build. Edition guard, migrations, fresh-install fixes, an isolated dev runtime, test scaffolding. No user-visible features.
- **Backend:**
  - `backend/app/core/edition.py` (new): `is_law()` and `validate_law_posture()`. Startup refuses on: auth off, weak or missing secret, `CORS=*`, voice auth off, `nim` image backend, prompt and payload logging taps on, `public` profile env.
  - `backend/app/main.py`: call the guard; disable `/docs` and `/openapi.json` in Law (kept in dev via `LEGAL_DEV_OPENAPI=1`); content-free log formatter in Law.
  - `backend/app/core/config.py`: declare `ECHOMIND_EDITION`, `LEGAL_*` settings, and the undeclared knobs from R-13.
  - `backend/app/core/db.py`: **fix DB-1** (add `page_start`/`page_end` to the `CREATE` and an `ALTER` for existing DBs) and **fix DB-2** (move the boardroom `ALTER`s after the `CREATE`). These are shared fixes, proposed for cherry-pick to `main`.
  - `backend/app/legal/{__init__.py, migrations/runner.py, migrations/0001_foundation.up.sql, .down.sql}`, with the runner called from `init_db()` only when the edition is Law.
- **Voice:**
  - `voice/app/config.py`: edition flag.
  - `voice/app/session.py`: `VOICE_ABLATION_MODE`, `/tmp/echomind_ablation_mode` and the `/tmp/dump_payload` tap become unreachable in Law (D-016).
  - Transcript-content logs drop to DEBUG with a content-free formatter in Law.
- **Ops:**
  - `docker-compose.law.yml`, `.env.law.example`.
  - `scripts/law/dev_backend.sh`, `dev_frontend.sh`, `asr_container.sh` (refuses to start if free memory is under 12 GB).
  - `.gitignore`: add `.law-data/` and `offline-bundle-*/`.
- **Frontend:** add Vitest, Testing Library and jsdom (dev); `vitest.config.ts`; a `test` script; fix Tailwind `content` to include `./verticals/**` and `./law/**`.
- **CI:** `.github/workflows/ci.yml` gains a `backend-law` job (`pytest tests/law`) and a `frontend-test` job (`vitest run`).
- **Fixtures:** `eval/fixtures/law/` with the recreated fictional MSA and playbook as Markdown, plus a provenance note. `eval/golden/law.jsonl` keeps working against them.
- **Tests:**
  - `backend/tests/law/test_edition_posture.py`: each insecure setting refuses start.
  - `test_fresh_db_schema.py`: fresh DB has the required columns; upload works on a fresh DB.
  - `test_migrations.py`: up, down, idempotent, checksum mismatch refuses.
  - `voice/tests/test_law_edition.py`: ablation and dump unreachable.
- **Acceptance:**
  - All existing suites still pass (50 backend, 131 voice, tsc).
  - New tests pass.
  - The Law dev backend boots against `.law-data` and ingests a fixture on a fresh DB.
  - The golden eval law subset runs against the Law dev backend, with the result recorded.
- **Rollback:** revert the PR. The migration's down script drops only the new tables. The DB-1/DB-2 fixes are additive.

#### S1.1 Identity and sessions (SEC-01) — est. 5–7 d
- **Backend:**
  - `backend/app/legal/identity/{sessions.py, mfa.py (TOTP RFC 6238, stdlib), passwords.py (policy, lockout, back-off), csrf.py, obo.py, oidc.py (flagged off)}`.
  - Migration `0002_identity`.
  - `core/auth.py`: in Law, delegate to server-side sessions. `seed_admin` never logs a password; it writes a one-time bootstrap token to a 0600 file instead.
  - `routes/auth.py`: login → MFA challenge → session cookie `__Host-echomind_s` (Secure, HttpOnly, SameSite=Strict); CSRF cookie and header; session list and revoke; password change; MFA enroll and verify; OBO issuance (internal audience).
  - `main.py` auth guard uses sessions in Law.
  - WS helper `require_ws_session()` is used by `transcribe.py`'s WS, with a 60 s re-validation task.
- **Voice:**
  - `voice/app/auth_check.py` validates the session through the backend.
  - `voice/app/session.py` obtains and refreshes the OBO token and sends `Authorization: Bearer <obo>` on RAG calls (fixes the 401 at `session.py:1524-1530`).
  - In Law the voice service also refuses to start unless `VOICE_AUTH_ENABLED=1`.
- **Frontend (minimal, restyled in S1.3):**
  - `services/backend.ts`: CSRF header; global 401 → re-auth event.
  - `components/Login.tsx`: MFA step and enrollment QR (rendered locally, no external QR service).
- **Tests:**
  - L-AUTH: route-table anonymous denial (HTTP and WS); idle, absolute and revoked expiry; revoke-on-role-change; lockout; CSRF; MFA required; recovery code single-use; OBO audience and expiry; voice RAG works under auth (integration test in the ASR container).
- **Acceptance:** SEC-01 criteria met. OIDC is **BLOCKED** (P-011), with the interface and unit tests against a stub IdP clearly labelled as such.
- **Rollback:** feature-flagged by edition; the down migration drops new tables and columns are left nullable.

#### S1.2 Matters, memberships, policy engine (SEC-02 part 1) — est. 5–7 d
- **Backend:**
  - `backend/app/legal/policy.py` (§4.2); `legal/matters/{models.py, repo.py, routes.py}`; `legal/errors.py` (non-disclosing errors).
  - Migration `0003_matters`.
  - Org bootstrap CLI: `python -m app.legal.cli bootstrap-org`.
- **Tests:**
  - Policy unit matrix (role × membership validity × matter status × sensitivity): about 200 parametrized cases.
  - API contract tests.
  - **T01 fixture**: org with 2 clients, 4 matters (M-101 Alpha, M-114 Alpha, M-202 Beta restricted, M-215 Beta), 5 users, a revoked and an expired membership. Lists, by-ID routes and counts reveal nothing outside membership, and the timing shape is comparable.
- **Acceptance:** policy matrix green; T01 (matters scope) green.
- **Rollback:** down migration; routes are behind the edition flag.

#### S1.3 Law workspace shell and design system (SEC-02 UI) — est. 6–9 d
- **Frontend:**
  - `frontend/law/**` per DESIGN_SYSTEM §9: tokens, glass, self-hosted fonts, Lucide icons, router, `MatterContext`, typed API client with matter-keyed TanStack Query caches.
  - `AppSidebar`, `MatterSwitcher`, `Toolbar`, `CommandPalette`, `StatusPill`, `EmptyState`, `ErrorState`, `Skeleton`, `Toast`.
  - Screens: Login and MFA (restyled), Home, Matters list, Matter overview, Members (lead only), Not-available state; `/__design` catalogue (dev only).
  - `App.tsx` renders `LawApp` before the legacy hooks when the edition is Law.
  - The marketing `LawExperience` moves to a public `/about` route (not linked in the Law edition).
- **Dependencies to approve (🔒 new packages):** `react-router`, `@tanstack/react-query`, `lucide-react`, `@fontsource-variable/inter`, `@fontsource-variable/source-serif-4`, `qrcode` (local MFA QR); dev-only `vitest`, `@testing-library/react`, `jsdom`, `@playwright/test`.
- **Tests:**
  - Vitest interaction tests (switcher confirms with an active capture; caches purge on switch; items you can't use are absent; non-disclosing page).
  - axe accessibility checks.
  - Playwright E2E: sign in → see only M-101/M-114 → M-202 is never visible, and its URL shows "not available".
  - Visual snapshots in light, dark, reduced transparency and high contrast.
- **Acceptance:** design approved by the owner (🔒); E2E green; other verticals unchanged (smoke run).

#### S1.4 Ethical walls and break-glass (SEC-02 part 2) — est. 3–4 d
- **Backend:** migration `0004_walls`; policy steps 3 and 6; two-person approvals; `acl_epoch` bump; routes.
- **Frontend:** Admin → Ethical walls (request, approve, exceptions); break-glass request and approve banner. An active break-glass session shows a persistent `--attention` banner.
- **Tests:**
  - A walled user is denied even as lead or firm admin.
  - An exception grants access only while valid.
  - Break-glass never crosses a wall.
  - All of it is audited.
- **Acceptance:** walls enforced in policy unit tests and in T01.

#### S1.5 Audit v2 (SEC-04) — est. 3–5 d
- **Backend:**
  - Migration `0005_audit` (triggers).
  - `legal/audit/{record.py, chain.py, verify.py, export.py, anchor.py}` with daily anchor files.
  - `audit.record()` is called by the policy engine (all denies, content reads) and by services.
  - Optional syslog/CEF forwarder (disabled by default; enabling it needs network approval 🔒).
- **Frontend:** Admin → Audit (filters, chain badge, JSONL/CSV export); Matter → Activity.
- **Tests:**
  - Completeness matrix (each action emits exactly one event).
  - A tamper attempt aborts.
  - An out-of-band edit is detected by the verifier.
  - Planted canary content never appears in audit or app logs (log-scan test).
- **Acceptance:** SEC-04 criteria.

#### S1.6 Matter documents and partitioned retrieval (SEC-03, RAG-01 base) — est. 7–10 d
- **Backend:**
  - Migrations `0006_documents`, `0007_ownership`.
  - `legal/retrieval/{partition_index.py, scope.py, registry.py}`; refactor `rag/index.py` and `rag/sparse.py` behind `PartitionIndex` (legacy default instance).
  - `legal/documents/{routes.py, ingest_job.py, blobs.py (AES-GCM, D-014 🔒 dependency `cryptography`)}` and `legal/jobs/worker.py`.
  - Law ingestion skips the global PII redaction (D-012) and stores PII findings as labels.
  - Upload limits (size, pages, type agreement).
  - File route sends `no-store`.
- **Frontend:** Documents library (import, versions, index status, labels read-only for now) and an Inspector shell.
- **Tests:**
  - **Canary leakage suite v1**: a unique token in each matter's documents; queries from every other principal across dense, sparse, keyword, rerank, evidence and citations return zero hits.
  - **T02**: revoke membership → next query 404, open stream ends with `scope_revoked`.
  - **T06**: the forbidden matter contains the answer, but the permitted query abstains.
  - Ingestion rolls back if the principal loses rights mid-job.
  - Legacy golden eval unchanged against a legacy fixture.
- **Acceptance:** SEC-03 for documents.

#### S1.7 Matter-scoping of existing modules (SEC-03 completion) — est. 6–9 d
- **Changes**, with the Law edition switch on each:
  - `chat.py`: `/ask*` require a matter in Law; history is filtered by owner and matter; the time-window shortcut is disabled; debug routes are off.
  - `docs.py`: legacy list, file, data-preview and delete-all return 404 in Law.
  - `transcribe.py` and `ws.py`: namespace ignored; transcripts are stamped with owner and matter; list and by-ID calls are scoped.
  - `silent_assistant/retrieval.py`: whole-KB fallback removed; scope required.
  - `boardroom/*`: owner and matter stamping, scoped retrieval in the report (fixes `service.py:792`).
  - `docgen`: `kb` and `chat` sources must belong to the matter.
  - voice: scope from the matter, server-side persona and prompt only.
- **New:** `/api/legal/matters/{id}/research/query` wrapping the existing pipeline with an `AuthorizedScope`; the frontend Research page v0 (findings plus sources from the existing citations).
- **Tests:**
  - Canary suite v2 covering every module above, including WS and voice.
  - T01 completed: list, search, chat, citation, thumbnail, WS, export and history.
- **Acceptance:** zero canary leaks across all modules. **This is the main security milestone.**

#### S1.8 Consent and capture control (SEC-05) — est. 4–6 d
- **Backend:**
  - Migration `0008_meetings`; `legal/meetings/{routes.py, consent.py, capture.py}`.
  - Gates in `ws.py _handle_start`, the Boardroom chunk upload and the voice handshake.
  - Withdrawal pushes a stop and rejects frames.
  - Boardroom chunk SHA-256 custody records.
  - Consent policy defaults (P-001).
- **Frontend:** Meetings list, Meeting page (Prepare / Consent / Record tabs), `ConsentPanel`, `RecordingCapsule`.
- **Tests (T03):**
  - Missing, withdrawn or stale consent refuses start: the WS gets `412 consent_required` before ASR loads.
  - Withdrawal mid-stream stops capture within one frame batch.
  - A race (start and withdraw concurrently) never captures after the withdrawal timestamp.
  - Timeline and audit are correct.
- **Acceptance:** SEC-05.

#### S1.9 Labels, retention, legal hold, deletion (SEC-07, SEC-06 labels) — est. 5–7 d
- **Backend:**
  - Migration `0009_records`; `legal/records/{labels.py, holds.py, retention.py, deletion.py}`.
  - **All** delete paths (legacy and legal) route through `DeletionService` in Law: soft delete → dry-run report (rows, chunks, partition vectors, blobs, analyses, exports) → approval → purge in a transaction with a hold re-check → verification. Retention runner stays dry-run until a policy is approved (P-002).
- **Frontend:** label editor (privilege and confidentiality, audited); Admin → Retention and holds; deletion request flow.
- **Tests (T04):** user, retention worker, system cleanup, direct service call and legacy route all refuse held items; purge verification finds no residue; restore from soft delete works.
- **Acceptance:** SEC-07 mechanics DONE. Policy values remain **NEEDS COUNSEL**.

#### S1.10 Export gateway v2 and redaction (SEC-06) — est. 6–8 d
- **Backend:**
  - Migration `0010_exports`; `legal/exports/{detect.py, plan.py, render_pdf.py, render_docx.py, sanitize.py, verify.py, watermark.py}`.
  - Detectors extend `core/export_gateway.py` with matter party names, matter numbers, account numbers and privilege markers.
  - PDF redaction removes content (PyMuPDF `add_redact_annot` + `apply_redactions`, then text-extraction verification).
  - DOCX output is rebuilt clean: comments, revisions, `docProps`, custom XML and hidden runs stripped.
  - Every Boardroom, docgen and document export in Law goes through this gateway.
- **Frontend:** `ExportPreview` (thumbnails, redaction boxes with reasons, approve, execute, download).
- **Tests (T09):** for each format, the redacted strings are absent from the extracted text, metadata, comments and revisions; unauthorized export is denied; the watermark is present; the audit chain is complete.
- **Acceptance:** SEC-06.

#### S1.11 DMS connector framework (SEC-08) — est. 4–6 d (+10–20 d per real provider once unblocked)
- **Backend:**
  - Migration `0011_connectors`; `legal/connectors/{base.py (DmsConnector protocol), local_folder.py, sync.py}`.
  - Encrypted config.
  - Fail-closed ACL intersection (sidecar `*.acl.json` for the local connector).
  - Removal → purge from the partition.
- **Frontend:** Admin → Connectors; document source badges; stale and failed states.
- **Tests:** T15 with LocalFolderConnector.
- **Acceptance:** framework DONE. **Provider BLOCKED (P-010)**: no mock is presented as working. 🔒 Credentials, licences and network approval are needed to unblock.

#### S1.12 Phase 1 gate — est. 3–5 d
- **Security suite:** T01, T02, T03, T04, T08 (injection fixtures: hidden-text PDF, instruction-laden DOCX; the model cannot reach other partitions or tools), T09, T11 (network-denied E2E: backend and frontend run in a network namespace with only the model servers allowed; outbound connections are captured and asserted empty), T13 (seeded upgrade, rollback, encrypted restore), T14 (legacy smoke: chat, Boardroom, voice, Document Studio, other verticals).
- **Outputs:** threat-model review; release-gate report in `docs/echomind-law/releases/phase1-gate.md` with the commit, models and test outputs.
- **Exit:** zero unauthorized matter disclosure and zero recording without consent (spec gate). 🔒 Owner sign-off.

### Phase 2 — Pilot core

| Slice | Content | Est. |
|---|---|---|
| **S2.1 Legal ingestion quality** (RAG-01) | Tesseract OCR with per-page provenance and confidence (D-010); DOCX structure (auto-numbering via `w:numPr` resolution, tables, footnotes, headers and footers; comments and tracked changes extracted as metadata, not body text); exact char offsets and page/line maps; clause segmentation (`§`, `12.3(b)`, Article, Schedule, Exhibit); defined-terms capture; hidden-text and injection detection flags; version status and "signed only"; async ingestion with progress. **Tests:** OCR fixture with known text (CER threshold), DOCX numbering fixture, offset round-trip (citation offsets highlight the exact quote). | 6–9 d |
| **S2.2 Legal research** (RAG-01) | Evidence contract `{answer, status, evidence[{doc_id, version_id, version_status, section, page, line?, char_range, quote, verification: verbatim\|normalized\|failed}], conflicts[], abstention_reason}`; post-generation **quote verification** (each cited quote must exist in the authorized chunk text, otherwise the claim is dropped or flagged); abstention when evidence is insufficient; conflict detection across versions; signed-only default; scope = matter (+ optional firm library); fixes R-4 (short legal terms), R-6 (pinpoints kept, off-by-one), R-7 (multi-source answers), R-8 (Lawyer persona server-side, no FMR text); paragraph-level streaming; Research UI (FindingSheet, Pinpoint, Inspector with highlighted quote, version switcher, "Copy with citations"). **Tests:** T05, T06, T08; legal golden set ≥60 plus a held-out set ≥30 (never used in tuning); citation precision; abstention accuracy; p50/p95 latency. | 6–8 d |
| **S2.3 Compare and contradictions** (RAG-02) | Clause alignment (numbering + embeddings) → per-clause diff → explanation citing both quotes; cross-document conflict scan (amounts, dates, parties, notice periods); Compare UI. **Tests:** seeded contract pair. | 3–5 d |
| **S2.4 Legal transcription** (LIVE-01) | Verbatim mode (D-017); roster-based N speakers; wall-clock anchors; matter-vocabulary correction with provenance; `transcript_revisions` UI; pause and resume; WER and entity harness (TTS-generated scripts with legal names, dates, amounts and negations; AMI subset only after the licence check). **Tests:** WER and entity accuracy reported per scenario; revision provenance. | 5–7 d |
| **S2.5 Silent Assist legal packs** (LIVE-02) | `legal_client_call`, `legal_negotiation`, `legal_governance` packs (inconsistency with the signed agreement, unauthorized commitment, privileged or sensitive disclosure, unresolved action, quorum, conflict, authority); evidence-relative wording (D-015); conservative thresholds; Confirm/Dismiss with reason; telemetry (alerts/hour, dismiss rate, latency); `AlertCard` UI. **Tests:** scripted mock calls with planted issues: precision, recall on critical items, false alerts per hour, latency. Thresholds signed off by a lawyer (P-008). | 5–7 d |
| **S2.6 Boardroom legal mode** (MEET-01) | Before: board-pack research and agenda. During: consent, pause, flags. After: map-reduce over the **full** transcript; minutes, resolutions, actions (owner, deadline), unresolved items with timecode pinpoints; fixes the timestamp drop (`service.py:863`) and speaker reconciliation across batches; approval before export; custody hashes; hold and retention. **Tests:** fixture meeting; approval enforced; hold respected. | 5–7 d |
| **S2.7 Review and approval framework** | `ai_artifacts`, `review_approvals`; `ReviewBanner`, `ApprovalDialog`; Reviews queue page; approval hash binding (the approved content hash must match what is exported). | 3–4 d |
| **S2.8 Phase 2 gate** | Legal eval suite in `eval/law/` (golden + held-out + canaries + latency), plus `evaluation_runs` records; lawyer review session using the scoring rubric; latency benchmark on the target hardware. 🔒 Load-test window approval. Gate report. | 4–6 d |

### Phase 3 — Document workflows

| Slice | Content | Est. |
|---|---|---|
| **S3.1 Legal drafting** (DOC-01) | Template registry with versions and approvals; presets (engagement letter, NDA, demand letter, internal memo, contract review report, board minutes); field extraction with provenance; **no-invention writer** (gaps marked `[[NEEDS INPUT: …]]`; the "plausible specifics" instruction is not used for legal templates); DOCX renderer (python-docx, styles from the firm template); PDF via existing reportlab; Drafting UI. **Tests:** every factual field sourced or flagged; DOCX/PDF open and round-trip through LibreOffice. | 5–7 d |
| **S3.2 Clause library and playbooks** (DOC-02a) | `clause_playbooks`/`clause_versions` (standard, fallbacks, owner, jurisdiction, approval); playbook import from the firm library; deviation classification (Standard / Fallback / Outside playbook) with source; Clause library UI. | 4–6 d |
| **S3.3 Redlines and Word tracked changes** (DOC-02b) | Word-level diff → suggested revision → HTML redline; **D-011 spike**: OOXML `w:ins`/`w:del` with author and date, plus comments; T10 automated checks plus LibreOffice round trip plus 🔒 **manual Microsoft Word verification by a human**. If the spike fails, DOCX tracked changes are BLOCKED and the HTML/PDF redline ships. | 5–8 d |
| **S3.4 Citation and quote checker** (DOC-03) | Parser (internal: §, clause, exhibit, schedule; external: U.S.C., C.F.R., reporters); internal resolution to exact passages; altered-quote detection (normalized diff); external authority **UNVERIFIED** (P-009); report UI. **Tests:** T07 with planted fictional authorities and altered quotes. | 4–6 d |
| **S3.5 Defined terms and cross-references** (DOC-04) | Definitions, usage, undefined, unused, duplicate and inconsistent terms; numbering continuity; broken and outdated references; offsets and proposed fixes (approval required). **Tests:** planted errors. | 3–5 d |
| **S3.6 Obligations tracker** (DOC-05) | Extraction with evidence spans; human confirmation; timezone; status; optional ICS export (off by default). **Tests:** fixture contracts. | 4–5 d |
| **S3.7 Due diligence** (DOC-06) | Checklists; data-room scan job; missing-document list; issue register (preliminary vs verified); reviewer trail. **Tests:** synthetic data room. | 4–6 d |
| **S3.8 Governance checks** (DOC-07) | Cap table vs approvals, delegation matrix, policy vs bylaws, IP assignment; advisory findings. **Tests:** fictional company fixtures. | 3–5 d |
| **S3.9 Phase 3 gate** | Planted-error suite, format round trips, human sign-off. | 3–4 d |

### Phase 4 — Optional (each behind its own flag; separate business approval 🔒)

| Slice | Content | Est. |
|---|---|---|
| S4.1 Intake assistant (OPT-01) | Scripted administrative intake (voice or text), disclaimer, conflict-name screening against client and party lists (minimized output: "possible conflict, refer to conflicts team"), consent before recording, lawyer-approved replies | 5–7 d |
| S4.2 Summaries and translation (OPT-02) | Timecode and page/line-cited summaries with reviewer edits; translation **BLOCKED** until a local MT model and licence are approved; original always preserved | 3–5 d (+MT) |
| S4.3 Time-entry drafts (OPT-03) | Suggestions from audited activities; user edits and approves; CSV export only | 3–4 d |
| S4.4 Regulatory monitoring (OPT-04) | **BLOCKED**: needs a licensed source and an approved scheduled channel; disabled in air-gap mode | — |
| S4.5 Four-perspective panel (OPT-05) | Deal / risk-compliance / litigator / specialist plus moderator; independent citations; disagreement matrix; missing-fact stop; partner approval; background queue; benchmark | 6–9 d |

### OPS track (continuous)

- **OPS-01 runbooks:**
  - Law compose and `.env.law` reference.
  - Upgrade and rollback (migration down scripts, image tags).
  - Encrypted backup and restore including `/data/legal` (partitions, blobs, audit anchors), with restore using the consistent SQLite snapshot (fixes the existing gap).
  - Offline prepare/export covering every model (fixes R-15: wrong default model, SDXL/VibeVoice missing).
  - LUKS setup (🔒 host root).
  - SSO and connector runbooks.
  - Disaster-recovery drill (T13).
- **OPS-02 harness:**
  - CI: backend-law, voice, frontend unit, Playwright E2E (seeded), log-scan, route-annotation test.
  - Eval runs recorded with commit, model and dataset versions.
  - Load test script (T12) for concurrent transcription + research + redline, reporting p50/p95, memory and failures.
  - Release-gate report generator.

## 8. Test strategy

| Layer | Tooling | Where |
|---|---|---|
| Policy unit matrix | pytest parametrized | `backend/tests/law/test_policy_*.py` |
| Repository and filters | pytest + temp SQLite | `backend/tests/law/test_repo_*.py` |
| API contracts | FastAPI TestClient + OpenAPI snapshot | `backend/tests/law/test_api_*.py` |
| WebSocket consent and auth | TestClient WS (ASR stubbed by an adapter fake **clearly in test code only**) plus a real-ASR integration run in the container | `backend/tests/law/test_ws_*.py` |
| Retrieval leakage (canaries) | pytest over every retrieval mode | `backend/tests/law/test_leakage_*.py` |
| Prompt injection | Fixture documents and assertions on scope, tool calls and outputs | `backend/tests/law/test_injection.py` |
| Export residue | PyMuPDF and python-docx extraction, plus `zipfile` inspection of DOCX parts | `backend/tests/law/test_export_*.py` |
| React interaction and a11y | Vitest + Testing Library + axe | `frontend/law/**/*.test.tsx` |
| E2E journey | Playwright vs native dev backend with seeded data | `frontend/e2e/law/*.spec.ts` |
| Legal quality | `eval/law/run_law_eval.py`: golden, held-out, abstention, citation precision, latency | `eval/law/` |
| Offline egress | network-namespaced run + connection capture | `scripts/law/egress_test.sh` |
| Load and recovery | `scripts/law/load_test.py`, `scripts/law/dr_drill.sh` | `scripts/law/` |

**Spec test IDs → slices:**
- T01 → S1.2/S1.3/S1.7
- T02 → S1.6/S1.7
- T03 → S1.8
- T04 → S1.9
- T05, T06 → S1.6/S2.2
- T07 → S3.4
- T08 → S1.12/S2.1/S2.2
- T09 → S1.10
- T10 → S3.3
- T11 → S1.12
- T12 → S2.8
- T13 → S1.12/OPS-01
- T14 → every gate
- T15 → S1.11

**Demo journey (spec §11 and §17)** becomes a Playwright spec plus a manual walkthrough in `docs/echomind-law/DEMO_WALKTHROUGH.md`. It is written in S1.3 and grows with each slice. Seed data: `python -m app.legal.cli seed-demo` (fictional: Voss-Lindqvist AB / M-101 MSA renewal; Alder Finch Ltd / M-114; walled Beta Corp / M-202).

## 9. Rollback strategy

- **Code:** every slice is a PR into `feature/echomind-law`, so reverting the PR rolls it back. `main` stays untouched (D-001).
- **Schema:** each migration has a tested `down`. Columns added to legacy tables are nullable and ignored by legacy code. A pre-migration backup (`scripts/backup_data.sh` with encryption) is mandatory before applying migrations to any non-dev DB. 🔒
- **Runtime:** the edition flag is off, so legacy behaviour is unchanged. Per-feature flags (`LEGAL_FEATURE_<NAME>`) cover Phase 2–4 features.
- **Data:** partitions are rebuildable from blobs and chunks (`python -m app.legal.cli reindex --matter …`). Audit anchors allow the chain to be verified after a restore.

## 10. Effort estimate and assumptions

| Phase | Effort (eng-days, Claude-assisted) |
|---|---|
| Phase 1 (S1.0–S1.12) | 60–87 |
| Phase 2 (S2.1–S2.8) | 37–53 |
| Phase 3 (S3.1–S3.9) | 35–52 |
| Phase 4 (optional) | 22–32 (+MT/regulatory when unblocked) |
| OPS track | 10–15 (spread) |
| **Total** | **≈164–239** |

**Assumptions:**
- One human reviewer is available for PR review within 1–2 days.
- Counsel answers the P-items before their slices.
- Development happens on the shared Spark under D-002.
- No firm DMS or IdP integration inside these numbers.
- Estimates include tests and documentation but not lawyer review time.

**Critical path:** S1.0 → S1.1 → S1.2 → S1.6 → S1.7 → S1.8 → S1.12. Counsel decisions P-001/P-002 gate *policy values*, not *mechanisms*.

## 11. Top risks and dependencies

| Risk | Impact | Mitigation |
|---|---|---|
| Shared host with production (memory, compose name collision) | Production outage or data mix | D-002 isolation; memory check scripts; never use production volumes |
| Retrieval refactor regresses legacy quality | Other verticals degrade | Default-instance delegation; legacy golden eval on fixtures before and after |
| SQLite contention in a multi-user pilot | Latency, `SQLITE_BUSY` | T12 measurement; D-004 switch gate to Postgres + RLS |
| OCR accuracy on poor scans | Wrong facts | Provenance, confidence flags, image-beside-text verification |
| Word tracked changes infeasible or fragile | DOC-02 partial | Spike plus stop condition (D-011); HTML/PDF redline fallback |
| Policy decisions delayed (consent, retention) | Pilot launch blocked | Mechanisms ship with the safest defaults; values are configuration |
| GPU contention (ASR + LLM + OCR + reranker) | Live-capture latency | Job queue priorities; OCR on CPU; measure in T12 |
| Licence issues (Piper GPL, SDXL-Turbo, NVIDIA model terms) | Distribution risk | LICENSE_REGISTER review before the pilot; SDXL off in Law |

## 12. Definition of Done (per feature)

DONE only when **all** of the following hold:
- the UI, API, service and data path works end-to-end on authorized sample data;
- failure paths are handled;
- authorization is enforced in the backend **and** in retrieval;
- state persists correctly;
- feature tests and relevant regressions pass;
- user and maintainer docs are updated;
- no unexpected outbound calls occur (egress check);
- a manual demo is possible.

Otherwise the feature is reported as PARTIAL, DEMO ONLY or BLOCKED, with the reason.

## 13. First vertical slice to approve (spec §15.6)

**PR-1 = S1.0, then PR-2 = S1.1 backend, then PR-3 = S1.1 voice and frontend.**

### PR-1 (S1.0) changed files (planned)
```
backend/app/core/edition.py                         (new)
backend/app/core/config.py                          (edit: edition + LEGAL_* + declare R-13 knobs)
backend/app/core/db.py                              (edit: DB-1, DB-2 fixes; call legal migrations in Law)
backend/app/main.py                                 (edit: posture guard, /docs off in Law, log formatter)
backend/app/legal/__init__.py                       (new)
backend/app/legal/migrations/runner.py              (new)
backend/app/legal/migrations/0001_foundation.up.sql / .down.sql (new)
backend/tests/law/conftest.py                       (new)
backend/tests/law/test_edition_posture.py           (new)
backend/tests/law/test_fresh_db_schema.py           (new)
backend/tests/law/test_migrations.py                (new)
voice/app/config.py, voice/app/session.py           (edit: D-016 gates in Law)
voice/tests/test_law_edition.py                     (new)
frontend/package.json, frontend/vitest.config.ts    (edit/new: test runner)
frontend/tailwind.config.js                         (edit: content globs)
docker-compose.law.yml, .env.law.example            (new)
scripts/law/dev_backend.sh, dev_frontend.sh, asr_container.sh (new)
eval/fixtures/law/*.md (+ PROVENANCE.md)            (new, fictional)
.github/workflows/ci.yml, .gitignore                (edit)
docs/echomind-law/PROGRESS.md, TEST_RESULTS.md      (edit)
```

**Migrations:** `0001_foundation` (`schema_migrations`, `organizations`). Additive; down drops both tables.

**Security checks:** posture refusal matrix; no `/docs` in Law; voice taps unreachable; log formatter strips content; fresh-DB ingest works.

**Test commands:**
```bash
cd backend && ECHOMIND_DATA_DIR="$(mktemp -d)" python3 -m pytest tests -q
cd voice && python3 -m pytest tests -q
cd frontend && npx tsc --noEmit -p . && npx vitest run
scripts/law/dev_backend.sh   # boots Law dev backend on :8110 against .law-data (fictional fixtures)
python3 eval/run_eval.py --base http://localhost:8110/api --set law   # existing flags (eval/run_eval.py:245-246)
```

**Needs approval before PR-1 starts:** D-001 (branch), D-002 (dev runtime), D-003 (edition guard), D-004 (migration approach). New dev dependencies: vitest, @testing-library/react, jsdom.
