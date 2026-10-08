# EchoMind Law — Codebase Audit (Phase 0)

> **Scope.** This is an evidence-based audit of the existing EchoMind Enterprise repository. It was written before any EchoMind Law code exists and records what is actually in the repository.
> **Snapshot.** Commit `95782e4` on `main` (tag `v1.4.1` plus QMS-only commits), audited 2026-10-08. File:line references are valid at that commit.
> **Method.** Four parallel code read-throughs (RAG; transcribe/boardroom/docgen/auth; frontend; voice/infra/eval). High-impact claims were re-verified by hand and are marked ✔. Code was trusted over docs.
> **Companion files.** [FEATURE_MATRIX.md](FEATURE_MATRIX.md) (per-feature gap table) · [THREAT_MODEL.md](THREAT_MODEL.md) · [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) · [DECISIONS.md](DECISIONS.md) · [TEST_RESULTS.md](TEST_RESULTS.md) · [LICENSE_REGISTER.md](LICENSE_REGISTER.md) · [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md)

---

## 1. Repository facts

| Item | Value |
|---|---|
| Local root | `/home/echomind/Documents/echomind/law/echomind-enterprise` (fresh clone of `github.com/RW2523/echomind-enterprise`) |
| Branch / HEAD | `main` @ `95782e4` — "docs(qms): ISO 9001 evidence package rebuilt after approval" |
| Working tree | Clean at clone time. Untracked: `docs/echomind-law/` (this plan) and `frontend/node_modules/` (gitignored). |
| Releases | Tags `v1.4.0` (2026-09-22) and `v1.4.1` (2026-10-06). The commits after v1.4.1 are QMS/CI only. |
| Other branches | 18 remote branches, all 70–224 commits behind `main` and at most 6 ahead. All are stale; `main` is the only source of truth. |
| Size | About 48k lines of Python/TS/TSX. The largest files are `backend/app/rag/advanced.py` (3,831), `voice/app/session.py` (2,723), `backend/app/transcribe/ws.py` (1,248) and `frontend/hooks/useLiveTranscription.ts` (1,173). |
| Top-level dirs | `backend/` FastAPI app · `frontend/` React 19 SPA · `voice/` FastAPI speech-to-speech · `nemotron_asr/` shared ASR package · `livetranscript/` **standalone legacy app, not used by the stack** · `docker/` (trtllm, ollama, unused sglang shim) · `eval/` golden eval + paper experiments · `scripts/` ops · `docs/` (incl. 196-file ISO 9001 QMS) · `marketing_doc/` |

## 2. Runtime environment (this machine) — ⚠ shared with production

| Fact | Evidence | Consequence for the Law build |
|---|---|---|
| The host is an NVIDIA **DGX Spark GB10**, driver 580.159.03, 121 GB unified memory, **about 107 GB in use**. | `nvidia-smi`, `free -g` ✔ | A second full EchoMind stack (another LLM plus ASR plus SDXL) **will not fit**. |
| The **live production EchoMind stack** runs here from `/home/echomind/Documents/echomind/echomind-enterprise` (same commit `95782e4`). It uses compose project `echomind-enterprise`, fixed container names `echomind-*`, ports 3000/3443/8002/8355/11435, and a running `echomind-cloudflared` public tunnel. | `docker ps`, `docker inspect` ✔ | **This clone's folder is also named `echomind-enterprise`.** Running `docker compose up` here would reuse the same project name, containers and `echomind_data` volume, **replacing production and mixing law data into it**. The Law build must run under its own project name, ports and volume (DECISIONS D-002). |
| Production runs with `AUTH_ENABLED=1`. | `docker exec echomind-backend printenv AUTH_ENABLED` ✔ | Confirms the auth-on code path is the deployed one. |
| Shared model servers are reachable from the host: TRT-LLM `localhost:8355` (Qwen3-30B-A3B-FP4) and Ollama `localhost:11435` (`nomic-embed-text`, `bge-m3`, `qwen2.5:7b-instruct`, …). | `curl` ✔ | The Law dev backend can share inference without loading a second LLM. |
| Host Python 3.13 has faiss 1.13, rank-bm25, PyMuPDF, pypdf, python-docx 1.1.2, python-pptx, reportlab, sentence-transformers and torch (CPU). **NeMo is missing.** | import probe ✔ | Everything except live ASR can run natively for development. ASR work runs in a container built from the existing backend image. |
| LibreOffice (`soffice`) and Tesseract are installed on the host. | `which` ✔ | DOCX round-trip verification (T09/T10) and OCR spikes are possible locally. |

## 3. Architecture as built

```
Browser SPA (React 19, no router) ──nginx :3000/:3443──┬── /api/*   → backend (FastAPI :8000, GPU)
                                                       └── /voice/* → voice   (FastAPI :8000 WS, GPU)
backend ─► trtllm :8355 (Qwen3-30B-A3B NVFP4, OpenAI API) · ollama :11434 (nomic-embed-text)
        ─► in-process: FAISS(+BM25) indexes, cross-encoder reranker (CPU), Nemotron streaming ASR,
           VibeVoice ASR subprocess (Boardroom), SDXL-Turbo (Document Studio images)
voice   ─► trtllm (LLM), Nemotron + Parakeet (STT), Piper/Kokoro (TTS), backend /api/chat/ask-voice-stream (RAG)
storage ─► Docker volume echomind_data:/data — SQLite (WAL), FAISS/BM25 JSON files, uploads/, boardroom audio, models
```

| Module | Main code | What it does today |
|---|---|---|
| Knowledge Chat | `backend/app/api/routes/chat.py`, `backend/app/rag/advanced.py`, `backend/app/rag/index.py`, `frontend/components/KnowledgeChat.tsx` | Hybrid FAISS+BM25 (weighted RRF), cross-encoder rerank, relevance gate, persona prompts and NDJSON streaming. Citations give file, section, page and a snippet. |
| Ingestion | `backend/app/api/routes/docs.py:143-197`, `backend/app/rag/parse.py`, `backend/app/rag/chunking/*` | PDF (PyMuPDF/pypdf), DOCX (paragraph text only), PPTX, TXT and MD. **No OCR** (`docs.py:171-175` ✔). PII redaction is applied to every document (`rag/chunking/sanitize.py:12-19` ✔). |
| Live Transcription + Silent Assistant | `backend/app/transcribe/ws.py`, `backend/app/silent_assistant/*` | Nemotron streaming ASR over WS, sentence-level claim verification with verbatim proof and record lookups. The `LEGAL` profile exists (`silent_assistant/profiles.py:140-167` ✔). |
| Boardroom | `backend/app/boardroom/service.py`, `backend/app/api/routes/boardroom.py` | Chunked audio upload → VibeVoice diarization (Nemotron fallback) → LLM report → Silent Assistant checks → PDF/PPTX export. |
| Document Studio | `backend/app/docgen/*`, `frontend/components/DocumentStudio.tsx` | 23 built-in templates (README says 18), including `contract_review` and `legal_brief`. Plan→write→assemble pipeline. **PDF/PPTX export only, no DOCX.** |
| Voice | `voice/app/session.py`, `voice/app/server.py` | Full-duplex STT→LLM→TTS with barge-in, tool routing and backend RAG. |
| Auth / audit / export | `backend/app/core/auth.py`, `core/audit.py`, `core/export_gateway.py` | Local users, HS256 JWT cookie, roles `admin`/`user`, per-user `tenant` string. Mutations-only activity log. Advisory regex PII scanner. |
| Vertical packs | `frontend/packs.ts`, `frontend/verticals/*` | Subdomain → namespace/persona/theme. The `law` pack exists; `LawExperience.tsx` is a **marketing/demo landing page**, not a workspace. |

## 4. Route inventory and current authorization

With `AUTH_ENABLED=1`, the global middleware (`backend/app/main.py:158-168`) only requires a valid token on HTTP `/api/*`. Role checks exist **only** on `/api/auth/users*`, `/api/auth/audit` and `/api/auth/usage` (`routes/auth.py:63-109`). WebSockets are outside the middleware. FastAPI `/docs` and `/openapi.json` are public.

| Family | Endpoints | Tenant/namespace scoped? | Notes |
|---|---|---|---|
| Chat | `POST /api/chat/ask`, `/ask-stream`, `/ask-voice`, `/ask-voice-stream` | **Yes**, via `_effective_ns` (`chat.py:27-35`), but only when auth is on and the user has a tenant | The transcript time-window shortcut `_fetch_transcripts_since_hours` (`chat.py:355-377` ✔) bypasses scoping completely. |
| Chat history | `GET /api/chat/list`, `GET/DELETE /api/chat/{id}`, `POST /chat/create` | **No** | Chats have no owner column, so any user sees every chat. |
| Debug | `POST /api/chat/debug-retrieval`, `/eval-contextual-retrieval` | **No** | Searches the whole KB. |
| Documents | `GET /api/docs/list`, `/{id}/file`, `/data-preview`, `/usage`; `DELETE /api/docs/{id}`; `POST /delete-all`, `/upload-toc`, `/add-sample-transcripts` | **No** (`docs.py:133-139` ✔) | Any authenticated user can run `delete-all`. File responses send `Cache-Control: public, max-age=86400` (`docs.py:217-218` ✔). |
| Upload | `POST /api/docs/upload` | **Yes**, forced to the user's tenant (`docs.py:146-150`) | |
| Transcripts | `/api/transcribe/list`, `/transcripts/{id}` (GET/PATCH/DELETE), `/analysis`, `/assistant`, `/chunks/{id}/preview`, `/store`, `/tags`, `/refine`, `/speak`, `/scenarios` | **No** | |
| Live WS | `WS /api/transcribe/ws` | Cookie checked (`transcribe.py:129`). **The client chooses the namespace** from a hard-coded allow-list (`ws.py:707`). | No consent gate. |
| Boardroom | `/api/boardroom/sessions*` (11 routes) | **No.** The namespace is set freely via PATCH. | Report KB search ignores the namespace (`boardroom/service.py:792`). |
| Document Studio | `/api/docgen/*` (8 routes) | **No.** `kb` mode reads any `doc_ids` and `chat` mode reads any `chat_id`. | |
| Auth/admin | `/api/auth/*` | Admin role on users/audit/usage only | |
| Export | `POST /api/export/evaluate` | n/a | Advisory only; not wired into any real export. |
| Voice | `WS /voice/ws`, `POST /voice/speak`, `/voices/*` | Opt-in `VOICE_AUTH_ENABLED` checks at handshake only | **Voice→backend RAG calls send no credential** (`voice/app/session.py:1524-1530` ✔). With auth on, grounded voice answers should get 401 (traced in code; not exercised live). |

## 5. Data storage as built

- **SQLite** at `/data/echomind.sqlite` with WAL mode (`core/db.py:238-240` ✔). The schema is created by `CREATE TABLE IF NOT EXISTS` plus try/except `ALTER TABLE` on every boot. There is **no migration versioning**, no ORM and no Alembic. A Postgres migration is documented as planned but not executed (`docs/POSTGRES_MIGRATION.md`).
- **Tables:** `documents`, `chunks`, `users`, `activity_log`, `revoked_tokens`, `chats`, `messages`, `transcripts`, `book_sections`, `section_references`, `transcript_analysis`, `transcript_segments`, `assistant_entities`, `assistant_subjects`, `assistant_records`, `boardroom_sessions`, `docgen_jobs`, `docgen_templates`. Full column lists are in the backend map in §13.
- **No ownership columns.** Chats, transcripts, Boardroom sessions and docgen jobs have no `user_id`. Documents carry a namespace only inside `meta_json`.
- **Files** live under `/data`: `faiss*.index` and `*_meta.json` (the **full chunk metadata is duplicated in JSON**), `uploads/{doc_id}.{ext}` (raw originals, **unredacted, unencrypted**), `boardroom/<sid>/chunk_*` (raw audio, unhashed), `auth_secret.key`, and model caches.
- **Encryption at rest:** none in the application. No volume encryption is configured by the repo. Backups (`scripts/backup_data.sh`) are **unencrypted**.

### 5.1 Fresh-install defects (blocking for a new Law deployment) — verified by running `init_db()` on an empty directory ✔

| # | Defect | Evidence | Effect |
|---|---|---|---|
| DB-1 | `chunks` is created without `page_start`/`page_end`, yet ingestion inserts them. | `rag/index.py:538` vs `core/db.py:9,25`; the fresh-DB probe shows `chunks = [id, doc_id, chunk_index, text, source_json, contextualized_text]` | **Every document upload fails on a fresh database.** Production only works because its database predates the change. |
| DB-2 | `ALTER TABLE boardroom_sessions ADD COLUMN scenario/namespace/speaker_map_json` runs **before** `CREATE TABLE boardroom_sessions`. | `core/db.py:170` vs `:197` | On the first boot Boardroom queries fail with "no such column" until a restart. |

## 6. Namespace / isolation analysis (the spec's "filter BEFORE retrieval" test)

Isolation today is **one namespace string per chunk, checked after search**. `rag/index.py:_ns_ok` (`:36-40`) runs on dense, BM25, keyword and exact paths, and within a namespace the fetch is widened to `max(50k, 512, 0.5×N)` before filtering (`index.py:43-68`).

| Path | Filtered? | Before or after search | Verdict for Law |
|---|---|---|---|
| Document dense (FAISS) / BM25 | Yes | **After** (one global index) | ❌ The spec requires pre-retrieval partitioning. Post-filtering leaks timing (the paper's E3 timing side channel) and scores. |
| Transcript dense/BM25 | Yes | After; `search_transcript_only` does not widen the fetch (`index.py:872`) | ❌ |
| `search_document_restricted` / sparse variant | **No** (`index.py:1000-1083`) | — | ❌ Only a later safety net catches it. |
| Section / TOC / summary / glossary indexes, cross-ref graph | **No** (global) | — | ❌ Disable for Law, or partition. |
| Keyword `LIKE` fallback | Yes | After (full-table scan) | ❌ |
| Silent Assistant retrieval | Falls back to the **whole KB** when the namespace returns nothing (`silent_assistant/retrieval.py:48-54`; `advanced.py:3006-3009`) | — | ❌ Ethical-wall breach by design. |
| Transcript time-window path | **No** (`chat.py:355-377` ✔) | — | ❌ |
| Boardroom report retrieval | **No** (`boardroom/service.py:792`) | — | ❌ |
| Docgen `kb` / `chat` sources | **No** | — | ❌ |
| List / history / file endpoints | **No** | — | ❌ The existence of other matters' data is visible. |
| Voice RAG | Namespace is client-supplied; a bare `set_context` resets it to the whole KB (`useVoiceConnection.ts:744`, `session.py:1050-1137`) | — | ❌ |
| Caches | The query-embedding cache holds no content (OK). The PDF prefetch cache is in browser memory. The reranker has no cross-request cache. | — | ⚠ Re-check when partition caches are added. |

**Conclusion:** the namespace model is a *vertical label*, not a security boundary. A matter-level ethical wall needs (a) partitioned indexes, (b) a policy engine consulted by every route/WS/job, and (c) ownership plus matter columns on every artifact. → Plan slices S1.2, S1.4, S1.6, S1.7.

## 7. Security posture map

| Area | Current state | Evidence |
|---|---|---|
| Auth default | `AUTH_ENABLED=0` in code and compose defaults (production overrides it to 1) | `core/config.py:230`, `docker-compose.yml` |
| Sessions | HS256 JWT, 720-minute TTL, cookie `httponly`, `samesite=lax`, **no `Secure`**; revocation via a hash denylist | `core/auth.py:54-131`, `routes/auth.py:32` |
| MFA / SSO / lockout | None | `core/auth.py` |
| Admin bootstrap | Without `AUTH_ADMIN_PASSWORD`, a random admin password is **written to the log**. With it, the admin password is reset on every boot. | `core/auth.py:206-227` |
| Roles | `admin`, `user` only | `routes/auth.py:81` |
| CORS | `*` by default; credentials only when not `*` | `core/config.py:235`, `main.py:147-154` ✔ |
| Public ingress | `cloudflared` is profile-gated (`profiles: ["public"]`) in compose but **running on this host** | `docker-compose.yml:264-273`, `docker ps` ✔ |
| Host-published ports | trtllm `8355`, ollama `11434/11435`, voice `8002` (unauthenticated `/speak`) | `docker-compose.yml` |
| nginx | No CSP/HSTS/X-Frame-Options, no HTTP→HTTPS redirect, self-signed cert | `frontend/nginx.conf` |
| Audit | `activity_log`: mutations only; **GETs, WS and downloads are not logged**; best-effort; no tamper evidence; logins recorded as "anonymous" | `core/audit.py`, `main.py:175-191` |
| Logs with content | Voice logs transcript previews (`stt_nemotron.py:323-328`) and the listen-only buffer at WARNING (`session.py:1871-1876`). Debug taps exist: `/tmp/dump_payload` (`llm_openai_stream.py:179-185`), `LLM_LOG_PAYLOAD`, `ECHOMIND_LOG_FULL_PROMPTS`. | voice + backend |
| Eval backdoor in production code | `VOICE_ABLATION_MODE` or `/tmp/echomind_ablation_mode` can make voice **speak invented legal facts** ("liability cap is around fifty thousand dollars") | `voice/app/session.py:32-43, 227-238` |
| Outbound egress | Default is offline (`HF_HUB_OFFLINE=1`, `VOICE_OFFLINE=1`). Docgen `nim` image backend and `NVIDIA_API_KEY` are cloud egress paths when selected. | `docgen/images.py:48,391` |
| Prompt injection | Retrieved content is fenced as untrusted in prompts; E4 containment held, but users are never told | `advanced.py`, `eval/paper/results/SUMMARY.json` |
| Supply chain | `ollama/ollama:latest` and `cloudflare/cloudflared:latest` unpinned; TRT-LLM `1.2.0rc6`; Piper `piper1-gpl` (GPL); `livetranscript` pulls NeMo `@main` | compose, Dockerfiles |
| Backups | sha256-verified tarballs, 7 generations, **unencrypted**; restore does not use the consistent SQLite snapshot | `scripts/backup_data.sh`, `restore_data.sh` |

## 8. Integration inventory

| Integration | Status |
|---|---|
| Document Studio output formats | PDF (reportlab) and PPTX (python-pptx). **No DOCX output. No tracked changes, comments or redlines anywhere** (grep for `w:ins`/`w:del`/track-changes returns nothing ✔). python-docx 1.1.2 is used for *parsing* only. |
| Diarization | Boardroom: VibeVoice-ASR-HF subprocess (30-minute timeout, 60-minute batches, speaker IDs not reconciled across batches); Nemotron and spectral-clustering fallback. Live mode: manual two-role toggle only. |
| Transcript storage | Text only in live mode (no audio). The stored text is normalized lossily (punctuation repair, lowercase, suffix de-duplication, short-token drops); see backend map §7. Not deposition-grade. |
| Boardroom report | LLM reads only the first 6,000 characters of the transcript; timestamps dropped in checks (`boardroom/service.py:806, 863`). |
| DMS connector | **None.** |
| SSO / OIDC | **None.** |
| Consent | **None.** grep `consent` returns zero hits ✔. |
| Retention / legal hold | **None.** All deletes are hard deletes. |
| OCR | **None.** |

## 9. Models in use (see LICENSE_REGISTER.md)

Qwen3-30B-A3B-FP4 (TRT-LLM) · nomic-embed-text (Ollama) · cross-encoder/ms-marco-MiniLM-L-6-v2 · NVIDIA nemotron-speech-streaming-en-0.6b · Parakeet-TDT-0.6b-v2 · VibeVoice-ASR-HF · Piper voices (5) · Kokoro-82M · SDXL-Turbo. ASR is English-only.

## 10. Evaluation and tests baseline (details in TEST_RESULTS.md)

| Suite | Result at `95782e4` (2026-10-08) |
|---|---|
| Backend unit tests (`backend/tests`, 8 files) | **50 passed** |
| Voice unit tests (`voice/tests`) | **131 passed** |
| Frontend type-check (`tsc --noEmit`) | **Clean** |
| Golden eval (`eval/run_eval.py`, 52 items; law = 7) | **Not re-run in Phase 0.** The law corpus (`Master Services Agreement (DEMO).md`, `Contract Review Playbook (DEMO).md`) is **not in the repo**. The last recorded run (2026-09-22) scored **9/52** because the corpus was missing; the best ever was 50/52 (2026-07-30). Running it against production would write chats into the production DB, so it needs explicit approval. |
| Untested areas | WS protocols, Silent Assistant verifier/rules, auth middleware/RBAC/tenant scoping, audit, Boardroom diarization/report, docgen pipeline, schema migration, the 2.7k-line voice session. No frontend tests exist (no runner configured). |

## 11. Other defects and tech debt relevant to Law

| # | Issue | Evidence |
|---|---|---|
| R-1 | PII redaction rewrites **every** ingested document (emails, phones, any 10+ digit number, `AA123456` patterns), destroying e-discovery content, Bates numbers and account references in contracts. | `rag/chunking/sanitize.py:12-19` ✔, `pipeline.py:187` |
| R-2 | Page numbers for flat chunks are estimated from chunk position, not real offsets. No char/line offsets are stored. | `rag/chunking/pipeline.py:324-332` |
| R-3 | DOCX parsing drops tables, headers/footers, footnotes, comments, tracked changes and **auto-numbered clause numbers**. | `rag/parse.py:273-275` |
| R-4 | Inputs of two words or fewer (e.g. "force majeure") are classified as small talk, so retrieval is skipped. | `rag/advanced.py:189-208` |
| R-5 | "Summarize the deposition transcript" is hijacked by the time-window shortcut. | `chat.py:319-352` |
| R-6 | Citation "Document N" → filename mapping is off by one when the evidence block is prepended. Inline pinpoints such as "(Clause 4.2, p. 7)" are **stripped** from answers (backend `strip_inline_refs`, frontend `stripInlineRefs`). | `advanced.py:3394-3488`, `KnowledgeChat.tsx:133-146` |
| R-7 | Winner-take-all source selection: one answer cannot combine a transcript and a contract. | `advanced.py:1706-1722` |
| R-8 | The default persona fallback is "Financial Advisor" (DoD FMR), and insufficient-context messages mention DoD FMR. | `advanced.py:75-81, 2212-2227` |
| R-9 | Docgen writers are told to invent "plausible specifics" where the source is thin, which is unacceptable for legal drafting. | `docgen/pipeline.py:464-466` ✔ |
| R-10 | Book/TOC/cross-ref code is DoD-FMR-shaped: section-ID validator accepts only `\d{4,6}`; nothing handles "§", `12.3(b)`, Bates or "17 U.S.C. § 107". | `rag/metadata_validation.py`, `rag/cross_ref_graph.py:30-45` |
| R-11 | Uploads run synchronously (sequential per-chunk embedding, full BM25 rebuild and metadata JSON rewrite per upload). No size limit or background job. | `rag/index.py`, `rag/sparse.py` |
| R-12 | Frontend: no router (no deep links), no 401 handling, `tailwind.config.js` content globs **omit `verticals/`** (vertical pages render partly unstyled in builds), App mounts the chat hook even under a vertical page (creates stray chats), dark-only theme, persona metadata duplicated in four places, `cyan-*` hard-coded about 90 times. | frontend map |
| R-13 | Several config knobs are read via `getattr(settings, …)` but never declared, so environment overrides silently do nothing (`RAG_NS_FETCH_FRACTION`, `RAG_MAX_CITATIONS`, `TRANSCRIPT_ANALYSIS_CONCURRENCY`, …). | `core/config.py` |
| R-14 | Voice: wake word stored globally across users (`wake_word_storage.py`); the client-built system prompt is overwritten server-side; the 2.7k-line session is untested. | voice map |
| R-15 | Offline prep: `docker-compose.prepare.yml` defaults trtllm to Llama-3.1-8B (wrong model); SDXL/VibeVoice are not covered by prepare/export; `OFFLINE_DEPLOYMENT.md` is stale (Kyutai/Whisper). | infra map |
| R-16 | Stale docs: `docs/CAPABILITIES.md` (Llama, "no authentication"), `docs/reference/ENDPOINTS.md` (38 lines), the voice README. | docs |

## 12. Gap summary (detail per backlog ID in FEATURE_MATRIX.md)

| Status | Backlog IDs |
|---|---|
| **Partial — reusable foundation** | SEC-01 (local auth/JWT), SEC-04 (mutation log), SEC-06 (regex PII scan), RAG-01 (hybrid RAG + citations), LIVE-01 (ASR, two-role labels), LIVE-02 (`LEGAL` profile with 2 rules), MEET-01 (Boardroom pipeline), DOC-01 (`contract_review`/`legal_brief` templates, PDF/PPTX), OPS-01 (backup/restore scripts), OPS-02 (golden eval + CI unit tests) |
| **Missing** | SEC-02, SEC-03, SEC-05, SEC-07, SEC-08, RAG-02, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, OPT-01, OPT-02, OPT-03, OPT-04, OPT-05 |
| **Incompatible (must change, not just extend)** | Namespace-as-tenant isolation model; global PII redaction at ingest; lossy transcript normalization for legal use; docgen "plausible specifics" instruction; hard deletes everywhere; client-chosen namespaces on WS/voice; Silent Assistant whole-KB fallback |

## 13. Detailed subsystem maps

Full subsystem maps (endpoints, WS message lists, every env var, all table columns) were produced during the audit and are condensed above. These numbers matter for later slices:

- **RAG constants:** chunk 450/120 tokens, `TOP_K` 15, rerank 25→15, relevance threshold 0.45, CE gate −10.5 / −8.0, RRF k=60 with dense/sparse 0.6/0.4 (query-class blended), context budget 24k chars, ≤6 sections × 4 chunks, ≤6 citations.
- **Live WS** server messages: `loading, ready, session, warning, partial, segment, analysis_start, analysis, analysis_done, entity, subject, record, scenario_suggest, session_title, final, stored, stored_combined, refined, overloaded, error`.
- **Voice WS** server messages: `hello, context_ack, profile_update, event, cancel, partial_transcript, asr_final, assistant_text_partial, assistant_phrase, assistant_text, assistant_sources, audio_out, memory_event, memory_info, listen_buffer, error`.
- **Silent Assistant tiers:** T0 filler gate → T1 entities/records → claim gate → T2 retrieval → T3 LLM verify, with server-side proof rules (verbatim quotes only; no citation means unverified; `ASSISTANT_MIN_VERDICT_CONF` 60).
- **Docgen:** plan (9k source chars) → parallel section writers (6k source chars each) → normalize → optional images. Template registry at `docgen/templates.py:25-596`.
