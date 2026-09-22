# SOP-11 — Customer Property and Data Handling

| Field | Value |
|---|---|
| Document ID | SOP-11 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.5.3, 8.5.4, 7.5.3.2 |

---

## 1. Purpose

To define how Ajace AI identifies, verifies, protects and safeguards property belonging to
customers and to external providers while it is under Ajace AI's control or in use by the EchoMind
Enterprise platform, and how loss, damage or unsuitability of that property is reported to the
customer.

This is the procedure that matters most to the organisation's customers. EchoMind Enterprise exists
because defence, government, legal, finance and healthcare organisations cannot send their material
to a third-party AI service. Everything below should be read against that promise.

## 2. Scope

All customer property handled by the platform in a deployment, and all customer material handled by
Ajace AI during pre-sales, customisation and support.

**The controlling fact of this procedure is that the platform runs on hardware the customer owns
and controls.** In the standard deployment Ajace AI holds no customer data at all: the data is on
the customer's machine, in Docker volumes the customer administers. Ajace AI's obligation under
8.5.3 is therefore primarily one of *design* — the platform must protect the property placed into
it — and secondarily one of *conduct* during any support or customisation engagement in which
Ajace AI is given access to real customer material.

This procedure does not cover the customer's own obligations to their data subjects or regulators.
It states what the platform does, so that the customer can discharge those obligations.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Managing Director | Owns this procedure. Agrees with each customer, before any engagement, whether Ajace AI will be given access to real customer material and on what terms. Makes and records every report to a customer of loss, damage or unsuitability of their property. |
| Lead Engineer | Ensures that the data-handling behaviour described in §5 to §11 remains true of the code, and raises a nonconformity under `SOP-10` when a change makes any statement in this procedure false. |
| Customer's system administrator (customer role, stated here for the boundary) | Administers the host, the Docker volumes, host-level access control, backups, and physical security of the machine. Ajace AI does not perform these in a customer deployment. |

## 4. Identification of customer property (8.5.3)

Customer property in this platform is not only the files a user uploads. Four categories are in
scope, and the third and fourth are the ones most often missed.

| Category | What it is |
|---|---|
| **A — Material supplied by the customer** | Documents uploaded to the knowledge base; audio streamed for transcription or recorded in a boardroom session; text typed into chat, into Document Studio prompts, or into Silent Assistant |
| **B — Material produced by the platform from that material** | Transcripts (raw and polished), transcript analyses and segments, extracted entities, subjects and records, meeting reports, generated documents, chat history |
| **C — Derived representations** | Text chunks, vector embeddings and the FAISS indexes, the BM25 sparse index and its metadata, the glossary and section indexes, the cross-reference graph |
| **D — Personal data of the customer's own people** | Usernames and password hashes in `users`; and in `activity_log`, the username, role, HTTP method, path, status and **IP address** of every API mutation |

**Category C is customer property.** An embedding or an index entry is derived from the source text
and can reveal it: the `chunks` table holds the full extracted text of every uploaded document
(`backend/app/core/db.py:9`), and the FAISS metadata files hold the text and provenance alongside
the vectors. Treating indexes as "just derived artefacts" — for example, excluding them from a
deletion request or copying them off the host for debugging — would be a disclosure of the source
material. They are handled exactly as category A.

**Category D is customer property too**, and it is personal data. It is created by the platform
rather than supplied by the customer, which is precisely why it is easy to overlook when scoping a
deletion or a retention period.

## 5. Inventory: what is held, and where

Everything below lives inside one Docker volume, `echomind_data`, mounted at `/data`
(`docker-compose.yml:101-102`; base path `backend/app/core/config.py:5,10`). The single-volume
design is deliberate and is stated to operators at `docs/USER_MANUAL.md:1150-1158`.

**SQLite database — `/data/echomind.sqlite`** (`backend/app/core/config.py:11`). Seventeen tables
are created at startup in `backend/app/core/db.py`. Those holding customer property:

| Table | Contents | Category | Path |
|---|---|---|---|
| `documents` | Uploaded document metadata: filename, type, timestamp, namespace | A | `backend/app/core/db.py:8` |
| `chunks` | **The full extracted text of every uploaded document**, split into chunks, with source metadata | A | `:9` |
| `users` | Username, password hash, role | D | `:11` |
| `activity_log` | Timestamp, username, role, method, path, status and **IP address** per API mutation | D | `:17` |
| `chats` | Chat titles and conversation summaries | B | `:26` |
| `messages` | **Every chat message**, user and assistant | A, B | `:31` |
| `transcripts` | `raw_text` and `polished_text` of every transcript | B | `:33` |
| `transcript_segments` | Per-segment transcript text | B | `:131` |
| `transcript_analysis` | Derived analysis of transcripts | B | `:91` |
| `assistant_entities`, `assistant_subjects`, `assistant_records` | Entities, subjects and records extracted by Silent Assistant | B | `:137`, `:143`, `:149` |
| `boardroom_sessions` | Meeting session state and reports | B | `:195` |
| `docgen_jobs`, `docgen_templates` | Document Studio jobs, prompts and templates | A, B | `:208`, `:223` |
| `book_sections`, `section_references` | Section index and cross-references over customer material | C | `:61`, `:76` |

**Files on disk in the same volume:**

| Path | Contents | Category | Evidence |
|---|---|---|---|
| `/data/uploads/` | The original uploaded source files, retained so the front end can render them | A | `backend/app/api/routes/docs.py:24,30-32` |
| `/data/boardroom/{session_id}/` | Raw meeting audio chunks, up to `BOARDROOM_MAX_CHUNKS` 4000 × `BOARDROOM_MAX_CHUNK_BYTES` 25 MB — roughly 5.5 hours per session | A | `backend/app/boardroom/service.py:37,79`; `backend/app/core/config.py:146,148` |
| `/data/faiss.index`, `faiss_transcript.index`, `faiss_section.index`, `faiss_glossary.index` and the matching `*_meta.json` and `sparse_*_meta.json` | Embedded customer text and its provenance | C | `backend/app/core/config.py:12-18,198-202` |
| `/data/docgen_images/`, generated documents | Output produced from customer prompts and material | B | `backend/app/docgen/images.py:46` |
| `/data/auth_secret.key` | The JWT signing secret, auto-generated and persisted | — (not customer property, but a secret protecting it) | `backend/app/core/auth.py:58` |
| `/data/hf_cache/`, `/data/docgen_models/` | Model weights. **Not** customer property | — | `backend/app/boardroom/service.py:41`; `docker-compose.yml:96` |

**Live voice audio** is streamed over a WebSocket and processed in memory. The persisted artefacts
are the transcripts and, for boardroom sessions, the audio chunk files above.

## 6. Isolation between customers and between knowledge bases

Isolation in EchoMind Enterprise is **by knowledge-base namespace, enforced at retrieval time** —
not by separate databases, separate volumes or separate processes. An auditor should understand the
mechanism exactly, because its limits matter more than its presence.

| Element | Where | What it does |
|---|---|---|
| `_active_namespace` context variable | `backend/app/rag/index.py:29`, setter at `:32-33` | Holds the namespace for the current request; async-safe |
| `_ns_ok` predicate | `backend/app/rag/index.py:37-40` | `((src or {}).get("namespace") or "default") == ns` — applied inside every index search path |
| Per-request namespace binding | `backend/app/api/routes/chat.py:382, 422, 498, 560` | `set_active_namespace(_effective_ns(request, inp.namespace))` |
| **The actual security boundary** | `backend/app/api/routes/chat.py:25-33` (`_effective_ns`) | When authentication is on and the user is bound to a tenant, the user's tenant namespace is **forced**, so a tenant user cannot read another tenant's knowledge base by sending a different namespace |
| Ingest-side tagging | `backend/app/api/routes/docs.py:144-181`, tenant forcing at `:145-150` | An uploaded document is tagged with the uploader's tenant namespace |
| Candidate-pool widening | `backend/app/rag/index.py:44-57` | Searches rank globally and then filter; a fixed pool starved small namespaces as the corpus grew, so the pool scales with index size |

**Stated plainly: the tenant boundary is only enforced when `AUTH_ENABLED` is true.** With
authentication off — which is the default (`backend/app/core/config.py:230`;
`docker-compose.yml:79`) — `_effective_ns` returns whatever namespace the client sent, unverified.
In that configuration the namespace is a *partition*, not a *security boundary*: it keeps knowledge
bases separate for users who ask honestly, and stops nothing else.

`frontend/packs.ts:1-9` maps a subdomain to a namespace, persona and theme. This is
presentation-layer selection and **is not a security control**; it is client-side and can be
overridden by a query parameter, as the comment in that file states.

The isolation mechanism has failed once in the recorded history, and the failure was found by
measurement rather than by report — commit `4e27109`, worked through in `SOP-10` §10 Example A, and
a related keyword-grep namespace leak in `ebd232f`. Post-fix verification across all five retrieval
paths recorded 0 out-of-namespace hits in 359 checks
(`eval/paper/results/SUMMARY.json`, experiment E3b).

## 7. What must never leave the customer's perimeter, and how that is enforced

Nothing in categories A to D may leave the customer's hardware. This is not a policy statement
layered over a connected product; it is how the runtime is built.

| Control | Evidence |
|---|---|
| All inference is local — chat, embeddings, speech-to-text and text-to-speech all run in containers on the customer's host | `docker-compose.yml` service definitions for `trtllm`, `ollama`, `backend` and `voice`; `README.md:130` |
| No telemetry, analytics or usage reporting of any kind | `README.md:130`; the only usage record is the local `activity_log` table |
| Open-weight models served locally, not called as a hosted API | `docker-compose.yml:73-76` (`nomic-embed-text` via Ollama), `:5` and `:140` (Qwen3-30B-A3B-FP4 via TensorRT-LLM at `http://trtllm:8355`) |
| Offline model resolution — no runtime downloads | `docker-compose.yml:84-85` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; `:170-171` disables the runtime Piper voice download |
| Front-end crash reports go to the local backend, not to a third party | `frontend/components/ErrorBoundary.tsx:36` posts to `/api/client-error`, handled at `backend/app/main.py:198-206` |
| The one outbound path in the product is disabled | Document Studio image generation supports a `nim` backend that calls an NVIDIA-hosted endpoint (`backend/app/docgen/images.py:14-15`). It is **not enabled**: the shipped configuration is `DOCGEN_IMAGE_BACKEND=diffusers` (`docker-compose.yml:94`), on-device SDXL-Turbo. A `comfyui` backend also exists and points at a local server by default |

**Consequence for change control.** Setting `DOCGEN_IMAGE_BACKEND=nim` would send a document prompt
derived from customer material to a third-party endpoint. Any change that enables it, or that adds
a new outbound call, is a change to the core value proposition and to this procedure, and must be
raised under `SOP-02` §9 and approved by the Managing Director before it is made.

`cloudflared` (`docker-compose.yml:218-228`) is the one component that intentionally reaches
outward, and it runs only under the `public` compose profile. It is not part of an on-premises
customer deployment.

## 8. Protection of customer property (7.5.3.2, 8.5.4)

Honest separation of what is in place from what is not.

**In place:**

| Control | Evidence |
|---|---|
| The offline, on-premises posture — the principal protective control, and the reason the other gaps are survivable | §7 above |
| Password storage: PBKDF2-HMAC-SHA256, 200,000 iterations, per-user salt, constant-time verification | `backend/app/core/auth.py:24,30-31,37-40` |
| Session tokens: HS256 JWT with a secret auto-generated and persisted outside the source tree | `backend/app/core/auth.py:45,58`; lifetime `AUTH_TOKEN_TTL_MIN` 720 minutes (`backend/app/core/config.py:232`) |
| Namespace isolation at retrieval, when auth is on | §6 |
| Retrieved content fenced as untrusted data, so a document cannot instruct the model | `backend/tests/test_prompt_guards.py` |
| Duplicate-upload guard, preventing the same customer bytes being indexed twice | `backend/app/api/routes/docs.py:152-166` |
| Boardroom chunk directory path containment against traversal | `backend/app/boardroom/service.py:238-239` |
| Customer data is never committed to version control | `SOP-01` §9 |
| Activity log records who did what, from which IP, for every API mutation | `backend/app/main.py:154-163`; `backend/app/core/audit.py:13-23` |

**Not in place:**

| Absent control | Current state | Consequence |
|---|---|---|
| **Encryption at rest** | None anywhere. The SQLite database is plain, `/data/uploads` is plain, the FAISS indexes and their metadata are plain. No encryption mechanism is configured for `echomind_data` in `docker-compose.yml` | Anyone with host, volume or disk access reads all customer content directly. The customer's own full-disk encryption is the only mitigation, and it is outside Ajace AI's control |
| Authentication on by default | `AUTH_ENABLED` defaults to `0` (`backend/app/core/config.py:230`; `docker-compose.yml:79`) | With the default configuration there is no user identity, so §6's tenant boundary does not apply and `activity_log` records `"anonymous"` (`backend/app/core/audit.py:18`) |
| Authentication over WebSockets | The enforcement middleware at `backend/app/main.py:138-147` is HTTP-only; the comment at `:135` states "WebSocket endpoints are not gated here yet". The voice service has its own gate, `VOICE_AUTH_ENABLED`, also defaulting to `0` (`voice/app/auth_check.py:14`; `docker-compose.yml:145`) | The live-transcription WebSocket is not covered by the HTTP auth guard even when auth is enabled |
| Restrictive CORS | `CORS_ALLOW_ORIGINS` defaults to `*` (`backend/app/core/config.py:235`) | Any origin may call the API from a browser |
| Rate limiting | `RATE_LIMIT_PER_MIN` defaults to `0`, i.e. off (`backend/app/core/config.py:236`) | No protection against bulk extraction or resource exhaustion |
| Activity-log integrity | The middleware is best-effort inside a bare `except Exception: pass` (`backend/app/main.py:161-162`), and the insert itself swallows failures at debug level (`backend/app/core/audit.py:1,21-22`) — deliberately, so logging "never breaks a request" | The audit trail is not guaranteed complete. This is a defensible engineering trade-off and an undefensible audit record; it must be stated to any customer who relies on the log |

For a public deployment, Cloudflare Access is the compensating control and is described as
**mandatory**, not optional, at `docs/PUBLIC_DEPLOYMENT.md` §"Step 4"; `docker-compose.yml:221`
carries the same warning inline.

> **Note on a stale comment.** `docker-compose.yml:221` and `docs/PUBLIC_DEPLOYMENT.md:5` state
> that "EchoMind has no built-in auth". That is out of date and contradicts `README.md:133`, which
> states that authentication is opt-in and that local accounts and JWT ship in the box. The README
> is accurate. The stale comments are a class N6 nonconformity under `SOP-10` §4.

## 9. Deletion of customer property on request

Deletion today is **per-object and operator-driven**: an authorised user deletes a named object
through the API or the user interface. There is no bulk erasure, no data-subject-request workflow
and no scheduled deletion.

| Endpoint | Deletes | Evidence |
|---|---|---|
| `DELETE /api/docs/{doc_id}` | The document, its chunks and embeddings from the index, and the stored source file (best-effort; a failure to remove the file is logged at WARNING) | `backend/app/api/routes/docs.py:220-235` |
| `DELETE /api/transcribe/transcripts/{id}` | The transcript row **and** its knowledge-base documents, chunks and embeddings. Matches both identifier forms, because auto-stored transcripts are indexed under a different filename — "else embeddings are orphaned (audit H1)" | `backend/app/api/routes/transcribe.py:213-246` |
| `DELETE /api/boardroom/sessions/{id}` | The session row and the whole `/data/boardroom/{id}/` audio chunk directory | `backend/app/api/routes/boardroom.py:73-79`; `backend/app/boardroom/service.py:163-176` |
| `DELETE /api/chat/{chat_id}` | A chat and its messages | `backend/app/api/routes/chat.py:156` |
| `DELETE /api/docgen/jobs/{job_id}` | A Document Studio job | `backend/app/api/routes/docgen.py:209` |
| `DELETE /api/auth/users/{username}` | A user account | `backend/app/api/routes/auth.py:87` |

**What a deletion request must cover.** Because of §4 category C, deleting a document is not
complete unless the chunks and the index entries go with it. The transcript endpoint is the model
here: it deliberately matches both identifier forms specifically so that embeddings are not left
orphaned. Any new deletion path must meet the same standard, and this is a design-review item under
`SOP-05`.

**Not covered by any endpoint:** the `activity_log` table, which holds usernames and IP addresses
(category D) and has no deletion or pruning path at all.

## 10. Retention

**Stated plainly: no retention policy exists.** There is no time-based deletion of transcripts,
documents, messages, boardroom audio or activity-log entries; no scheduled cleanup task; no
retention statement in any shipped document; and no data-subject-request or erasure process.
Deletion is per-object and manual (§9). The only time-to-live anywhere in the system is the
authentication token lifetime, `AUTH_TOKEN_TTL_MIN` 720 minutes
(`backend/app/core/config.py:232`).

This is aggravated by unbounded accumulation. `AUTO_STORE_INTERVAL_SEC` defaults to 60
(`backend/app/core/config.py:95`), so a live transcript is auto-stored into the knowledge base every
60 seconds while recording. `eval/paper/results/SUMMARY.json` (experiment E2) records that at one
point only 421 of 17,514 chunks were content — approximately 97.6% of the index was
auto-accumulated transcript. That is a retention problem before it is a retrieval problem: material
the customer never chose to keep is retained indefinitely and indexed.

A retention policy, when written, must specify at minimum:

1. a retention period for each object class in §5 — uploaded documents, transcripts, boardroom
   audio, chat messages, generated documents, activity-log entries — expressed as a period, not as
   "until deleted";
2. whether the period is configurable per deployment, and where;
3. the mechanism that enforces it, and what it does on failure;
4. that expiry deletes the derived representations in category C along with the source;
5. a shorter, separately stated period for `activity_log`, because it is personal data that no user
   asked to create;
6. what happens at end of contract or decommissioning — how the customer confirms erasure;
7. how a customer exercises an individual erasure request through the operator.

Until such a policy exists and is implemented, Ajace AI must state to every customer, in the
requirements record under `SOP-04`, that the platform retains everything indefinitely and that
retention is the customer's responsibility to enforce operationally.

## 11. Preservation, backup and recovery (8.5.4)

**Stated plainly: the only volume containing customer data has no backup procedure. This is the
highest-priority gap in this procedure.**

`echomind_data` holds everything listed in §5. `scripts/export_offline_bundle.sh` exports the two
*model* volumes — `ollama_data` (`:17,50`) and `trtllm_hf_cache` (`:26,58`) — plus the container
images (`:37`). It does not reference `echomind_data`, and no other script in `scripts/` backs it
up. The volumes it does export are the reproducible ones:
`docs/USER_MANUAL.md:1158` states the operator's real backup priority is `echomind_data` precisely
because the model volumes can be rebuilt from the prepare step — but the manual gives no procedure
for doing it.

Consequently there is:

- no backup procedure for customer data;
- no restore procedure;
- no verification that a backup is restorable;
- no stated recovery point objective or recovery time objective.

The preservation controls that *do* exist are the ones internal to the running system: the
duplicate-upload guard (`backend/app/api/routes/docs.py:152-166`) prevents corruption of the index
by double ingestion; the transcript delete path prevents orphaned embeddings
(`backend/app/api/routes/transcribe.py:225-227`); and the volume is preserved across rebuilds, which
was explicitly verified during the clean rebuild recorded in commit `724fb98`. None of these
survive loss of the volume.

Because the volume is on customer hardware, backup is operationally the customer's task. That does
not discharge Ajace AI's obligation: the product must ship a documented, tested backup and restore
procedure for the operator to follow, and the absence of one is Ajace AI's nonconformity, not the
customer's.

## 12. Loss, damage or unsuitability of customer property (8.5.3)

ISO 9001:2015 8.5.3 requires that when customer property is lost, damaged or otherwise found to be
unsuitable for use, Ajace AI reports this to the customer and retains documented information on
what occurred.

| Event | Report to the customer | Record |
|---|---|---|
| Customer data lost, corrupted or irrecoverable | Immediately, by the Managing Director, stating what was affected, over what period, and what can be recovered | `FRM-05` at severity S1 under `SOP-10` §5 |
| Content crossing a namespace or tenant boundary | Immediately — the customer cannot detect this themselves | `FRM-05` at severity S1; see `SOP-10` §10 Example A |
| Customer material reaching any destination outside their perimeter | Immediately, with the destination, the volume and the period named | `FRM-05` at severity S1 |
| Uploaded material found unsuitable — for example a scanned PDF from which no text can be extracted | To the user at the point of upload; a 422 with an explanatory message is returned today (`backend/app/api/routes/docs.py:168-172`) | No further record required |
| Customer material handled by Ajace AI during support or customisation, and then lost or disclosed | Immediately, by the Managing Director | `FRM-05` at severity S1 |

**There is no customer-notification procedure today** — no defined recipient, timescale, content or
channel. §12 states the obligation; `SOP-10` §12 records the absence of the mechanism as the
highest-priority gap in that procedure. The two entries describe the same missing control and must
be closed together.

## 13. Records

| Record | Location | Retention |
|---|---|---|
| Customer property inventory | §5 of this document; revision history in Git | Life of the QMS + 3 years (`SOP-01` §8) |
| Agreement on whether Ajace AI may access real customer material | Customer requirements record under `SOP-04` | Life of the contract + 3 years |
| Report of loss, damage or unsuitability of customer property | `forms/FRM-05_Nonconformity_and_CAPA_Record.md`, logged in `registers/REG-04_Nonconformity_and_CAPA_Log.md` | 3 years after closure |
| Deletion performed on customer request | `activity_log` table in the deployment (method, path, user, timestamp, IP) — in the customer's system, **not** in Git | Per the customer contract |
| Customer data held in a deployment | The `echomind_data` volume on customer hardware | Per the customer contract; see §10 |
| Retention and backup requirements agreed with a customer | Customer requirements record under `SOP-04` | Life of the contract + 3 years |
| Review of this procedure against the code | Management review minutes under `SOP-15` | 3 years |

## 14. Current state and gaps

**Stated plainly: the platform's protection of customer property rests almost entirely on one
control — it never leaves the customer's machine. That control is real, verifiable and strong. Very
little else is in place, and the organisation should not claim otherwise.**

| Gap | Current state | Consequence |
|---|---|---|
| No encryption at rest | SQLite, uploads and FAISS indexes are all plain on disk in `echomind_data`; nothing in `docker-compose.yml` configures encryption | Host, volume or disk access exposes all customer content. The single most significant technical gap in this procedure |
| No backup of the only volume holding customer data | `scripts/export_offline_bundle.sh` covers `ollama_data` and `trtllm_hf_cache` only; `docs/USER_MANUAL.md:1158` names `echomind_data` as the real priority but gives no procedure | Total, unrecoverable loss of customer data on volume loss. Highest-priority gap |
| No restore procedure and no backup verification | None exists in `scripts/` or `docs/` | A backup that has never been restored is not a backup |
| No retention policy | §10 | Everything is kept for ever, including material auto-accumulated without a user decision |
| No data-subject-request or bulk-erasure process | Deletion is per-object and manual (§9) | A customer cannot demonstrate erasure to their own regulator using the platform alone |
| `activity_log` has no deletion or pruning path | No endpoint and no scheduled task touches it (`backend/app/core/db.py:17`; `backend/app/core/audit.py`) | Usernames and IP addresses accumulate indefinitely with no way to remove them |
| Tenant boundary conditional on an off-by-default setting | `_effective_ns` applies only when `AUTH_ENABLED` is true (`backend/app/api/routes/chat.py:25-33`); default is `0` | In the default configuration, namespace is a partition and not a security boundary. Must be stated in writing to every customer |
| WebSocket endpoints not covered by the HTTP auth guard | `backend/app/main.py:135,138-147` | Live transcription is reachable without the HTTP session check even when auth is on |
| Wildcard CORS and no rate limiting by default | `backend/app/core/config.py:235,236` | Bulk extraction is unimpeded on a network-reachable deployment |
| Activity log is best-effort | `backend/app/main.py:161-162`; `backend/app/core/audit.py:21-22` | The audit trail cannot be asserted to be complete |
| No customer-notification procedure | §12 | The 8.5.3 reporting obligation has no mechanism behind it |
| Unbounded knowledge-base growth from auto-stored transcripts | `AUTO_STORE_INTERVAL_SEC` 60 (`backend/app/core/config.py:95`); `eval/paper/results/SUMMARY.json` E2 records 421 content chunks of 17,514 | Retention, storage and retrieval quality all degrade together |
| Stale security statements in shipped files | `docker-compose.yml:221` and `docs/PUBLIC_DEPLOYMENT.md:5` contradict `README.md:133` | An operator may under- or over-estimate the protection in place |

Each gap above is carried into `registers/REG-03_Risk_Register.md` with a score, and into
`ISO9001_Gap_Analysis.md`.
