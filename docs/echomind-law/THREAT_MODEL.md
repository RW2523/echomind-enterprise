# EchoMind Law — Threat Model & Security Policy (v0.1, Phase 0)

> Status: **Draft for review.** Written from the audited code at `95782e4`. Each threat lists its current exposure (with evidence), the planned mitigation, the plan slice that delivers it, and the test that proves it. "Release-blocking" means a single failure blocks any Law release (spec §10).

## 1. Assets and trust boundaries

**Assets, in order of harm if leaked:**
1. Client matter content: documents, versions, transcripts, audio, chats, AI drafts.
2. The **existence** of a walled matter or client: the name alone can breach an ethical wall.
3. Privilege and work-product labels and reviewer notes.
4. Credentials: sessions, MFA secrets, connector tokens, the master key.
5. The audit trail.
6. Model and system prompts.

**Trust boundaries:**
- Browser ↔ nginx (TLS).
- nginx ↔ backend/voice (Docker network).
- backend ↔ model servers (trtllm, ollama; shared on the dev host).
- backend ↔ storage (`/data`).
- backend ↔ DMS/IdP (optional, network-approved).
- Retrieved content ↔ LLM prompt: **untrusted data inside a trusted prompt.**
- Background job ↔ initiating user.

**Actors:**
- An authorized attorney acting outside their matter.
- A walled-off insider.
- A paralegal with a limited role.
- A malicious or compromised admin.
- A malicious document author (opposing party, data room).
- A participant who has not consented to recording.
- A compromised connector or IdP.
- A network attacker on the firm LAN.

## 2. Security principles (policy)

1. **Deny by default.** Every Law endpoint, WS and file response needs an authenticated session *and* a policy decision. The only exceptions are login, health and version.
2. **Authorize before retrieval.** The retrieval API takes an `AuthorizedScope`. Partitions outside it are never opened (D-005).
3. **Non-disclosure.** A forbidden or non-existent matter, document, chat or transcript gets the same `404 not_found` body, with a similar code path and timing.
4. **Least privilege everywhere.** Roles × matter membership × walls; the export right is separate; admins **cannot read matter content** without audited break-glass.
5. **Evidence over assertion.** The AI never presents unverifiable statements as fact. Citations resolve only after authorization, and quotes are checked verbatim.
6. **No silent egress.** No cloud inference, telemetry or update calls. Any outbound integration is a named, approved, audited exception.
7. **Logs carry metadata, not content.**
8. **Fail closed.** Unknown ACL, stale connector sync, missing consent or a failed audit write → deny.

## 3. Threats

| # | Threat | Current exposure (evidence) | Mitigation | Slice | Test |
|---|---|---|---|---|---|
| T-01 | **Cross-matter IDOR**: guessing or using IDs of another matter's docs, chats, transcripts or sessions | Lists and by-ID routes are unscoped: `docs.py:133`, `/{id}/file`, `chat.py /list,/{id}`, transcribe and Boardroom routes, docgen `doc_ids`/`chat_id` | Ownership and `matter_id` on every artifact; central `authorize(principal, action, resource)`; non-disclosing 404 | S1.2, S1.7 | T01 (release-blocking) |
| T-02 | **Retrieval leakage across matters** (dense, BM25, keyword, rerank, evidence, citations) | Global index with post-filter (`rag/index.py:36-68`); unfiltered `search_document_restricted`; global section/TOC/glossary indexes | Per-matter partitions; `AuthorizedScope` required; auxiliary indexes off in Law | S1.6 | T01, T06, canary suite (release-blocking). **S1.6 built for documents:** canary suite v1 passes for 7 people × 4 matters × 4 channel combinations, plus T02 (next query 404), T06 (retrieval abstains), a scope that cannot be forged and is re-checked, a misplaced index file never served, and Law text kept out of the legacy stores. **S1.7:** every other module is closed in the Law edition (D-025; every legacy route gives 404, live capture 4404); research and voice answer only from the matter's partition; canary suite v2 passes for every person × matter through research, streaming and the voice on-behalf-of path. |
| T-03 | **Whole-KB fallback** in Silent Assistant | `silent_assistant/retrieval.py:48-54`, `advanced.py:3006-3009` | Remove fallback in Law; an empty result means "No matching source in this matter" | S1.7 | T06 |
| T-04 | **Existence leak** via counts, autocomplete, recent widgets, usage bars, errors, timing | Unscoped lists; `/api/docs/usage`; recorded timing side channel (eval E3) | Per-scope counts only; no global usage for non-admins; constant-shape 404s; partitioned search removes the cross-matter work-size signal | S1.2, S1.3, S1.6 | T01 (timing assertion: forbidden vs. missing within noise band) |
| T-05 | **Stale ACL / revocation lag** (caches, open streams, WS sessions) | Stateless JWT valid for 12 h; no ACL epochs | Server-side sessions (D-006); per-matter `acl_epoch` checked per request, per streamed chunk and every 60 s on WS; matter-keyed client caches purged | S1.1, S1.4, S1.6 | T02 (release-blocking). **Built through S1.7:** sessions are revoked on role removal; search re-checks the live policy before and after each query; research streams re-check every 2 s and end with `scope_revoked`; T02 passes for the next query and for an open stream. The policy check is a live query, not a cached epoch. |
| T-06 | **Indirect prompt injection** in documents or transcripts (hidden text, white-on-white, metadata) instructing the model to reveal other matters, call tools or exfiltrate | Fencing exists; containment measured (E4) but never shown to users | Keep fencing; retrieval is pre-authorized, so other matters cannot be reached even if the model obeys; the LLM gets no tools that cross scope; strip hidden text at ingest and flag it; detected injection attempts shown as a document warning | S1.6, S2.1, S2.2 | T08 (release-blocking). **S1.7 built:** sources are escaped and fenced (a document cannot close its fence, as tested); the system prompt says quoted text is never instructions; the model has no tools and sees only the matter's passages. Hidden-text stripping and warnings: S2.1–S2.2 |
| T-07 | **Hallucinated or fabricated citations and quotes** | Retrieval-level citations; no claim→source check; docgen told to invent "plausible specifics" (`docgen/pipeline.py:464-466`) | Evidence contract with verbatim quote verification; abstention; legal docgen forbids invention; citation checker marks external authority UNVERIFIED | S2.2, S3.1, S3.4 | T05, T06, T07 |
| T-08 | **Recording without consent** / consent race (start before consent lands; consent withdrawn mid-stream) | No consent concept; capture starts on WS `start` | Server-side consent gate inside the WS `start` handler, Boardroom chunk upload and voice handshake, all evaluated against the consent ledger in one transaction; withdrawal pushes a stop and rejects further frames | S1.8 | T03 (release-blocking). **S1.8 built:** the capture socket opens only after fresh, explicit consent from every current participant (4412 before speech recognition starts). One lock per meeting orders a withdrawal against frame acceptance and transcript storage, so nothing is captured after the withdrawal time (race test). Withdrawal is open to any matter member. Consent and capture timelines are append-only. Pauses are recorded. |
| T-09 | **Export bypass**: direct file URLs, Boardroom/docgen exports, browser cache | `/api/docs/{id}/file` with `Cache-Control: public`; Boardroom/docgen exports do not pass through the gateway | All downloads go through `/api/legal/exports/*` or an authorized file route with `Cache-Control: no-store`; watermarking; audit | S1.10 | T09 (release-blocking for privileged docs) |
| T-10 | **Redaction failure**: text under black boxes, metadata, comments, hidden runs, tracked changes, embedded objects | Advisory regex only; nothing applied to files | True content removal (PDF: rasterize or remove glyph runs; DOCX: rebuild without removed runs, strip `docProps`, comments, revisions, custom XML); post-render verification by text extraction | S1.10 | T09 |
| T-11 | **Log leakage** of transcripts, prompts, document text | Voice INFO/WARNING transcript logs; dump taps; full-prompt env flags | Law edition: content-free log formatter, startup refusal if taps are set (D-003/D-016), log scrubbing test | S1.0, S1.5 | Log-scan test in CI (greps for planted canary strings) |
| T-12 | **Admin abuse**: an admin reads matters or alters the audit | Admins have no content separation; the audit log is mutable and best-effort | Firm admin ≠ matter access; break-glass with reason, second approver, time limit and prominent audit; hash-chained append-only audit with triggers and daily anchors | S1.4, S1.5 | Audit tamper test; break-glass test. **S1.5 built:** triggers refuse edits; the verifier catches an edited event, a removed event, a rewritten chain (anchor mismatch) and a removed tail; firm admins see matter references, not numbers. Residual risk: anchors sit on the same disk until the off-host forwarder is approved. |
| T-13 | **Namespace collision / client-chosen scope** | Live WS namespace chosen by the client (`ws.py:707`); voice `set_context` resets to the whole KB | Scope derived server-side from `matter_id` plus authorization; client hints ignored; Law voice requires a matter | S1.7 | T01 (WS cases) |
| T-14 | **OCR hallucination** (misread numbers or dates presented as facts) | No OCR today | OCR provenance and confidence on chunks; low-confidence spans flagged in citations; "verify against image" UI | S2.1 | OCR fixture test |
| T-15 | **Compromised or misconfigured DMS connector** (over-broad ACLs, stale sync, injected docs) | No connector | Source ACL ∩ matter ACL; stale or unknown ACL denies; signed sync runs; connector writes go only to its matter partition; credentials encrypted | S1.11 | T15 |
| T-16 | **Session theft / CSRF** | Cookie lacks `Secure`; `SameSite=Lax`; token in response body | `__Host-` Secure Strict cookie; CSRF double-submit for state-changing requests; no token in the body for the Law edition | S1.1 | Auth contract tests |
| T-17 | **Unauthenticated LAN services** | trtllm `:8355`, ollama `:11434/11435`, voice `:8002/speak` published on the host; FastAPI `/docs` public | Law compose publishes only nginx; model servers internal-only (firm deployment); `/docs` disabled in Law | S1.0 | Port-scan check in the deployment test |
| T-18 | **Malicious uploads** (zip bombs, polyglots, path traversal, oversized files) | Magic-byte check exists; no size limit; synchronous processing | Size and page limits, MIME + magic + extension agreement, ClamAV optional (offline signatures), background ingestion with resource limits, safe filenames | S1.6, S2.1 | Fuzz and limit tests. **S1.6 built:** size limit (413); page limit; name, declared type and content must agree; a .docx must be a real Word package and stay under its unpacked limit (zip bomb refused); text must be UTF-8 without NULs; password-protected PDFs refused; safe filenames; background indexing. ClamAV and resource-limited parsing: S2.1 |
| T-19 | **Data egress via features** (image NIM backend, HF downloads, update checks) | `nim` backend possible; `HF_HUB_OFFLINE=1` already | Law edition refuses `nim`; a network-denied E2E run checks for zero unexpected outbound connections | S1.0, S1.12 | T11 (release-blocking) |
| T-20 | **Deletion of held data** (user, timer, system cleanup, `delete-all`) | Hard deletes; `delete-all` open to any user | Single deletion service that checks holds; `delete-all` disabled in Law; FK-like reconciliation of derived artifacts (chunks, indexes, analyses, exports, backups list) | S1.9 | T04 (release-blocking). **S1.9 built:** one deletion service; a held matter refuses deletion requests, an approval racing a new hold (re-checked inside the purge transaction; the request becomes `blocked`), a direct service call and the retention runner (which never purges; it only creates requests); legacy delete routes are closed (D-025); purge verification covers the database, the matter index, stored files and research quotes (T04). Backups still hold purged data until they rotate (documented). |
| T-21 | **Backup exposure** | Unencrypted tarballs, optional off-host copy | Encrypted backups (D-014); restore test | OPS-01 | T13 |
| T-22 | **Model / system prompt override from the client** | Voice system prompt built on the client (overwritten server-side today); persona strings client-chosen | Law: persona and prompts server-side only; the client sends no prompt text except the question | S1.7, S2.2 | Contract test |
| T-23 | **Evaluation backdoor** making voice assert invented legal facts | `VOICE_ABLATION_MODE` and `/tmp` file switch (`voice/app/session.py:227-238`) | Hard-disabled in Law (D-016) | S1.0 | Unit test asserting unreachable |

## 4. Release gate (security)

A Law release is blocked by **any** failure in:
- T01, T02 (ACL and revocation)
- T03 (consent)
- T04 (holds)
- T08 (injection containment)
- T09 (redaction residue)
- T11 (offline egress)
- the audit-chain verification

Results are recorded in TEST_RESULTS.md with the commit hash. Failures are never suppressed or skipped to reach green.

## 5. Open items requiring counsel or owner input

P-001 (consent policy), P-005 (prompt logging), P-006 (export policy), P-011 (IdP), D-014 (LUKS on target host).
