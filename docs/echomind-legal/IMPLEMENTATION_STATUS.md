# EchoMind Legal — implementation status

Statuses: **Not started**, **In progress**, **Tested** (executed tests pass, evidence given), **Blocked** (needs a person,
root or a decision). "Tested (Law)" = delivered earlier on `feature/echomind-law` with its own evidence
(docs/echomind-law/TEST_RESULTS.md) and re-run on this branch. Branch: `feature/echomind-legal-chat`.

## What a person can do now

Sign in → **Chat** opens: "How can I help?", the matter in the header, four suggestions. In one conversation a lawyer can
ask what matters today (a briefing from records, at most three items), ask questions answered only from the matter's
documents with verified, clickable citations, find potential inconsistencies between documents and resolve them with a
reason, open any cited passage beside the conversation (re-verified against the stored text, highlighted), and reopen
past conversations from the rail. Matters, Drafts (what waits for review, your own drafts) and Activity (your own
actions, documents being read) are the other destinations; every existing Law page still works inside the shell, and
the firm's pages are in the account menu. Screens: [`screens/`](screens/).

## Baseline on this branch before any change (2026-10-26)

| Suite | Command | Result |
|---|---|---|
| Backend, all | pinned image `echomind-law-backend:latest`, `--network none`, `python -m pytest tests -q` | **1402 passed, 3 skipped, 1 failed** (12 min 17 s). The failure, `test_t09_…reads_back_in_libreoffice…`, is `FileNotFoundError: 'soffice'`: LibreOffice is not in the pinned image (CI installs it). |
| Frontend type-check | `npx tsc --noEmit -p .` | clean |
| Frontend unit | `npx vitest run` | **184 passed** (37 files) |

## Blueprint MVP (docs/echomind-legal/11) against this branch

| Ticket | Item | Status | Evidence |
|---|---|---|---|
| LEGAL-001 | Repo audit, current architecture, gap matrix | **Tested** | CURRENT_ARCHITECTURE.md, GAP_ANALYSIS.md (commands and outputs quoted) |
| LEGAL-002 | Firm/matter/membership schema | **Tested (Law)** | migrations 0002–0006; `test_law_matters_api.py` |
| LEGAL-003 | RBAC, ethical walls, deny by default, 404 for hidden | **Tested (Law)** | `test_law_policy.py` (incl. new `findings.review`), `test_law_walls.py`, `test_law_scoping.py` |
| LEGAL-004 | Source ingestion: hash, versions, OCR, anchors | **Tested (Law)** | `test_law_documents.py`, `test_law_extract.py`, `test_law_structure.py`; live: 9 demo documents imported and indexed |
| LEGAL-005 | Citation resolver: opaque ids, re-verified spans, tamper refusal | **Tested** | `test_law_counsel.py`: exact characters returned; foreign/forged ids 404; changed stored text 409; revoked member 404; UI: panel highlights the span, a 409 is shown as such |
| LEGAL-006 | Search filtered by authorization before scoring | **Tested (Law)** | `AuthorizedScope` + per-matter partitions (`test_law_retrieval.py`); chat: a document of another matter named by id is not used |
| LEGAL-007 | Answer verifier + abstention | **Tested (Law)**, reused unchanged | research `_answer`; chat tests for answered/partial/abstained; live: the Rivera question abstained |
| LEGAL-008 | Chat shell (Quiet Counsel) | **Tested** | `frontend/law/counsel/CounselShell.tsx`; `counsel.test.tsx` (four destinations, four suggestions, private, real capture state); live screenshots desktop, 390 px, dark |
| LEGAL-009 | Trusted UI component contract and registry | **Tested** | server `counsel/schemas.py` (unknown types, extra fields, >3 components, >3 items, foreign paths, out-of-range highlights rejected); client `validate.ts` (unknown → inert, hostile paths not offered, markup shown as text) |
| LEGAL-010 | Cards, source panel, actions with CAS + idempotency | **Tested** | `POST /api/legal/counsel/actions`: stale version 409 with current state (UI shows it), replay by key, forged component/action refused, read-only 403 (UI offers no controls), closed matter 409; Esc closes the panel and focus returns to the chip (unit and live) |
| LEGAL-011 | Chronology | Not started | (obligations give lawyer-confirmed dates: Tested (Law); the briefing counts unconfirmed ones) |
| LEGAL-012 | Discrepancy resolution | **Tested** | `legal_discrepancies` + reviews written once (trigger), audited `discrepancy.review`, hidden when a quoted document is not readable, amendments flagged as possible supersessions; live: found and explained in the browser |
| LEGAL-013 | Draft editor | **Tested (Law)** in its own pages; chat points there | `test_law_drafting.py`; the chat's `draft` intent offers the real Drafting page only to roles that may draft |
| LEGAL-014 | Export approval | **Tested (Law)** | `test_law_exports.py` (T09) |
| LEGAL-015 | Backup and restore | **Tested (Law)** | `app/legal/backup`, `test_law_dr.py`; the chat's tables are in the same database and purge |
| LEGAL-016 | Security / prompt-injection suite | **Tested (Law)** + chat cases | `test_law_injection.py`; chat: injected notes withheld from the model, no extra words, no other matter; live: `INJECTION-CANARY` never shown |
| LEGAL-017 | Offline acceptance | Backend tests run with `--network none`; live UI made **no** request to a non-local host; enforcement **Blocked (owner, root)** | `sudo scripts/law/egress_lock.sh` (D-071) |
| LEGAL-018 | Advanced agents (Council, GraphRAG, MCP) | Not started (Review panel exists, off: Tested (Law)) | — |
| — | Purge covers the chat | **Tested** | `test_purging_a_document_removes_what_the_chat_kept_of_it`; purge verification counts the chat's tables |
| — | Conflict finder recognises delivery dates | **Tested** | `compare/facts.py` anchor `delivery`; `test_law_compare.py` green |
| — | Content-Security-Policy | **Tested** (config) | `frontend/nginx.law.conf`: `default-src 'self'` …, validated with `nginx -t`; the built bundle references no CDN |
| — | Strict TypeScript for new code | **Tested** | `tsconfig.counsel.json` (`strict`), also in CI |

## Live run on the DGX Spark (2026-10-26, real model, isolated dev backend)

Native dev backend (`scripts/law/dev_backend.sh`, fresh `.law-data`, 127.0.0.1 only) with the demo world and the Legal
demo matters loaded through the real ingestion pipeline (`demo_load.py --data-dir .law-data --only MAT-1042,MAT-2033`).
Model `nvidia/Qwen3-30B-A3B-FP4` (TensorRT-LLM) and `nomic-embed-text`, both shared with production, ~10 GB free.

| Turn (alice, MAT-1042) | End | Seconds |
|---|---|---|
| What matters today? | completed (briefing from records) | 0.1 |
| Find conflicting dates or amounts | completed: 5 potential inconsistencies across 8 documents | 4.4 |
| Which contractual deadline applied? | completed, 2 verified citations | 5.7 |
| Could the witnesses be describing different events? | completed (partial), 3 verified citations | 6.0 |
| Let other case materials inform this | completed (notice, no other matter named) | 0.0 |
| What does the Rivera note say about settlement authority? | **abstained** | 1.1 |
| Summarize the draft meeting notes | completed, injection noted, canary absent | 5.7 |

Seconds are request-to-last-event, single user, measured once: not a benchmark.

**Findings to act on (not hidden):**
1. *Wrong inference with a real quote.* "Which contractual deadline applied?" was answered "the amendment did not change the
   delivery deadline", citing the amendment's clause 2.1 ("all other terms remain in force"). The quote is verified; the
   conclusion is wrong, because retrieval did not bring clause 1.1 of the amendment, which moves delivery to May 20. A verified
   quote proves the words exist, not the inference. **Fixed (amendment-aware answers):** `app/legal/research/amendments.py`
   finds, in the documents' own words, a readable document of the same matter that replaces or changes a cited clause
   ("Clause 4.2 of the Services Agreement is deleted and replaced …"), adds that clause and the clause saying when it applies to
   the sources, marks it for the model (`amends="… clause 4.2" amendment="signed"`, prompt rule 7, prompt version
   `research-s2.3`), and returns `amendments` with the answer (signed or not, its own words on when it applies, whether the
   answer cited it). An answer relying on wording a **signed** amendment replaced, without citing the amendment, is `partial`
   and the chat and research page say so and offer the amending clause. Stored with research history (migration 0037) and
   removed by a purge. Evidence: `test_law_amendments.py` (11, incl. the live failure replayed with a scripted model, unreadable
   and other-matter amendments never used, unsigned amendments said to be unsigned), research page Vitest (2), and the real
   model on new evaluation items: golden `g-amend-storage-fee`, `g-amend-storage-unchanged` **2/2**, held-out
   `h-amend-storage-from` **1/1** (final code: eval/law/reports/20261026-233235-golden.md, 20261026-233253-heldout.md).
   **Re-asked live** on MAT-1042 with the real model: "Which contractual deadline applied?" → *answered*: "initially … May 15,
   2026 [1]. However, this was amended by Amendment No. 1 (DEMO), which replaced the deadline with May 20, 2026 [5]. The
   amendment became effective on May 14, 2026, when both parties signed it [2]" — three verified quotes; the note names clause
   1.1 and "Open the amendment" opens its exact characters. A passage is matched on every clause it covers, and "relies on the
   replaced wording" is decided by where the answer's verified quote lands (a quote from an unchanged clause of the same
   passage is not flagged).
2. *The conflict check over-proposes.* A contract's deadline ("by May 15") against a witness's event date ("took place on
   May 12") was proposed as a potential inconsistency. **Mitigated:** such a pair is now recognised (`duty_vs_event`), ranked
   after real conflicts and shown with "they may not contradict each other: check whether the event met the deadline"; it
   stays `potential` until a person resolves it. Amendments are handled the same way (`amends`).
3. *Retrieval missed the shipment log* for "Could the witnesses be describing different events?" (the answer used both
   witnesses and the meeting notes). The blueprint's expected answer cites it.

## Test runs on this branch

| Run | What | Result |
|---|---|---|
| C1 | `test_law_counsel.py` + policy, route table, audit, records, compare, migrations, schema, settings | **701 passed** (1 min 49 s, `--network none`) |
| C2 | Frontend after the shell change: existing suites updated for the new navigation | **184 passed** |
| C3 | Frontend with the chat tests | **202 passed** (38 files); `tsc` main and strict clean |
| C4 | `test_law_counsel.py` final | **48 passed** |
| C5 | Live API journey + live browser journey (Playwright, real model) | all steps passed; no console errors, no remote requests, no horizontal overflow at 390 px, focus returns after Esc |
| C7 | Backend, all, with amendment-aware answers (final code) | **1479 passed, 3 skipped, 1 failed** (LibreOffice absent, as at baseline) |
| C8 | Frontend after amendment-aware answers | **204 passed**; `tsc` main and strict clean |
| EV1 | Research evaluation, new amendment items, real model (`run_law_eval.py --only …`) | golden **2/2**, held-out **1/1**; 0 leaks; quote verification 1.0 |
| C6 | Backend, all (`--network none`) after the chat work | **1467 passed, 3 skipped, 2 failed**: LibreOffice absent (environment, as at baseline) and the nginx drift test (the new CSP lines; test updated to allow exactly them, then its module + the chat suite **59 passed**) |
| E1 | Browser E2E, full (36 journeys + the new chat journey), real model | 28 passed, 9 failed: all from the shell change (old sidebar links, sign-out, the queue link, an "Ask" name clash) — fixed |
| E2 | Browser E2E, full | 30 passed, 7 failed: 6 refused at sign-in by the per-address throttle (the test helper reused single-use codes in a faster run) — helper fixed; 1 `versions` |
| E3 | Browser E2E, full | **36 passed, 1 failed** (19.5 min). The new **EchoMind Legal chat journey passed**. `versions`: with "Latest drafts" the research model abstained in the full-suite matter (many competing contracts); research code is unchanged on this branch |
| E4 | `versions` journey alone, twice | **2 passed** — the E3 failure is real-model variance (it also passed in E1), recorded rather than hidden |

## How to run the demo locally (fictional data only)

```bash
LAW_SEED_DEMO=1 LEGAL_DEMO_SIGNIN=1 scripts/law/dev_backend.sh          # 127.0.0.1:8110, fresh .law-data
python3 scripts/law/demo_load.py --base http://127.0.0.1:8110/api --data-dir "$PWD/.law-data" --only MAT-1042,MAT-2033
LAW_FRONTEND_HOST=localhost scripts/law/dev_frontend.sh                  # http://localhost:5174 → "Enter the demo as" alice
```
Demo sign-in is for this local, fictional setup only; it is off by default and must stay off with real data.

## Next, in order

1. ~~Amendment-aware answers~~ — done (finding 1 above). Still open: research notes sent for review do not yet carry the
   amendment links; amendments that change a clause without naming its number are not detected.
2. **Chronology card** (LEGAL-011): dated events with their sources and uncertain dates, built from obligations and the
   conflict scan's dated facts; an "explained" discrepancy (dispatch vs arrival) becomes two events.
3. **Draft from the conversation** (LEGAL-013 in chat): the `draft` intent opens a right-hand editor over the existing drafting
   engine (firm templates, sourced fields, second-lawyer approval), instead of pointing to the Drafting page.
4. **Guided questions** (03 registry): one contextual form with at most three questions for ambiguous requests.
5. **Owner and counsel items carried from the Law programme**: egress lock (root), off-host key copies, lawyer review sessions,
   P-001…P-015 (docs/echomind-law/PROGRESS.md). Restarting the stopped pilot with this branch is the owner's decision
   (`scripts/law/deploy_pilot.sh`).
