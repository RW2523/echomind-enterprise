# EchoMind Legal — implementation status

Statuses: **Not started**, **In progress**, **Tested** (executed tests pass, evidence given), **Blocked** (needs a person,
root or a decision). "Tested (Law)" = delivered earlier on `feature/echomind-law` with its own evidence
(docs/echomind-law/TEST_RESULTS.md) and re-run on this branch. Branch: `feature/echomind-legal-chat`.

## What a person can do now

Sign in → **Chat** opens: "How can I help?", the matter in the header, four suggestions. In one conversation a lawyer can
ask what matters today (a briefing from records, at most three items), ask questions answered only from the matter's
documents with verified, clickable citations, find potential inconsistencies between documents and resolve them with a
reason, build a chronology of the matter (every date the documents state, in order, each opening its exact words), open any
cited passage beside the conversation (re-verified against the stored text, highlighted), and reopen
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
| LEGAL-011 | Chronology | **Tested** | `app/legal/counsel/chronology.py`, `timeline` component: built from the documents' stated dates (the conflict scan's reader), never by the model. Document dates, deadlines and events in order; recollections and hedged dates marked; partial or relative dates ("the 20th", "yesterday") listed without a date, never converted; a clause a signed amendment replaced marked "Replaced"; a resolved inconsistency's reason on both events; register deadlines with their review state (confirmed only by a lawyer). `test_law_chronology.py` (11: order and exact sources, unreadable documents left out, explained pair, register states, malformed events refused, purge withdraws it, header dates / date ranges / headings); `counsel.test.tsx` (timeline card); live below |
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

## Design refresh "Chambers" (2026-10-27)

Taken from the owner's reference mosaic (navy and gold, serif headings, glass cards, a long sidebar, a home of action
tiles) and kept to what the product really does:

- **Shell**: a navy rail in both appearances (brand, New chat, Chat / Matters / Drafts / Activity, the chosen matter's own
  pages, other matters, recent conversations); a header with the matter, a search field that opens the existing command
  palette, the privacy and real recording states, and the person with their firm role.
- **Home**: "How can I help you today?" over an original line drawing of a courthouse (inline SVG, nothing fetched); eight
  tiles, each either one of the chat's real answers (briefing, conflict check, chronology, summary, a question) or a real
  page of the matter the person may open (Meetings, Drafting, Review panel or Research); a composer with **Choose documents**
  (answer only from up to ten of the matter's read documents; the server re-checks each) and the matter; four example
  questions.
- **Cards**: glass cards with icon headers; numbered priority items coloured by kind; side-by-side evidence with A/B marks;
  the chronology as a dated rail with year headings and coloured kinds (deadline, document date, recollection).
- **Matter overview, at a glance** (`counsel/MatterGlance.tsx`): the open inconsistencies (each opens in the chat, asked
  once with its id), the next dated deadlines (overdue first; "Not confirmed" until a lawyer confirms) and the latest
  documents, from the discrepancy register, the obligations register and the library. A part the person may not see is
  left out rather than shown as an error.
- **Not copied, on purpose**: "Use web sources" (the product is offline), risk and confidence percentages, "AI Analysis
  Complete" badges (no such measures exist; the blueprint forbids fake verification), Calendar across matters, Time &
  Billing (time entries exist behind a flag and appear in the rail when on; billing does not), a People & Relationships
  graph and firm-wide content search (no backend for either). Each would be a button that does nothing.
- Light, dark, 390 px and reduced-transparency all render; contrast values are in the header of `counsel.css`.

## Several matters in one conversation, and answers you can work with (2026-10-28)

The owner asked for conversations about one case or several together, and for answers in the spirit of ChatGPT's
"Intelligent UI" (interactive charts, tappable controls, rendered progressively). Built on the existing trusted component
registry: the server still chooses and validates every component; the model writes only text.

- **One matter or several (up to five).** The question box's matter button is a multi-select. Each matter is authorized
  when the conversation starts (`POST /api/legal/counsel/conversations {"matter_ids": [...]}`), on every read and on every
  turn; losing any one closes the whole conversation (404) and drops it from the list. A highly restricted matter is only
  ever discussed on its own (409, and the picker enforces it). Matters of different clients get a notice to keep each
  client's information to its own work. Migration 0038 (`legal_conversation_matters`); a conversation's own matter is
  still `legal_conversations.matter_id`, so existing conversations are unchanged.
- **Every source names its matter.** Research searches each matter under its own `AuthorizedScope`, finds amendment links
  inside that matter, and labels every passage; the prompt tags each source `matter="…"` and asks the model to say which
  matter each statement comes from and never combine them (prompt `research-s2.5`; a one-matter prompt is byte-identical
  to before). Citations are stored and re-verified under their own matter. An answer about several matters is kept in its
  conversation only, never in one matter's research history (so one matter's history never holds another's quotes).
- **Per-intent behaviour:** the briefing and the chronology are built per matter and merged with labels; the conflict
  scan runs inside each matter only (documents of different matters are never compared with each other); a comparison
  card from another matter of the conversation carries that matter, and resolving it acts under it; "Compare these
  cases" shows a `case_table`.
- **Interactive answers:** the chronology has filters (kind, matter) and a month chart whose bars filter the list; the
  cases table sorts by any column and each row opens the matter; cards are sent as `ui.partial` as soon as they are
  validated (shown while the turn still runs), then the full envelope.
- **Found in Chrome and fixed:** popovers opened from the glass composer or header were placed and clipped relative to
  that surface (`backdrop-filter` makes an element the frame of fixed children) — they now render on `<body>`; a quote that
  itself contains quotation marks was shown as raw text — the parser now reads it whole.

## Commercial-readiness review (2026-10-28)

Phase 0 audited security, operations and the interface read-only (three audits, baseline below); the owner then asked for
everything to be implemented. Each finding below is **Tested** with the named tests, **Done (ops)** where it is a script
or document checked by running it, or **Owner** where it needs a decision, root or a contract. Detail and the go/no-go are
in the review report shared with the owner (not in this public repository).

| Area | Finding | Status | Evidence |
|---|---|---|---|
| Demo | Demo sign-in could run beside real data; administrators one click away; cross-site entry | **Tested** | backend refuses LEGAL_DEMO_SIGNIN without LEGAL_PROFILE=demo + seed marker, or with SSO; firm-wide powers never one-click; Origin checked; demo sessions end when off — test_law_demo_signin, test_law_edition_posture |
| Deletion | A purge missed chat copies in conversations about several matters | **Tested** | purge by document id — test_law_counsel_multi |
| Access | Saved answers stayed readable after a document's access list narrowed; action replay before re-check | **Tested** | withheld whole on read; replay after checks — test_law_counsel, test_law_scoping |
| Access | `set_acl` with an empty list did not restrict an open document | **Tested** | test_law_counsel (withheld answer) |
| Abuse | No per-person model limit; unbounded bodies before parsing; giant OCR pages | **Tested** | LEGAL_MODEL_REQUESTS_PER_MIN (20), LEGAL_MAX_BODY_BYTES (2 MiB), OCR pixel cap — test_law_limits |
| Data | Purged text in SQLite free pages; intake-link tokens in the activity log; rollback below 0036/0038 | **Tested** | secure_delete, token mask, refusal — test_law_limits |
| Roles | An administrator could make managing partners (two approve walls and emergency access) | **Tested** | only a managing partner appoints another — test_law_matters_api, admin-users.test |
| Jobs | Queued summaries/diligence ran after the requester left; a disabled account's answer went on | **Tested** | test_law_late_access |
| Files | Redline generation parsed stored Word XML with entity resolution possible | **Tested** | hardened parser — test_law_late_access |
| Secrets | Authenticator secrets stored in clear | **Tested** | sealed with the master key, old ones sealed on use — test_law_identity_units |
| Network | Plain HTTP served the whole app on the LAN; no HSTS | **Tested** | HTTP on 127.0.0.1 only; HSTS on HTTPS — test_law_pilot_config |
| Claims | "No cloud service", "nothing leaves it" regardless of configuration | **Tested** | wording from `model_location` — test_law_auth_api, counsel.test |
| Logs | Tracebacks could carry document text | **Tested** | message replaced by its length — test_law_edition_posture |
| Interface | Design pass, keyboard, screen readers, phone, states, source panel, glance, tabs, home | **Tested** | vitest (audit-wording, keyboard, admin-users, counsel); screenshots light/dark/390 px |
| Evaluation | Gate always exited 0; held-out set not frozen | **Tested** | thresholds.json (proposed), `--gate`, frozen checksum — test_law_eval_gate |
| Answers | Declines shown as "partly supported"; 3 held-out abstention items too few to measure | **In progress** | rule + second frozen held-out set (EV6, EV7): golden and held-out pass; held-out 2 fails (abstention 9/15, answers 11/15, one attribution error) |
| Answers | Verified quotes from the wrong document ("the Kestrel agreement is governed by English law" on the NDA's and lease's words) | **In progress** | attribution, standing and party checks (EV8-EV11): golden, held-out and the unseen held-out 4 pass every bar; development sets held-out 2 and 3 still miss answer/precision/quote bars |
| Operations | Deploy without backup; rollback broken by migrations; no restore script; backups silently stopped; no RPO/RTO | **Done (ops)** | deploy_pilot.sh, backup_pilot.sh, restore_pilot.sh (drilled on scratch volumes), RUNBOOK, INSTALL, INCIDENT_RESPONSE |
| Operations | Containers ran with full capabilities | **Tested** | no-new-privileges, cap_drop ALL, pids — test_law_pilot_config; images smoke-tested |
| Demo | Ran from an uncommitted script on the host network, code writable, health check probed production | **Done (ops)** | scripts/law/demo_up.sh; the public demo now runs that way |
| CI | No dependency audit | **Done** | `dependencies` job |
| Infra | Full-disk encryption (the database holds extracted text) and the egress lock (needs root) are host settings | **Owner** | check_disk_encryption.sh and egress_lock.sh --check report them; deploy refuses either missing unless accepted |
| Release | Thresholds, pricing/contracts, customer model, cross-client multi-matter policy | **Owner** | thresholds marked "proposed" |
| Open | Non-root containers and read-only roots for the pilot; master-key rotation tool; signed audit anchors; browser E2E and evaluations in CI (need the model); actions pinned by SHA | **Not started** | listed in the report with effort |

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
4. *An answerable question about what a witness said was withheld (2026-10-27).* "When did Witness B say the delivery took
   place?" came back "The documents in this matter do not answer that." **Cause:** not the server — a spy on every model
   call showed 19 byte-identical prompts (same temperature, no concurrent calls) and the model alone choosing the
   abstention once; the earlier "server vs script" gap was small samples. The model reads "when did X *say* …" as "when did X
   speak" and withholds: 10–15 times in 15 on the E2E matter, 10 in 10 for "When did Jonas Ekberg say the chiller alarm
   sounded?" in the evaluation corpus (an answer opening "The documents state …" and the abstention sentence share their
   first words, so sampling at 0.1 also flips plain questions about 1 time in 20). A system-prompt rule helped only partly
   (7 in 10 still withheld) and made unanswerable questions drift into "not stated, but here is related text" (held-out
   abstention 2/3 → 1/3), so it was not kept. **Fix (prompt `research-s2.4`):** such a question alone gets its plain reading
   beside it — "(That is: when, according to Witness B, the delivery took place.)" (`prompt.account_reading`); every other
   prompt is byte-identical. Measured: 10/10 → 0/10 and 10/15 → 1/15 withheld for the answerable ones; the unanswerable
   "When did Jonas Ekberg say the chiller was repaired?" still withheld 10/10. Also: bare "[1] [1] [1]" markers after a
   not-stated sentence no longer count as support (`routes._stands`). Evaluation: a fictional witness statement and four
   ACCOUNT items added (golden 3, held-out 1); results in the test table (EV2–EV4). The chat journey asks the natural
   question again, without a second try.

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
| C9 | `test_law_chronology.py` (final code) | **11 passed** (15 s, `--network none`) |
| C10 | Frontend with the chronology card | **206 passed** (38 files); `tsc` main and strict clean |
| C11 | Backend, all, with the chronology | **1489 passed, 3 skipped, 1 failed** (13 min 46 s; LibreOffice absent, as at baseline) |
| L2 | Live, through the public demo URL (alice, MAT-1042, Playwright): "Build a chronology of events" | 17 events (15 dated, 2 not) from 6 documents: 8 shown first, "Show all" reveals the rest and the two undated ones ("the 20th", "the 12th"); the email's "Sent:" lines, the shipment's "May 13, 2026 to May 14, 2026" as one event, Witness A marked Recollection + Uncertain with alice's explanation, the contract's May 15 deadline marked Replaced, the amendment's May 20 deadline; Inspect opened the exact characters; no console errors |
| EV1 | Research evaluation, new amendment items, real model (`run_law_eval.py --only …`) | golden **2/2**, held-out **1/1**; 0 leaks; quote verification 1.0 |
| C6 | Backend, all (`--network none`) after the chat work | **1467 passed, 3 skipped, 2 failed**: LibreOffice absent (environment, as at baseline) and the nginx drift test (the new CSP lines; test updated to allow exactly them, then its module + the chat suite **59 passed**) |
| E1 | Browser E2E, full (36 journeys + the new chat journey), real model | 28 passed, 9 failed: all from the shell change (old sidebar links, sign-out, the queue link, an "Ask" name clash) — fixed |
| E2 | Browser E2E, full | 30 passed, 7 failed: 6 refused at sign-in by the per-address throttle (the test helper reused single-use codes in a faster run) — helper fixed; 1 `versions` |
| E3 | Browser E2E, full | **36 passed, 1 failed** (19.5 min). The new **EchoMind Legal chat journey passed**. `versions`: with "Latest drafts" the research model abstained in the full-suite matter (many competing contracts); research code is unchanged on this branch |
| E4 | `versions` journey alone, twice | **2 passed** — the E3 failure is real-model variance (it also passed in E1), recorded rather than hidden |
| C12 | Frontend after the design refresh and the matter glance | **208 passed** (38 files); `tsc` main and strict clean |
| E5 | Browser E2E, full, after the design refresh (alternate ports, the demo stayed up) | **34 passed, 3 failed** (24.4 min): `research` (no passages: transient, passed on re-run with its documents journey, 14 s), `versions` (two correct "reads differently" warnings — 8.1 and 4.1 — matched one test filter: filter narrowed to 8.1), the chat's source step (finding 4) |
| E6 | `documents` + `research`; `versions`; the Legal chat journey | **all passed** after the two test fixes |
| E7 | Browser E2E, full, final code | **36 passed, 1 failed** (19.4 min): `versions`, its "Latest drafts" step (no 500,000 in the evidence) — the same real-model variance as E3; the journey passed alone in E4 and E6 |
| EV2 | Research evaluation before S2.4 (prompt `research-s2.3`), full sets | golden 72: answers 62/66, abstentions 5/6, 3 false abstentions, 0 leaks; held-out 33: 29/30, 2/3, 0 leaks (reports 20261027-021309, -021454). The new ACCOUNT items on that prompt: held-out `h-account-said-time` **withheld** (-021529, -021543) |
| EV3 | With a system-prompt account rule (not kept) | golden 75: 66/68, 5/7; held-out 34: 29/31, **1/3** — unanswerable questions drifted into "not stated, but…" (-022015, -022202) |
| EV4 | Final: `research-s2.4` (a reading beside account questions only) and marker-only claims support nothing | golden 75: answers 65/68, abstentions 5/7, 1 false abstention, 0 leaks, quotes verified 0.98; held-out 34: 29/31, 2/3, 0 leaks; **all four ACCOUNT items pass** (-025933, -030117). Before the marker fix: golden 65/68, 6/7; held-out 30/31, 2/3 (-023608, -023748). Every non-ACCOUNT prompt is byte-identical to `research-s2.3`; their run-to-run swings (salary, non-compete, attendance) occur in the baseline runs too |
| E8 | Legal chat journey, asking "When did Witness B say the delivery took place?" with no second try | **2 of 2 passed** (15 s each); 0 of 3 before the fix |
| C13 | Backend: evidence, injection, amendments, counsel (and evaluations, retrieval) | **118 passed** (136 with evaluations and retrieval), pinned image, `--network none` |
| C14 | Several matters + interactive answers: backend all (`--network none`) | **1503 passed, 3 skipped, 1 failed** (14 min; LibreOffice absent, as at baseline); then the quote-parser change: evidence, injection, counsel, multi-matter, amendments, chronology, retrieval **151 passed** |
| C15 | Frontend after several matters | **217 passed** (38 files); `tsc` main and strict clean |
| L5 | Live, public demo, Chrome: MAT-1042 + M-301 | the two chosen in the question box; "Compare these cases" table; a 43-event chronology filtered by matter and month; "What delivery deadlines and late-delivery remedies apply in each case?" answered per matter, **6 verified quotes**, every source tagged with its matter; no console errors, no failed requests. Browser E2E for the new journey is written; the full E2E run follows the UI simplification |
| C16 | Commercial-readiness review: backend, all, final code (`f32da8d`, pinned image, `--network none`) | **1537 passed, 4 skipped, 0 failed** (14 min). T09 now skips where LibreOffice is absent and is required in CI |
| C17 | Frontend, final code | `tsc` main and strict clean; **231 passed** (41 files), twice in a row; single timeouts seen only while the browser suite loaded the host, passing alone |
| EV5 | Research evaluation **gate** (`--gate`, clean tree `e5d670f`, frozen held-out) | golden 75: answers 63/68, abstentions 5/7, precision 0.924, quotes verified 0.989, 0 leaks, p95 5.9 s — **passed**. Held-out 34: 27/31, **abstentions 1/3 (bar 2/3) — FAILED**, precision 0.97, quotes 1.0, 0 leaks, p95 4.9 s (reports 20261028-025039, -025228). Code on the answer path is unchanged since `cb96553`; the misses are the model writing "not stated, but…" with a quote instead of declining |
| E9 | Browser E2E, full, first run after the review | 34 passed, 4 failed: rail aria-labels clashed with fields named "Matters"/"Activity" (fixed: text kept for screen readers), carol's matter now also in the header (locator fixed), the two-matter journey |
| E10 | Browser E2E, full | **37 passed, 1 failed**: the two-matter journey chose M-114, which an earlier journey walls alice off (it had only passed alone); it now opens its own second matter and passes after the wall journey (18 s) |
| E11 | Browser E2E, full, final code (`84fcc64`) | **37 passed, 1 failed** (19.8 min): `conversation` — the real model declined the liability-cap question ("The documents in this matter do not answer that"), a false abstention; the journey passed in E9 and E10 on near-identical code |
| D2 | Restore drill (`restore_pilot.sh` steps) on scratch volumes with a demo archive | archive verified, previous data kept aside, restored, audit chain verified (728 events, 2 anchors) |
| L6 | Hardened public demo (`demo_up.sh`) through the tunnel | entry as alice, `model_location` this_server, a cited research answer, transcription 3.2 s and speech; one-click list without administrators; a cross-site entry refused (403) |
| EV6 | Abstention work (2026-10-28): a second held-out set frozen **first** (`d014a78`: 15 unanswerable near-misses incl. 1 T06, 15 answerable look-alikes), then a deterministic rule developed on golden only (`d5d066a`, `research/prompt.py` `opens_not_stated`, `only_context`, `missing_terms`) | golden trials during development: abstentions 7/7, 6/7, 7/7, 5/7, 6/7 while the rule grew; replayed over the stored answers of 11 golden runs it turns 9 declines into abstentions and loses no correct answer |
| EV7 | Gate on the clean commit `d5d066a`, each set once | golden **passed** (answers 65/68, abstentions **7/7**, 0 leaks, precision 0.927, quotes 1.0, p95 6.1 s); held-out **passed** (30/31, 2/3, 0 leaks, 0.983, 1.0, 6.4 s); held-out 2 **FAILED** (answers 11/15, abstentions 9/15, 0 leaks, precision 0.91, quotes 1.0, p95 4.5 s). Reports 20261028-131259, -131454, -131620. Not tuned against: misses there are (a) "not stated" then a positive side fact the rule cannot tell from an answer (4), (b) a count where every word of the question is in the document (1), (c) **an attribution error**: the Kestrel agreement said to be governed by English law, with verified quotes from the NDA and the lease, (d) the model declining or misreading four answerable questions. Held-out 2 has now been seen; the next change needs a fresh held-out set |
| EV8 | Attribution work (2026-10-28): held-out 3 frozen **first** (`2fabfe5`: 15 unanswerable, 13 of them cross-document traps, 14 answerable look-alikes); then `research/attribution.py` and two checks in `_answer` (`ac801b3`), developed on golden and held-out 2 | replayed over the stored answers of 16 runs: 2 sentences flagged, both wrong; a model second pass measured on 153 answers and **not** added (one wording withheld 78/141 correct answers, the other caught 0/7 declines); golden trial 67/68, 7/7, 0 leaks |
| EV9 | Gate on the clean commit `ac801b3`, each set once | golden **passed** (65/68, 7/7, 0 leaks, precision 0.928); held-out **passed** (30/31, **3/3**); held-out 2 failed (11/15, 10/15); **held-out 3, first run: failed by one item** — answers 12/14 (passes), abstentions 11/15 (bar 12/15), 0 leaks, precision 0.909, quotes 0.955. Three traps withheld as `other_documents`. Misses: a sentence claiming the Kestrel agreement "states" the NDA's London-courts clause (misattributed, so not standing, but nothing else stands either and the card shows "partly supported" instead of declining); "not stated" then the MSA's 48-hour clause or the MSA's and NDA's confidentiality terms; Voss-Lindqvist's next board meeting given as Kestrel's; two false abstentions (minutes approval date, statement date). Held-out 3 has now been seen. Reports 20261028-182514, -182659, -182825, -182955 |
| EV10 | Standing and parties (2026-10-28): held-out 4 frozen **first** (`09539da`: 15 unanswerable, 7 of them party mix-ups and 6 cross-document traps, 13 answerable look-alikes); then three checks in `_answer` (`832ea90`), developed on golden and held-out 1-3 | replayed over stored development answers: 2 more correct declines, no correct answer lost; golden trial 63/68, 7/7, 0 leaks (none of the new checks fired on golden) |
| EV11 | Gate on the clean commit `832ea90`, each set once | golden **passed** (65/68, 7/7, 0 leaks); held-out **passed** (28/31, 3/3); held-out 2 failed (answers 11/15, precision 0.872; abstentions 13/15 now pass); held-out 3 failed (quotes verified 0.889; abstentions **15/15**, answers 13/14); **held-out 4, first run: passed every bar** — answers 13/13, abstentions 12/15, 0 leaks, precision 0.923, quotes 0.952; withheld as other_party 2, other_documents 4. Its misses: "What budget did Northbridge approve for Project Lantern?" answered with Voss-Lindqvist's board budget (no possessive, so no party check), an insurance premium not stated, and the lease's break notice read as a non-renewal notice. Reports 20261028-200542 … -201153 |
| L3 | Live, public demo (2026-10-27): meeting recorded in a real browser with spoken audio as the microphone (voice service clip), backend in the Law image with speech recognition on the CPU (the GPU is held by the host's models) | consent from 3 participants, then the transcript appeared live every ~5 s; Silent Assist raised **action**, **commitment to the other side** and **privileged or sensitive information with a counterparty** alerts quoting the words said (commitment and disclosure need the speaker marked: with "Not marked", only the action rule applies, by design) |
| L4 | Live, public demo: the conversation assistant asked aloud | the clip transcribed (first call 17.6 s including loading the model), answered from MAT-1042's documents with citations, read aloud by the voice service |

## How to run the demo locally (fictional data only)

```bash
LAW_SEED_DEMO=1 LEGAL_DEMO_SIGNIN=1 scripts/law/dev_backend.sh          # 127.0.0.1:8110, fresh .law-data
python3 scripts/law/demo_load.py --base http://127.0.0.1:8110/api --data-dir "$PWD/.law-data" --only MAT-1042,MAT-2033
LAW_FRONTEND_HOST=localhost scripts/law/dev_frontend.sh                  # http://localhost:5174 → "Enter the demo as" alice
```
Demo sign-in is for this local, fictional setup only; it is off by default and must stay off with real data.
**Open demo, no sign-in step (owner's request, 2026-10-27):** `LEGAL_DEMO_SIGNIN=1 LEGAL_DEMO_SIGNIN_FROM=anywhere
LEGAL_DEMO_AUTO=alice` — the sign-in page enters as that demo person by itself, so a visitor opens straight into the
workspace. Only a demo seed person can be named; `/signin?manual=1` still shows the page (the administrator's way in); a
failed entry is tried once, not looped. Every visitor is the same person and sees the same conversations. Never with real
data: unset the three to restore sign-in. Tests: `test_law_demo_signin.py` (open demo), `signin-demo.test.tsx` (2).

## Next, in order

1. ~~Amendment-aware answers~~ — done (finding 1 above). Still open: research notes sent for review do not yet carry the
   amendment links; amendments that change a clause without naming its number are not detected.
2. ~~Chronology card~~ (LEGAL-011) — done. Still open: no date filter or per-party view; events are dates the reader
   recognises in English month-name or numeric form; a document with more than 60 dated events shows the first 60 and
   says how many more.
3. **Draft from the conversation** (LEGAL-013 in chat): the `draft` intent opens a right-hand editor over the existing drafting
   engine (firm templates, sourced fields, second-lawyer approval), instead of pointing to the Drafting page.
4. **Guided questions** (03 registry): one contextual form with at most three questions for ambiguous requests.
5. **Owner and counsel items carried from the Law programme**: egress lock (root), off-host key copies, lawyer review sessions,
   P-001…P-015 (docs/echomind-law/PROGRESS.md). Restarting the stopped pilot with this branch is the owner's decision
   (`scripts/law/deploy_pilot.sh`).
