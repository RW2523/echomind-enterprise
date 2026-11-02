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

## Chat controls and the rest of the card registry (2026-11-02)

The owner asked for the chat to work like other assistants, and for everything listed as missing to be built. The
server still chooses and validates every component; the model writes only text.

| Area | What was built | Status | Evidence |
|---|---|---|---|
| Formatted answers | Headings, bulleted and numbered lists, tables and bold, parsed into React text (`counsel/format.ts`); no HTML, links or images; a marker without a source is still dropped | **Tested** | `counsel-controls.test.tsx` (hostile link, image and markup stay words) |
| Copy | The answer, the source list with each quote's verification, a partly supported or calculated note, to the clipboard (`counsel/copy.ts`) | **Tested** | `counsel-controls.test.tsx` |
| Ask again, edit and resend | Another version of the turn (migration 0039 `parent_turn_id`); every shown answer stays on record; "‹ 2 / 3 ›" switches; what follows stays with its version; a follow-up is rewritten from its own path only | **Tested** | `test_law_counsel_controls.py` (versions, own path, foreign ids 404, migration backfill and refused rollback); `counsel-controls.test.tsx` |
| Rating | Thumbs up/down, a reason and note on thumbs down; one per answer and asker (`legal_counsel_feedback`); the audit keeps the note's length only | **Tested** | `test_law_counsel_controls.py`; no screen reads ratings yet (owner decision: who reviews them) |
| Conversations | Rename, delete (off the list; kept as the matter's work product, purge and retention still apply), search by title or question with the matching words | **Tested** | `test_law_counsel_controls.py` (`%` and `_` literal, other people's never); `counsel-controls.test.tsx` |
| Question box | Add a document (contributors only, as a draft, from the menu or by dropping a file; asking waits until it is read); dictate where voice is on (the "Ask by voice" transcription, not kept) | **Tested** | `counsel-controls.test.tsx` (read-only member not offered; file type refused before sending; no microphone where voice is off) |
| `checklist` | Open obligations from the register; tick done, reopen, confirm a date (after an explicit step) through the register's own change (`obligations/routes.change_obligation`), at the item's version | **Tested** | `test_law_counsel_cards.py`; `counsel-cards.test.tsx` |
| `approval_request` | The review queue's items in the conversation's matters; approve (the person confirms they read that version) or ask for changes (a reason) through the review route's own rules (`reviews/routes.decide_review`); approving never files or sends | **Tested** | `test_law_counsel_cards.py` (gaps block a legal draft, author and readers cannot decide, replay, no filing) |
| `guided_questions` + `document_preview` | "Draft …" asks up to three questions with defaults (template, title, instructions), then writes with the Drafting engine; the draft opens beside the chat to correct (each save a new version) and waits for a second lawyer | **Tested** | `test_law_counsel_cards.py` (answers checked against the options asked, another conversation's form refused, newest version shown) |
| `council_summary` | "Challenge our position …" (panel on, contributors) asks the question and the specialist area, queues the panel (`panel/routes.start`); the card follows the run and shows where the perspectives differ | **Tested** | `test_law_counsel_cards.py`, `counsel-cards.test.tsx` (stops asking when done) |
| `numeric_chart` | Amounts the documents state, read by rule; one currency per chart; never added or converted; an amount whose digits do not match its value is left out; each bar opens its sentence | **Tested** | `test_law_counsel_cards.py` (other matter excluded, no model call) |
| `graph_excerpt` | Companies read by rule and the documents naming them ("party to" or "named in"), each edge with its sentence; roles only from the matter's records; a list view for screen readers and phones | **Tested** | `test_law_counsel_cards.py`; `counsel-cards.test.tsx` |
| `live_transcript` | The meeting being recorded (or the latest transcript): state, consent count, latest segments; follows for ten minutes; never records, pauses or stops (that stays with the person capturing) | **Tested** | `test_law_counsel_cards.py` (deleted transcript leaves the card) |
| Reading a card again | `GET /api/legal/counsel/components/{id}`: the owner's card, rebuilt from its records; cards drawn from changing records store references only | **Tested** | `test_law_counsel_cards.py` |

Intents added (rules, no model): `tasks`, `approvals`, `live`, `amounts`, `parties`; `draft` and `deliberate` now ask
their questions. A question about what a document says ("What deadlines does the lease set?") stays a research question.

Test runs: backend full suite **1608 passed, 4 skipped** (1610 after the voice work) (pinned image, `--network none`); frontend **261 passed**
(43 files); `tsc` main and strict clean.

Off in the demo, so not shown there: dictation (`LEGAL_CONVERSATION`) and the review panel (`LEGAL_PANEL`).

Screenshots (`chat-2026-11-02/`, fictional DEMO documents on a throwaway stack, real model): the question box, a formatted
answer with its toolbar, asked again (versions), rating, editing, the checklist before and after a tick, approvals, the
three questions and the draft beside the chat, amounts, parties, the meeting transcript, the panel's two questions, its
run and its result, history search and delete, and a phone-width chart. Found while taking them and fixed: a class name
clash that stacked the checklist, the version counter wrapping, list bullets missing, the template guess choosing
"Engagement letter" for a letter before claim, the chat not following a card that arrives without text, and the
document picker showing a row revision as the version.

## Voice mode: the original EchoMind voice conversation, in the chat (2026-11-02)

The owner asked for the original EchoMind speech-to-speech bot, Silent Assistant and live transcript in this application
(plan approved 2026-11-02; decisions: a separate CPU voice container reusing `voice/app`; on the public demo only after the
owner reviews it). This part is the voice bot; the meeting transcript and Silent Assistant follow separately.

| Area | What was built | Status | Evidence |
|---|---|---|---|
| Voice service, Law mode | `voice/app` (webrtcvad, adaptive end of turn, Nemotron partials, Parakeet final words, phrase-by-phrase Piper speech, barge-in) asks every spoken question as a **turn of the open chat conversation** (`?conversation=`), with the person's on-behalf-of token, so it is answered only from the matter's documents with checked quotes and saved with its sources; speaks the checked paragraphs without source marks, a card's server summary, and calculated dates labelled as calculated; tells the browser which turn was saved. No speculative answer, no ambient listening, no voice commands or recaps, no "legal advisor" greeting; a failed turn is said once, never asked again behind the person's back; a question right after an interruption waits for the stopped turn | **Tested** | `voice/tests/test_law_voice_chat.py` (9) and the voice suite **160 passed** (Law voice image) |
| Backend | On-behalf-of tokens reach one more route only: asking a turn in one of the person's own conversations | **Tested** | `test_law_counsel_controls.py` (service token: own conversation only; read, rename, create and actions refused; dead after sign-out) |
| Chat | A Voice button beside Send (where voice is on, for a matter); the voice panel takes the question box's place: an orb following who speaks, the words being heard, what is said, mute and end; saved answers appear in the thread as ordinary turns; the header shows the open microphone; audio frames from a static AudioWorklet (`/law-voice-framer.js`, the CSP allows no blob modules) | **Tested** | `counsel-voice.test.tsx` (4); frontend **265 passed**; `tsc` main and strict clean |
| Deployment | `scripts/law/demo_up.sh voice`: the Law voice image on the CPU, code read-only, read-only root, every capability dropped, 10 GB, `oom-score-adj` 800, on the demo's own network with the backend, published on 127.0.0.1 only; nginx passes `location = /voice/ws` and nothing else (D-062 holds for its HTTP routes); the backend's read-aloud uses it on the private network instead of production's voice service; meeting transcription uses the pilot's look-ahead (13 / 1120 ms, D-072). `DEMO_PREFIX`/`DEMO_API_PORT`/`DEMO_VOICE_PORT` run a private copy | **Tested** | `test_law_pilot_config.py` (only the socket is passed; the demo never calls port 8002); private copy below |
| Read-aloud speed | The voice service built a new Piper voice per `/speak` and synthesised on its event loop; now one cached voice in a worker thread | **Tested** | 0.95 s first call, then 0.13–0.16 s |

**End to end on a private copy** (law-image backend on 8121 with fictional DEMO documents, voice container on 8122, real
models, Chromium with a fake microphone playing a Piper-spoken question): "When does the initial term of the Master
Services Agreement end?" was heard cleanly, answered from the agreement with its source and the calculated end date
(1 March 2029, said as calculated), saved in the conversation, and shown in the thread. End of speech to final words
**0.8 s**, to first audio (a hold phrase) **1.4 s**, to the answer itself **3.9 s** (CPU speech, Parakeet final words; with
Nemotron final words the final decode took 5.2 s and inserted stray full stops, so Parakeet is on). A second question spoken
over the answer stopped the playback and was answered. Screenshots in `voice-2026-11-02/`.

## Live meeting transcript and the Silent Assistant, from the original EchoMind (2026-11-02)

The rest of the same request: the original EchoMind live transcript and Silent Assistant, inside the Legal meeting
capture (consent, custody, pause and withdrawal unchanged).

| Area | What was built | Status | Evidence |
|---|---|---|---|
| Live line | Capture sends rate-limited `partial` messages with the words not yet committed, joined as the original joins them (`meetings/stitch.py` reuses `transcribe/session_state.py`'s chunk-boundary rules); display only, never stored, never sent once capture stops for any reason; the record keeps the model's text unchanged (segments are immutable, corrections are revisions) | **Tested** | `test_law_transcripts.py` (shown joined, stored as produced, nothing after a withdrawal); `transcript.test.tsx` |
| Silence commit | A sentence followed by silence is a segment at once, even short; silence is audio with no new words from the model, at least 1.6 model chunks (word times lag by the look-ahead) | **Tested** | `test_law_transcripts.py` (short sentence then silence; lagging word times not taken for silence) |
| Statement checks | `assist/claims.py`: the original's tiers 2–3 for any statement a document could settle (its legal-profile classifier); the research search under the capturing person's own scope (signed versions, per-document access); one model call citing a numbered passage; an alert only when the model's quote is in that passage; a contradiction is critical, agreement only when the agreement or a clause was named, "not addressed" raises nothing; figures stay with the rule engine, including the first half of a figure the segmenter split ("…the cap is five." then "Hundred thousand dollars"; a sentence ending on a clause number is still checked); passages and the statement fenced, text addressed to an AI withheld; every document quote re-checked against its version (`engine.verify_source`, previously unused) before it is stored | **Tested** | `test_law_assist_claims.py` (6: contradiction with exact words, invented quote / not addressed / unnamed agreement raise nothing, questions and asides skipped, another matter never offered, injected text withheld) |
| Live view | Record view in two columns: transcript with "Checking against the documents…" on the sentence being checked and each alert marked under its sentence (a link to its card); Silent Assist and "Mentioned so far" (parties, amounts, dates, periods, clauses, spoken clause numbers included) beside it | **Tested** | `assist.test.tsx`; frontend **267 passed** |

**End to end on the private copy** (law-image backend with CPU speech-to-text and the pilot's look-ahead, real models,
Chromium with a fake microphone playing two Piper-spoken fictional statements): the live line followed the speech;
"Under the agreement, either party can assign it to anyone without the other party's consent" raised **"Said differently
in Master Services Agreement (DEMO) (cl. 14.1)"** about 3 s after it was said, quoting "Neither party may assign this
Agreement without the prior written consent of the other party…"; "clause twelve" was listed as mentioned. Screenshots
in `meetings-2026-11-02/`. Still as before: the recogniser's stray full stops can split a sentence across two segments
once it passes 12 words (the live line shows it whole); names it mishears ("Harrogate") are fixed with the matter
vocabulary suggestions.

## The conversation assistant is the chat, and the walnut-and-gold palette (2026-11-02)

The owner's next request: the conversation assistant tied to the chat as in ChatGPT, answering properly, and a brown, gold
and black look to try. Checked first with a spoken eight-question conversation on the private copy (fictional data, real
models), which found what is fixed below.

| Area | What was built | Status | Evidence |
|---|---|---|---|
| One assistant | Every "talk" entry — the matter's tab (now "Talk by voice"), the home tile and command palette, the chat home's link, the old `/m/…/conversation` address — opens voice mode in a new chat about that matter; the separate push-to-talk page is gone. The voice button is offered without a chosen matter and asks which one first | **Tested** | `counsel-voice.test.tsx`, `navigation.test.tsx`, `counsel.test.tsx` |
| Live in the thread | While voice is on, the spoken question and the answer as it is spoken appear in the conversation at once; the saved turn (sources, copy, rating) takes their place. The voice panel keeps the orb, the state and the words being heard | **Tested** | `counsel-voice.test.tsx` |
| Read aloud | Every answer has "Read aloud" (where voice is on), said as voice mode says it — the answer without its source marks, "partly supported" when so, calculated dates as calculated, a card's own summary — and again to stop | **Tested** | `counsel-voice.test.tsx` |
| Small talk | "Hello, can you hear me?" was answered "The documents in this matter do not answer that." Now any utterance made only of greetings, checks, thanks, acknowledgements, goodbyes or "what can you do" gets a short reply with no search and no model; a question in any part still goes to the documents; spoken without "the details are on your screen" | **Tested** | `test_law_counsel.py` (intents table, replies, no model call); voice tests |
| Declined for words like "someone else" | "Can either party assign the agreement to someone else?" was declined because no document says "someone" or "else", though clause 14.1 answers it. Words for anyone or anything (someone, anyone, something, else…) no longer count as missing words of the question; "Can someone else chair the board?" is still declined on "chair" and "board" | **Tested** | `test_law_evidence.py`; spoken: now answered from cl. 14.1 |
| Voice says what the chat shows | Calculated dates as a sentence ("the current term ends on 1 March 2029, and the last day to give notice of non-renewal is 1 December 2028", still introduced as calculated, not stated); after a partly supported answer, "Part of that has no verified source in the documents; it is marked in the chat"; when the server sets aside an answer whose paragraphs were already said, "On checking, that is not given as the answer" and its reason (not when what was said already declined) | **Tested** | voice tests (162 passed) |
| Walnut and gold | A second token set in `counsel.css` (black rail, parchment and paper, brown actions, gold for the current place and evidence), default; "Navy colours" in the profile menu switches back, kept per browser; the few colours written in place now read tokens. DESIGN.md updated | **Tested** | WCAG AA pairs in the `counsel.css` header; screenshots |

Tried and withdrawn: dropping an opening "the documents do not address…" when the next, cited sentence seemed to answer.
It removed true qualifiers ("The documents do not specify a penalty. However, they state delivery is due by May 15 [1]"
would have read as if the date answered the penalty question), so research's own rules stay in charge of such openings.

**Spoken end to end on the private copy** (Piper-spoken fictional questions into Chromium's microphone, real models, CPU
speech recognition): greetings, "What can you do?", "What is waiting for my approval?" and goodbyes answered in 0.3–0.6 s
after the words were heard, with no hold phrase; document questions answered with their sources in about 4–8 s (term end
with calculated dates, payment terms across two agreements, parties, assignment from cl. 14.1); a late-payment "penalty"
the documents do not provide was declined aloud and in the chat alike; the weather was declined. Each spoken question and
its answer appeared in the thread as they were said. Still to know: the recogniser sometimes hears "assign" as "a sign"
(the matter vocabulary used for meetings is not yet applied to voice mode), and a lone one-word "Hello." can be too short
to be taken as speech. Screenshots in `palette-2026-11-02/`. Tests: backend **1631 passed, 4 skipped** (all of
`backend/tests`, `--network none`); voice service **162 passed**; frontend **266 passed**; `tsc` main and strict clean.


## The public demo's data: every feature ready to use (2026-11-02)

The owner's request: demo data with which every feature can be used and experienced end to end. The public demo had 14
matters but documents in only three; M-101 held a visitor's "sample_msa_test" and nothing else, and 49 test conversations.
It now runs on a fresh, fully fictional world built by three committed scripts (the old folder is kept as a backup):

- `scripts/law/demo_seed.sh <folder>`: sign-in secrets, `seed-demo` (people, matters, templates, playbook, an ended client
  call with its transcript and Silent Assist alerts), the newest evaluation reports, 48 regulatory updates dated to today;
- `scripts/law/demo_load.py`: the practice cases and all 64 fictional documents of 14 matters (M-101 MSA renewal, M-114
  Series B diligence, M-130 base corpus with a signed and a draft version, M-301–306 six practice cases, MAT-1042 Morgan v.
  Apex, the restricted M-202 / M-215 / MAT-2033, R-001 real public law), every one searchable;
- `scripts/law/demo_enrich.py`: through the product's own API, as the lawyer who would do it — alice (the one-click
  visitor) joins the six practice cases (the restricted matters stay closed to her); obligations read from six agreements,
  two confirmed with their dates; conflict scans (MAT-1042: May 12 / 15 / 20); three drafts from firm templates (carol's
  engagement letter waits for alice); a research note sent for review; a playbook check of the MSA, a citation check that
  flags an authority that cannot be real, a definitions check finding the planted duplicate "Price"; two summaries; a
  signed-versus-draft comparison; two due-diligence runs with quoted issues; a review-panel question ready for a partner;
  time suggestions; two enquiries (one naming Harrowgate, so screening refers it to the conflicts team); and chats showing
  a briefing, sourced answers, the May 12 / May 15 comparison with its follow-up, a chronology and an amounts chart.

Found and fixed while building it:

| Found | Fix | Status | Evidence |
|---|---|---|---|
| "Could these refer to different events?" after the May 12 / May 15 comparison (demo script step 4) was declined: follow-ups saw only plain answers, not comparisons | A comparison turn joins the follow-up history as its cards' titles and documents (those the person may still see); the follow-up is then answered from the documents as any question: "The shipment log states that the pickup … occurred on May 12, 2026 [1]. The arrival … May 15, 2026 [1]" | **Tested** | `test_law_counsel.py` |
| A signed agreement uploaded just before its redraft was never read (its job found it no longer current), so it had no text to read or compare and was missing from answers from signed versions | Every version of a document that is not deleted is read; the existing step makes only the searchable versions (current + latest signed) active | **Tested** | `test_law_versions.py` (signed answer from v1, versions compared) |
| A failed first request through the tunnel opened the standard EchoMind app instead of Legal | The edition is asked again after a network or server error before the standard edition is assumed | **Tested** | `edition.test.ts` |

Checked on a private copy, then on the public demo: every one of 37 feature pages opened at desktop, phone and dark with
no console error, failed call or sideways scroll (two network-change blips from the host during one public run); voice
through the public address answered a greeting, the term end with calculated dates, what waits for approval and the
liability caps with their sources. Tests: backend **1633 passed, 4 skipped** (all of `backend/tests`, `--network none`),
frontend **268 passed**, voice **162 passed**.

## Review of the rest of the app (2026-11-02)

The last part of the same request: every main page opened at desktop (1360 px), phone (390 px) and dark, on the private
copy with fictional data, recording console errors, failed API calls and anything wider than the screen (including rows
that only scrolled inside the main column), then looked at. No console errors and no failed calls on any page; what was
wrong is fixed below.

| Found | Fix | Status | Evidence |
|---|---|---|---|
| "Ask by voice" never read a long answer aloud: the 2,000-character limit counted the quoted citations the answer carries, so a long cited answer was refused before they were removed | The request takes up to 20,000 characters; what is *said* is at most 2,000, cut at a sentence end with "The rest of the answer is on screen." | **Tested** | `test_law_conversation.py` (a 1,550-character answer says 50; a 4,000-character one stops at a sentence) |
| Phone: the meeting page was 424 px wide (its tabs set the width of the whole column) and the document reader 577 px | The page column can shrink (`minmax(0, 1fr)`; pages with their own columns keep them); the meeting tabs scroll inside their strip | **Tested** | review script: 0 pages wider than 390 px |
| Phone: with no matter chosen, the account button was pushed off the right edge | The matter button gives way (it was set never to shrink) | **Tested** | header measured: ends at 382 of 390 px |
| Reader: its five buttons ran off a phone and broke into two-line labels on a desktop; the title sat beside its breadcrumb on a phone | Buttons keep one-line labels and wrap as whole buttons; on a phone the title sits under its breadcrumb | **Tested** | screenshots |
| Reader, research and inspector side panels started 76 px lower than the page and never stuck while scrolling: liquid.css makes every paper `position: relative`, which left their `top: 76px` as an offset | The sticky rules name `.lw-paper` too, so they win; the panels line up and stick again | **Tested** | positions measured: all three columns start together, `position: sticky` |
| Drafts: in the narrow "Yours, with others" column (and on a phone) the date took a column, leaving titles a few words wide | A narrow list puts the date under the title (a container query, so it follows the list's width, not the screen's) | **Tested** | screenshots |

| A frontend test failed about one run in four: the due-diligence page loads lazily and its first lookup waited only 1 s | 5 s, as the other lazily loaded pages' tests already do | **Tested** | frontend **267 passed**, three runs in a row |

Left as they are, on purpose: wide tables (documents, drafting, meetings) scroll sideways inside their panel on a phone;
the reader shows a Markdown file's own characters (`#`, `>`), because quotes are matched to the document's exact text.

**Full browser journeys** (Playwright, fresh seeded backend on spare ports, real model): **37 of 38 passed** in 19 min.
The "conversation" journey failed once — the real model answered "The documents in this matter do not answer that." for
the liability-cap question late in the full run, when the matter also holds the earlier journeys' filed notes — and passed
when run on its own (23 s). Backend **1621 passed, 4 skipped** (all of `backend/tests`, `--network none`). Screenshots in
`review-2026-11-02/`.

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
| Answers | Declines shown as "partly supported"; 3 held-out abstention items too few to measure | **In progress** | rule + frozen held-out sets (EV6-EV20): over three runs golden, held-out, 3 and 5 pass; the unseen held-out 6 and 7 fail on declines, where the answer says "not stated" and then gives the related value ("appointed from 1 June 2026" for when it ends); showing such answers as declines is an owner decision with its cost measured (EV19) |
| Answers | Verified quotes from the wrong document ("the Kestrel agreement is governed by English law" on the NDA's and lease's words) | **In progress** | attribution, standing, party and concept checks (EV8-EV13): on the unseen held-out 5, party actions 6/6, cross-document traps 3/3, clauses for another purpose 4/6 withheld; held-out 4's pass on its first run did not repeat on the second (EV14); quote checks no longer fail a definition quoted without its quotation marks (EV18: held-out 3 quotes 0.921 → 0.984) |
| Evaluation | One run per set moves by 2-4 items on the model's sampling alone, so a single gate run cannot show a change of that size | **Done (proposed)** | EV15: `--repeat N` scores N runs together with their spread and the items that changed; thresholds.json proposes `gate_runs` 3 (a gate with fewer is refused); temperature 0 measured and rejected (runs move as much). The run count is part of the owner's threshold sign-off |
| QA 2026-10-29 | RAG-002: a verified termination quote shown as "Not verified" (written under §3's number) | **Tested** | quote cited to the one document that holds it; test_law_qa_20261029, test_law_evidence; QA_FIX_2026-10-29.md |
| QA 2026-10-29 | RAG-001: "These details are found in Section 2" listed as unsupported | **Tested** | pointer to a verified clause, and a statement its explicit attribution carries, are not unsupported; live and in history |
| QA 2026-10-29 | RAG-003: term end and non-renewal deadline not derived; "cl. 5" from a heading quote | **Tested** | research/dates.py: calculated, labelled, with assumptions, facts by clause (31 Jan 2027, 17 Dec 2026 from §§1, 6); a heading supports nothing; test_law_dates |
| QA 2026-10-29 | UX-001: generic decline hid "no penalty is mentioned" | **Tested** | the decline says what is missing, bounded, without a source; no unsupported list on a decline |
| QA 2026-10-29 | CHAT-001: "Which documents are available?" searched | **Tested** | intent `documents`, from the records under documents.read and per-document access lists |
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
| EV12 | Party actions and a clause for another purpose (2026-10-28): held-out 5 frozen **first** (`60fb2b7`: 16 unanswerable — 6 party actions asked without a possessive, 6 clauses for another purpose, 3 cross-document traps, 1 T06 — and 12 answerable look-alikes); then a party's own acts in `attribution.party_records` and a concept check (`concepts_elsewhere`, reason `not_in_cited`) (`4ec8bbc`), developed on golden and held-out 1-4 | replayed over stored development answers: 6 more correct declines (party acts 1, concepts 5), no correct answer lost; a word-level version was measured first and rejected (126 correct answers lost); golden trial 63/68, 7/7, 0 leaks; backend suite 1555 passed, 0 failed |
| EV13 | Gate on the clean commit `4ec8bbc`, each set once | golden **passed** (64/68, 7/7, 0 leaks, precision 0.927, quotes 1.0, p95 6.0 s); held-out **failed** (28/31, **abstentions 0/3**); held-out 2 failed (answers 12/15, abstentions 11/15); held-out 3 failed (quotes verified 0.857; abstentions 15/15); held-out 4, second run, failed (13/13, abstentions 10/15, precision 0.833); **held-out 5, first run: failed** — answers 10/12 (bar 11/12), abstentions **14/16 (passes**: party actions 6/6, other clause 4/6, cross-document 3/3, T06 1/1), 0 leaks, precision 0.879, quotes verified 0.944. Withheld on held-out 5 as other_party 3, other_documents 4; not_in_cited fired once (held-out 2). Held-out 5 misses: "not stated" followed by the MSA's renewal clause or the rent review date; two answerable items declined because the model hedged ("does not specify what the fee estimate excludes … only 'excluding VAT and disbursements'"; the board's size); one answer citing three sources for one fact. Held-out 5 has now been seen. Reports 20261028-211600 … -212354 |
| EV14 | Code or model? Nine unanswerable items declined on `832ea90` were answered on `4ec8bbc` (held-out 1, 2 and 4) | the change only adds reasons to withhold, so it cannot turn a decline into an answer; the answer texts differ (answers at temperature 0.1): this time the model followed "not stated" with a true, cited but unrelated fact (Thomas Reyes as General Counsel, for "who chairs Harrowgate's board"), which stands. Rerun four times each on `4ec8bbc`: **declined 22/36**, per item 0/4 to 4/4 (never: the CEO item and the side-letter termination item). A single run of a 12-16 item set moves by 2-4 items on sampling alone. Reports 20261028-212734 … -213058 (repeat runs, not gates) |
| EV15 | Sampling noise (2026-10-29): every development set run three times at answer temperatures 0.1 and 0.0 (`--repeat 3`; `973af20` / `c5dd663`, the same answer path) | from run to run, at 0.1 / 0.0: golden answers 64-66 / 65-66 of 68 and declines 7/7 every run; held-out declines 1-2 / 1-3 of 3; held-out 2 declines 10-11 / 9-11 of 15; held-out 3 12-14 / 11-13; held-out 4 11-14 / 12-14; held-out 5 12-13 / 12-15 of 16; answers move by up to 2. Items whose outcome changed between runs: 33 at 0.1, 29 at 0.0. Pooled scores of the two temperatures differ only within that spread, and the answer text varies even at 0.0 (the shared model server is not deterministic), so the temperature stays 0.1 (now `LEGAL_RESEARCH_TEMPERATURE`) and thresholds.json proposes `gate_runs` 3. The first held-out 2 run crashed: the per-person model limit (review F4) refused the runner's burst of quick declines; the runner now lifts it for its own throwaway backend and stops on any reply that is not a stream (`c5dd663`). Reports 20261028-233941 … 20261029-003627 |
| EV16 | The "not stated, but …" hedge (2026-10-29): held-out 6 frozen **first** (`840293f`: 15 unanswerable — 14 that invite "not stated" and then a true but unrelated fact, 1 T06 — and 17 answerable look-alikes, 6 worded to invite a wrong decline); then a rule in `_answer` (`8e11c31`): after a not-stated opening, a fact that stands is set aside when the documents it cites hold none of two or more of the question's own words | replayed over stored development answers: on one missing word 36 more correct declines and 29 correct answers lost (rejected); on two, 24 and 2. On the fresh three-run answers of EV15: 15 more correct declines, 2 lost. Not addressed: flat refusals of answerable questions by the model ("The documents in this matter do not answer that"), most of the remaining false declines. Backend suite 1559 passed, 0 failed |
| EV17 | Gate on the clean commit `8e11c31`, each set once, **three runs pooled** | golden **passed** (188/204, 21/21, 0 leaks, precision 0.95, quotes 0.981, p95 5.9 s; the rule declined the carve-out question in all 3 runs: "caps" and "exclusions" are not the MSA's words); held-out **passed** (83/93, **9/9**, was 5/9 and 6/9 before the rule); held-out 2 failed (answers 34/45, declines 35/45, precision 0.849); held-out 3 failed (quotes verified 0.921); held-out 4 failed (precision 0.882; declines 42/45); held-out 5 **passed** (32/36, 40/48); **held-out 6, first runs: failed on declines only** — 33/45 (bar 0.8), answers 48/51, 0 leaks, precision 0.917, quotes 0.961. Never declined: an appointment's end date ("appointed with effect from 1 June 2026"), which regulator clears the acquisition ("subject to regulatory clearance"), the rent set at review ("reviewed on 1 April 2030 to the open market rent"): the asked words are in the cited document, the asked value is not, which no word check can see. Held-out 6 has now been seen. Reports 20261029-004949 … -011454 |
| EV18 | Citation precision and quotes (2026-10-29): held-out 7 frozen **first** (`c699d35`: 16 unanswerable — 15 where the asked words are in a document but the asked value is not, 1 T06 — and 15 answerable look-alikes, 8 where other documents share the asked terms); diagnosis over the EV17 reports | **Quotes:** 29 failed; 16 were a definition quoted without the marks around its term ("Client Data means all data …"), the rest real misquotes ("law" for "laws", paraphrase inside quotation marks, a quote credited to another document). Quote matching now ignores quotation marks as it ignores Markdown (`7862077`). **Precision:** of 51 items citing an unexpected document, some cite a second document that also states the fact (Amendment No. 1 confirming the other terms stand, the minutes naming Petra Voss as General Counsel), which the frozen sets list as unexpected and stay as frozen; the rest add other documents' clauses for comparison to a question that names one document. Three runs of golden and held-out 1-6 with the quote fix and a prompt note for such questions ("answer from that document"): quotes verified up on every set (held-out 3 0.921 → 0.985, held-out 4 and 6 → 1.0), precision held-out 2 0.849 → 0.931, but declines down 8 on held-out 3-6. On the 103 questions that name a document, three runs each: declines 82/96 (EV17) → 76 (note) → 74 (a reworded note that also said to decline), answers 190 → 199/213. The note gave up declines for true comparisons, so it was taken out (`384e975`, prompt research-s2.5 again); dropping other documents' citations after the answer streamed was measured (held-out 2 0.849 → 0.891) and not taken. Backend suite 1560 passed, 0 failed. Reports 20261029-140425 … (development) |
| EV19 | The value hedge, an owner decision: "When does her appointment end?" — "The documents do not state when … She was appointed with effect from 1 June 2026 [1]" | measured on the EV17 gate answers: showing every answer that opens "not stated" as a decline with its passages would add 30 correct declines and show 31 correct answers as declines (their fact still visible beneath); declines would pass on every set (0.93-1.0) and answers would fall below the bar on golden (0.897) and held-out 3-6 (0.72-0.84). On held-out 7 after its gate (EV20): declines 25/48 → 38/48, answers 41/45 → 39/45. Not shipped. A narrower option: keep the status and say "Not stated in the documents — related passages" on such a card instead of "Partly supported: some sentences have no verified source" (counsel/cards.tsx), which describes neither case. |
| EV20 | Gate on the clean commit `384e975`, each set once, three runs pooled | golden **passed** (192/204, 21/21, 0 leaks, precision 0.941, quotes 0.982); held-out **passed** (85/93, 8/9); held-out 2 failed (answers 34/45, declines 35/45; precision 0.911 now passes); **held-out 3 passed** (39/42, 42/45, quotes 0.984, was 0.921); held-out 4 failed (precision 0.886: the second documents that also state the fact); held-out 5 **passed** (31/36, 39/48); held-out 6 failed (declines 33/45); **held-out 7, first runs: failed on declines** — 25/48 (bar 0.8), answers 41/45, 0 leaks, precision 0.976, quotes 0.987. Its misses: "not stated" then the related value (13 of 23), the same said with a named document as the subject ("The Calder & Wynne engagement letter does not specify an expiration date", which the not-stated check reads only for "the documents/sources"), and two wrong readings (the notice date of 3 September 2026 given as the day Pellucid agreed to buy; Schedule 4's address taken from the clause that refers to it). Held-out 7 has now been seen. Reports 20261029-151337 … -154335 |
| EV21 | 2026-10-29 functional QA fixes (`fix/legal-qa-20261029`, `8671958` … `0a73950`), gate on `a31e71b`, three runs pooled | golden, held-out and held-out 3 **passed** as before; held-out 2, 4, 6, 7 failed as before; held-out 5 failed by one decline (38/48, bar 0.8; it passed at 39/48): the model's not-stated hedge, with 0 of 60 citations re-pointed when rerun. Net against `384e975`: answers +2, declines +1, quotes verified 1.0 on six sets, 0 leaks. Backend suite 1588 passed, 0 failed; vitest 236 passed; before/after screenshots and the handoff in docs/echomind-legal/QA_FIX_2026-10-29.md. Deployed to the demo 2026-10-29 at the owner's request (`77c8389`), checked through the public address |
| L3 | Live, public demo (2026-10-27): meeting recorded in a real browser with spoken audio as the microphone (voice service clip), backend in the Law image with speech recognition on the CPU (the GPU is held by the host's models) | consent from 3 participants, then the transcript appeared live every ~5 s; Silent Assist raised **action**, **commitment to the other side** and **privileged or sensitive information with a counterparty** alerts quoting the words said (commitment and disclosure need the speaker marked: with "Not marked", only the action rule applies, by design) |
| L4 | Live, public demo: the conversation assistant asked aloud | the clip transcribed (first call 17.6 s including loading the model), answered from MAT-1042's documents with citations, read aloud by the voice service |

## How to run the demo locally (fictional data only)

Rebuilding the public demo's data (2026-11-02): `scripts/law/demo_seed.sh <new folder>`, a backend on it
(`DEMO_PREFIX=echomind-legal-test DEMO_API_PORT=8121 DEMO_DATA_DIR=<folder> DEMO_PUBLIC_URL=http://localhost:5181
scripts/law/demo_up.sh api`), then `demo_load.py --base http://127.0.0.1:8121/api --data-dir <folder>` and
`demo_enrich.py` with the same arguments; check it, stop that backend, and put the folder in place of `.law-data` before
`scripts/law/demo_up.sh api` for the demo (keep the old folder outside the repository).

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
