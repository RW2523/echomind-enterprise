# EchoMind Legal — implementation status

Statuses: **Not started**, **In progress**, **Tested** (executed tests pass, evidence given), **Blocked** (needs a person,
root or a decision). "Tested (Law)" = delivered earlier on `feature/echomind-law` with its own evidence
(docs/echomind-law/TEST_RESULTS.md) and re-run on this branch. Branch: `feature/echomind-legal-chat`.

## What a person can do now

Sign in → **Chat** opens: a greeting, the matter in the header, four suggestions. In one conversation a lawyer can
ask what matters today (a briefing from records, at most three items), ask questions answered only from the matter's
documents with verified, clickable citations, find potential inconsistencies between documents and resolve them with a
reason, build a chronology of the matter (every date the documents state, in order, each opening its exact words), open any
cited passage beside the conversation (re-verified against the stored text, highlighted), and reopen
past conversations from the rail. Matters, Drafts (what waits for review, your own drafts) and Activity (your own
actions, documents being read) are the other destinations; every existing Law page still works inside the shell, and
the firm's pages are in the account menu. Screens: [`screens/`](screens/).
Since 2026-11-07 a firm also runs its cases here: **New case** (details, team, other side, documents), **Import cases** from
a spreadsheet, a **Pipeline** of matters by stage, each matter's **stage**, **case updates** and **plan**, and "what's the
latest on this case" in the chat (section "Case management" below).

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
| LEGAL-018 | Advanced agents (Council, GraphRAG, MCP) | Council iteration 1 (manual Council) and 2 (discovery) on the review panel (off: `LEGAL_PANEL`): **Tested**; actions, autonomy: Not started; GraphRAG, MCP: Not started | section "Counsel Intelligence: the Council" |
| — | Purge covers the chat | **Tested** | `test_purging_a_document_removes_what_the_chat_kept_of_it`; purge verification counts the chat's tables |
| — | Conflict finder recognises delivery dates | **Tested** | `compare/facts.py` anchor `delivery`; `test_law_compare.py` green |
| — | Content-Security-Policy | **Tested** (config) | `frontend/nginx.law.conf`: `default-src 'self'` …, validated with `nginx -t`; the built bundle references no CDN |
| — | Strict TypeScript for new code | **Tested** | `tsconfig.counsel.json` (`strict`), also in CI |

## The external review's twelve points (2026-11-07)

The owner shared an outside "Immediate fix list" of 12 points. Each was checked against the code, then fixed and measured.
Measurements use the demo's 165-question chat set (`eval/law/run_chat_keys.py`, a private copy of the demo, run noise
about ±4), the golden and held-out 10 research sets, and the new stage-by-stage retrieval evaluation.

| # | Point | What was found | What changed | Status |
|---|---|---|---|---|
| 1 | Accuracy 126/165 | Rule 6 asked for "a few sentences" and answers stopped at 700 tokens, so multi-part answers dropped totals and dates. The chat-set runner lived only in a scratch folder and matched "36,000" inside "360,000" | Rule 6 asks for each part of the question, every figure, date, total and party asked; 1100 tokens. Runner in the repo, numbers matched whole | **Tested**: chat set 127 before, 126, 125 and 122 in three runs after: no change beyond run noise (two passes of the same 15 items on the same code gave 5 and 9, so the noise is larger than ±4); golden 63/68 → 65/68 answers, false declines 4 → 2; held-out 10 answers 0.48–0.54 → 0.64 (one run of 28) |
| 2 | Retrieval metrics | Only answers were scored | `eval/law/run_retrieval_eval.py`: Recall@5/10, MRR@10, nDCG@10 per stage, labelled by the answer keys' exact evidence, and the stage where each failed item lost its evidence | **Tested** (report `20261107-152311-retrieval-baseline`) |
| 3 | Context: definitions, exceptions | Amendments were already linked; definitions only when defined twice; no neighbouring passages or "subject to" clauses outside calculations | `LEGAL_RESEARCH_CONTEXT`: the rest of a cut clause, definitions of terms used, the clause a passage is subject to, each its own cited source, capped | **Tested**, off by default: 125/165 with it against 126 without, and slower |
| 4 | Claim support | A quote's presence was checked, not whether it supports the sentence; conditions ignored | `research/support.py`: each cited sentence against its passage (one model call, with the question and the parties' roles); an unsupported sentence is listed as unsupported; a condition the answer leaves out ("subject to", "unless", "provided that") is noted in the document's words, by rule | **Tested** (`test_law_support.py`, 9) |
| 5 | Cross-matter access | Per-feature tests only; the route-table test covered anonymous callers | `test_law_route_sweep.py`: 151 routes and the capture socket as an outsider, a walled lead, a revoked member and through another matter. Four gaps closed: a connector link on a closed matter, another matter's document id at the redlines route, a hidden deletion request in the review queue, one-chunk documents in the query-embedding cache. A fifth, outside the routers: a folder linked while the matter was open kept syncing into it after it closed; the sync now skips matters that are not open | **Tested** (merged from `legal-route-sweep`; `test_law_connectors.py`) |
| 6 | Streaming unverified text | A paragraph whose quotes failed was still streamed; streamed text looked final; a withdrawn answer was swapped silently | Such a paragraph is held back (`answer.held`); the stream is shown as a draft; a withdrawn draft says so; the research page no longer shows a ruled answer beside the model's | **Tested** |
| 7 | Offline vs regulatory feeds | No network mode; the feed fetch ran regardless | `LEGAL_NETWORK_MODE` disconnected (default) or restricted with `LEGAL_EGRESS_ALLOWLIST`, checked at start-up for every address the backend calls, shown to users; the feed fetch runs only in restricted mode with both hosts allowed | **Tested** (`test_law_edition_posture.py`) |
| 8 | Hardening | MFA on but could be turned off; the egress lock was lost at reboot; images unverified; restore test opt-in | The edition refuses MFA off outside a demo; `egress_lock.sh --install-service`; image manifests (`build_images.sh`, `verify_images.sh`, checked by `deploy_pilot.sh`); restore test every 7 days. Recovery drill on a copy of the demo data: 2,339 audit events, 73 of 73 versions open and match | **Tested**; disk encryption and installing the egress lock: **Blocked** (owner, root) |
| 9 | Reranker | — | Benchmarked: the MiniLM cross-encoder alone ranked evidence below BM25 alone (Recall@5 0.574 against 0.618); fusing its rank with BM25's gives 0.631, MRR@10 0.532 (`LEGAL_RERANK_FUSION=sparse`). bge-reranker-large did not finish one search in 4 minutes on the CPU (MiniLM 0.75 s) | **Tested**; a larger reranker needs the GPU |
| 10 | Ambiguous questions | Several documents of a kind were answered for each, silently; two parties sharing a name were not noticed | The answer comes with "Only <document>" or "Only <party>" asks and a note. Periods ("the rent", when it changes) are not detected | **Tested** (`test_law_ambiguity.py`) |
| 11 | DGX performance | No chat, upload, Council or GPU numbers | `load_test.py`: chat, uploads, Council, GPU, host CPU and memory sampling. On the private copy, 3 minutes per level, chat p50 / p95 to the checked answer: one person 6.1 / 16.3 s; two with uploads 9.0 / 15.2 s (searchable 5.0 / 8.0 s); three chatting with a Council run and uploads 12.9 s p95, Council 79–92 s, uploads 74–84 s behind the Council (one job worker). GPU about 94 % busy while answering; host memory 28–30 GB free. **Found:** at four at once the backend passed its 512-process limit (every CPU model call's own thread team, "libgomp: Thread creation failed") and restarted; `demo_up.sh` now caps each call at 8 threads and allows 2048 processes: peak 195, no restart. The model quota (20 requests a minute a person) refuses a person faster than that, as designed | **Tested** |
| 12 | Approval workflows | Covered by unit and browser tests, not one live chain | `scripts/law/workflow_check.py`: enquiry to matter, draft approval, export, wall, emergency access, audit chain, all PASS on a private copy; an author's own approval is now audited as a refusal and an ended break-glass grant listed as ended | **Tested** (merged from `legal-workflow-check`) |

Where the 38 failed chat items lost their evidence before these changes: 14 had it in the context and answered wrong or
incomplete, 11 were ranked out by the reranker, 7 had part of it, 5 are follow-ups (not measured), 1 was lost before
reranking. After them the remaining failures are mostly answers missing one detail the key asks for and answer-key wording
(e.g. "does not accurately state" for an expected "incorrect"); the key is unchanged so runs stay comparable.

Tests on the merged code: backend 1866 passed, 4 skipped (LibreOffice); frontend 273 passed, both TypeScript configurations clean; Playwright 39 of 39 (one journey timed out at sign-in under load and passed alone in 9 s).

## Case management and the full review before the webinar (2026-11-07)

Owner's request: add a case, upload documents and cases, a place to post what is happening in a case, plan the work in a
pipeline; then a full end-to-end review of every part with every error fixed. Commits 7306c0f and the review fixes after it.

| Item | What was built | Status | Evidence |
|---|---|---|---|
| New case | `/matters/new` in three steps: client (or a new one), the next free number, title, practice area, jurisdiction, track and starting stage, description, sensitivity; the team (colleagues, each checked against ethical walls on the client) and the other side (party register, so conflict checks find it); then documents. Also from the matter picker | **Tested** | `test_a_new_matter_takes_its_team_parties_track_and_stage`, `…walled_from_the_client…`; `casework.test.tsx`; journey "Case management" |
| Import cases | CSV (Excel saves as CSV; openpyxl is not in the image) of up to 200 rows; every row previewed with its problems (duplicate number, missing title, unknown stage, sensitivity); only good rows opened, the importer as lead, other parties added, one audit event per matter and new client | **Tested** | `test_a_csv_is_previewed_row_by_row…`, `test_only_lawyers_who_open_matters…` |
| Many documents at once | Several files, a browser folder or a zip (Documents → Upload many, the new case's last step, up to 10 in the chat composer). A zip is unpacked by the server with limits (200 files, 10× the upload size, a per-entry expansion ratio, nested zips and password-protected entries skipped); each file goes through the same checks and is recorded as its own import | **Tested** | `test_a_zip_imports_each_supported_file…`, `test_a_zip_is_refused…` |
| Case updates | Migration 0042 `legal_matter_updates`: kind, date it happened, text, linked document; append-only by trigger (only the link to a deleted document may be cleared); corrections are new updates naming the original; read by every member, posted by lead/attorney/paralegal; refused on closed matters; audit carries kinds and counts, never the text | **Tested** | `test_updates_are_posted…`, `test_an_update_is_never_changed…`, `test_an_update_records_the_past…` |
| Plan | `legal_matter_tasks` + append-only events: task, key date, hearing, filing deadline; due date, assignee (a member), stage; version-checked changes; a stage's standard tasks on request. Dates are a person's, never a model's (CLAUDE.md rule 7); shown in "Next deadlines" apart from document deadlines | **Tested** | `test_tasks_are_kept…`, `test_a_stages_standard_tasks_are_added_once` |
| Pipeline and stages | Three fixed tracks (`matters/pipeline.py`); a stage move (lead/attorney) is also an update and can record the day it happened (`effective_on`); `/pipeline` lists only visible matters, with lead, days in stage, next task, last update; existing matters got a track from their practice area | **Tested** | `test_a_stage_move_is_made_by_the_lawyers…`, `test_the_pipeline_shows_only…`, `test_a_closed_matter_takes_no…` |
| Chat: where the case stands | Intent `case_status` ("what's the latest on this case", "status update", "what stage…"); card built from the records by rule; "open tasks" shows the plan first | **Tested** | parametrised intent test, `test_the_chat_shows_where_the_case_stands…`; live card on the private copy |
| Demo data | `scripts/law/demo_casework.py`: stages, 3–7 updates and a plan for nine demo matters, dates counted from the run; idempotent; called by `demo_enrich.py` | **Tested** | private copy: 9 matters filled, a second run changed nothing |

**Review (private copy of the demo with this code, every demo role):**
- **Every-page crawl:** 8 roles (attorney, paralegal, managing partner, firm admin, records admin, intake, conflicts,
  reviewer), 506 pages (every firm page, every tab of every visible matter, a document reader each), desktop, phone and dark.
  - No 5xx, no page errors, no error text, no horizontal overflow at 390 px.
  - The only console errors were 403s for pages opened by URL that the person is not offered. The new case page made two
    of these calls itself for people who cannot open matters; fixed.
- **Playwright (38 journeys + the new one) on spare ports:** 33 of 39 passed first. The six failures were:
  - five tests out of date with the redesigned shell: the greeting, "Council", "Talk by voice" and the matter button's
    label. Updated;
  - one real: a ticked plan task stayed unticked until the server answered. Fixed to tick at once.
  - On rerun all six pass.
- **Fixed in the review:**
  - The e2e backend could not start: the evaluations importer refused the numbered held-out research reports
    (`heldout2`…`heldout10`). It now treats them as research sets (`test_each_suite_is_named…`).
  - Converting an enquiry onto an existing client did not check that the person may see that client; a client hidden by
    a wall now answers 404 (`test_an_intake_is_never_converted_onto_a_client_the_person_cannot_see`).
  - A worked-out figure's label that says who stated it ("Total claimed amount" for GBP 19,340 at the policy rates) is
    kept only when the step's own figures come from a sentence saying so; otherwise "Step n"
    (`test_a_label_saying_who_stated…`).
  - Seeded templates, playbooks and checklists showed "by someone"; they now say "from the starter set".
  - Icons in the plan, the pipeline cards and the chat card sat on their own line; the pipeline's track tabs ran off a
    phone screen.
- **Meetings (synthetic speech into the capture socket):**
  - The live transcript and partial line work: 5 segments and 54 words from 16 s of speech, and an action alert.
  - CPU speech recognition misheard "five hundred thousand dollars" as "five hundred", so the misstated cap was not
    flagged. Known limit; the demo uses the best measured settings (D-072).
- **Chat, the 165-question set** (private copy, real model, alice): 126 / 165, against 118 in the last full run.
  - 16 fixed, 8 lost. By kind: facts 38 → 41 / 45, follow-ups 4 → 7 / 12, not-stated 16 → 18 / 21,
    cross-document 11 → 13 / 16, calculations 13 → 10 / 16.
  - Re-asked, 3 of the 5 lost calculation and conflict items pass. The two that fail again give a correct answer short
    of what the key asks:
    - the register's class totals without the overall 10,520,000: the planner added each class but not the two
      together;
    - USD 55,800 without the May 20 and June 10 dates.
  - The calculation labels were unchanged in both, so this is run-to-run variance, not the label rule.
- **Deployed to the public demo 2026-11-07 (d0c7547):**
  - Migration 0042 applied on start; the case histories loaded; a reset snapshot taken.
  - Live through the public URL: `demo_health.py` READY.
  - Voice: a spoken question heard (13 words in 3.5 s), answered with the USD 250,000 cap, and read aloud (48 s).
  - Every run-sheet step: the case status, the 45-event delivery chronology, the liability cap, the inconsistency scan
    (15), the letter-before-claim questions, and GBP 19,340 for M-302.
  - The demo was then restored to the snapshot, so the checks left nothing behind.
- **New scripts:**
  - `scripts/law/demo_reset.sh snapshot|restore|status` puts the demo back after a rehearsal. Tested on the private copy:
    a row added after the snapshot is gone after a restore, and the API is healthy at the same URL.
  - `scripts/law/demo_health.py` is a pre-webinar check: containers, model, voice, public page, sign-in, a case status, a
    chat answer and disk.
  - `scripts/law/demo_check.py` takes `LAW_CHECK_BASE`/`LAW_CHECK_CONTAINER`.
  - Playwright takes `E2E_API_PORT`/`E2E_WEB_PORT`.
- **A stable public address, 2026-11-07: https://legal.echomind-ajace.com** (owner's request).
  - A named Cloudflare tunnel, `echomind-legal-demo`, on the owner's zone replaces the quick tunnel. The quick tunnel's
    random trycloudflare.com address changed every time it restarted.
  - It has one route: `legal.echomind-ajace.com` to the demo's nginx on 127.0.0.1:3125. Anything else answers 404.
    `law.` was left free for the main product's law edition.
  - Its credentials and config live in `~/.echomind-legal-demo-tunnel` (mode 700), outside the repository.
  - Production's tunnel for echomind-ajace.com is a separate tunnel with its own token, and was not touched.
  - Set up once:
    - the owner runs `cloudflared tunnel login`, which writes `~/.cloudflared/cert.pem` for the zone;
    - `scripts/law/demo_tunnel.sh setup legal.echomind-ajace.com` creates the tunnel, its DNS record and its config. It
      is idempotent and never overwrites an existing DNS record;
    - `scripts/law/demo_up.sh tunnel` runs it: read-only, every capability dropped, restart unless-stopped.
  - Restarting the tunnel keeps the address.
  - The API and voice take `DEMO_EXTRA_ORIGINS`, so they also accept the old quick-tunnel address while people still have
    it. `demo_reset.sh` keeps that setting when it restarts the API.
  - The old quick tunnel still runs as `echomind-legal-demo-quicktunnel`, set not to restart, so it ends at the next reboot.
  - Checked through the new address:
    - `demo_health.py` READY: sign-in, the case status card, and a cited chat answer;
    - the voice socket opens for both addresses, and a foreign origin is refused (403);
    - the API answers 401 without a session.

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

## "Courthouse": the redesigned workspace (2026-11-05)

The owner sent a UI concept and two photographs ("I don't want a dull UI … professional, not AI-made"). Built from them, with
every control doing something real and nothing the records do not back (`DESIGN.md`, rewritten for the new direction):

| Built | Status | Evidence |
|---|---|---|
| The room: the courthouse photograph behind the light workspace (faded into parchment where the work is; only above the tabs on a matter's pages), the gavel behind the dark one and at the rail's foot; local assets in `frontend/public/law/` | **Tested** | screenshots light/dark/390 px (session scratch `ui/final`) |
| Walnut rail: the AJACE mark and "EchoMind Legal", Chat, Matters, the current matter's Documents, Council and Transcription ("Live" while recording), Drafts, Activity; recent chats; the current matter and the person at its foot | **Tested** | `counsel.test.tsx` (rail links, the current-matter button), `keyboard.test.tsx` |
| Top bar without a band: search (⌘K), privacy, recording, a bell counting what waits for the person's decision | **Tested** | `counsel.test.tsx` |
| Chat: serif greeting by the person's clock, the conversation on one paper stage with the composer docked, answers under the AJACE mark | **Tested** | `counsel.test.tsx` (greeting, axe clean) |
| Context column (≥ 1361 px, hidden while the source panel is open): Context (matter, latest documents, questions found for the Council or the starters, talk by voice), Sources (the passages this conversation cites, each reopened by the server), Actions (waiting, deadlines not confirmed, inconsistencies open) | **Tested** | vitest suite; a document without dates no longer breaks the shared list |
| "Review panel" is called "Council" everywhere | **Tested** | `navigation.test.tsx` |

Not built from the concept, on purpose: the model-made "Risk level: High" meter and a "Council Reviewed" badge on ordinary
answers (scores and badges the records do not back), and "Workflows / Knowledge / Reports", which are not features. Frontend
**260 passed**; `tsc` main and strict clean.

## Counsel Intelligence: the Council (2026-11-05)

The owner asked to make the "Counsel Intelligence" idea workable and build it. The spec (four iterations, each with a gate) is
kept outside the public repository until the owner decides on an IP filing. It builds on the review panel (S4.5, D-057), which
stays off unless the firm turns it on (`LEGAL_PANEL`).

**Iteration 1, the manual Council.** Existing code touched: `panel/run.py`, `panel/routes.py`, `counsel/deliberation.py`,
`counsel/schemas.py`, the panel pages and the `council_summary` card. Migration 0040 adds `roles_json` and `depth` to
`legal_panel_reviews`; reviews asked before it read as the four default perspectives, focused.

| Built | How | Status | Evidence |
|---|---|---|---|
| Perspectives chosen for the question | Rules, not the model (`run.assemble`), first match wins: a challenge ("strongest argument against", "weaknesses", "opposing") → lead counsel, an opposing reading, an evidence investigator, the specialist; evidence (witness, statement, timeline, inconsistent) → lead, opposing, evidence, contract specialist; clauses (amendment, clause, obligation, deadline, breach) → lead, opposing, contract, specialist; strategy (settle, exposure, "should we") → lead, opposing, risk, specialist; anything else → the four default perspectives. Kept with the question, so a rerun with the lawyer's facts asks the same ones. "Opposing reading" is an analytical role that speaks for no real person | **Tested** | `test_rules_choose_the_perspectives_for_the_question_and_keep_them_with_it` |
| Challenge round | The opposing reading and the evidence investigator (else the litigator) read the first perspective's answer, only its sentences whose quotes verified, fenced as material, and object with quotes from the sources; "NO OBJECTION", or objections with no verified quote, count as none | **Tested** | `test_challengers_read_only_the_verified_sentences_of_the_first_answer` |
| Rebuttal at "deep" depth | Only when an objection cites a verified quote the challenged answer did not; the first perspective answers the objections, checked as any answer; the chair reads the reply in place of the first answer | **Tested** | `test_deep_adds_a_rebuttal_only_when_an_objection_found_something_new` |
| The chair | The moderator reads only verified sentences of the answers and objections, never the sources, and writes a one-sentence conclusion, at most three points (with the roles that support each; unknown roles dropped), gaps and the matrix. A figure in the conclusion or the points that no answer, objection or cited clause gave is flagged ("Check: 900,000 is in no answer") | **Tested** | `test_the_chair_reads_only_verified_sentences_and_its_conclusion_is_checked` |
| Inconclusive | Fewer than two verified quotes in the answers: no challenge, no chair, no conclusion; the page and card say so | **Tested** | `test_too_few_verified_quotes_end_the_run_as_inconclusive`; vitest |
| Evidence snapshot | The artifact's inputs pin each passage read (chunk, version, sha256 prefix) | **Tested** | same test |
| Conclusion first | Panel page: the chair's conclusion, what matters most (with roles), gaps, then the answers, the challenge round, the reply, the matrix (columns follow the roles that answered); the chat card leads with the conclusion; depth in the ask form and the chat form (third question) | **Tested** | `panel.test.tsx` (8), `counsel-cards.test.tsx` (12), axe clean; `test_the_chat_card_shows_the_conclusion_first`; `test_the_review_panel_is_queued_from_the_chat_when_the_firm_has_it` (depth stored) |
| Kept as it was | Missing facts stop the run and ask the lawyer; continuing without them is recorded; a partner other than the asker approves; the audit keeps counts, never the question; chosen documents; text addressed to an AI withheld | **Tested** | the 19 earlier panel tests, one updated for the challenge call |

Not changed from the panel: every perspective is the same model, and the notice says they are not independent experts. Cost:
the default four perspectives now make 6 model calls (4 answers, 1 challenge, the chair); a challenge question 7; "deep" up to
8. Opinions and challenges are stored in the review's artifact (versioned, visible only to those who may read it, approved
through the review route) rather than a separate table. Not yet measured with the real model: a live run waits until the shared
GPU is free of the RAG gate. Backend suite **1680 passed**, 4 skipped; frontend **259 passed**; `tsc` main and strict clean.

**Iteration 2, Council Discovery.** Existing code touched: `jobs/worker.py`, `counsel/discrepancies.py`, `obligations/routes.py`,
`panel/routes.py`, `counsel/briefing.py`, `counsel/schemas.py`, the panel page; new `council/discovery.py`. Migration 0041 adds
`legal_council_questions` and `question_id` on `legal_panel_reviews`. Settings: `LEGAL_COUNCIL_DISCOVERY` (on; only with the
panel) and `LEGAL_DISCOVERY_DEADLINE_DAYS` (30).

| Built | How | Status | Evidence |
|---|---|---|---|
| Questions from the records, by rule | Three detectors, no model: an unresolved potential inconsistency ("Are “15 May 2026” in Services Agreement and “20 May 2026” in Amendment No. 1 genuinely inconsistent?"), a sentence that amends another agreement's clause ("How does Amendment No. 1 change clause 4.2 of the Services Agreement?"), a deadline read from a clause, not confirmed, within the window ("What must be ready before … under clause 4.2 of …?"); each keeps its sources (document, version, characters) and opens at them | **Tested** | `test_questions_come_from_the_records_by_rule_and_the_model_is_never_called` |
| Suggested once; dismissals hold | Fingerprint = detector + exact sources; a dismissal needs a reason and holds until the sources change (a moved deadline is a new question); a finding resolved or confirmed makes its question stale | **Tested** | `test_a_question_is_suggested_once_and_a_dismissal_holds_until_its_sources_change`, `test_a_question_whose_finding_is_gone_becomes_stale` |
| Who sees a question | Only someone who may read every document it quotes (document ACLs too); at most three, highest score first (10 × severity + deadline urgency, fixed in the code) | **Tested** | `test_only_people_who_may_read_every_source_see_a_question` |
| Convening | A person with `ai.draft` convenes: a Council run asked in the question's words, linked both ways, roles by the same rules; the audit keeps lengths and kinds, never the words | **Tested** | `test_convening_puts_the_question_to_the_council_in_its_own_words` |
| When it runs | A `discovery.run` job after each document is read, after the register or the obligations change, daily for open matters, and on "Look again"; one queued run per matter covers them all | **Tested** | `test_discovery_runs_after_a_document_is_read` |
| Where it shows | The panel page ("Questions worth putting to the Council": priority, reason, sources, Convene / Dismiss with a reason) and the briefing ("3 questions worth putting to the Council") | **Tested** | `test_the_briefing_offers_the_questions`; `panel.test.tsx` (axe clean) |
| Off switch | Off with the panel or `LEGAL_COUNCIL_DISCOVERY`: no listing, 404 on the routes, nothing queued | **Tested** | `test_discovery_is_off_with_the_panel_or_its_own_switch` |

Not yet: model-checked conflict candidates in the background (iteration 4, with budgets and yielding to live work), and the
chronology detectors (single-source and hedged dates). Backend suite **1688 passed**, 4 skipped; frontend **260 passed**.

**The Council's interface, redesigned (2026-11-07).** The owner sent a reference image with six Council layouts and asked
for the Council to be "more clear and clean". Built from the ideas the data supports, on its own branch until the owner
decides. Existing code touched: `panel/run.py`, `panel/routes.py`, `counsel/deliberation.py`, `counsel/schemas.py`,
`counsel/orchestrator.py` (wording), the two panel pages, the `council_summary` card and its validator; new
`frontend/law/ui/council.tsx` and `frontend/law/theme/council.css`. Migration 0043 adds `progress_json` to
`legal_panel_reviews`.

| Built | How | Status | Evidence |
|---|---|---|---|
| The run records where it is | `progress_json`: the step (answers, challenge, reply, chair), the roles that have finished it and the steps passed through; never what anyone wrote (the answers stay in the artifact, under its visibility rules). Reset when facts are given; a failed run keeps the step it failed at | **Tested** | `test_the_run_records_each_step_and_the_roles_that_finished_it_never_what_they_wrote`, `test_a_focused_run_records_no_reply_and_an_inconclusive_one_stops_after_the_answers`, `test_giving_facts_starts_the_flow_again_and_a_failed_run_keeps_the_step_it_failed_at`, `test_the_progress_shown_is_a_known_step_and_known_roles_only` |
| Discussion flow | A stepper on the session page: Question, Initial views ("2 of 4 answered"), Challenge, Responses, Synthesis. While the run works it follows the recorded step; once it ends it reads the stored result. A step the run did not take says why ("Not part of a focused run", "Skipped: fewer than two verified quotes", "Skipped: most perspectives said facts were missing", "Not needed: no objection cited anything new"); a stop or failure is shown at its step. Each step opens its part of the discussion | **Tested** | `panel.test.tsx` (flow cases); real runs on a private stack (screenshots) |
| Overview | The Council's view first (the chair's sentence, or why there is none), then the question in the centre with the perspectives around it: role icon and name (never a face), what the role looks at (the run's own role description), its opening words (the first verified sentence of its answer, quoted), and facts about it: citations, "Objected to the … answer" / "No objection", "Says facts are missing", sentences without a source. While running, each seat says answered / answering now / waiting, and nothing it wrote is shown before the run ends | **Tested** | `panel.test.tsx` (axe clean) |
| Discussion | Answers, objections, the reply and the chair in order, as messages. Each citation marker is a gold chip; one chip per source passage under each message; uncited sentences collapsed ("Cites nothing: 2 sentences"), as on answer cards | **Tested** | `panel.test.tsx` |
| Cited passage | A chip opens the passage, read again from the document version on the server (the reader route, a recorded read), with the quoted words in gold, the verification in words, document, version, clause, page, who cited it, and "Open at the quote"; focus returns to the chip | **Tested** | `panel.test.tsx` |
| Evidence | Each cited document, its cited places (clause, page, the quote) and which perspective used each (initial view, objection, reply); a quote inside another quote of the same version is one place; quotes not found are counted, never listed | **Tested** | `panel.test.tsx` |
| Result | The Council's view with what matters most (supporting roles named), where they differ (one card per issue on which they differ, every position named; the whole matrix as a table under it), the facts still missing (the chair's gaps and the facts each perspective asked for), next steps, and the partner review that already existed | **Tested** | `panel.test.tsx` (axe clean) |
| Next steps into the plan | Suggested by rule from what the run left open (gaps, facts asked for, figures in no answer: "Confirm whether …", "Check the figure … against the documents"). The person edits each and adds it with the case-plan route (`POST /legal/matters/{m}/tasks`), so it is a person's task; "In the plan" once there. Only for the matter's lawyers and paralegals on an open matter; others see the list | **Tested** | `panel.test.tsx` |
| Council home | Ask form, the usual perspectives, how a session runs, questions found in the records, sessions with who sat on each and the step a running one is on | **Tested** | `panel.test.tsx` |
| Chat card | "Council · area", who sits on it (initials and name), the step a running one is on ("Step 3 of 5 · Challenge …"); the server schema and the client validator both take `roles` and `step`; "review panel" wording in the chat now says "Council" | **Tested** | `counsel-cards.test.tsx`, `test_the_chat_card_names_the_council_and_the_step_a_running_one_is_on` |

Not built from the reference, on purpose: the boardroom view (it needs people's faces), "Council settings" (there is nothing
to set beyond the area and depth already asked), a running "2 min ago" feed (the answers are shown only once checked), and
any confidence or "verified" badge. Keyboard: the views are WAI-ARIA tabs (arrows, Home, End), the steps and chips are
buttons; at 390 px the stepper becomes a list and the round table a column.

## The whole RAG system evaluated again, and what it found fixed (2026-11-03)

The owner asked for the RAG system to be evaluated again, end to end ("it should be smart and answer properly"). Two
measures: the Phase 2 gate on `16671f0` (every research set three times, EV22), and a new chat-level set of 66 items / 80
turns asked through the chat itself (intent routing, follow-up rewriting, cards) on a private copy of the demo data, each
answer read by hand: 61/80 right; no restricted-matter leak, injections refused, conflicting documents flagged. Retrieval
alone found the asked fact for 177 of 185 answerable research items; 30 of 47 answerable misses had the fact in front of
the model. Every fix below was developed on those sets; held-out 8 and a fresh chat set were written first by a separate
pass, frozen, and not read until the fixes were measured on them.

| Found | Fix | Status | Evidence |
|---|---|---|---|
| Follow-ups asked as written: the rewrite prompt said to return a follow-up that "already stands alone" unchanged, and the model returned "Was that late under the contract?" and "Does the draft SPA say the same?" as they were, so the documents were searched for "that" | The prompt names the words that point back (it, that, they, then, the same, the other one, "And …?") and shows three examples from other matters (`research/followup.py`) | **Tested** | offline on the 14 failed follow-ups: 12 resolved, stand-alone questions kept their meaning; `test_law_followup.py` |
| "How does the liability cap in the reseller agreement compare with the one in the MSA?" got the conflict scan's card about notice periods: any "compare" went to the scan | Conflict words still go to the scan, and "compare" does when the documents' facts are compared ("Compare the May 12 and May 15 claims", "the witness accounts"); a term in named documents is a question (`counsel/intent.py`) | **Tested** | `test_intents` |
| "Thanks, that's really helpful." and "What can you do for me on this matter?" were researched and declined | Small talk covers praise in other words and "what can you (help me with \| do for me) on this matter/case" | **Tested** | `test_intents` |
| A contract's parties by name, its clauses by role: "What subscription fee does Orbis pay?" declined over "The Customer shall pay the Provider …", "What does Harrowgate indemnify Voss-Lindqvist against?" never retrieved "Provider shall indemnify Client" (rank 9) | The roles a preamble defines (`attribution.roles_of`: "Harrowgate Systems Inc., a Delaware corporation ("Provider")") are read once per version; the search reads "Harrowgate (Provider)" (`with_roles`, from the documents that define every party named), and the model is given the opening passage of each source document that defines a party the question names (at most two, read under the same scope and version policy as a search, quoted and checked like any source; a note only then; prompt research-s2.7). The first version (S2.6) put the roles in an attribute (`parties="Crestmoor Estates Ltd is the Landlord"`); the model quoted it, a quote of no document, and quotes verified fell on four sets (held-out 1.0 → 0.838) | **Tested** | indemnity clause rank 9 → 2; `test_law_evidence.py` (opening given, its quote verified, none for a question naming no party) |
| A 600-character window of an 896-character notice left out "Please direct any questions to our General Counsel, Thomas Reyes" and "Your account team will not change" | Research gives the model each passage whole (up to 1,800 characters around its best window); the window still ranks it and marks the sentence the reader opens on; drafting, Silent Assist and playbooks keep exact windows (`retrieval/search.py` `wide`) | **Tested** | `test_law_evidence.py` |
| "The supply agreement with Alder Retail" answered GBP 120,000 with a verified quote from the other supply agreement (Northwind and Tidewater): titles did not tell them apart | A document named by its party ("the agreement with Alder Retail", "Harrowgate's MSA") also searches the documents that party is party to; a sentence that names it and cites a document that party is not party to does not stand, and an answer that rests only on such documents is withheld (`attribution.named_parties`, `routes._answer`) | **Tested** | `test_law_evidence.py` |
| "The consultancy agreement … is governed by the laws of Sweden.\n\n[2]", [2] being the reseller agreement: the sentence was unsupported, the bare marker supported nothing, the card read "partly supported" | A bare marker in a paragraph of its own cites the sentence before it, so the claim is checked as one (withheld: other documents); an answer with nothing standing is "unverified", never "partly supported" | **Tested** | `test_law_evidence.py` |
| "The documents do not state when her appointment ends. She was appointed from 1 June 2026 [1]" read "Partly supported: some sentences have no verified source", which describes neither (EV19's narrower option) | Such an answer's card says "Not stated in the documents: related passages are cited" (`not_stated` on source_answer; a decline can never carry it) | **Tested** | `test_law_counsel.py`, `counsel.test.tsx` |

Not changed: the decline rules (EV16), the threshold for what counts as declined, the quote checks. Retrieval measured as
research now searches: the asked fact in the passages for **182 of 185** answerable research items (was 177), none lost.

Measured on the unseen sets (EV23). **The chat set** (60 conversations, 80 turns, same questions and data copy, read by
hand): 44/80 right on `16671f0`, **59/80** on the fixes; no citation outside the matter on either. Fixed: compare routing
(3), follow-ups (5), and six single answers; worse: one calculation (5% where 5 × 5% was asked) and one "do not specify"
before the right clause. **Held-out 8** (role-party and cross-document questions): answers 25 → 30 of 48, declines 30 →
29, trap values in answers 11 → 6, no leak; still far below the bars. **The gate as a whole is level**: answers 533 →
532 of 603 and declines 268 → 270 of 354 over every set, where about one item in six changes between the three runs
alone. Quotes verified 0.96-1.0 on every set. The single-question gate does not ask follow-ups or route chat, which is
where the fixes moved most; what it still fails on is mostly the model's "not stated, but …" answer (EV19), which the
card now labels honestly but which is still scored as an answer that is not a decline. Held-out 1-8 and both chat sets
have now been seen; another round needs a new blind set first.

## The remaining failures fixed, measured on a new blind set (2026-11-04)

The owner asked for a new blind set and the remaining issues fixed. A separate pass wrote held-out 9 (32 research items)
and a third chat set (60 conversations, 79 turns) and both were frozen (`aadab70`) before any fix was measured on them;
every fix below was developed on the sets already seen. Of the last gate's 155 failures, 61 were answers that opened
"the documents do not state …", then cited something related, and so were not declines; 44 were the reverse, the model
saying "not stated" before the answer or instead of it.

| Found | Fix | Status | Evidence |
|---|---|---|---|
| "Not stated, but …": a decline with context ("The lease does not say whether the deposit earns interest. The deposit is GBP 21,000 [1]") and a hedged answer ("The documents do not state who Harrowgate's General Counsel is … Thomas Reyes is the General Counsel [1]") read alike (EV19) | The passages the answer's facts cite are asked one narrow question with no answer of their own to defend: what is asked, which words give it, YES or NO (`research/stated.py`). NO declines; YES keeps the answer and drops the "Not stated" card label; a YES counts only with words the passages hold; no verdict changes nothing. Asked only of answers that open "not stated" or with a named document's silence ("The Northbridge NDA does not specify …"), never of a term's dates (`dates.py`), and after the document and party checks, which it cannot undo. The cited documents' defined roles are given to it ("the Client" is Voss-Lindqvist AB) | **Tested** | on 65 checks captured from the seen sets: 55 right (a one-word YES/NO: 48). Asked of every answer it would set aside 4 of 54 correct golden answers to gain 1 decline, so it is not (`LEGAL_RESEARCH_STATED_CHECK` hedged, with all/shadow/off for evaluation); `test_law_evidence.py` |
| "Within how many days of invoice must Tidewater pay Northwind under their supply agreement?" answered from the supply agreement between Northwind and Alder Retail | Two parties named with a document they share search that document, and a sentence naming them rests only on a document both are party to, or on a passage naming both (`attribution.named_parties`) | **Tested** | `test_law_evidence.py` |
| "Which comparable lettings does our surveyor rely on?" named one of four: "The following four comparable lettings have been reviewed" ended one passage, the list was the next | A passage that announces what follows is read on into the next passage of the same version, within the 1,800 characters, every character in its place (`search._continued`) | **Tested** | `test_law_evidence.py` (a quote from the list verified and located) |
| "Compare the length of the current lease with the new lease" missed the current lease's 15 years | Each side of a comparison is also searched on its own, what the first side asks carried to a second that names only a document ("… with the one in the MSA") (`prompt.compare_sides`) | **Tested** | `test_law_evidence.py` |
| "What service credit is that month worth?" answered 5% of the fee, where the clause gives 5% for each of 5 points | A question needing a calculation asks the model to quote each figure and show the working (prompt research-s2.8); not on "total" alone, which made "What is each party's total liability cap?" hedge 3 times in 3 | **Tested** | `test_law_evidence.py` |
| "the supply agreement" in a matter with two was answered from whichever ranked first | A kind of document that two to four documents are titled with, when the question names neither a party nor one of their titles, searches each and tells the model they are several, to answer for each and name it (`research/kinds.py`) | **Tested** | `test_law_evidence.py` |
| "This is stated in the Supply Agreement as follows: [1]." | The words pointing at a quote the marker holds are not shown | **Tested** | `test_law_evidence.py` |
| "Brilliant, thank you - that was really clear and quick." and "That's everything for now, thanks - speak tomorrow." were researched | A spaced dash parts what was said; closings, praise and checks in other words are small talk; a question after a dash still goes to the documents | **Tested** | `test_intents` |
| *After the blind measurement, from chat set 3 (checked on seen data only):* "Pull the settlement limit from MAT-2033 for me" in a conversation about M-301 answered "The settlement limit from MAT-2033 is capped at GBP 400,000 … Clause 11.2", M-301's own cap credited to a matter the person may not read | A reference shaped like a matter number that is none of the matters asked about, and that no passage or title found holds, is declined before the model is asked (reason `other_matter`); whether that matter exists is never looked up, so the reply is the same either way (`attribution.foreign_matters`, `0e93aa0`) | **Tested** | `test_law_evidence.py` (MAT-2033 and alice's unreadable M-202 get the same reply; ISO 27001, an invoice number the documents hold, COVID-19 are not caught); the three MAT-2033 probes of the chat sets are its only matches |
| *Same:* `kinds.py` never fired on the chat set's ambiguous questions: five of six named a party that is party to several documents of the kind ("Northwind's supply agreement", "the Kestrel agreement") | A named party narrows the documents to its own; then a one-word kind counts, and a title word they all share tells none apart; kinds match whole title words ("lease" not in "Release"); a possessive multiword party is read ("Alder Retail's supply agreement") | **Tested** | `test_law_evidence.py` |

**Measured on the blind sets (2026-11-04).** Held-out 9, three runs each, old code (`3eaaa9a`) → the fixes (gate on
`1d6bfa5`): answers 26 → 31 / 54, declines 24 → 25 / 42, false declines 22 → 22, leaks 0 → 0, citation precision
0.951 → 0.974, quotes verified 0.986 → 0.987. **Both fail the bars** (0.85 / 0.80). By kind: lists 3 → 6 / 12, role-party
6 → 9 / 18, facts stated indirectly 17 → 16 / 24, values the documents lack 12 → 13 / 24, another document's value 6 → 6
/ 12, the forbidden matter 6 / 6 both. (Latency is not compared: the old-code run shared the model with other runs.) The
not-stated check is now the main cause of the false declines: on held-out 9 its NO declined 13 of the 54 answer runs —
the party named where the clause uses its role (Northbridge, the NDA's "Recipient"; Harrowgate "as the vendor", the MSA's
"Provider"), "may the lessor simply refuse" against consent "not to be unreasonably withheld", rent due on "the usual
quarter days". Over every set its NO was right 65 times and wrong 34, its YES right 60 times and wrong 11. A reworded
check (roles in other words, a person's rank, the document's own general words, a limit on a right) replayed on all 226
captured checks did worse (seen 187 → 183 right, held-out 9 7 → 5) and was not kept. Another document's value is still
counted wrong when the model's text names it while declining ("The Northbridge NDA specifies the courts of London …, but
this document is not the Unit 7 Lease"): the card shows the decline, the streamed text showed the value.
Chat set 3 graded by hand, old → new (pass / soft / fail of 79 turns): 45/8/26 → 50/7/22; automatic grader 43/8/28 →
50/5/24. Small talk 3 → 6 is not blind: the brief to the set writer suggested its phrasings; without it 42 → 44 of 73.
Citations outside the matter 0 both; latency p50/p95 4.38/10.52 → 4.98/11.26 s.

**The seen sets** (gate on `1d6bfa5` against `04f8d1e`; answers, declines): golden 190 → 188 / 204, 21/21; held-out
1 88 → 89 / 93, 3 → 8 / 9 (**now passes**); 2 36 → 34, 33 → 40 / 45; 3 40 → 37 / 42, 41 → 45 / 45; 4 34 → 35 / 39, 40 → 43
/ 45; 5 32 → 30 / 36 (**now fails**, 0.833), 41 → 44 / 48; 6 45 → 45 / 51, 37 → 39 / 45; 7 37 → 41 / 45, 25 → 32 / 48; 8
30 → 30 / 48, 29 → 28 / 48. Over the nine: answers 532 → 529 / 603, declines 270 → 300 / 354, false declines 42 → 52;
leaks 0. Passed: golden, held-out 1, 3, 6 and the compare, assist, minutes, transcription and injection suites. Latency
p50/p95 2.8-4.2 / 5.4-9.7 s.

**After the blind measurement** (the two fixes marked so above, `0e93aa0`, seen data only): the 8 chat-set-3 items they
address, automatic grader 2 → 4 / 8 (the MAT-2033 question declined; "the date of the Kestrel agreement" answered with the
agreement's and its amendment's dates); still wrong there: "Which law governs Voss-Lindqvist's agreement with
Harrowgate?" declined, one of Harrowgate's two letters dated, the April credit's two figures, Beta Corp's merger.
A kind no longer counts a document titled as made to another ("Amendment No. 1 to the Kestrel storage agreement"), which
had made 16 seen-set questions about the Kestrel storage agreement ambiguous (`326a20d`); replayed over every research
question of the eval sets, the two fixes now change none of them. A golden run on `0e93aa0` (whose changes touch no golden
answer) gave 182/204 against 188/204 the hour before: on the same answers a run of three varies by ±6 items.

**The final gate** (on `65e4e57`, against the gate on `1d6bfa5`; answers, declines): golden 188 / 204 both, 21/21; held-out
1 89 → 88 / 93, 8/9; 2 34 → 36, 40 → 39 / 45; 3 37 → 38 / 42, 45/45; 4 35/39 both, 43 → 45 / 45; 5 30 → 32 / 36 (**passes
again**), 44 → 47 / 48; 6 45 → 47 / 51, 39 → 37 / 45; 7 41 → 39 / 45, 32 → 28 / 48; 8 30/48 both, 28 → 39 / 48; 9 31 → 35 /
54, 25 → 29 / 42. Over the nine seen before this round answers 529 → 533 / 603, declines 300 → 309 / 354, false declines
52 → 44; with held-out 9, false declines 74 → 59, trap values in answers 15 → 10, leaks 0. Passed: golden, held-out 1, 3,
5, 6 and every other suite; failed: held-out 2 (answers 0.80), 4 (precision 0.882, quotes 0.918: three quotes the model
reworded, one copying "Calder &amp; Wynne" as the source is escaped for it, one dropping the `("Client")` of the opening,
refused by the verifier and shown unverified), 7 (declines 0.583: declines that open with the related context, "The
supervising partner … is Helena Marsh … There is no information …"), 8 and 9. Latency p50/p95 2.5-3.9 / 5.4-7.4 s. No
question of the gate was declined as another matter and the replay shows the other changes route none differently, so the
two gates differ only by the model's run-to-run variance: about ±8 answers and ±13 declines over the gate, ±4 within one
set (held-out 9: 31 → 35 with nothing on its path changed). The blind gain on held-out 9 (answers +5, declines +1) is
within that, so it does not show the fixes helping beyond doubt; the false declines the not-stated check causes are still
the largest class.

## The not-stated check's wrong declines, measured on a new blind set (2026-11-05)

The owner asked for a new blind set and the not-stated check's wrong declines fixed. A separate pass wrote held-out 10 (44
research items: 12 stated in other words than asked, 12 asked by a party's name where the document uses defined roles, 4
lists; 10 where a related value is stated, 4 cross-document traps, 2 T06) and it was frozen (`8e807da`) before any fix was
measured on it; the fixes below were developed on the sets already seen (captured checks replayed, search wordings
compared, seen-set runs).

| Found | Fix | Status | Evidence |
|---|---|---|---|
| The check copied the right words and ended NO: "Did the board propose an interim dividend?" over "The board resolved not to propose an interim dividend"; "does not include tax advice", "exclusive of VAT", consent "not to be unreasonably withheld". NO was the question's answer, not whether it was given | The verdict is GIVEN or NOT GIVEN, and the check is told that words settling a yes-or-no question give it whichever way they settle it; a YES or NO reply is no verdict (`research/stated.py`) | **Tested** | replayed on 244 checks captured from the seen sets: right 193 → 211, wrong NOs 29 → 13, wrong YESes 13 → 16 (a replay varies by about ±3); `test_law_evidence.py` |
| A mutual NDA names its parties "VL" and "Northbridge", then speaks only of "the Recipient" and "the Discloser", roles it defines for "a Party" and "the other"; the check was told no role for Northbridge | Roles whichever party takes are read from the sentence defining them and given to the check with it (`attribution.taken_roles`) | **Tested** | `test_law_evidence.py` |
| "May Northbridge use what Voss-Lindqvist tells it for some other project?" never retrieved "The Recipient shall use Confidential Information only for the Purpose": the names drew the search to passages naming the companies | A question mentioning a party is also searched with the parties put as "the party", among the documents they are party to, two passages added after the others (`attribution.without_parties`, `routes`) | **Tested** | on the 16 seen role-party questions the search missed the clause 4 times; asked without the names, all 4 at rank 0-1; `test_law_evidence.py` |
| An opening passage found by a search lost the prompt's note on openings | An opening already among the sources is marked as the opening | **Tested** | `test_law_evidence.py` |

**Measured on the blind set.** Held-out 10, five runs each, old code (`8e807da`) → the fixes (`93b2f92`): answers 53 → 76 /
140 (0.38 → 0.54), declines 54 → 53 / 80, false declines 70 → 40, leaks 0 → 0, citation precision 0.912 → 0.962, quotes
verified 0.988 → 0.972. Answers per run 9-12 / 28 before, 14-16 after: the ranges do not overlap, so unlike held-out 9 the
gain is beyond the run-to-run variance. **Both still fail the bars** (0.85 / 0.80). By kind: stated in other words 19 → 30 /
60, role-party 29 → 41 / 60, lists 5 → 5 / 20, related value stated 29 → 32 / 50, cross-document traps 20 → 15 / 20, T06 5 →
6 / 10 (0 leaks both). Of the 5 cross-document misses, 3 are declines (reason `other_documents`) counted wrong because their
text names the other document's value, and 2 are answers. Wrong answers by cause: the check's verdict 34 → 23 runs, the
model's own decline 36 → 17, answered without the asked fact 17 → 24 (5 of them kept by GIVEN). (Latency is not compared: both runs shared the
model with other runs.) Three result lines of the old-code run were seen once while checking its progress; nothing was
changed after.

**The gate** (on `a578891`, which changes only a test, against the gate on `65e4e57`; answers, declines): golden 188 → 190
/ 204, 21/21; held-out 1 88 → 89 / 93, 8/9; 2 36 → 38, 39/45 both; 3 38 → 39 / 42, 45/45; 4 35 → 37 / 39, 45 → 42 / 45; 5
32 → 34 / 36, 47 → 41 / 48; 6 47 → 46 / 51, 37 → 40 / 45; 7 39 → 44 / 45, 28 → 31 / 48; 8 30 → 37 / 48, 39 → 31 / 48; 9 35
→ 39 / 54, 29 → 26 / 42; held-out 10 40 / 84, 30 / 48 (13-14 / 28 a run, against 14-16 in the five runs above and 9-12 on
the old code). Over the ten sets seen before, answers 568 → 593 / 657, false declines 59 → 47, declines 338 → 324 / 396;
trap values in answers 10 → 10, leaks 0. Passed: golden, held-out 1, 3, 5, 6 and every other suite; failed: held-out 2
(answers 0.844, one item short), 4 (precision 0.829, quotes 0.947; it failed on both before, and its precision misses are
mostly a second right source the scorer does not expect: "Who is Voss-Lindqvist's General Counsel?" cited from the
engagement letter and the minutes), 7 (declines 0.646), 8, 9 and 10. The answers gained are beyond the run-to-run variance
(+25 against about ±8 over the gate); the declines lost (-14) are at its edge (±13). Of the 14 more wrong declines, 3 are
declines counted wrong for naming another document's value (10 against 7). Run again twice, the five seen items that fell
furthest declined 7 times in 10, and the check said NOT GIVEN in all six checks traced: they changed with the model's
wording, not the verdict. What stays wrong there is an answer that opens with the related fact ("The documents state that
VAT is added at the applicable rate to Calder & Wynne's fees. However, they do not specify the percentage"): the check is
asked only of answers that open "not stated", so never of these. Latency p50/p95 3.0-4.9 / 6.2-9.3 s (the models were
shared). Backend suite 1674 passed, 4 skipped.

**Not done.** Every bar is still unmet on six sets. On held-out 10 the largest causes are now an answer without the asked
fact (24 runs), the check's verdict (23) and the model's own decline (17); lists are 5 / 20. Left as found: answers that
open with a related fact are not checked; a verified quote introduced by a colon is not shown ("The source states: [1].",
15 of about 1,400 answers); a decline whose text names another document's value is scored wrong although it is shown as a
decline; one quote copied "&amp;" from the escaped prompt.

## Retrieval and citation brief, new demo data and an end-to-end run (2026-11-05)

The owner sent a "Retrieval & Citation Fix Brief" (P0: a question naming the Master Services Agreement was answered from the
advice memo's misquote of it, "net sixty (60) days" where the MSA says thirty; P1: four sentences quoting clauses 4.1, 4.2
and 4.3 shared one citation whose span was 4.1), then asked for full demo data and an end-to-end test of the chat on it.
Nothing below hard-codes a document, title, amount or clause; verification, authorization and fail-closed behaviour are
unchanged.

| Found | Fix | Status | Evidence |
|---|---|---|---|
| P0: the memo's passages outranked the MSA's own for "the payment terms in the Master Services Agreement", and its quote answered for the MSA | A document the question names is searched on its own and put first (`LEGAL_RESEARCH_NAMED_PASSAGES`), without the words naming it; another document's account of it is marked `secondary`; a title's own words come before "on/about/to/under …" ("Advice memo on the MSA" is a memo) | **Tested** | `test_a_named_document_is_searched_and_put_first…`, `test_inside_a_named_document_the_words_naming_it_are_not_searched_for` |
| P0: a quote another document gives as the named one's, which that one does not hold | Rule-checked against the named document's text, also in a memo "on" it that quotes "Clause 4.2" without naming it; the answer says so ("Checked against the documents: …"), the nearest passage of the named document is added, and a misquoting document cannot answer for it | **Tested** | `test_an_answer_about_the_named_document_from_another_documents_quote_of_it_does_not_stand`, `test_a_memo_on_the_named_document_quoting_its_clause_by_number…` |
| P1: one citation for several quoted places | One citation per verified span, each with the sentences it supports; each marker in the answer opens its own clause | **Tested** | `test_each_place_an_answer_quotes_has_its_own_citation…`, `answer-citations.test.tsx`; live: 4.1, 4.2, 4.3 open their own words; bob gets 404 |
| A clause named by number was answered from letters describing it | The clause's own passage, from the named document or the matter's contracts, is read first | **Tested** | `test_a_clause_named_by_number_is_read_from_the_contract_first` |
| End-to-end run: requiring the named document's own text withheld answers resting on minutes or a draft amendment that name it | Such a passage answers again, unless its document misquotes the named one | **Tested** | second half of `test_a_memo_on_the_named_document…` |
| End-to-end run: a list cut at "(a) … within the limits in paragraphs (b) to (d);" | A passage ending on a list item's ";" is read on into the next one | **Tested** | `test_officer_initials_are_searched_as_written_out_and_a_list_cut_at_a_semicolon_is_read_on` |
| End-to-end run: "the CFO" never matched "the Chief Financial Officer"; "Fjellstad Distribution AS" and "APEX INDUSTRIES, INC." were no parties; "THE PARTIES AGREE AS FOLLOWS" was one | Officer initials are searched written out; AS/ASA/Oy/AG/ApS and ", Inc." end a company name; capitalised boilerplate is not a party | **Tested** | same tests |
| Owner, after the deploy: "Run a definitions check on the Supply Agreement (DEMO)" went to research | A `definitions` intent runs the document page's rules-only check (S3.5) on the document named (its full title first) or chosen, keeps it like a page run (audited `check.definitions`), and shows each finding as a cited line opening its exact words, with "Open the definitions check"; with no single document named it asks which; "How is Price defined?" stays research (`counsel/definitions_check.py`) | **Tested** | `test_a_definitions_check_asked_in_the_chat_runs_the_rules…` (no model call, 4 verified citations, owner-bound), `counsel.test.tsx` (page path) |
| "Which price applies to Products?" was declined with both definitions of "Price" in its sources | A term the question asks about that the answering document defines twice, with different meanings, is found by rule: both definitions are read, the model is told the document is ambiguous, a note says so; if the model still declines, the two definitions, quoted and verified like any quote, are the answer | **Tested** | `test_a_term_the_document_defines_twice_is_answered_with_both_definitions…` |
| "What fee arrangement did Voss-Lindqvist ask Calder & Wynne for?" read two memo passages, never the engagement letter's fee paragraph | The search without the parties' names keeps its best two and the best passage of each of up to three documents the named parties share | **Tested** | `test_each_document_the_named_parties_share_gives_its_best_passage` (fails without the change) |
| The misquote check paired '(the "Agreement") … (the "Supplier")' as one quotation, and read a contract's sentence about itself as quoting its sister agreement | Quotation marks pair only around a quote; a document naming itself quotes nothing | **Tested** | `test_quotation_marks_pair_around_quotes_not_between_defined_terms` |

**Live check of the brief on a private copy** (five runs each): the memo's net sixty never answered for the MSA (5/5), net
thirty and 1% interest given (5/5), USD 18,500 3-4/5, "monthly in advance" 1-2/5 (the model's selection); "Does the advice
memo state the MSA's payment terms correctly?" contrasts sixty with thirty (3/3).

**Demo data** (`eval/fixtures/demo/`, all fictional, loaded by `scripts/law/demo_load.py`): M-101 (8 documents), M-114 (10),
MAT-1042 (12) and MAT-2033 (1), each with a presenter guide and an answer key whose every evidence string was checked against
its document (59 + 34 items), plus the 72 items of the M-301 … M-306 sets.

**End-to-end run through the chat** (alice, 165 questions, private copy, each answer scored against its key): 107 → 111 /
165 after the second round of fixes (M-101 16 → 20 / 29, M-114 17 → 20 / 30, MAT-1042 21 → 20 / 34, M-301 … M-306 53 → 51 /
72). 12 items fixed, 8 lost; the 8 rerun or read as run-to-run variance of the model (calculations, a signer named from the
email instead of the amendment), so the overall gain is within it. By kind: facts 38 / 45, not stated 17 / 21, role-party 16
/ 22, conflicts 6 → 9 / 14, calculations 9 / 16, cross-document 9 / 16, follow-ups 7 / 12, lists 4 / 9, amendments 1 / 4,
definitions 0 / 2, chronology 0 / 2. **After the definitions and fee fixes** (rows above; M-101, M-114, MAT-1042 again, 93 items): 60 → 65 / 93 (M-101 22 / 29, M-114 19 / 30, MAT-1042 24 / 34); the definitions check and the fee question now pass; "Which price applies?" gives both definitions and says the document is ambiguous. **Not done:** chronology (the card showed the matter's first 60 events, not the ones asked about) and calculations (the model's arithmetic): fixed in the next section; lists still depend on the model. Backend suite 1700 passed, 4 skipped (1696 before the definitions and fee fixes); frontend 262 passed; `tsc` clean. Golden research set on this code: answers 63 / 68, declines 7 / 7, quotes 1.0, citation precision 0.984 (the gate on
`a578891`: 190 / 204 and 21 / 21 over three runs), report 20261105-155937; the held-out sets were not rerun.

## Chronology and calculations from the end-to-end run (2026-11-05)

The owner asked for the chronology and calculation failures of the end-to-end run to be fixed. Nothing below names a
document, title, amount or clause of the demo data; verification, authorization and fail-closed behaviour are unchanged.

| Found | Fix | Status | Evidence |
|---|---|---|---|
| "Give me a chronology of the delivery, from dispatch to acceptance" showed the matter's first 60 of 174 events and never reached June | A chronology of something is the matter's chronology narrowed to it: the events in the passages a search for it finds, every message date (with its subject) of a thread one of them is in, and, for a range, those from the first event naming its start to the last naming its end; the summary says what it is about and how many the whole chronology has; "Build a chronology" is unchanged (`counsel/chronology.topic`, `_narrow`) | **Tested** | `test_a_chronology_of_something_is_the_matters_events_narrowed_to_it_and_its_range`, `test_a_chronology_of_something_names_it_and_its_range`, `test_a_chronology_of_the_whole_matter_has_no_focus` |
| "How did the delivery date come to be changed? Give the key dates." was answered from the amendment alone | "key dates", "key events", "dates in order" ask for a chronology, narrowed to the question | **Tested** | `test_a_message_date_reads_with_its_subject_and_a_thread_found_gives_every_message` |
| A message's date showed as "Date: May 4, 2026, 11:47"; a table row's date showed the whole table from its header | The date line is shown with the subject line that follows it; a dated table row is that row | **Tested** | same test; `test_header_dates_ranges_and_headings_read_as_people_write_them` still passes |
| Calculations: the model quoted figures and left the arithmetic out (the August service credit, 10% of USD 18,500), did it wrong (21 days after 1 October 2026 given as 21 October) or applied the clause in part (a 25% cap per month left out) | Figures are worked out by rule (`research/working.py`): a planning call writes the arithmetic as formulas over the numbers the sources state, stating no result; every number must be stated by a source (constants aside), its words and unit (currency, %, date, count) read where it is stated; money divided by a count is not a percentage, money is not added to a count or multiplied by money; the formulas are read token by token and worked out with exact decimals and calendar days, never evaluated as code; business days are never counted; a plan with a problem, or a condition that does not hold, is sent back once with what is wrong (a cap's own words quoted); a plan that says something is missing is never used, what it needs is searched for and it is planned again once. The answer model is given the results, labelled as worked out; if it still declines, the working, quoting the words of each number, is the answer. The card shows "Worked out from the documents' figures, not stated in them", each step, and each number's words opening in the reader | **Tested** | `test_law_working.py` (24: the capped rate per full point, rates × counts, calendar days and business days, unstated numbers and money mixed with counts refused, percentages on one scale, injected text gives no figure, the note in the prompt, the declined answer, the card); `counsel.test.tsx` (the card, copy, refused on a decline or off-workspace link) |
| The rule a calculation applies was often not among the sources: "a credit calculated in accordance with paragraph 3 of Schedule 2" with the schedule unread; letters working out their own figures crowded out the agreement's clause; the register's holdings never read | For a calculation question only: the provisions a source says the figure is worked out under are read (a schedule's paragraph found after the schedule's heading; a clause by number); a short document the question names is read whole; the matter's agreements' two best passages are read too | **Tested** | `test_the_provisions_a_figure_is_worked_out_under_are_found_in_the_sources`; live runs below |

**Live, on a private copy of the demo** (real model, alice, the same answer keys):
- **The 18 chronology and calculation items:** 9 → 13 / 18. Fixed: the August service credit (USD 1,850, 10% band of USD 18,500),
  the register's totals by class, both chronology requests, 83 nights and 12 weekend days (GBP 19,340, not the GBP 18,400
  claimed) and the response deadline (22 October 2026, not 21 October).
- **Lost there:** the Hollins holding and the liquidated-damages period both came back right when asked again (variance).
- **Scored as a fail but answered right:** the most Orbis could get (GBP 3,000; the scorer finds "36,000" inside "360,000"
  once it drops commas).
- **Still wrong:**
  - **M-301 service credits (GBP 91,000 with the 25% cap applied to each month):** the planner applies the cap only on its
    retry; in the run it gave GBP 98,000.
  - **The Series B change of control:** answered correctly (21.2%, below 50%) without stating the 14,120,000 issued shares
    the key looks for.
- **The whole set, 165 questions:** 118 / 165 (111 in the last full run; 116 taking each item's latest earlier result); 15
  fixed, 13 lost.
  - By kind: calculations 9 → 13 / 16, chronology 0 → 2 / 2, definitions 2 / 2, follow-ups 8 → 4 / 12.
  - 12 of the 13 lost questions never reach the new code: they ask nothing the working is for, so their path is unchanged and
    the loss is run-to-run variance.
  - The 13th, a follow-up ("What share of the issued shares will that be after completion?"), is worked out right (21.25%)
    when asked on its own.
- **Planning call:** 2 to 6 s on the shared GPU; at most three calls when it is sent back or something is searched for.

**Deployed to the public demo 2026-11-07 (0b8564d).** The live check found that the card showed every condition as
"holds", even one the figures fail ("the total matches the GBP 18,400 claimed: holds" when it comes to GBP 19,340). Fixed
the same day: the card and the copied answer show the check's own result, and the server schema and the client both refuse
a check whose result is anything but "holds" or "does not hold" (`test_rates_times_counts_and_a_comparison_the_question_asks_for`,
`counsel.test.tsx`). Live through the public URL: the August service credit worked out (3 figures), 83 nights and 12
weekend days were worked out to GBP 19,340 in two of three askings (in the first the plan gave no working and the answer
was the model's own, marked partial; the third was marked unverified because the model wrote "Source 1" instead of its
citation numbers), and the delivery chronology has 45 events from May 4 to June 12.

**Tests:** backend suite 1737 passed, 4 skipped (1736 before the last change to how unreadable checks are handled);
`test_law_working.py` 26, chronology 22; frontend 264 passed; `tsc` clean. The held-out research sets were not rerun.

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
| EV22 | The whole RAG system again (2026-11-03): gate on `16671f0`, three runs pooled; held-out 8 written and frozen first (`78d88c4`); a 66-item / 80-turn chat set asked through the chat on a private copy of the demo data | golden, held-out and held-out 3 **passed** (193/204 and 21/21; 86/93 and 8/9; 39/42 and 38/45); held-out 2 (32/45, 35/45), 4 (37/39, 40/45, precision 0.842), 5 (33/36, 37/48), 6 (45/51, 35/45) and 7 (43/45, 24/48) failed, as on 10-29. Chat set by hand 61/80. Root causes: follow-ups searched as asked, "compare" routed to the conflict scan, small talk researched, a contract's parties named while its clauses use roles, cut passages, a document named by its party answered from another, the "not stated, but …" answer. Reports 20261103-103827 … -112259 |
| EV23 | The fixes (`a7fc503`, `adba314`; section "The whole RAG system evaluated again") measured on the unseen held-out 8 and a fresh chat set, then the gate on `04f8d1e` | **Own regression caught and fixed:** S2.6 gave each source a `parties="…"` attribute, which the model quoted (a quote of no document): quotes verified on held-out 1.0 → 0.838 and held-out 4 0.985 → 0.826 in the gate on `a7fc503` (20261103-124310 …); `adba314` gives the document's own opening passage instead. **Scorer:** held-out 8 lists same-matter trap values as forbidden, and the scorer counted a correct decline citing them as a T06 leak (15 on old and new code alike); T06 items still scan every event, other forbidden values are checked in the answer (`trap_values_in_answers`; no earlier set affected). **Final gate on `04f8d1e`:** golden (190/204, 21/21, quotes 1.0), held-out 3 (40/42, 41/45), 5 (32/36, 41/48) and **6 (45/51, 37/45, first pass)** passed; held-out 1 failed on declines (88/93, **3/9**: the Beta Corp settlement answered with this matter's payment terms, "not stated" then the related value), 2 (36/45, 33/45), 4 (34/39, 40/45, precision 0.878), 7 (37/45, 25/48: "do not specify the role … present as Chair") and 8 (30/48, 29/48, 6 trap values, 0 leaks) failed. Over every set answers 533 → 532 / 603 and declines 268 → 270 / 354 against `16671f0`; quotes 0.96-1.0; p95 4.9-7.1 s. Chat set by hand 44 → 59/80, 0 citations outside the matter; still wrong there: an answer from the wrong supply agreement when the question says "their supply agreement" (no party named), "And under the reseller agreement?" rewritten without its subject, closings and praise researched, role-party declines (lessor/tenant), calculations. Backend suite 1601 passed. Reports 20261103-124310 … -150402 |
| EV24 | A new blind set and the remaining failures (2026-11-04): held-out 9 (32 items) and chat set 3 (60 conversations / 79 turns) written by a separate pass and frozen (`aadab70`) before the fixes (`109ae34`; section "The remaining failures fixed, measured on a new blind set") were measured on them against the old code `3eaaa9a`; then two fixes from chat set 3 (`0e93aa0`, `326a20d`) and the final gate on `65e4e57` | **Blind, held-out 9:** answers 26 → 31 / 54, declines 24 → 25 / 42, false declines 22 → 22, leaks 0; both fail the bars. **Chat set 3 by hand:** 45/8/26 → 50/7/22 of 79, 0 citations outside the matter. **Final gate:** golden, held-out 1, 3, 5, 6 and every other suite passed; held-out 2, 4 (quotes 0.918, reworded by the model and refused), 7 (declines 0.583), 8 and 9 (35/54, 29/42) failed; over the nine sets seen before answers 529 → 533 / 603, declines 300 → 309 / 354 against the gate on `1d6bfa5`, the two gates differing only by run-to-run variance (±8 / ±13 over the gate, ±4 within a set), which the blind gain is within. Main remaining cause: the not-stated check's NO on answerable questions (the party named where the clause uses its role, a right limited by a condition, the document's own general words): NO right 65 / wrong 34. A reworded check did worse on replay and was not kept. Backend suite 1672 passed, 4 skipped. Reports 20261104-125706 … -175555 |
| EV25 | The not-stated check's wrong declines (2026-11-05): held-out 10 (44 items) written by a separate pass and frozen (`8e807da`) before the fixes (`93b2f92`; section "The not-stated check's wrong declines, measured on a new blind set") were measured on it, five runs each against the old code; then the gate on `a578891` | **Blind, held-out 10:** answers 53 → 76 / 140, declines 54 → 53 / 80, false declines 70 → 40, leaks 0, precision 0.912 → 0.962, quotes 0.988 → 0.972; 9-12 → 14-16 answers a run, beyond the run-to-run variance; both fail the bars. Role-party 29 → 41 / 60, stated in other words 19 → 30 / 60, lists 5 → 5 / 20, cross-document traps 20 → 15 / 20 (3 of the 5 declines scored wrong for naming the other value). **Gate:** golden, held-out 1, 3, 5, 6 and every other suite passed; held-out 2 (38/45, 0.844), 4 (precision 0.829, quotes 0.947), 7 (declines 0.646), 8, 9 and 10 (40/84, 30/48) failed; over the ten sets seen before answers 568 → 593 / 657, false declines 59 → 47, declines 338 → 324 / 396 against the gate on `65e4e57` (the declines lost at the edge of the ±13 variance; the items that fell furthest declined 7 times in 10 when run again, the check NOT GIVEN in all six traced). Main remaining causes on held-out 10: an answer without the asked fact, the check's verdict, the model's own decline; answers that open with a related fact are never checked. Backend suite 1674 passed, 4 skipped. Reports 20261105-114328 … -140635 |
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
