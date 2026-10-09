# EchoMind Law — Test Results Log

> Only real, executed results are recorded here, with the commit and command. Nothing is estimated.

## Run 0 — Baseline (Phase 0), 2026-10-08

- **Commit:** `main@95782e4` (fresh clone, no modifications)
- **Host:** DGX Spark GB10 (aarch64), driver 580.159.03, Python 3.13.11 (host), Node 22.23.1 / npm 10.9.8
- **Note:** CI uses Python 3.12 and Node 20; the host versions differ.

| Suite | Command | Result |
|---|---|---|
| Backend unit | `cd backend && ECHOMIND_DATA_DIR=<scratch> python3 -m pytest tests -q -rs` | **50 passed**, 3 deprecation warnings (SWIG), 1.34 s |
| Voice unit | `cd voice && python3 -m pytest tests -q -rs` | **131 passed**, 0.14 s |
| Frontend type-check | `cd frontend && npm ci && npx tsc --noEmit -p .` | **Clean (exit 0)** |
| Fresh-DB schema probe | `ECHOMIND_DATA_DIR=<empty> python3 -c "from app.core.db import init_db; init_db()"` then `PRAGMA table_info` | **Defects confirmed:** `chunks` lacks `page_start`/`page_end` (DB-1); `boardroom_sessions` lacks `scenario`/`namespace`/`speaker_map_json` on first boot (DB-2) |
| Golden eval (`eval/run_eval.py`) | — | **Not run.** The law corpus is not in the repo, and running it against production would write chat rows into the production DB. Scheduled for S1.0 against the Law dev backend with committed fictional fixtures. Last recorded historical runs: 50/52 (2026-07-30), 49/52 (2026-08-06), 9/52 (2026-09-22, corpus missing). |

**Not covered by any existing automated test** (see CODEBASE_AUDIT §10): WebSocket protocols, Silent Assistant verifier and rules, auth middleware, RBAC and tenant scoping, audit, Boardroom diarization and report, the docgen pipeline, schema migrations, and the voice session.

## Run 1 — S1.0 Law foundations, 2026-10-08

- **Branch / commit:** `law/s1.0-foundations` @ `56478af` (plus the CI and docs commit that records this run)
- **Host:** same as Run 0.

| Suite | Command | Result |
|---|---|---|
| Backend unit (all) | `cd backend && ECHOMIND_DATA_DIR=$(mktemp -d) python3 -m pytest tests -q` | **94 passed** (50 legacy + 44 new Law) |
| Backend Law only | `… pytest tests/law -q` | **44 passed**: fresh-DB schema (6), posture guard and startup in a real interpreter (22), migrations (13), settings declarations (3) |
| Voice unit | `cd voice && python3 -m pytest tests -q` | **143 passed** (131 legacy + 12 new Law) |
| Frontend | `cd frontend && npx tsc --noEmit -p . && npx vitest run` | tsc **clean**; Vitest **5 passed** (2 files) |
| Tests-first evidence | new schema tests run before the fix | 6/6 **failed** on `main` code (DB-1: `chunks missing ['page_end','page_start']`; DB-2: `no such column: scenario`), then passed after the fix |
| Tailwind build check | `npx vite build` before/after, then grep the CSS | `sm:text-6xl`, `max-w-6xl`, `lg:grid-cols-4`, `blur-[140px]`: **0 → 1** occurrence each |
| Law compose render | `scripts/law/compose.sh config` (dummy secrets) | Renders project `echomind-law`; containers `echomind-law-{backend,voice,frontend}`; ports 3110/3553 only; no trtllm/ollama/cloudflared; GPU hidden; code mounted `:ro`. With empty secrets: **refuses** (`required variable LAW_AUTH_SECRET is missing a value`). **Stack not started** (first `up` needs approval). |
| Law dev backend, fresh DB | `scripts/law/dev_backend.sh` (127.0.0.1:8110) | Boots; `/health` 200; `auth_enabled: true`; `/docs` and `/openapi.json` **404**; anonymous `/api/docs/list` **401**; migration `0001_foundation` applied |
| Fresh-DB ingestion | `python3 scripts/law/load_law_fixtures.py` | Both fixtures ingested (2 + 3 chunks); DB-1 fixed end to end |
| Law golden set | `ECHOMIND_EVAL_USER=admin ECHOMIND_EVAL_PASSWORD=… python3 eval/run_eval.py --base http://127.0.0.1:8110/api --set law` | **6/7** twice (`eval/reports/eval_20261008-172730.json`, `…-173020.json`). Fail: `law-msa-renewal-vs-playbook` (cites only the playbook; the cross-document item also failed in the July and August runs). Latency 4.9–11.5 s per question (CPU reranker on the host, LLM shared with production). |
| Log-leak scan (T-11) | grep the dev backend log for 13 fixture terms (party names, amounts, clause terms) | **Before the fix: leaks found.** Question text with client names in `rag/advanced.py` and `rag/evidence_gate.py` lines. **After the root-handler redaction filter: 0 hits** for all 13 terms; lines read `question=<81 chars>`. |
| Dev frontend proxy | `scripts/law/dev_frontend.sh` (127.0.0.1:5174) | SPA 200 at `/?vertical=law`; `/api/auth/config` proxied to the Law backend |

**Not exercised in S1.0:** the Law compose stack (`up` is gated on approval, D-002), ASR, voice and Boardroom inside the Law edition, and CI on GitHub (nothing pushed).

## Run 2 — S1.1 Identity and sessions, 2026-10-08

- **Branch / commit:** `law/s1.1-identity` @ `acf6904` (base `feature/echomind-law@2a91cf2`)

| Suite | Command | Result |
|---|---|---|
| Backend unit (all) | `cd backend && ECHOMIND_DATA_DIR=$(mktemp -d) python3 -m pytest tests -q` | **144 passed** (50 legacy + 94 Law) |
| Backend Law only | `… pytest tests/law -q` | **94 passed**. New in S1.1: identity units 21 (incl. RFC 6238 vectors), auth API 21, route table 4, posture +4 |
| Voice unit | `cd voice && python3 -m pytest tests -q` | **150 passed** (131 legacy + 19 Law) |
| Frontend | `npx tsc --noEmit -p . && npx vitest run` | tsc **clean**; Vitest **13 passed** (4 files: session fetch 4, Login 4, packs 4, tailwind 1) |
| Tests-first evidence | new API/route tests before the wiring | **24 failed, 1 passed** before; all pass after |
| Deny by default | `test_every_http_route_denies_anonymous_callers` | Every API route in the live route table (>60 method/path pairs) returns **401** anonymously except login, logout, config, version, health. Every WebSocket closes **1008** anonymously. |
| Live: upgrade path | dev backend on the S1.0 `.law-data` DB | Migration `0002_identity` applied on boot; admin gained `firm_admin` |
| Live: anonymous | `curl` | `/api/docs/list`, `/api/chat/list`, `/api/auth/me`, `/api/transcribe/list`, `POST /api/client-error`: all **401** |
| Live: login | `curl -i` | `__Host-echomind_s`: HttpOnly, Secure, SameSite=strict, Path=/, Max-Age=600 (pending MFA). `__Host-echomind_csrf`: Secure, SameSite=strict, not HttpOnly. Body `{"status":"mfa_enrollment_required"}` (no token). Wrong password: generic message. |
| Live: dev loop with MFA | `load_law_fixtures.py`, then `run_eval.py --set law` | First sign-in enrolled TOTP (key and recovery codes saved 0600); fixtures already present (dedup); law set **6/7** (`eval/reports/eval_20261008-180831.json`), unchanged |
| Live: voice → backend (D-013) | voice `auth_check.BackendAuth` against the dev backend, real LLM | Session seen; OBO minted; `POST /api/chat/ask-voice` with OBO → **200**, answer contains "250,000"; same token on `/api/docs/list` → **401**; no token → **401**; after sign-out → **401** and `still_valid()` False. **The audit's voice-401 defect is fixed.** |
| Live: log scan | grep the dev backend log | **0** hits for party names, amounts and clause terms, admin password, TOTP key, recovery codes and OBO tokens |

**Not exercised:** an in-browser walkthrough of the new sign-in screen (covered by component tests and the live API; the browser walkthrough comes with the S1.3 shell); the voice WebSocket with real audio (needs the isolated compose stack, which has not been started); OIDC (blocked, P-011).

## Run 3 — S1.2 Matters, memberships, policy engine, 2026-10-08

- **Branch / commit:** `law/s1.2-matters` @ `3b8574f` (base `feature/echomind-law@0caea7e`)

| Suite | Command | Result |
|---|---|---|
| Backend unit (all) | `cd backend && ECHOMIND_DATA_DIR=$(mktemp -d) python3 -m pytest tests -q` | **562 passed** (50 legacy + 512 Law) |
| Policy matrix | `pytest tests/law/test_law_policy.py` | **398 passed**. Every matter action × matter role × open/closed/archived; non-member and missing identical for every action; every firm action × firm role; export right; unknown actions fail closed. Expectations are written from the plan's §4.1 tables, independently of the engine. |
| T01 over HTTP | `pytest tests/law/test_law_matters_api.py` | **20 passed**. Each user sees exactly their live matters; hidden ≡ missing on GET/PATCH/memberships (same 404 body); hidden and missing run the same 2 queries; revoked and expired memberships grant nothing; firm admin sees no matter content; clients visible only through matters; another organisation invisible even with a stray membership row; members lacking a right get 403; removal takes effect at once; history kept on role change; last lead protected; inactive colleagues refused; creation rights; no matter under an invisible client; optimistic concurrency; closed matters accept only reopening; firm-role removal signs the user out; role management is firm-admin only, without self-lockout. |
| Route table | `pytest tests/law/test_law_route_table.py` | The 12 new `/api/legal` routes are covered automatically: all **401** anonymously |
| Voice / frontend | as before | **150 passed**; tsc clean, **13 passed** (unchanged by this slice) |
| Live: upgrade | dev backend on the S1.1 dev DB | `0003` applied; 1 organisation created; 0 users without an organisation |
| Live: API walk-through | `law_client` as `admin` and a new colleague `a.rahman` | Adding a role revoked 0 sessions. Matter M-101 created (creator = lead). The colleague saw `[]`. Hidden vs random matter id: **identical** `404 {"detail":"Not Found"}`. After being added the colleague saw `M-101/attorney`. Managing members as attorney: **403**. After removal: **404**. Firm-role removal: **1 session revoked**, then **401**. `acl_epoch` 3. |
| Live: log scan | grep the dev backend log | **0** hits for the client name, matter title, practice area, jurisdiction and the new user's password |

**Honesty note:** the 20 API tests were written before the routes existed, but they were not executed in a failing state first; the policy matrix was written alongside the engine. The S1.0/S1.1 tests-first evidence stands as recorded above.

## Run 4 — S1.3 Law workspace shell, design system, onboarding hardening, 2026-10-08

- **Branch / commit:** `law/s1.3-workspace` @ `6f8da72` (base `feature/echomind-law@ee67615`)

| Suite | Command | Result |
|---|---|---|
| Backend unit (all) | `cd backend && ECHOMIND_DATA_DIR=$(mktemp -d) python3 -m pytest tests -q` | **573 passed** |
| D-021 onboarding | `pytest tests/law/test_law_onboarding.py` | Before implementation: **9 failed** (8 new tests plus the updated S1.1 enrolment test). After: **8 passed**. Covers: link required (missing, guessed or another user's link → 403); single-use and 72 h expiry; a new link replaces older ones; firm may relax the requirement; admin-created account gets a link and must change its password (403 `password_change_required` elsewhere until done); admin reset of two-step signs the person out and issues a new link; no link for an already-enrolled account (409); bootstrap admin link written 0600. |
| Demo seed | `pytest tests/law/test_law_demo_seed.py` | **2 passed**: refuses without `LEGAL_ALLOW_DEMO_SEED=1`; builds the T01 world (alice {M-101, M-114}, bob {M-202, M-215}, carol {M-101}); idempotent |
| Voice | `cd voice && python3 -m pytest tests -q` | **150 passed** |
| Frontend unit | `cd frontend && npx tsc --noEmit -p . && npx vitest run` | tsc **clean**; Vitest **28 passed** (5 files). Law workspace 15: only server-listed matters; hidden ≡ missing page; matter by number; absent controls for non-lead/non-admin; switching drops the previous matter's cache; confirmation while work is in progress; signed-out → `/signin?next=`; forced password change first; axe checks on Matters, Members and Sign-in (0 violations; colour contrast is not computable in jsdom); monograms. Sign-in 5: enrolment only through a link with QR and recovery codes; no-link explanation; no open redirect; server message on failure. |
| Browser E2E (Chromium 1248) | `cd frontend && npx playwright test` | **3 passed**: alice sees exactly M-101/M-114 and never M-202/M-215; the hidden matter's page text is identical to a non-existent one; the switcher lists only her 2 matters; bob sees the highly restricted matter; sign-out ends the session in every tab. Real sign-in with TOTP; Secure `__Host-` cookies accepted on `http://localhost`. |
| Bundle | `npx vite build` | The Law workspace is a separate lazy chunk (218 KB JS, 16 KB CSS); fonts are bundled woff2; **no external font or CDN references** in the bundle |
| Visual review | screenshots in `docs/echomind-law/design/screens/` | The first pass found 3 defects: a black canvas (legacy `index.html` background), the switcher list trapped under the content (glass stacking context), and "VA" instead of "VL" monograms. All fixed and re-shot (`af501bc`). |

**Not exercised:**
- The compose stack.
- Production build served by nginx (dev server used).
- Mobile widths. The rail layout below 860 px is styled but not screenshot-tested.
- Colour-contrast measurement (jsdom cannot compute it; to be checked with Playwright and axe on real rendering in a later slice).

## Run 5 — S1.4 Ethical walls and break-glass, 2026-10-08

- **Branch / commit:** `law/s1.4-walls` @ `3dbb6f5`

| Suite | Result |
|---|---|
| Backend (all) | **591 passed** after the backend commit, then re-run green with the frontend commit. `test_law_walls.py` (18): **17 failed before** the implementation, **all pass after**. Covers: a wall hides a matter even from its lead (GET, list, memberships); a client wall covers every matter of the client and the client itself; no effect until a second managing partner approves (requester and attorneys refused); release needs two people; exceptions only once approved by someone else, end on expiry or revocation, at most 90 days; walls visible only to managing partners; an attorney can only protect what they can see; break-glass gives time-limited read-only access (no edits, no membership changes), cannot be self-approved, ends early or on expiry, never reaches highly restricted matters, never crosses a wall, is capped at 240 minutes, reveals nothing about the requested number, and is limited to firm admins (request) and managing partners (approve); walled, hidden and missing matters still run the same 2 policy statements. |
| Bug found by the new tests | A client's creator could still see a client they were walled from. Fixed. |
| Frontend | tsc **clean**; Vitest **32 passed** (+4: walls/emergency screens absent without the role; approval offered only to the other managing partner; emergency-access banner with "End now"; break-glass request body). |
| Accessibility bug found | Field hints were part of the label's accessible name. Fixed with `aria-describedby`. |
| Browser E2E | **5 passed**. New: (4) morgan requests a wall screening alice from M-114, sees no approve button, nakamura approves, and alice — the matter's lead — no longer sees M-114 (list and URL); (5) fiona requests emergency access to M-215 by number, morgan sees the resolved matter and approves, fiona sees the read-only banner and no edit controls, ends it, and the matter becomes "not available" again. Screenshots 10 and 11 added. |

## Run 6 — S1.5 Audit trail, 2026-10-08

- **Branch / commits:** `law/s1.5-audit`:
  - `3fcf8a3` backend
  - `8080275` frontend

| Suite | Result |
|---|---|
| Backend (all) | **613 passed**, which includes the 22 audit tests in `test_law_audit.py`. The first 12 were written before the implementation and **all 12 failed** first. The other 10 were added during the build: completeness, failure behaviour, anchors and the CLI. Coverage: <ul><li>the chain verifies;</li><li>the database refuses UPDATE/DELETE;</li><li>an out-of-band edit is detected at the right event;</li><li>a removed tail and a fully re-hashed chain are caught by the anchor;</li><li>details keep identifiers only;</li><li>the anchor file is 0600;</li><li>sign-in outcomes are recorded without passwords or typed unknown usernames;</li><li>refusals are recorded while the response stays the non-disclosing 404;</li><li>matter, membership, wall and role changes are each recorded;</li><li>every break-glass access is recorded, including on routes that are not audited;</li><li>one action writes exactly one event;</li><li>422 and CSRF failures are recorded;</li><li>other modules get a generic `api.request` event with no body content;</li><li>when the trail cannot be written, the API returns 500 and no data;</li><li>the verifier CLI exits 1 on a broken chain;</li><li>canary text appears in neither the trail nor the logs;</li><li>the firm trail is for governance roles only, and firm admins see no matter numbers;</li><li>matter activity is for the lead only (403 for a member, 404 for a non-member);</li><li>verify and export work, and the export is itself recorded;</li><li>a structural test fails if any state-changing `/api/legal` or `/api/auth` route is not audited.</li></ul> |
| Voice | **150 passed**. No change. |
| Frontend | tsc **clean**. Vitest **40 passed**, 8 of them new: <ul><li>the trail is absent without a governance role;</li><li>events read as sentences, refusals are marked in words, and axe is clean;</li><li>a firm admin sees matter references, not numbers;</li><li>filters and paging go to the server;</li><li>the integrity check shows "Intact" or "Broken" with the first affected event;</li><li>matter activity is lead-only and axe-clean;</li><li>a matter is not shown when its recorded open fails;</li><li>the banner says access is recorded.</li></ul> |
| Bugs found by the browser test | <ul><li>**An emergency-access visit left no read event.** The workspace took the matter from the list and never called the recorded read. Fixed by D-023: entering a matter performs the recorded read first.</li><li>**The lead's activity page showed "a matter"** instead of the matter's own number. Fixed.</li></ul> |
| Visual review | Screenshot 12 (audit trail) and 13 (matter activity). The first pass wrapped timestamps onto two lines; fixed and re-shot. |
| Browser E2E | **6 passed**. New test (6): <ol><li>morgan filters the trail to emergency access and sees fiona's request and his own approval, then runs the integrity check and gets "Intact";</li><li>bob, the lead of M-215, sees "fiona viewed M-215" marked as emergency access;</li><li>alice gets "not available" on the audit trail.</li></ol> |
