# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); slice branch `law/s1.0-foundations` |
| Phase | Phase 1: S1.0–S1.4 merged into `feature/echomind-law` (standing approval to build and merge slices in order) |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations | `0001`–`0005` (Law edition only) |
| Production impact | None. Production stack untouched; only read-only `docker ps`/`inspect`/`curl` probes were run. |

## Session log

### Session 1 — 2026-10-08
- Cloned the repo. Four parallel subsystem audits plus manual verification of high-impact findings.
- Baseline tests: backend 50/50, voice 131/131, frontend `tsc` clean. Fresh-DB defects DB-1 and DB-2 confirmed.
- Discovered that production shares this host and has the same compose project name. Isolation decision D-002 proposed.
- Wrote CODEBASE_AUDIT, FEATURE_MATRIX, THREAT_MODEL, DESIGN_SYSTEM (+ `design/prototype.html`), IMPLEMENTATION_PLAN, DECISIONS, LICENSE_REGISTER, TEST_RESULTS and this file.

### Session 1 (continued) — S1.0 Law foundations
Commits on `law/s1.0-foundations` (base `feature/echomind-law@5039c13`):
- `8e26751` fix(db): fresh installs get chunks page columns and boardroom meta columns (DB-1, DB-2)
- `03a8e13` feat(law): edition flag with startup posture guard (D-003)
- `c55a458` feat(law): versioned schema migrations with checksums and down scripts (D-004)
- `fdbaeba` fix(config): declare 15 settings the code read but env could not override (R-13)
- `d5d188a` feat(law): voice posture guard, eval switches off, spoken content out of logs (D-003, D-016)
- `9a52939` build(frontend): Vitest test runner; Tailwind now scans vertical pages (R-12)
- `0d56398` ops(law): isolated compose stack and native dev scripts (D-002)
- `8954c08` fix(law): redact free-text log arguments in the Law edition (T-11)
- `56478af` test(law): committed fictional law fixtures, fixture loader, law eval baseline
- then: ci: frontend unit tests; docs: S1.0 results

**As built vs. plan (S1.0):**
- `schema_migrations` is owned by the runner; `0001_foundation` creates only `organizations`.
- `asr_container.sh` became `scripts/law/compose.sh`, a wrapper for the isolated compose stack that refuses `up` below 12 GB free.
- Content-free logging is a root-handler redaction filter rather than per-call-site edits. A live log scan proved the per-call-site approach had missed lines.
- Dev ports moved because other stacks on this host hold them: 8100 is `cio-vllm-1`, and 3100 is `cio-loki-1`. The dev backend now uses 8110 and the compose frontend 3110/3553.
- No separate `backend-law` CI job: the existing backend and voice jobs run `pytest tests`, which includes the Law tests. The frontend job gains `vitest run`.

**Shared fixes worth cherry-picking to `main` for production:**
- `8e26751` (fresh-install DB defects)
- `fdbaeba` (R-13 env knobs)
- the Tailwind glob part of `9a52939`

These change legacy behaviour only by fixing bugs: vertical pages become fully styled, and env overrides start working.

**Production impact this session:** none. Only these touched shared resources:
- read-only reuse of the host-published LLM (:8355) and embeddings (:11435), for about 14 questions and 5 chunk embeddings;
- one public model download into the host HF cache: `cross-encoder/ms-marco-MiniLM-L-6-v2`, the same model production uses.

### Session 1 (continued) — S1.1 Identity and sessions
- 2026-10-08: owner approved merging S1.0 → merged `2a91cf2`. No cherry-pick to `main` and no compose `up` (not approved).
- Commits on `law/s1.1-identity`:
  - `4a88f02` identity building blocks
  - `1be9500` backend wiring
  - `c2ac0cd` voice
  - `b052424` frontend
  - `acf6904` dev tooling
- **Delivered:**
  - Server-side sessions: hashed tokens, 30 min idle / 10 h absolute, rotation after MFA.
  - TOTP MFA with recovery codes.
  - Lockout (MFA guesses count) and the password policy.
  - Uniform sign-in failures.
  - `__Host-` cookies and CSRF.
  - Deny-by-default guard (the client-error sink now needs a session).
  - Session list and revoke; password change signs out elsewhere.
  - Admin disable, enable and revoke.
  - Law-safe admin bootstrap (never logged).
  - WebSocket session, Origin check and 60 s revocation re-check (transcribe and voice).
  - Voice on-behalf-of tokens, **fixing the audit's voice→RAG 401**.
  - Frontend MFA sign-in, CSRF fetch wrapper and session-expiry return to sign-in.
  - Dev tooling with MFA.
- **As built vs. plan (S1.1):**
  - **OIDC not built.** It is BLOCKED (P-011), and verifying the IdP's RS256 tokens needs a crypto library that is not yet approved. The plan's "stub IdP" tests were skipped rather than shipping a fake.
  - **Revoke-on-role-change** is not wired: there is no role-change API until S1.2. Disable, delete, password change and admin password reset do revoke sessions.
  - **MFA secret** is stored base32 in `users.mfa_secret`. Encryption at rest follows D-014 (LUKS on the host, plus app-level encryption once `cryptography` is approved).
  - The voice service in the Law edition no longer receives `AUTH_SECRET` (least privilege). Its posture requires `VOICE_ALLOWED_ORIGINS` instead of a shared secret.
  - Admin routes still check the legacy `role == "admin"`. The S1.2 policy engine replaces this with firm roles.
- **Found while building:** backend app shutdown hangs because the CUDA-fault watchdog's thread never finishes. This is pre-existing and unrelated to Law. It was raised as a separate task (chip "Fix backend shutdown hang from CUDA watchdog"), and tests avoid running the app lifespan meanwhile.
- **Residual risks to address later:**
  - Anyone holding a new user's initial password can enrol their own authenticator at first sign-in. Planned mitigation (S1.3 admin UX): an admin-issued one-time enrolment link and a forced password change on first sign-in.
  - Throttling is per account (lockout); there is no per-IP sign-in throttle. `RATE_LIMIT_PER_MIN` exists but is off.

### Session 1 (continued) — S1.2 Matters, memberships, policy engine
- 2026-10-08: owner approved merging S1.1 → merged `0caea7e`.
- Commit `3b8574f` on `law/s1.2-matters`.
- **Delivered:**
  - Migration `0003_matters`.
  - `legal/policy.py`: deny-by-default `decide()`/`authorize()`, with hidden matters answering exactly like missing ones.
  - `legal/matters/` repo and routes, `legal/identity/routes.py` for firm roles, `legal/orgs.py` for the default organisation.
  - Principals now carry `org_id`; in the Law edition, admin routes use the `firm_admin` firm role.
  - **Completes S1.1's revoke-on-role-change.**
- **As built vs. plan (S1.2):**
  - **Matter-level changes:** membership changes bump `matters.acl_epoch` but do not revoke sessions. Every request re-evaluates the policy, so removal is immediate for HTTP. Open streams and sockets will compare the epoch from S1.6 on.
  - **Firm-role changes:** removing a firm role revokes all of that person's sessions; adding one does not.
  - **Single organisation:** created on first boot (`LEGAL_ORG_NAME`). Everything is org-scoped, but there is no API to create more organisations; it is not needed for a single-firm pilot.
  - **Who can see a client:** its creator, or anyone through a visible matter. Firm-wide client lists (conflict checks) are deliberately not offered; that is OPT-01.
  - **Audit events** for matter, membership and role changes are not written yet. They come with the hash-chained audit trail in **S1.5**, which instruments these routes.
  - **Ethical walls and break-glass** are not in the engine yet; they come in S1.4. No UI yet; that is S1.3.

### Session 1 (continued) — S1.3 Law workspace shell and design system
- 2026-10-08: owner approved merging S1.2 → merged `ee67615`, and approved the S1.3 packages.
- Commits on `law/s1.3-workspace`:
  - `da43cc3` D-021 backend
  - `c564d90` workspace and design system
  - `af501bc` visual-review fixes
  - `6f8da72` demo seed, E2E and screenshots
- **Delivered:**
  - **Onboarding hardening (closes D-021):** enrolment links, forced password change, two-step reset.
  - **Edition switch:** `/api/auth/config` reports the edition, and the frontend loads the Law workspace as a lazy chunk before any legacy hook mounts.
  - **"Paper & Glass" design system** (`frontend/law/theme/law.css`).
  - **Routing and data:** react-router routes, session gate, matter context from the URL (hidden ≡ missing), matter-keyed caches dropped on leave and on session expiry, navigation guard.
  - **Pages:** Sign-in, Change password, Home, Matters, Matter overview, Members, Security, Users & roles, Not available, `/__design`.
  - **Demo seed CLI** and the Playwright E2E.
- **As built vs. plan (S1.3):**
  - **react-router 7, not 8.** Version 8 needs Node 22, while CI and the Docker image use Node 20.
  - **No Tailwind preset.** Design tokens are applied through scoped CSS classes (`lw-*`); Tailwind is used only for a few layout utilities.
  - **Display preferences** (theme, transparency) are kept in browser storage as a per-viewer convenience, not on the server as DESIGN_SYSTEM §4.2 suggested. A server-side preference can come later.
  - **No command palette (⌘K) yet.**
  - **Matter navigation shows only built sections** (Overview, Members). Documents, Research and the rest appear when their slices land, so nothing is shown that doesn't work.
  - **The marketing `LawExperience` page is not routed** in the Law edition (it was an unauthenticated demo page); the file remains for the legacy `law` vertical.
  - **E2E is not in CI yet.** It needs the backend's Python dependencies plus a Chromium download in the CI job; this is planned with OPS-02.
- **Design approval (🔒 S1.3 acceptance):** screenshots are in `docs/echomind-law/design/screens/` and were sent to the owner for review.

### Session 1 (continued) — S1.4 Ethical walls and break-glass
- S1.3 merged (`10e3f6f`) after design approval. The owner's standing approval to build and merge in order is recorded in DECISIONS.md.
- Commits:
  - `3915f1e` backend: migration `0005_walls`, policy steps for walls and break-glass in the single query, `/api/legal/walls*`, `/api/legal/break-glass*`
  - `3dbb6f5` screens, a11y fix, seed and E2E
- **As built:**
  - Managing partners are the walls authority and may see the firm's matter directory, but only on the walls screen (D-022).
  - Break-glass is requested by matter number with an identical response whether or not the number exists; it is resolved when a different managing partner approves.
  - The banner does not yet say actions are recorded: read-access audit lands in S1.5, and the copy will change then.

## Approvals

- 2026-10-08: **APPROVED** D-001, D-002, D-003, D-004 and D-019, committing the plan set, and the S1.0 dev dependencies (`vitest`, `@testing-library/react`, `jsdom`).
- Still 🔒 per plan:
  - the first `scripts/law/compose.sh up`;
  - any push or PR to GitHub;
  - merging `law/s1.0-foundations` into `feature/echomind-law`;
  - new runtime dependencies in S1.1+ (S1.1 needs none on the backend; `qrcode` for the MFA QR is S1.3);
  - counsel items P-001…P-015.

## Feature checklist (one row per backlog ID)

| ID | Status | Slice | Notes |
|---|---|---|---|
| SEC-01 | PARTIAL (local accounts complete) | S1.1, S1.3 | Local accounts: sessions, TOTP MFA (enrolment by admin link), forced password change, lockout, CSRF, deny-by-default, WS gating, voice OBO, revoke-on-role-change; browser sign-in tested end to end. OIDC BLOCKED (P-011) |
| SEC-02 | DONE (audit events pending S1.5) | S1.2–S1.4 | Clients, matters, memberships, policy engine with non-disclosure, firm roles, ethical walls with two-person approval and exceptions, break-glass; UI; T01 in backend and browser, walls and break-glass in the browser. Audit trail for these actions → S1.5 |
| SEC-03 | INCOMPATIBLE | S1.6–S1.7 | |
| SEC-04 | PARTIAL | S1.5 | S1.0: content-free logs in the Law edition (backend and voice, verified by log scan) |
| SEC-05 | MISSING | S1.8 | Policy values need counsel (P-001) |
| SEC-06 | PARTIAL (pre-existing) | S1.9–S1.10 | |
| SEC-07 | MISSING | S1.9 | Policy values need counsel (P-002/P-003) |
| SEC-08 | MISSING | S1.11 | Provider BLOCKED (P-010) |
| RAG-01 | PARTIAL (pre-existing) | S2.1–S2.2 | |
| RAG-02 | MISSING | S2.3 | |
| LIVE-01 | PARTIAL (pre-existing) | S2.4 | |
| LIVE-02 | PARTIAL (pre-existing) | S2.5 | |
| MEET-01 | PARTIAL (pre-existing) | S2.6–S2.7 | |
| DOC-01 | PARTIAL (pre-existing) | S3.1 | |
| DOC-02 | MISSING | S3.2–S3.3 | Word tracked changes need a spike and a manual Word check |
| DOC-03 | MISSING | S3.4 | External authority stays UNVERIFIED (P-009) |
| DOC-04 | MISSING | S3.5 | |
| DOC-05 | MISSING | S3.6 | |
| DOC-06 | MISSING | S3.7 | |
| DOC-07 | MISSING | S3.8 | |
| OPT-01 | MISSING | S4.1 | Needs business approval |
| OPT-02 | PARTIAL (pre-existing) | S4.2 | MT model BLOCKED |
| OPT-03 | MISSING | S4.3 | Needs business approval |
| OPT-04 | BLOCKED | S4.4 | Licensed source plus network approval |
| OPT-05 | MISSING | S4.5 | Needs business approval |
| OPS-01 | PARTIAL | S1.0 + OPS | S1.0: versioned migrations with down scripts + CLI; isolated compose stack. LUKS needs host root |
| OPS-02 | PARTIAL | S1.0, gates | S1.0: Law pytest suite, Vitest, committed law fixtures, law eval baseline 6/7 |

## Known risks right now

- **Production shares this host.** Never run `docker compose` in this clone without `-p echomind-law -f docker-compose.law.yml` (which does not exist yet; created in S1.0).
- Memory headroom is about 14 GB. ASR containers need a memory check first.

## NEXT EXACT ACTION

Start **S1.5 Audit v2** on `law/s1.5-audit`:
1. Migration `0006_audit`: append-only, hash-chained `audit_events` with no-update/no-delete triggers.
2. `legal/audit` modules: record, chain, verify, anchors, export.
3. Instrument policy decisions (every deny and content read) and every Law mutation route.
4. Screens: Admin → Audit (firm-wide metadata) and Matter → Activity (leads).
5. Change the break-glass banner copy to say access is recorded.
