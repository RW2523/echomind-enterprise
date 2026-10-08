# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); slice branch `law/s1.0-foundations` |
| Phase | Phase 1: S1.0 (`2a91cf2`) and S1.1 (`0caea7e`) merged; **S1.2 complete** on `law/s1.2-matters` (not yet merged) |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations | `0001_foundation`, `0002_identity`, `0003_matters` (Law edition only; applied in local `.law-data` dev DB) |
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
| SEC-01 | PARTIAL (local accounts complete) | S1.1 | Local accounts: sessions, TOTP MFA, lockout, CSRF, deny-by-default, WS gating, voice OBO. All tested, plus live API checks. OIDC BLOCKED (P-011). Revoke-on-role-change → S1.2. In-browser walkthrough → S1.3 shell |
| SEC-02 | PARTIAL | S1.2–S1.4 | S1.2: clients, matters, memberships, policy engine, non-disclosure, firm roles: done and tested (T01 backend). Walls and break-glass → S1.4; UI → S1.3; audit events → S1.5 |
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

1. Owner reviews S1.2 (`git log feature/echomind-law..law/s1.2-matters`) and approves merging it.
2. Start **S1.3 Law workspace shell and design system** on `law/s1.3-workspace`.
   - 🔒 It needs approval of the new frontend packages: `react-router`, `@tanstack/react-query`, `lucide-react`, `@fontsource-variable/inter`, `@fontsource-variable/source-serif-4`, `qrcode`, and dev-only `@playwright/test` and `axe-core`.
   - First failing test: `frontend/law/app/MatterSwitcher.test.tsx` (switching matter purges matter-keyed caches and asks for confirmation during an active capture).
   - Then the tokens and glass layer, router, sign-in restyle with an authenticator QR code, Home, the Matters list, Matter overview and Members, the non-disclosing "not available" page, and `/__design`.
   - Also build the S1.1 hardening (D-021): admin-issued one-time enrolment links and a forced password change at first sign-in.
