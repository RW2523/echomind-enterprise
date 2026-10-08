# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); slice branch `law/s1.0-foundations` |
| Phase | Phase 1: **S1.0 complete** (branch `law/s1.0-foundations`, not yet merged into `feature/echomind-law`) |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations | `0001_foundation` (Law edition only; applied in local `.law-data` dev DB) |
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
| SEC-01 | PARTIAL | S1.1 | S1.0: Law edition refuses insecure startup (backend and voice); API docs hidden. OIDC BLOCKED on P-011 |
| SEC-02 | MISSING | S1.2–S1.4 | |
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

1. Owner reviews S1.0 (diff `feature/echomind-law..law/s1.0-foundations`) and approves merging it into `feature/echomind-law`, and optionally cherry-picking the shared fixes to `main`.
2. Start **S1.1 Identity and sessions** on branch `law/s1.1-identity`. First failing test: `backend/tests/law/test_law_auth_routes.py::test_every_law_route_and_ws_denies_anonymous`, generated from the route table, HTTP and WebSocket. Then server-side sessions (`0002_identity`), MFA (TOTP, stdlib), CSRF, lockout, WebSocket session checks, voice on-behalf-of token.
