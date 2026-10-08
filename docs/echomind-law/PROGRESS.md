# EchoMind Law — Progress Tracker

> **Every session starts by reading this file and DECISIONS.md.** Update at the end of each session: branch and commit, completed work, modified paths, migrations, test commands and results, risks, approvals needed, and the **NEXT EXACT ACTION**.

## Current state

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Repo / branch | `law/echomind-enterprise`. Integration branch `feature/echomind-law` (from `main@95782e4`); slice branch `law/s1.0-foundations` |
| Phase | Phase 0 approved 2026-10-08 → **Phase 1 / S1.0 in progress** |
| Plan documents | Committed to `feature/echomind-law` |
| Migrations applied | None |
| Production impact | None. Production stack untouched; only read-only `docker ps`/`inspect`/`curl` probes were run. |

## Session log

### Session 1 — 2026-10-08
- Cloned the repo. Four parallel subsystem audits plus manual verification of high-impact findings.
- Baseline tests: backend 50/50, voice 131/131, frontend `tsc` clean. Fresh-DB defects DB-1 and DB-2 confirmed.
- Discovered that production shares this host and has the same compose project name. Isolation decision D-002 proposed.
- Wrote CODEBASE_AUDIT, FEATURE_MATRIX, THREAT_MODEL, DESIGN_SYSTEM (+ `design/prototype.html`), IMPLEMENTATION_PLAN, DECISIONS, LICENSE_REGISTER, TEST_RESULTS and this file.

## Approvals

- 2026-10-08: **APPROVED** D-001, D-002, D-003, D-004 and D-019, committing the plan set, and the S1.0 dev dependencies (`vitest`, `@testing-library/react`, `jsdom`).
- Still 🔒 per plan: first `docker compose -p echomind-law … up`, any push or PR to GitHub, new runtime dependencies in S1.1+, counsel items P-001…P-015.

## Feature checklist (one row per backlog ID)

| ID | Status | Slice | Notes |
|---|---|---|---|
| SEC-01 | PARTIAL (pre-existing) | S1.1 | OIDC BLOCKED on P-011 |
| SEC-02 | MISSING | S1.2–S1.4 | |
| SEC-03 | INCOMPATIBLE | S1.6–S1.7 | |
| SEC-04 | PARTIAL (pre-existing) | S1.5 | |
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
| OPS-01 | PARTIAL (pre-existing) | S1.0 + OPS | LUKS needs host root |
| OPS-02 | PARTIAL (pre-existing) | S1.0, gates | |

## Known risks right now

- **Production shares this host.** Never run `docker compose` in this clone without `-p echomind-law -f docker-compose.law.yml` (which does not exist yet; created in S1.0).
- Memory headroom is about 14 GB. ASR containers need a memory check first.

## NEXT EXACT ACTION

After owner approval of D-001 to D-004:
1. `git switch -c feature/echomind-law`
2. Commit the plan set (only if approved).
3. Branch `law/s1.0-foundations`.
4. Start S1.0 with the tests first: `backend/tests/law/test_fresh_db_schema.py` (expected to fail on DB-1/DB-2), then the fixes.
