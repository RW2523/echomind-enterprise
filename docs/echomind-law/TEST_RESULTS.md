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
