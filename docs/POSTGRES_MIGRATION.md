# SQLite → Postgres Migration Runbook (#6)

> **Status: planned, not executed.** This is a scheduled, maintenance-window task — **do not run it
> inline on the live system.** SQLite's single-writer model is the scaling wall (the new
> `activity_log` + chats now contend with RAG writes), so Postgres is the right move *before*
> onboarding multiple real tenants. But the cutover touches every DB call + the live data, so it
> needs a backup, a dry-run, and a rollback path.

## Why
- **SQLite = one writer.** Under concurrency, writes serialize and block. Fine for the demo
  (~8 users), not for multi-tenant production.
- Postgres gives real concurrency, connection pooling, and (with **pgvector**) a path to move the
  FAISS index into the DB for per-tenant vector collections.

## Scope — what moves
Relational tables (all in `backend/app/core/db.py` `init_db`): `documents`, `chunks`,
`chats`, `messages`, `transcripts`, `book_sections`, `section_references`, `transcript_analysis`,
`boardroom_sessions`, `docgen_jobs`, `docgen_templates`, `users`, `activity_log`.

**Vectors are a separate decision:** the FAISS index files (`faiss.index`, `faiss_meta.json`, …)
can either (a) stay as files (simplest — migrate only relational data now), or (b) move to
**pgvector** collections later (better per-tenant isolation/scale). Recommend (a) first.

## Code changes required (the refactor)
1. **DB abstraction.** Introduce `DB_BACKEND=sqlite|postgres` in `core/config.py`. Make
   `core/db.get_conn()` return a Postgres connection (psycopg) when set. Keep SQLite as default.
2. **Port SQLite-specific SQL.** Grep and fix:
   - `INSERT OR IGNORE` → `INSERT ... ON CONFLICT DO NOTHING`
   - `?` placeholders → `%s` (psycopg) — or adopt SQLAlchemy Core to abstract both.
   - `ALTER TABLE ... ADD COLUMN` guards (the `try/except` migrations) → use `IF NOT EXISTS`.
   - `COALESCE`, `CREATE TABLE IF NOT EXISTS`, indexes — mostly portable.
   Files with raw SQL: `core/db.py`, `core/auth.py`, `core/audit.py`, `rag/index.py`,
   `transcribe/analyzer.py`, `transcribe/store_to_db.py`, `api/routes/*`. **Recommend SQLAlchemy
   Core** to avoid hand-porting every statement and to get both backends from one code path.
3. **Connection pooling** (psycopg_pool / SQLAlchemy engine) — replaces per-call `sqlite3.connect`.

## Migration steps (maintenance window)
1. **Announce downtime.** Put the app in read-only or stop it.
2. **Backup**: copy `${DATA_DIR}/echomind.sqlite` + all `faiss*`/`*_meta.json` files offsite.
3. **Stand up Postgres** (a compose service + volume; pgvector image if going that route).
4. **Create schema** in Postgres (run the ported `init_db`).
5. **Copy data** with a one-shot script: for each table, `SELECT *` from SQLite → batch `INSERT`
   into Postgres (preserve ids/timestamps). JSON columns (`source_json`, `meta_json`, …) copy as
   text/`jsonb`.
6. **Dry-run + verify**: row counts per table match; spot-check a chat, a doc's chunks, users,
   `activity_log`. Run a few `/api/chat/ask` queries against the Postgres-backed instance on a
   side port.
7. **Cutover**: set `DB_BACKEND=postgres` + the connection env, recreate backend. FAISS files stay
   on the same volume (unchanged) for option (a).
8. **Smoke test**: login, chat (whole-KB + a vertical namespace), upload, Live Transcription,
   audit/usage panels.

## Rollback
Keep the SQLite file untouched during the window. If anything fails, set `DB_BACKEND=sqlite` and
recreate — you're back to the pre-migration state instantly (no data written to SQLite during the
window).

## Effort / risk
- **Effort:** ~2–4 focused days (mostly the SQLAlchemy/port refactor + the copy script + testing).
- **Risk:** high if rushed (data loss, subtle SQL incompatibilities). Low if done with the backup +
  dry-run + rollback above.
- **Recommendation:** do this as its own branch + PR, dry-run against a *copy* of the live DB, then
  schedule the ~30-min cutover window.
