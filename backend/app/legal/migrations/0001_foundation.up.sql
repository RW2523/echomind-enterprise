-- 0001 foundation: tenant root for the Law edition.
-- Conventions (IMPLEMENTATION_PLAN §3): UUIDv4 text ids; UTC ISO-8601 timestamps with milliseconds.
-- schema_migrations itself is owned by the runner.
CREATE TABLE organizations(
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  created_by TEXT
);
