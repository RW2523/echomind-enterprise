-- 0003 matters (S1.2, SEC-02): clients, matters and time-bounded matter memberships.
-- Memberships are never deleted: removal sets revoked_at/revoked_by/reason, so history is kept.
-- matters.acl_epoch increments on every membership change; long-lived streams compare it (S1.6).
CREATE TABLE clients(
  id         TEXT PRIMARY KEY,
  org_id     TEXT NOT NULL,
  name       TEXT NOT NULL,
  code       TEXT,
  status     TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived')),
  created_at TEXT NOT NULL,
  created_by TEXT,
  updated_at TEXT,
  version    INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX ux_clients_code ON clients(org_id, code) WHERE code IS NOT NULL;

CREATE TABLE matters(
  id                  TEXT PRIMARY KEY,
  org_id              TEXT NOT NULL,
  client_id           TEXT NOT NULL REFERENCES clients(id),
  number              TEXT NOT NULL,
  title               TEXT NOT NULL,
  practice_area       TEXT,
  jurisdiction        TEXT,
  status              TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','archived')),
  sensitivity         TEXT NOT NULL DEFAULT 'restricted' CHECK(sensitivity IN ('standard','restricted','highly_restricted')),
  retention_policy_id TEXT,
  acl_epoch           INTEGER NOT NULL DEFAULT 1,
  created_at          TEXT NOT NULL,
  created_by          TEXT,
  updated_at          TEXT,
  version             INTEGER NOT NULL DEFAULT 1,
  UNIQUE(org_id, number)
);
CREATE INDEX ix_matters_client ON matters(client_id);

CREATE TABLE matter_memberships(
  id         TEXT PRIMARY KEY,
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK(role IN ('lead','attorney','paralegal','reviewer','read_only')),
  can_export INTEGER NOT NULL DEFAULT 0,
  valid_from TEXT NOT NULL,
  valid_to   TEXT,
  granted_by TEXT,
  granted_at TEXT NOT NULL,
  revoked_by TEXT,
  revoked_at TEXT,
  reason     TEXT
);
CREATE INDEX ix_mm_user ON matter_memberships(user_id);
CREATE INDEX ix_mm_matter ON matter_memberships(matter_id);
-- At most one live membership per person per matter.
CREATE UNIQUE INDEX ux_mm_live ON matter_memberships(matter_id, user_id) WHERE revoked_at IS NULL;
