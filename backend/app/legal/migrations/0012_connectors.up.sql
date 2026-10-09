-- 0012 connectors (S1.11, SEC-08): document-management connectors. A firm administrator configures
-- a connector (its settings are encrypted with the master key); a matter's lead links a folder to
-- the matter; sync imports through the normal encrypted, partitioned, audited path.
-- Fail closed: a synced document is visible only to people its source permissions name AND who
-- are live members of the matter, and only while its connector has synced successfully recently.
CREATE TABLE legal_connectors(
  id                  TEXT PRIMARY KEY,
  org_id              TEXT NOT NULL,
  kind                TEXT NOT NULL CHECK(kind IN ('local_folder')),
  name                TEXT NOT NULL,
  config_sealed       TEXT NOT NULL,
  status              TEXT NOT NULL CHECK(status IN ('active','disabled')),
  sync_minutes        INTEGER NOT NULL DEFAULT 15 CHECK(sync_minutes >= 1),
  stale_after_minutes INTEGER NOT NULL DEFAULT 60 CHECK(stale_after_minutes >= 1),
  last_sync_at        TEXT,
  last_ok_at          TEXT,
  last_status         TEXT CHECK(last_status IN ('ok','partial','failed')),
  last_error          TEXT,
  created_by          TEXT NOT NULL,
  created_at          TEXT NOT NULL
);

CREATE TABLE legal_connector_links(
  id           TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES legal_connectors(id),
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  remote_path  TEXT NOT NULL,
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  removed_at   TEXT,
  UNIQUE(connector_id, matter_id, remote_path)
);
CREATE INDEX ix_links_matter ON legal_connector_links(matter_id);

CREATE TABLE legal_connector_runs(
  id           TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES legal_connectors(id),
  started_at   TEXT NOT NULL,
  finished_at  TEXT,
  status       TEXT CHECK(status IN ('ok','partial','failed')),
  report_json  TEXT
);
CREATE INDEX ix_runs_connector ON legal_connector_runs(connector_id, started_at);

-- Where a document came from, and who its source says may see it.
ALTER TABLE legal_documents ADD COLUMN connector_id TEXT;
ALTER TABLE legal_documents ADD COLUMN link_id TEXT;
ALTER TABLE legal_documents ADD COLUMN remote_id TEXT;
ALTER TABLE legal_documents ADD COLUMN remote_version TEXT;
ALTER TABLE legal_documents ADD COLUMN acl_restricted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE legal_documents ADD COLUMN source_state TEXT;
CREATE INDEX ix_documents_remote ON legal_documents(link_id, remote_id);

CREATE TABLE legal_document_acl(
  document_id TEXT NOT NULL REFERENCES legal_documents(id),
  user_id     TEXT NOT NULL,
  PRIMARY KEY(document_id, user_id)
);

-- A document removed at its source is hidden at once and queued for the usual approved purge.
CREATE TABLE legal_deletions_new(
  id                TEXT PRIMARY KEY,
  org_id            TEXT NOT NULL,
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  object_type       TEXT NOT NULL CHECK(object_type IN ('document','meeting_transcript')),
  object_id         TEXT NOT NULL,
  object_label      TEXT,
  object_label_sha256 TEXT NOT NULL,
  reason            TEXT NOT NULL,
  origin            TEXT NOT NULL CHECK(origin IN ('person','retention','connector')),
  soft_deleted      INTEGER NOT NULL DEFAULT 0,
  status            TEXT NOT NULL CHECK(status IN ('requested','purged','rejected','restored','blocked')),
  requested_by      TEXT NOT NULL,
  requested_at      TEXT NOT NULL,
  decided_by        TEXT,
  decided_at        TEXT,
  decision_note     TEXT,
  purged_at         TEXT,
  report_json       TEXT NOT NULL,
  verification_json TEXT
);
INSERT INTO legal_deletions_new (id, org_id, matter_id, object_type, object_id, object_label, object_label_sha256, reason, origin, soft_deleted, status, requested_by, requested_at, decided_by, decided_at, decision_note, purged_at, report_json, verification_json) SELECT id, org_id, matter_id, object_type, object_id, object_label, object_label_sha256, reason, origin, soft_deleted, status, requested_by, requested_at, decided_by, decided_at, decision_note, purged_at, report_json, verification_json FROM legal_deletions;
DROP TABLE legal_deletions;
ALTER TABLE legal_deletions_new RENAME TO legal_deletions;
CREATE INDEX ix_deletions_matter ON legal_deletions(matter_id, status);
CREATE INDEX ix_deletions_object ON legal_deletions(object_type, object_id);
