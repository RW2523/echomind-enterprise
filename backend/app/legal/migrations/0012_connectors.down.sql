CREATE TABLE legal_deletions_new(
  id                TEXT PRIMARY KEY,
  org_id            TEXT NOT NULL,
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  object_type       TEXT NOT NULL CHECK(object_type IN ('document','meeting_transcript')),
  object_id         TEXT NOT NULL,
  object_label      TEXT,
  object_label_sha256 TEXT NOT NULL,
  reason            TEXT NOT NULL,
  origin            TEXT NOT NULL CHECK(origin IN ('person','retention')),
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
INSERT INTO legal_deletions_new (id, org_id, matter_id, object_type, object_id, object_label, object_label_sha256, reason, origin, soft_deleted, status, requested_by, requested_at, decided_by, decided_at, decision_note, purged_at, report_json, verification_json) SELECT id, org_id, matter_id, object_type, object_id, object_label, object_label_sha256, reason, CASE origin WHEN 'connector' THEN 'person' ELSE origin END, soft_deleted, status, requested_by, requested_at, decided_by, decided_at, decision_note, purged_at, report_json, verification_json FROM legal_deletions;
DROP TABLE legal_deletions;
ALTER TABLE legal_deletions_new RENAME TO legal_deletions;
CREATE INDEX ix_deletions_matter ON legal_deletions(matter_id, status);
CREATE INDEX ix_deletions_object ON legal_deletions(object_type, object_id);
DROP TABLE legal_document_acl;
DROP INDEX ix_documents_remote;
ALTER TABLE legal_documents DROP COLUMN source_state;
ALTER TABLE legal_documents DROP COLUMN acl_restricted;
ALTER TABLE legal_documents DROP COLUMN remote_version;
ALTER TABLE legal_documents DROP COLUMN remote_id;
ALTER TABLE legal_documents DROP COLUMN link_id;
ALTER TABLE legal_documents DROP COLUMN connector_id;
DROP TABLE legal_connector_runs;
DROP TABLE legal_connector_links;
DROP TABLE legal_connectors;
