ALTER TABLE legal_artifact_filings DROP COLUMN as_new_version;
DELETE FROM legal_review_approvals WHERE artifact_id IN (SELECT id FROM legal_ai_artifacts WHERE kind NOT IN ('minutes','research_note','legal_draft'));
DELETE FROM legal_ai_artifacts WHERE kind NOT IN ('minutes','research_note','legal_draft');
CREATE TABLE legal_ai_artifacts_old(
  id               TEXT PRIMARY KEY,
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  kind             TEXT NOT NULL CHECK(kind IN ('minutes','research_note','legal_draft')),
  subject_type     TEXT NOT NULL CHECK(subject_type IN ('meeting','research_turn','draft')),
  subject_id       TEXT NOT NULL,
  version          INTEGER NOT NULL,
  status           TEXT NOT NULL CHECK(status IN ('draft','approved','superseded','withdrawn')),
  content_json     TEXT NOT NULL,
  content_sha256   TEXT NOT NULL,
  inputs_json      TEXT NOT NULL DEFAULT '{}',
  created_by       TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  origin           TEXT NOT NULL CHECK(origin IN ('model','person')),
  document_id      TEXT,
  UNIQUE(subject_type, subject_id, kind, version)
);
INSERT INTO legal_ai_artifacts_old SELECT id, matter_id, kind, subject_type, subject_id, version, status, content_json, content_sha256,
  inputs_json, created_by, created_at, origin, document_id FROM legal_ai_artifacts;
DROP TRIGGER artifacts_content_fixed;
DROP INDEX ix_artifacts_review;
DROP INDEX ix_artifacts_subject;
DROP TABLE legal_ai_artifacts;
ALTER TABLE legal_ai_artifacts_old RENAME TO legal_ai_artifacts;
CREATE INDEX ix_artifacts_subject ON legal_ai_artifacts(subject_type, subject_id, kind, version);
CREATE INDEX ix_artifacts_review ON legal_ai_artifacts(matter_id, status);
CREATE TRIGGER artifacts_content_fixed BEFORE UPDATE OF content_json, content_sha256, created_by, version ON legal_ai_artifacts
  BEGIN SELECT RAISE(ABORT, 'artifact content is fixed; a change is a new version'); END;
