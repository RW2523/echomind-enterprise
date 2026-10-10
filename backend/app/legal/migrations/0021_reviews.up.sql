-- S2.7 — the review and approval framework (D-043).

-- AI artifacts gain a second kind: a research answer sent for review by the person who asked.
CREATE TABLE legal_ai_artifacts_new(
  id               TEXT PRIMARY KEY,
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  kind             TEXT NOT NULL CHECK(kind IN ('minutes','research_note')),
  subject_type     TEXT NOT NULL CHECK(subject_type IN ('meeting','research_turn')),
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
INSERT INTO legal_ai_artifacts_new SELECT id, matter_id, kind, subject_type, subject_id, version, status, content_json, content_sha256,
  inputs_json, created_by, created_at, origin, document_id FROM legal_ai_artifacts;
DROP TRIGGER artifacts_content_fixed;
DROP INDEX ix_artifacts_subject;
DROP TABLE legal_ai_artifacts;
ALTER TABLE legal_ai_artifacts_new RENAME TO legal_ai_artifacts;
CREATE INDEX ix_artifacts_subject ON legal_ai_artifacts(subject_type, subject_id, kind, version);
CREATE INDEX ix_artifacts_review ON legal_ai_artifacts(matter_id, status);
CREATE TRIGGER artifacts_content_fixed BEFORE UPDATE OF content_json, content_sha256, created_by, version ON legal_ai_artifacts
  BEGIN SELECT RAISE(ABORT, 'artifact content is fixed; a change is a new version'); END;

-- Filings: which approved version became which document version, with the body's hash, so an export
-- of that document can show the approval and refuse a copy changed after it. Never updated. The
-- approval is copied here: the filed document outlives its draft (a purge of a source it quotes
-- removes the draft, never the filed document), and goes only with that document.
CREATE TABLE legal_artifact_filings(
  id               TEXT PRIMARY KEY,
  artifact_id      TEXT NOT NULL,
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  kind             TEXT NOT NULL,
  artifact_version INTEGER NOT NULL,
  document_id      TEXT NOT NULL,
  version_id       TEXT NOT NULL,
  body_sha256      TEXT NOT NULL,
  content_sha256   TEXT NOT NULL,
  approved_by      TEXT,
  approved_at      TEXT,
  filed_by         TEXT NOT NULL,
  filed_at         TEXT NOT NULL
);
CREATE INDEX ix_filings_document ON legal_artifact_filings(document_id);
CREATE TRIGGER filings_no_update BEFORE UPDATE ON legal_artifact_filings
  BEGIN SELECT RAISE(ABORT, 'filings are never changed'); END;
-- Minutes filed under S2.6 (the document's first version is what was filed).
INSERT INTO legal_artifact_filings (id, artifact_id, matter_id, kind, artifact_version, document_id, version_id, body_sha256, content_sha256,
                                    approved_by, approved_at, filed_by, filed_at)
  SELECT lower(hex(randomblob(16))), a.id, a.matter_id, a.kind, a.version, a.document_id, v.id, v.sha256, a.content_sha256,
         (SELECT r.by FROM legal_review_approvals r WHERE r.artifact_id = a.id AND r.decision = 'approved' ORDER BY r.at DESC LIMIT 1),
         (SELECT r.at FROM legal_review_approvals r WHERE r.artifact_id = a.id AND r.decision = 'approved' ORDER BY r.at DESC LIMIT 1),
         v.uploaded_by, v.uploaded_at
  FROM legal_ai_artifacts a JOIN legal_document_versions v ON v.document_id = a.document_id AND v.number = 1
  WHERE a.document_id IS NOT NULL;

-- Exports are bound to the text their redactions were planned on.
ALTER TABLE legal_exports ADD COLUMN source_sha256 TEXT;
