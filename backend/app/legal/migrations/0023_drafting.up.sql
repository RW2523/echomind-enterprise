-- S3.1 — firm templates and legal drafts (DOC-01, D-045).

-- Firm templates: the firm's own precedents, versioned. A version's wording and fields never change;
-- its firm styles may be attached while it is a draft. A managing partner other than the author
-- approves a version (binding its hash) before anyone drafts from it.
CREATE TABLE legal_templates(
  id          TEXT PRIMARY KEY,
  org_id      TEXT NOT NULL,
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK(kind IN ('engagement_letter','nda','demand_letter','memo','contract_review','board_minutes','custom')),
  preset_key  TEXT,
  created_by  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  UNIQUE(org_id, preset_key)
);
CREATE TABLE legal_template_versions(
  id                 TEXT PRIMARY KEY,
  template_id        TEXT NOT NULL REFERENCES legal_templates(id),
  version            INTEGER NOT NULL,
  status             TEXT NOT NULL CHECK(status IN ('draft','approved','retired')),
  body               TEXT NOT NULL,
  fields_json        TEXT NOT NULL,
  styles_name        TEXT,
  styles_sha256      TEXT,
  styles_wrapped_key TEXT,
  content_sha256     TEXT NOT NULL,
  created_by         TEXT NOT NULL,
  created_at         TEXT NOT NULL,
  approved_by        TEXT,
  approved_at        TEXT,
  UNIQUE(template_id, version)
);
CREATE TRIGGER template_wording_fixed BEFORE UPDATE OF body, fields_json, version, created_by, template_id ON legal_template_versions
  BEGIN SELECT RAISE(ABORT, 'a template version is fixed; a change is a new version'); END;
CREATE TRIGGER template_styles_until_approved BEFORE UPDATE OF styles_name, styles_sha256, styles_wrapped_key, content_sha256 ON legal_template_versions
  WHEN OLD.status <> 'draft'
  BEGIN SELECT RAISE(ABORT, 'an approved template version is fixed'); END;

-- Drafts are AI artifacts (S2.7): a third kind.
CREATE TABLE legal_ai_artifacts_new(
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
INSERT INTO legal_ai_artifacts_new SELECT id, matter_id, kind, subject_type, subject_id, version, status, content_json, content_sha256,
  inputs_json, created_by, created_at, origin, document_id FROM legal_ai_artifacts;
DROP TRIGGER artifacts_content_fixed;
DROP INDEX ix_artifacts_review;
DROP INDEX ix_artifacts_subject;
DROP TABLE legal_ai_artifacts;
ALTER TABLE legal_ai_artifacts_new RENAME TO legal_ai_artifacts;
CREATE INDEX ix_artifacts_subject ON legal_ai_artifacts(subject_type, subject_id, kind, version);
CREATE INDEX ix_artifacts_review ON legal_ai_artifacts(matter_id, status);
CREATE TRIGGER artifacts_content_fixed BEFORE UPDATE OF content_json, content_sha256, created_by, version ON legal_ai_artifacts
  BEGIN SELECT RAISE(ABORT, 'artifact content is fixed; a change is a new version'); END;
