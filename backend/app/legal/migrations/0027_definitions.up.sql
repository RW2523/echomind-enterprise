-- S3.5 — defined terms and cross-references (DOC-04, D-049). A check is kept with its document; lawyers'
-- notes on its findings are append-only. Fixes become redlines (S3.3), not changes made here.
CREATE TABLE legal_definition_checks(
  id           TEXT PRIMARY KEY,
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  document_id  TEXT NOT NULL,
  version_id   TEXT NOT NULL,
  results_json TEXT NOT NULL,
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX ix_definition_checks_document ON legal_definition_checks(matter_id, document_id, created_at);
CREATE TABLE legal_definition_notes(
  id         TEXT PRIMARY KEY,
  check_id   TEXT NOT NULL REFERENCES legal_definition_checks(id),
  finding_id TEXT NOT NULL,
  note       TEXT NOT NULL,
  by         TEXT NOT NULL,
  at         TEXT NOT NULL
);
CREATE INDEX ix_definition_notes ON legal_definition_notes(check_id);
CREATE TRIGGER definition_notes_no_update BEFORE UPDATE ON legal_definition_notes
  BEGIN SELECT RAISE(ABORT, 'notes are append-only'); END;
