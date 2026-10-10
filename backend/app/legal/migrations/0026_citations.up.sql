-- S3.4 — the citation and quote checker (DOC-03, T07, D-048). A check is kept with its document; lawyers' notes
-- on its items are append-only.
CREATE TABLE legal_citation_checks(
  id           TEXT PRIMARY KEY,
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  document_id  TEXT NOT NULL,
  version_id   TEXT NOT NULL,
  results_json TEXT NOT NULL,
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX ix_citation_checks_document ON legal_citation_checks(matter_id, document_id, created_at);
CREATE TABLE legal_citation_notes(
  id        TEXT PRIMARY KEY,
  check_id  TEXT NOT NULL REFERENCES legal_citation_checks(id),
  item_id   TEXT NOT NULL,
  note      TEXT NOT NULL,
  by        TEXT NOT NULL,
  at        TEXT NOT NULL
);
CREATE INDEX ix_citation_notes ON legal_citation_notes(check_id);
CREATE TRIGGER citation_notes_no_update BEFORE UPDATE ON legal_citation_notes
  BEGIN SELECT RAISE(ABORT, 'notes are append-only'); END;
