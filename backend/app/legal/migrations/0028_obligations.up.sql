-- S3.6 — obligations and deadlines (DOC-05, D-050). Read from a document by rules (or added by a lawyer from its
-- exact words); proposed until a lawyer confirms. Every change is an event with who, when and why.
CREATE TABLE legal_obligations(
  id            TEXT PRIMARY KEY,
  matter_id     TEXT NOT NULL REFERENCES matters(id),
  document_id   TEXT NOT NULL,
  version_id    TEXT NOT NULL,
  clause        TEXT,
  kind          TEXT NOT NULL,
  party         TEXT,
  action_text   TEXT NOT NULL,
  when_text     TEXT,
  rule_json     TEXT NOT NULL DEFAULT '{}',
  due_date      TEXT,
  due_note      TEXT,
  timezone      TEXT NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('proposed','confirmed','done','dismissed')),
  owner_id      TEXT,
  evidence_json TEXT NOT NULL,
  source        TEXT NOT NULL CHECK (source IN ('rules','model','person')),
  confirmed_by  TEXT,
  confirmed_at  TEXT,
  version       INTEGER NOT NULL DEFAULT 1,
  created_by    TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX ix_obligations_matter ON legal_obligations(matter_id, due_date);
CREATE INDEX ix_obligations_document ON legal_obligations(document_id);
CREATE TABLE legal_obligation_events(
  id            TEXT PRIMARY KEY,
  obligation_id TEXT NOT NULL REFERENCES legal_obligations(id),
  action        TEXT NOT NULL,
  changes_json  TEXT NOT NULL DEFAULT '{}',
  reason        TEXT,
  by            TEXT NOT NULL,
  at            TEXT NOT NULL
);
CREATE INDEX ix_obligation_events ON legal_obligation_events(obligation_id, at);
CREATE TRIGGER obligation_events_no_update BEFORE UPDATE ON legal_obligation_events
  BEGIN SELECT RAISE(ABORT, 'obligation history is append-only'); END;
