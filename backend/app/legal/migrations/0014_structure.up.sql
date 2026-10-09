-- S2.1b clauses and defined terms (RAG-01; D-035). Offsets point into legal_version_text.text.
CREATE TABLE legal_clauses(
  id         TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES legal_document_versions(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  number     TEXT NOT NULL,
  label      TEXT NOT NULL,
  heading    TEXT,
  level      INTEGER NOT NULL,
  parent     TEXT,
  char_start INTEGER NOT NULL,
  char_end   INTEGER NOT NULL,
  page       INTEGER
);
CREATE INDEX ix_legal_clauses_version ON legal_clauses(version_id, char_start);

CREATE TABLE legal_defined_terms(
  id         TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES legal_document_versions(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  term       TEXT NOT NULL,
  char_start INTEGER NOT NULL,
  char_end   INTEGER NOT NULL,
  term_start INTEGER NOT NULL,
  term_end   INTEGER NOT NULL
);
CREATE INDEX ix_legal_terms_version ON legal_defined_terms(version_id);
