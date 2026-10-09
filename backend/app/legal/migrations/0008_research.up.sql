-- 0008 research (S1.7, SEC-03 / RAG-01): matter-scoped research questions and their answers.
-- History is private to its owner within the matter. Cited sources are stored as references
-- (document, version, offsets) plus the quoted window, so an answer can be re-checked later.
CREATE TABLE legal_research_turns(
  id            TEXT PRIMARY KEY,
  org_id        TEXT NOT NULL,
  matter_id     TEXT NOT NULL REFERENCES matters(id),
  owner_id      TEXT NOT NULL,
  question      TEXT NOT NULL,
  answer        TEXT,
  citations_json TEXT,
  status        TEXT NOT NULL CHECK(status IN ('answered','abstained','scope_revoked','error')),
  auth          TEXT,
  created_at    TEXT NOT NULL,
  finished_at   TEXT
);
CREATE INDEX ix_research_owner ON legal_research_turns(matter_id, owner_id, created_at);
