CREATE TABLE legal_research_turns_old(
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
INSERT INTO legal_research_turns_old (id, org_id, matter_id, owner_id, question, answer, citations_json, status, auth, created_at, finished_at)
  SELECT id, org_id, matter_id, owner_id, question, answer, citations_json,
         CASE WHEN status IN ('partial','unverified') THEN 'answered' ELSE status END, auth, created_at, finished_at
  FROM legal_research_turns;
DROP INDEX ix_research_owner;
DROP TABLE legal_research_turns;
ALTER TABLE legal_research_turns_old RENAME TO legal_research_turns;
CREATE INDEX ix_research_owner ON legal_research_turns(matter_id, owner_id, created_at);
