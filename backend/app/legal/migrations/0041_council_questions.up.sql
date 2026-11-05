-- EchoMind Counsel Intelligence, iteration 2 (owner's request 2026-11-05): questions worth putting to the Council, found by
-- rules in the matter's own records (council/discovery.py), never written by the model. The fingerprint (detector and exact
-- sources) keeps a question from being suggested twice; a dismissal keeps its reason; a question whose finding is gone is stale.
CREATE TABLE legal_council_questions(
  id               TEXT PRIMARY KEY,
  org_id           TEXT NOT NULL,
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  origin           TEXT NOT NULL CHECK(origin IN ('discovered','manual')),
  detector         TEXT NOT NULL,
  fingerprint      TEXT NOT NULL,
  title            TEXT NOT NULL,
  reason           TEXT NOT NULL,
  priority         TEXT NOT NULL CHECK(priority IN ('high','medium')),
  score            REAL NOT NULL DEFAULT 0,
  sources_json     TEXT NOT NULL DEFAULT '[]',
  status           TEXT NOT NULL CHECK(status IN ('suggested','convened','dismissed','stale')),
  review_id        TEXT,
  decided_by       TEXT,
  decided_reason   TEXT,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  UNIQUE (matter_id, fingerprint)
);
CREATE INDEX ix_council_questions_matter ON legal_council_questions(matter_id, status, score);
-- The question a Council run was convened from (NULL for a question a lawyer typed).
ALTER TABLE legal_panel_reviews ADD COLUMN question_id TEXT;
