-- S4.5 — the four-perspective review panel (OPT-05, D-057). A review is a question asked of a matter, run as a background
-- job; each run (and each run again with the lawyer's facts) is a new version of its panel artifact, approved by a partner.
CREATE TABLE legal_panel_reviews(
  id          TEXT PRIMARY KEY,
  org_id      TEXT NOT NULL,
  matter_id   TEXT NOT NULL REFERENCES matters(id),
  question    TEXT NOT NULL,
  specialist  TEXT NOT NULL,
  facts_json  TEXT NOT NULL DEFAULT '[]',
  status      TEXT NOT NULL CHECK(status IN ('queued','running','done','needs_facts','failed','not_allowed')),
  artifact_id TEXT,
  created_by  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  finished_at TEXT
);
CREATE INDEX ix_panel_matter ON legal_panel_reviews(matter_id, created_at);
