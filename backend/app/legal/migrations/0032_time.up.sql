-- S4.3 — time-entry drafts from audited activity (OPT-03, D-056). One suggested entry per person, matter and day;
-- the person edits, approves or discards it. Approved entries are exported as CSV; nothing is sent anywhere.
CREATE TABLE legal_time_entries(
  id                  TEXT PRIMARY KEY,
  org_id              TEXT NOT NULL,
  user_id             TEXT NOT NULL,
  matter_id           TEXT NOT NULL REFERENCES matters(id),
  work_date           TEXT NOT NULL,
  source              TEXT NOT NULL CHECK(source IN ('suggested','manual')),
  minutes_suggested   INTEGER,
  narrative_suggested TEXT,
  sessions_json       TEXT NOT NULL DEFAULT '[]',
  counts_json         TEXT NOT NULL DEFAULT '{}',
  minutes             INTEGER NOT NULL CHECK(minutes > 0 AND minutes % 6 = 0),
  narrative           TEXT NOT NULL,
  edited              INTEGER NOT NULL DEFAULT 0,
  status              TEXT NOT NULL CHECK(status IN ('draft','approved','discarded')),
  created_at          TEXT NOT NULL,
  updated_at          TEXT,
  approved_at         TEXT,
  version             INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX ux_time_suggested ON legal_time_entries(user_id, matter_id, work_date) WHERE source = 'suggested';
CREATE INDEX ix_time_user_date ON legal_time_entries(user_id, work_date);
