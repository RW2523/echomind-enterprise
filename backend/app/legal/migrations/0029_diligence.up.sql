-- S3.7 — due diligence (DOC-06, D-051). Firm checklists are fixed versions approved by a managing partner other
-- than the author; a run matches a matter's documents to a checklist (a background job) and records issues, each
-- preliminary until a reviewer decides, with an append-only trail.
CREATE TABLE legal_dd_checklists(
  id             TEXT PRIMARY KEY,
  org_id         TEXT NOT NULL,
  family_id      TEXT NOT NULL,
  version        INTEGER NOT NULL,
  name           TEXT NOT NULL,
  content_json   TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('draft','approved','retired')),
  created_by     TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  approved_by    TEXT,
  approved_at    TEXT,
  UNIQUE (family_id, version)
);
CREATE TRIGGER dd_checklist_content_fixed BEFORE UPDATE OF content_json, content_sha256, version, family_id ON legal_dd_checklists
  BEGIN SELECT RAISE(ABORT, 'a checklist version is fixed; write a new version'); END;
CREATE TABLE legal_dd_runs(
  id            TEXT PRIMARY KEY,
  matter_id     TEXT NOT NULL REFERENCES matters(id),
  checklist_id  TEXT NOT NULL REFERENCES legal_dd_checklists(id),
  status        TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
  results_json  TEXT NOT NULL DEFAULT '{}',
  error         TEXT,
  created_by    TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  finished_at   TEXT
);
CREATE INDEX ix_dd_runs_matter ON legal_dd_runs(matter_id, created_at);
CREATE TABLE legal_dd_issues(
  id            TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL REFERENCES legal_dd_runs(id),
  matter_id     TEXT NOT NULL REFERENCES matters(id),
  item_id       TEXT NOT NULL,
  document_id   TEXT NOT NULL,
  version_id    TEXT NOT NULL,
  kind          TEXT NOT NULL,
  severity      TEXT NOT NULL CHECK (severity IN ('high','medium','low')),
  title         TEXT NOT NULL,
  detail        TEXT,
  evidence_json TEXT NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('preliminary','verified','dismissed')),
  reviewer_id   TEXT,
  decided_at    TEXT,
  version       INTEGER NOT NULL DEFAULT 1,
  created_at    TEXT NOT NULL
);
CREATE INDEX ix_dd_issues_run ON legal_dd_issues(run_id);
CREATE INDEX ix_dd_issues_document ON legal_dd_issues(document_id);
CREATE TABLE legal_dd_issue_events(
  id        TEXT PRIMARY KEY,
  issue_id  TEXT NOT NULL REFERENCES legal_dd_issues(id),
  action    TEXT NOT NULL,
  reason    TEXT,
  by        TEXT NOT NULL,
  at        TEXT NOT NULL
);
CREATE INDEX ix_dd_issue_events ON legal_dd_issue_events(issue_id, at);
CREATE TRIGGER dd_issue_events_no_update BEFORE UPDATE ON legal_dd_issue_events
  BEGIN SELECT RAISE(ABORT, 'the review trail is append-only'); END;
