-- S2.8 — evaluation runs (OPS-02, D-044): what was measured, on which code, models and dataset.
-- The summary only; each run's items stay in its report file under eval/law/reports.
CREATE TABLE legal_evaluation_runs(
  id              TEXT PRIMARY KEY,
  suite           TEXT NOT NULL,
  set_name        TEXT NOT NULL,
  commit_ref      TEXT,
  models_json     TEXT NOT NULL DEFAULT '{}',
  dataset_sha256  TEXT,
  summary_json    TEXT NOT NULL DEFAULT '{}',
  report          TEXT NOT NULL,
  report_sha256   TEXT NOT NULL UNIQUE,
  run_at          TEXT,
  imported_at     TEXT NOT NULL
);
CREATE INDEX ix_evaluation_runs ON legal_evaluation_runs(suite, set_name, run_at);
CREATE TRIGGER evaluation_runs_no_update BEFORE UPDATE ON legal_evaluation_runs
  BEGIN SELECT RAISE(ABORT, 'evaluation runs are a record; import a new report instead'); END;
