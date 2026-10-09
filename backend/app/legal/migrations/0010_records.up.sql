-- 0010 records (S1.9, SEC-07 and SEC-06 labels; policies P-002, P-003, P-004 safe defaults):
-- privilege labels, legal holds, approved deletion with verification, retention policies and runs.
ALTER TABLE matters ADD COLUMN closed_at TEXT;
ALTER TABLE legal_documents ADD COLUMN deleted_at TEXT;
ALTER TABLE legal_documents ADD COLUMN deletion_id TEXT;
ALTER TABLE legal_meetings ADD COLUMN transcript_deleted_at TEXT;
ALTER TABLE legal_meetings ADD COLUMN transcript_deletion_id TEXT;

-- Advisory privilege labels: every change is a row (the latest "set" is current; later
-- "suggested" rows are open suggestions). Purged with the object they label.
CREATE TABLE legal_labels(
  id          TEXT PRIMARY KEY,
  org_id      TEXT NOT NULL,
  matter_id   TEXT NOT NULL REFERENCES matters(id),
  object_type TEXT NOT NULL CHECK(object_type IN ('document','meeting')),
  object_id   TEXT NOT NULL,
  label       TEXT NOT NULL CHECK(label IN ('privileged','work_product','confidential','not_privileged','unreviewed')),
  kind        TEXT NOT NULL CHECK(kind IN ('set','suggested')),
  source      TEXT NOT NULL CHECK(source IN ('person','detector')),
  reason_code TEXT,
  set_by      TEXT,
  set_at      TEXT NOT NULL
);
CREATE INDEX ix_labels_object ON legal_labels(object_type, object_id, set_at);

-- Legal holds (P-003): requested by a records administrator, in force once a different person
-- approves; release likewise needs two people. Held material is never purged by any path.
CREATE TABLE legal_holds(
  id                   TEXT PRIMARY KEY,
  org_id               TEXT NOT NULL,
  name                 TEXT NOT NULL,
  reason               TEXT NOT NULL,
  status               TEXT NOT NULL CHECK(status IN ('requested','active','release_requested','released','rejected')),
  requested_by         TEXT NOT NULL,
  requested_at         TEXT NOT NULL,
  approved_by          TEXT,
  approved_at          TEXT,
  release_requested_by TEXT,
  release_requested_at TEXT,
  released_by          TEXT,
  released_at          TEXT
);
CREATE TABLE legal_hold_scopes(
  hold_id     TEXT NOT NULL REFERENCES legal_holds(id),
  object_type TEXT NOT NULL CHECK(object_type IN ('matter')),
  object_id   TEXT NOT NULL,
  PRIMARY KEY(hold_id, object_type, object_id)
);
CREATE INDEX ix_hold_scopes_object ON legal_hold_scopes(object_type, object_id);

-- Deletion requests: soft delete → dry-run report → approval by a second person → purge (hold
-- re-checked in the same transaction) → verification. The record outlives the purge; the
-- object's title is dropped at purge (only its SHA-256 is kept).
CREATE TABLE legal_deletions(
  id                TEXT PRIMARY KEY,
  org_id            TEXT NOT NULL,
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  object_type       TEXT NOT NULL CHECK(object_type IN ('document','meeting_transcript')),
  object_id         TEXT NOT NULL,
  object_label      TEXT,
  object_label_sha256 TEXT NOT NULL,
  reason            TEXT NOT NULL,
  origin            TEXT NOT NULL CHECK(origin IN ('person','retention')),
  soft_deleted      INTEGER NOT NULL DEFAULT 0,
  status            TEXT NOT NULL CHECK(status IN ('requested','purged','rejected','restored','blocked')),
  requested_by      TEXT NOT NULL,
  requested_at      TEXT NOT NULL,
  decided_by        TEXT,
  decided_at        TEXT,
  decision_note     TEXT,
  purged_at         TEXT,
  report_json       TEXT NOT NULL,
  verification_json TEXT
);
CREATE INDEX ix_deletions_matter ON legal_deletions(matter_id, status);
CREATE INDEX ix_deletions_object ON legal_deletions(object_type, object_id);

-- Retention (P-002): drafted by a records administrator, approved by a managing partner. Runs
-- are dry runs unless LEGAL_RETENTION_APPLY is on, and even then only create deletion requests.
CREATE TABLE legal_retention_policies(
  id               TEXT PRIMARY KEY,
  org_id           TEXT NOT NULL,
  object_type      TEXT NOT NULL CHECK(object_type IN ('document','meeting_transcript')),
  days_after_close INTEGER NOT NULL CHECK(days_after_close >= 0),
  note             TEXT,
  status           TEXT NOT NULL CHECK(status IN ('draft','approved','retired')),
  drafted_by       TEXT NOT NULL,
  drafted_at       TEXT NOT NULL,
  approved_by      TEXT,
  approved_at      TEXT
);
CREATE TABLE legal_retention_runs(
  id          TEXT PRIMARY KEY,
  org_id      TEXT NOT NULL,
  mode        TEXT NOT NULL CHECK(mode IN ('dry_run','apply')),
  run_by      TEXT,
  run_at      TEXT NOT NULL,
  report_json TEXT NOT NULL
);
