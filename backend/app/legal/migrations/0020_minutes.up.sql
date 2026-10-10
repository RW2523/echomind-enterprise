-- S2.6 Boardroom legal mode (MEET-01; D-042), with the review and approval foundation S2.7 extends.
-- Agenda: what the meeting will cover, and the documents each item relies on (the board pack).
CREATE TABLE legal_agenda_items(
  id                TEXT PRIMARY KEY,
  meeting_id        TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  position          INTEGER NOT NULL,
  title             TEXT NOT NULL,
  document_ids_json TEXT NOT NULL DEFAULT '[]',
  created_by        TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  removed_at        TEXT
);
CREATE INDEX ix_agenda_meeting ON legal_agenda_items(meeting_id, position);

-- Each segment records the agenda item the meeting was on when it was said (set at commit, like the speaker).
ALTER TABLE legal_transcript_segments ADD COLUMN agenda_item_id TEXT;

-- Capture marks gain "agenda": which item the meeting moved to, and when.
CREATE TABLE legal_capture_marks_new(
  id             TEXT PRIMARY KEY,
  capture_id     TEXT NOT NULL REFERENCES legal_captures(id),
  meeting_id     TEXT NOT NULL REFERENCES legal_meetings(id),
  kind           TEXT NOT NULL CHECK(kind IN ('speaker','degraded','recovered','gap','agenda')),
  participant_id TEXT,
  detail         TEXT,
  audio_ms       INTEGER,
  at             TEXT NOT NULL,
  by             TEXT
);
INSERT INTO legal_capture_marks_new SELECT id, capture_id, meeting_id, kind, participant_id, detail, audio_ms, at, by FROM legal_capture_marks;
DROP TRIGGER capture_marks_no_update;
DROP INDEX ix_capture_marks;
DROP TABLE legal_capture_marks;
ALTER TABLE legal_capture_marks_new RENAME TO legal_capture_marks;
CREATE INDEX ix_capture_marks ON legal_capture_marks(meeting_id, at);
CREATE TRIGGER capture_marks_no_update BEFORE UPDATE ON legal_capture_marks
  BEGIN SELECT RAISE(ABORT, 'capture marks are append-only'); END;

-- AI artifacts: drafts the model produced from matter content, versioned. Content is never edited in
-- place: a change is a new version. Each version's content is hashed; an approval binds that hash.
CREATE TABLE legal_ai_artifacts(
  id               TEXT PRIMARY KEY,
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  kind             TEXT NOT NULL CHECK(kind IN ('minutes')),
  subject_type     TEXT NOT NULL CHECK(subject_type IN ('meeting')),
  subject_id       TEXT NOT NULL,
  version          INTEGER NOT NULL,
  status           TEXT NOT NULL CHECK(status IN ('draft','approved','superseded','withdrawn')),
  content_json     TEXT NOT NULL,
  content_sha256   TEXT NOT NULL,
  inputs_json      TEXT NOT NULL DEFAULT '{}',
  created_by       TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  origin           TEXT NOT NULL CHECK(origin IN ('model','person')),
  document_id      TEXT,
  UNIQUE(subject_type, subject_id, kind, version)
);
CREATE INDEX ix_artifacts_subject ON legal_ai_artifacts(subject_type, subject_id, kind, version);
CREATE TRIGGER artifacts_content_fixed BEFORE UPDATE OF content_json, content_sha256, created_by, version ON legal_ai_artifacts
  BEGIN SELECT RAISE(ABORT, 'artifact content is fixed; a change is a new version'); END;

-- Review decisions: who approved (or asked for changes to) which content, by its hash. Append-only.
CREATE TABLE legal_review_approvals(
  id             TEXT PRIMARY KEY,
  artifact_id    TEXT NOT NULL REFERENCES legal_ai_artifacts(id),
  matter_id      TEXT NOT NULL REFERENCES matters(id),
  decision       TEXT NOT NULL CHECK(decision IN ('approved','changes_requested')),
  content_sha256 TEXT NOT NULL,
  reason         TEXT,
  by             TEXT NOT NULL,
  at             TEXT NOT NULL
);
CREATE INDEX ix_approvals_artifact ON legal_review_approvals(artifact_id);
CREATE TRIGGER approvals_no_update BEFORE UPDATE ON legal_review_approvals
  BEGIN SELECT RAISE(ABORT, 'review decisions are append-only'); END;
