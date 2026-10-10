DROP TRIGGER approvals_no_update;
DROP INDEX ix_approvals_artifact;
DROP TABLE legal_review_approvals;
DROP TRIGGER artifacts_content_fixed;
DROP INDEX ix_artifacts_subject;
DROP TABLE legal_ai_artifacts;
DELETE FROM legal_capture_marks WHERE kind = 'agenda';
CREATE TABLE legal_capture_marks_old(
  id             TEXT PRIMARY KEY,
  capture_id     TEXT NOT NULL REFERENCES legal_captures(id),
  meeting_id     TEXT NOT NULL REFERENCES legal_meetings(id),
  kind           TEXT NOT NULL CHECK(kind IN ('speaker','degraded','recovered','gap')),
  participant_id TEXT,
  detail         TEXT,
  audio_ms       INTEGER,
  at             TEXT NOT NULL,
  by             TEXT
);
INSERT INTO legal_capture_marks_old SELECT id, capture_id, meeting_id, kind, participant_id, detail, audio_ms, at, by FROM legal_capture_marks;
DROP TRIGGER capture_marks_no_update;
DROP INDEX ix_capture_marks;
DROP TABLE legal_capture_marks;
ALTER TABLE legal_capture_marks_old RENAME TO legal_capture_marks;
CREATE INDEX ix_capture_marks ON legal_capture_marks(meeting_id, at);
CREATE TRIGGER capture_marks_no_update BEFORE UPDATE ON legal_capture_marks
  BEGIN SELECT RAISE(ABORT, 'capture marks are append-only'); END;
DROP TRIGGER segments_no_update;
ALTER TABLE legal_transcript_segments DROP COLUMN agenda_item_id;
CREATE TRIGGER segments_no_update BEFORE UPDATE ON legal_transcript_segments
  BEGIN SELECT RAISE(ABORT, 'transcript segments are immutable; corrections are revisions'); END;
DROP INDEX ix_agenda_meeting;
DROP TABLE legal_agenda_items;
