-- S2.4 legal transcription (LIVE-01; D-017, D-040).
-- Segments keep the speech model's raw text and become immutable: corrections are revisions, which
-- record who changed what, when and why; the text shown is derived. Each segment also carries the
-- speaker marked when it was spoken (a roster participant) and its wall-clock times, so pauses show.
ALTER TABLE legal_transcript_segments ADD COLUMN speaker_participant_id TEXT;
ALTER TABLE legal_transcript_segments ADD COLUMN wall_start TEXT;
ALTER TABLE legal_transcript_segments ADD COLUMN wall_end TEXT;
CREATE TRIGGER segments_no_update BEFORE UPDATE ON legal_transcript_segments
  BEGIN SELECT RAISE(ABORT, 'transcript segments are immutable; corrections are revisions'); END;

CREATE TABLE legal_transcript_revisions(
  id                     TEXT PRIMARY KEY,
  segment_id             TEXT NOT NULL REFERENCES legal_transcript_segments(id),
  meeting_id             TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id              TEXT NOT NULL REFERENCES matters(id),
  kind                   TEXT NOT NULL CHECK(kind IN ('text','speaker')),
  text                   TEXT,
  speaker_participant_id TEXT,
  reason                 TEXT NOT NULL,
  source                 TEXT NOT NULL CHECK(source IN ('person','vocabulary')),
  provenance_json        TEXT NOT NULL DEFAULT '{}',
  by                     TEXT NOT NULL,
  at                     TEXT NOT NULL
);
CREATE INDEX ix_revisions_segment ON legal_transcript_revisions(segment_id, at);
CREATE INDEX ix_revisions_meeting ON legal_transcript_revisions(meeting_id);
CREATE TRIGGER revisions_no_update BEFORE UPDATE ON legal_transcript_revisions
  BEGIN SELECT RAISE(ABORT, 'transcript revisions are append-only'); END;

-- What happened while capturing, beyond start/pause/stop: who was speaking from when, and where the
-- transcript is degraded (the model fell behind) or has a gap (a transcription error). Append-only.
CREATE TABLE legal_capture_marks(
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
CREATE INDEX ix_capture_marks ON legal_capture_marks(meeting_id, at);
CREATE TRIGGER capture_marks_no_update BEFORE UPDATE ON legal_capture_marks
  BEGIN SELECT RAISE(ABORT, 'capture marks are append-only'); END;

-- A finalised transcript is also a matter document, so research, compare, holds and purges apply.
ALTER TABLE legal_meetings ADD COLUMN transcript_document_id TEXT;
ALTER TABLE legal_meetings ADD COLUMN transcript_finalised_at TEXT;
ALTER TABLE legal_meetings ADD COLUMN transcript_finalised_by TEXT;
