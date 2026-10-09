-- 0009 meetings (S1.8, SEC-05; policy P-001): meetings, participants, consent and capture.
-- Consent and capture timelines are append-only (triggers refuse UPDATE and DELETE): what was
-- agreed, withdrawn, started, paused and stopped is never rewritten. Raw audio is not stored by
-- default; each received batch of audio leaves a SHA-256 custody record instead.
CREATE TABLE legal_meetings(
  id             TEXT PRIMARY KEY,
  org_id         TEXT NOT NULL,
  matter_id      TEXT NOT NULL REFERENCES matters(id),
  title          TEXT NOT NULL,
  kind           TEXT NOT NULL DEFAULT 'other'
                 CHECK(kind IN ('client_call','internal','negotiation','interview','board','hearing','other')),
  scheduled_at   TEXT,
  status         TEXT NOT NULL DEFAULT 'planned' CHECK(status IN ('planned','ended')),
  consent_policy TEXT NOT NULL,
  created_by     TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  ended_at       TEXT,
  version        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX ix_meetings_matter ON legal_meetings(matter_id, created_at);

CREATE TABLE legal_meeting_participants(
  id         TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  name       TEXT NOT NULL,
  role       TEXT NOT NULL CHECK(role IN ('host','attorney','client','counterparty','witness','expert','other')),
  user_id    TEXT,
  added_by   TEXT NOT NULL,
  added_at   TEXT NOT NULL,
  removed_at TEXT,
  removed_by TEXT
);
CREATE INDEX ix_participants_meeting ON legal_meeting_participants(meeting_id);

CREATE TABLE legal_consent_events(
  id             TEXT PRIMARY KEY,
  meeting_id     TEXT NOT NULL REFERENCES legal_meetings(id),
  participant_id TEXT NOT NULL REFERENCES legal_meeting_participants(id),
  kind           TEXT NOT NULL CHECK(kind IN ('granted','withdrawn')),
  method         TEXT CHECK(method IN ('verbal_on_record','written','host_attestation')),
  statement_sha256 TEXT,
  recorded_by    TEXT NOT NULL,
  recorded_at    TEXT NOT NULL
);
CREATE INDEX ix_consent_meeting ON legal_consent_events(meeting_id, recorded_at);
CREATE TRIGGER consent_no_update BEFORE UPDATE ON legal_consent_events BEGIN SELECT RAISE(ABORT, 'consent events are append-only'); END;
CREATE TRIGGER consent_no_delete BEFORE DELETE ON legal_consent_events BEGIN SELECT RAISE(ABORT, 'consent events are append-only'); END;

CREATE TABLE legal_captures(
  id               TEXT PRIMARY KEY,
  meeting_id       TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id        TEXT NOT NULL REFERENCES matters(id),
  started_by       TEXT NOT NULL,
  started_at       TEXT NOT NULL,
  consent_snapshot TEXT NOT NULL,
  transcription    TEXT NOT NULL CHECK(transcription IN ('available','unavailable'))
);
CREATE INDEX ix_captures_meeting ON legal_captures(meeting_id);

CREATE TABLE legal_capture_events(
  id         TEXT PRIMARY KEY,
  capture_id TEXT NOT NULL REFERENCES legal_captures(id),
  kind       TEXT NOT NULL CHECK(kind IN ('started','paused','resumed','stopped','frame_rejected')),
  reason     TEXT,
  at         TEXT NOT NULL,
  by         TEXT
);
CREATE INDEX ix_capture_events ON legal_capture_events(capture_id, at);
CREATE TRIGGER capture_events_no_update BEFORE UPDATE ON legal_capture_events BEGIN SELECT RAISE(ABORT, 'capture events are append-only'); END;
CREATE TRIGGER capture_events_no_delete BEFORE DELETE ON legal_capture_events BEGIN SELECT RAISE(ABORT, 'capture events are append-only'); END;

-- Custody: one row per received batch of audio — its hash and size, not the audio.
CREATE TABLE legal_capture_custody(
  capture_id  TEXT NOT NULL REFERENCES legal_captures(id),
  seq         INTEGER NOT NULL,
  sha256      TEXT NOT NULL,
  bytes       INTEGER NOT NULL,
  received_at TEXT NOT NULL,
  PRIMARY KEY(capture_id, seq)
);

CREATE TABLE legal_transcript_segments(
  id         TEXT PRIMARY KEY,
  capture_id TEXT NOT NULL REFERENCES legal_captures(id),
  meeting_id TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  seq        INTEGER NOT NULL,
  t_start_ms INTEGER,
  t_end_ms   INTEGER,
  text       TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_segments_meeting ON legal_transcript_segments(meeting_id, capture_id, seq);
