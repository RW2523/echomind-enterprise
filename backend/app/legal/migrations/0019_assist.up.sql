-- S2.5 Silent Assist legal packs (LIVE-02; D-015, D-041). Alerts raised while a meeting is captured:
-- what was said (exact characters of a stored segment), what it was checked against (a verified
-- quote from a matter document, when there is one), the evidence-relative verdict, and the
-- confidence against the threshold in force. A person confirms or dismisses each alert, with a reason
-- for a dismissal; the decision is made once. Alerts are matter content: purged with the transcript.
ALTER TABLE legal_meetings ADD COLUMN assist_pack TEXT;      -- NULL: the default for the meeting's kind; 'off': no alerts

CREATE TABLE legal_assist_alerts(
  id                TEXT PRIMARY KEY,
  meeting_id        TEXT NOT NULL REFERENCES legal_meetings(id),
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  capture_id        TEXT,
  segment_id        TEXT REFERENCES legal_transcript_segments(id),
  pack              TEXT NOT NULL,
  rule              TEXT NOT NULL,
  verdict           TEXT NOT NULL CHECK(verdict IN ('differs','matches','needs_review')),
  severity          TEXT NOT NULL CHECK(severity IN ('critical','notice')),
  title             TEXT NOT NULL,
  said_json         TEXT NOT NULL,
  source_json       TEXT,
  explanation       TEXT,
  confidence        REAL NOT NULL,
  threshold         REAL NOT NULL,
  method            TEXT NOT NULL CHECK(method IN ('rule','model')),
  status            TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','confirmed','dismissed')),
  decided_by        TEXT,
  decided_at        TEXT,
  decision_reason   TEXT,
  heard_at          TEXT NOT NULL,
  raised_at         TEXT NOT NULL,
  document_ids_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX ix_assist_alerts_meeting ON legal_assist_alerts(meeting_id, raised_at);
