-- 0011 exports (S1.10, SEC-06; policy P-006 safe defaults): the only way matter content leaves
-- EchoMind Law as a file. Request (can_export) → redaction plan → approval by a second person →
-- render → verify → watermark → download; the output is stored encrypted and expires.
CREATE TABLE legal_exports(
  id                TEXT PRIMARY KEY,
  org_id            TEXT NOT NULL,
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  object_type       TEXT NOT NULL CHECK(object_type IN ('document','meeting_transcript')),
  object_id         TEXT NOT NULL,
  source_version_id TEXT,
  title             TEXT NOT NULL,
  format            TEXT NOT NULL CHECK(format IN ('pdf','docx')),
  purpose           TEXT NOT NULL,
  status            TEXT NOT NULL CHECK(status IN ('draft','pending_approval','ready','failed','rejected','expired')),
  plan_json         TEXT NOT NULL,
  requested_by      TEXT NOT NULL,
  requested_at      TEXT NOT NULL,
  submitted_at      TEXT,
  decided_by        TEXT,
  decided_at        TEXT,
  decision_note     TEXT,
  privileged_ack    INTEGER NOT NULL DEFAULT 0,
  watermark         TEXT,
  wrapped_key       TEXT,
  output_sha256     TEXT,
  output_bytes      INTEGER,
  verification_json TEXT,
  expires_at        TEXT,
  downloads         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_exports_matter ON legal_exports(matter_id, requested_at);
CREATE INDEX ix_exports_object ON legal_exports(object_type, object_id);
