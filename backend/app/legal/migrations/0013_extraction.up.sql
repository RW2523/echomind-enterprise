-- S2.1a legal ingestion quality (RAG-01; D-010, D-034).
-- The canonical text of each document version with its page, line and block maps, so every
-- citation offset points at exact characters; and the marks found while reading it (hidden text,
-- text addressed to an AI, tracked changes, comments, low-confidence OCR words).
CREATE TABLE legal_version_text(
  version_id   TEXT PRIMARY KEY REFERENCES legal_document_versions(id),
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  text         TEXT NOT NULL,
  pages_json   TEXT NOT NULL,
  blocks_json  TEXT NOT NULL,
  lines_json   TEXT NOT NULL,
  engines_json TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX ix_legal_version_text_matter ON legal_version_text(matter_id);

CREATE TABLE legal_marks(
  id         TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES legal_document_versions(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  kind       TEXT NOT NULL CHECK(kind IN ('hidden_text','instructions','tracked_insert','tracked_delete','comment','ocr_low_confidence')),
  char_start INTEGER,
  char_end   INTEGER,
  page       INTEGER,
  meta_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX ix_legal_marks_version ON legal_marks(version_id, kind);

-- Where reading has got to, for the progress shown while a document is read: {"stage": ...}.
ALTER TABLE legal_document_versions ADD COLUMN progress_json TEXT;
