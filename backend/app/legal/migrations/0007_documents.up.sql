-- 0007 documents (S1.6, SEC-03; decisions D-005, D-012, D-014): matter documents, their versions,
-- searchable passages and the background jobs that index them.
-- Law documents never use the legacy `documents`/`chunks` tables, so no legacy index rebuild can
-- ever pick them up. Each matter's passages are searched only through that matter's own partition.
CREATE TABLE legal_documents(
  id                 TEXT PRIMARY KEY,
  org_id             TEXT NOT NULL,
  matter_id          TEXT NOT NULL REFERENCES matters(id),
  title              TEXT NOT NULL,
  kind               TEXT NOT NULL DEFAULT 'other'
                     CHECK(kind IN ('contract','correspondence','pleading','evidence','memo','note','other')),
  status             TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','final','executed','received')),
  current_version_id TEXT,
  created_by         TEXT NOT NULL,
  created_at         TEXT NOT NULL,
  updated_at         TEXT,
  version            INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX ix_legal_documents_matter ON legal_documents(matter_id, created_at);

-- One row per uploaded file. The file itself is stored encrypted (AES-256-GCM) under a per-file
-- key; wrapped_key is that key wrapped by the deployment's master key (never stored here).
CREATE TABLE legal_document_versions(
  id           TEXT PRIMARY KEY,
  document_id  TEXT NOT NULL REFERENCES legal_documents(id),
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  number       INTEGER NOT NULL,
  filename     TEXT NOT NULL,
  mime         TEXT NOT NULL,
  size_bytes   INTEGER NOT NULL,
  sha256       TEXT NOT NULL,
  pages        INTEGER,
  wrapped_key  TEXT NOT NULL,
  uploaded_by  TEXT NOT NULL,
  uploaded_at  TEXT NOT NULL,
  index_status TEXT NOT NULL DEFAULT 'queued' CHECK(index_status IN ('queued','indexing','indexed','failed','superseded')),
  index_error  TEXT,
  chunk_count  INTEGER NOT NULL DEFAULT 0,
  indexed_at   TEXT,
  UNIQUE(document_id, number)
);
CREATE INDEX ix_legal_versions_matter ON legal_document_versions(matter_id);

-- Searchable passages: original text (D-012), offsets into the parsed text, pages, and the
-- embedding (float32) so a matter's index can be rebuilt from here without re-embedding.
CREATE TABLE legal_chunks(
  id          TEXT PRIMARY KEY,
  version_id  TEXT NOT NULL REFERENCES legal_document_versions(id),
  document_id TEXT NOT NULL REFERENCES legal_documents(id),
  matter_id   TEXT NOT NULL REFERENCES matters(id),
  chunk_index INTEGER NOT NULL,
  text        TEXT NOT NULL,
  char_start  INTEGER,
  char_end    INTEGER,
  page_start  INTEGER,
  page_end    INTEGER,
  vec         BLOB,
  active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX ix_legal_chunks_matter ON legal_chunks(matter_id, active);
CREATE INDEX ix_legal_chunks_version ON legal_chunks(version_id);

-- Personal-data findings recorded as labels (D-012): kind and offsets only, never the value.
CREATE TABLE legal_findings(
  id         TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES legal_document_versions(id),
  matter_id  TEXT NOT NULL REFERENCES matters(id),
  kind       TEXT NOT NULL,
  char_start INTEGER NOT NULL,
  char_end   INTEGER NOT NULL
);
CREATE INDEX ix_legal_findings_version ON legal_findings(version_id);

CREATE TABLE legal_jobs(
  id           TEXT PRIMARY KEY,
  kind         TEXT NOT NULL,
  org_id       TEXT NOT NULL,
  matter_id    TEXT,
  object_id    TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','running','done','failed','cancelled')),
  attempts     INTEGER NOT NULL DEFAULT 0,
  error        TEXT,
  created_at   TEXT NOT NULL,
  started_at   TEXT,
  finished_at  TEXT
);
CREATE INDEX ix_legal_jobs_status ON legal_jobs(status, created_at);
