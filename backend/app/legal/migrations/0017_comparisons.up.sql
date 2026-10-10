-- S2.3 comparisons of two versions and conflict scans across documents (RAG-02; D-039). Each is
-- private to the person who ran it within the matter. The result stores offsets into the versions'
-- text and the model's explanations; document_ids_json lets a purge find and remove every
-- comparison that quotes a document.
CREATE TABLE legal_comparisons(
  id                TEXT PRIMARY KEY,
  org_id            TEXT NOT NULL,
  matter_id         TEXT NOT NULL REFERENCES matters(id),
  owner_id          TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK(kind IN ('versions','scan')),
  status            TEXT NOT NULL CHECK(status IN ('done','partial','scope_revoked','error')),
  versions          TEXT,
  document_ids_json TEXT NOT NULL DEFAULT '[]',
  version_ids_json  TEXT NOT NULL DEFAULT '[]',
  result_json       TEXT NOT NULL DEFAULT '{}',
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL
);
CREATE INDEX ix_legal_comparisons_owner ON legal_comparisons(matter_id, owner_id, created_at);
