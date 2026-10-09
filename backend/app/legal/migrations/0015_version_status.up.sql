-- S2.1c: each version keeps its own status (draft, final, executed, received), so "v3 executed,
-- v2 draft" is known when answers must prefer signed text (S2.2). Existing versions take their
-- document's current status.
ALTER TABLE legal_document_versions ADD COLUMN status TEXT CHECK(status IN ('draft','final','executed','received'));
UPDATE legal_document_versions SET status = (SELECT d.status FROM legal_documents d WHERE d.id = legal_document_versions.document_id);
