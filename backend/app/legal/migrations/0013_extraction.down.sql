ALTER TABLE legal_document_versions DROP COLUMN progress_json;
DROP INDEX ix_legal_marks_version;
DROP TABLE legal_marks;
DROP INDEX ix_legal_version_text_matter;
DROP TABLE legal_version_text;
