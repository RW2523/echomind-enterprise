ALTER TABLE legal_panel_reviews DROP COLUMN question_id;
DROP INDEX IF EXISTS ix_council_questions_matter;
DROP TABLE IF EXISTS legal_council_questions;
