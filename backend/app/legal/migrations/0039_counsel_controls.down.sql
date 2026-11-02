DROP INDEX IF EXISTS ix_legal_counsel_feedback_matter;
DROP TABLE IF EXISTS legal_counsel_feedback;
ALTER TABLE legal_conversations DROP COLUMN deleted_at;
DROP INDEX IF EXISTS ix_legal_conversation_turns_parent;
ALTER TABLE legal_conversation_turns DROP COLUMN parent_turn_id;
