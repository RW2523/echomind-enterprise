-- EchoMind Legal chat controls (owner's request, 2026-11-02: "like ChatGPT"): asking again or editing a question makes
-- another version of that turn instead of overwriting it, so every answer that was shown stays on record; a person can
-- rename a conversation, take it off their list, and say whether an answer helped.

-- The turn this one follows in its conversation (NULL for a first question). Two turns with the same parent are versions
-- of one step: the newer was asked again, or with the question edited. Existing conversations are a single line.
ALTER TABLE legal_conversation_turns ADD COLUMN parent_turn_id TEXT;
UPDATE legal_conversation_turns SET parent_turn_id = (
  SELECT p.id FROM legal_conversation_turns p
   WHERE p.conversation_id = legal_conversation_turns.conversation_id AND p.seq = legal_conversation_turns.seq - 1);
CREATE INDEX ix_legal_conversation_turns_parent ON legal_conversation_turns(conversation_id, parent_turn_id);

-- Taken off its owner's list. The conversation is kept as matter work product (purge and the firm's retention rules
-- still apply to it); it is no longer listed, opened or continued.
ALTER TABLE legal_conversations ADD COLUMN deleted_at TEXT;

-- Whether an answer helped, from the person who asked. One per turn and person; changing it replaces it.
CREATE TABLE legal_counsel_feedback(
  turn_id     TEXT NOT NULL,
  owner_id    TEXT NOT NULL,
  org_id      TEXT NOT NULL,
  matter_id   TEXT NOT NULL,
  rating      TEXT NOT NULL CHECK (rating IN ('up','down')),
  reason      TEXT CHECK (reason IS NULL OR reason IN ('wrong_source','missed','not_asked','unclear','other')),
  note        TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  PRIMARY KEY (turn_id, owner_id)
);
CREATE INDEX ix_legal_counsel_feedback_matter ON legal_counsel_feedback(org_id, matter_id, updated_at);
