-- EchoMind Legal: a conversation may cover several of its owner's matters (owner's request, 2026-10-28). The conversation's
-- own matter (legal_conversations.matter_id) is the first; the others are listed here. Every read and every turn re-asks the
-- policy engine for each one, each answer names the matter of every source, and a highly restricted matter is only ever
-- discussed on its own (counsel/routes.py).
CREATE TABLE legal_conversation_matters(
  conversation_id TEXT NOT NULL REFERENCES legal_conversations(id),
  matter_id       TEXT NOT NULL REFERENCES matters(id),
  position        INTEGER NOT NULL,
  PRIMARY KEY (conversation_id, matter_id)
);
CREATE INDEX ix_legal_conversation_matters_matter ON legal_conversation_matters(matter_id);
