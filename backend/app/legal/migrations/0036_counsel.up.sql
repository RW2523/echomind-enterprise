-- EchoMind Legal chat ("Quiet Counsel", docs/echomind-legal/03 and 06): conversations about one matter, their turns,
-- the trusted UI components each turn showed, opaque citation references, and potential inconsistencies with the
-- people who resolved them. Every row names its matter; every read re-asks the policy engine.

-- A conversation belongs to one person and one matter. History is private to its owner.
CREATE TABLE legal_conversations(
  id          TEXT PRIMARY KEY,
  org_id      TEXT NOT NULL,
  matter_id   TEXT NOT NULL REFERENCES matters(id),
  owner_id    TEXT NOT NULL,
  title       TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX ix_legal_conversations_owner ON legal_conversations(owner_id, updated_at);
CREATE INDEX ix_legal_conversations_matter ON legal_conversations(matter_id);

-- One question and what was shown for it. The components live in legal_counsel_components (by id), so a purge
-- has one place to scrub.
CREATE TABLE legal_conversation_turns(
  id                 TEXT PRIMARY KEY,
  conversation_id    TEXT NOT NULL REFERENCES legal_conversations(id),
  matter_id          TEXT NOT NULL,
  owner_id           TEXT NOT NULL,
  seq                INTEGER NOT NULL,
  client_request_id  TEXT,
  question           TEXT NOT NULL,
  understood         TEXT,
  intent             TEXT NOT NULL,
  status             TEXT NOT NULL CHECK (status IN ('running','completed','abstained','failed','scope_revoked','canceled')),
  summary            TEXT,
  component_ids_json TEXT NOT NULL DEFAULT '[]',
  actions_json       TEXT NOT NULL DEFAULT '[]',
  research_turn_id   TEXT,
  created_at         TEXT NOT NULL,
  finished_at        TEXT,
  UNIQUE (conversation_id, seq),
  UNIQUE (owner_id, client_request_id)
);
CREATE INDEX ix_legal_conversation_turns ON legal_conversation_turns(conversation_id, seq);

-- A component as it was validated (schemas.py). `id` is the opaque data_ref the browser sends back with an action.
CREATE TABLE legal_counsel_components(
  id                TEXT PRIMARY KEY,
  conversation_id   TEXT NOT NULL,
  turn_id           TEXT NOT NULL,
  matter_id         TEXT NOT NULL,
  owner_id          TEXT NOT NULL,
  type              TEXT NOT NULL,
  props_json        TEXT NOT NULL,
  document_ids_json TEXT NOT NULL DEFAULT '[]',
  created_at        TEXT NOT NULL
);
CREATE INDEX ix_legal_counsel_components_turn ON legal_counsel_components(turn_id);

-- A citation the browser holds only by id. The quote is not stored: its offsets and SHA-256 are, so opening the
-- source re-reads the version's text and proves the passage is still the one that was cited.
CREATE TABLE legal_counsel_citations(
  id            TEXT PRIMARY KEY,
  turn_id       TEXT NOT NULL,
  matter_id     TEXT NOT NULL,
  owner_id      TEXT NOT NULL,
  n             INTEGER NOT NULL,
  document_id   TEXT NOT NULL,
  version_id    TEXT NOT NULL,
  char_start    INTEGER,
  char_end      INTEGER,
  quote_sha256  TEXT,
  verification  TEXT NOT NULL CHECK (verification IN ('verbatim','normalized','not_quoted','failed')),
  created_at    TEXT NOT NULL
);
CREATE INDEX ix_legal_counsel_citations_doc ON legal_counsel_citations(matter_id, document_id);

-- A potential inconsistency between two documents of one matter, proposed by the conflict scan. Only a person
-- moves it out of 'potential'. Matter-wide (not private): the resolution is matter work product.
CREATE TABLE legal_discrepancies(
  id              TEXT PRIMARY KEY,
  org_id          TEXT NOT NULL,
  matter_id       TEXT NOT NULL REFERENCES matters(id),
  fingerprint     TEXT NOT NULL,
  kind            TEXT NOT NULL,
  verdict         TEXT NOT NULL CHECK (verdict IN ('conflict','unclear')),
  method          TEXT NOT NULL CHECK (method IN ('model','rule')),
  status          TEXT NOT NULL CHECK (status IN ('potential','in_review','explained','confirmed','dismissed')),
  score           REAL NOT NULL DEFAULT 0,
  document_a_id   TEXT NOT NULL,
  document_b_id   TEXT NOT NULL,
  version_a_id    TEXT NOT NULL,
  version_b_id    TEXT NOT NULL,
  detail_json     TEXT NOT NULL,
  found_by        TEXT NOT NULL,
  version         INTEGER NOT NULL DEFAULT 1,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE (matter_id, fingerprint)
);
CREATE INDEX ix_legal_discrepancies_matter ON legal_discrepancies(matter_id, status);

-- Who changed a discrepancy's state, from what, to what and why. Written once, never changed (a purge of one of
-- its documents removes the discrepancy with its history; the audit trail keeps the fact that it happened).
CREATE TABLE legal_discrepancy_reviews(
  id              TEXT PRIMARY KEY,
  discrepancy_id  TEXT NOT NULL REFERENCES legal_discrepancies(id),
  matter_id       TEXT NOT NULL,
  from_status     TEXT NOT NULL,
  to_status       TEXT NOT NULL,
  note            TEXT,
  by_user         TEXT NOT NULL,
  at              TEXT NOT NULL
);
CREATE INDEX ix_legal_discrepancy_reviews ON legal_discrepancy_reviews(discrepancy_id, at);
CREATE TRIGGER discrepancy_reviews_no_update BEFORE UPDATE ON legal_discrepancy_reviews
BEGIN SELECT RAISE(ABORT, 'discrepancy reviews are written once'); END;

-- Which set of versions a person's scan already covered, so asking again does not call the model again.
CREATE TABLE legal_discrepancy_scans(
  matter_id     TEXT NOT NULL,
  versions_sha  TEXT NOT NULL,
  scanned_by    TEXT NOT NULL,
  documents     INTEGER NOT NULL,
  candidates    INTEGER NOT NULL,
  found         INTEGER NOT NULL,
  scanned_at    TEXT NOT NULL,
  PRIMARY KEY (matter_id, versions_sha)
);

-- Action idempotency (docs/echomind-legal/03 action contract): the same key from the same person answers the same.
CREATE TABLE legal_counsel_actions(
  owner_id         TEXT NOT NULL,
  idempotency_key  TEXT NOT NULL,
  component_id     TEXT NOT NULL,
  action_id        TEXT NOT NULL,
  result_json      TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  PRIMARY KEY (owner_id, idempotency_key)
);

-- When a person last had the matter briefed to them ("What matters today?").
CREATE TABLE legal_counsel_seen(
  user_id    TEXT NOT NULL,
  matter_id  TEXT NOT NULL,
  seen_at    TEXT NOT NULL,
  PRIMARY KEY (user_id, matter_id)
);
