-- 0006 audit (S1.5, SEC-04; decision D-008): append-only, hash-chained audit trail.
-- hash = SHA-256(prev_hash + canonical JSON of the event). Triggers refuse UPDATE and DELETE, and a
-- verifier recomputes the chain, so an edit made with direct database access is detected.
-- Events carry identifiers, actions, outcomes and counts — never matter content.
CREATE TABLE audit_events(
  seq            INTEGER PRIMARY KEY AUTOINCREMENT,
  id             TEXT NOT NULL UNIQUE,
  ts             TEXT NOT NULL,
  org_id         TEXT,
  actor_user_id  TEXT,
  actor_username TEXT,
  actor_roles    TEXT,
  auth           TEXT,
  session_ref    TEXT,
  ip             TEXT,
  action         TEXT NOT NULL,
  object_type    TEXT,
  object_id      TEXT,
  matter_id      TEXT,
  outcome        TEXT NOT NULL CHECK(outcome IN ('allow','deny','error')),
  reason         TEXT,
  details_json   TEXT,
  prev_hash      TEXT NOT NULL,
  hash           TEXT NOT NULL
);
CREATE INDEX ix_audit_ts ON audit_events(ts);
CREATE INDEX ix_audit_matter ON audit_events(matter_id, seq);
CREATE INDEX ix_audit_actor ON audit_events(actor_user_id, seq);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
