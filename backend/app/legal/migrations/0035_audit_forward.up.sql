-- S1.5 — forwarding the audit trail to the firm's SIEM (SEC-04, D-069). Off unless LEGAL_AUDIT_SYSLOG names a
-- destination. One row: the destination and the last event (seq) sent to it, moved on only after a send succeeds,
-- so a restart or an unreachable SIEM skips nothing. It holds a position in the trail, never event content.
CREATE TABLE legal_audit_forward_state(
  id           INTEGER PRIMARY KEY CHECK(id = 1),
  destination  TEXT NOT NULL,
  last_seq     INTEGER NOT NULL,
  updated_at   TEXT NOT NULL
);
