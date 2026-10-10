DROP TABLE legal_intake_replies;
DROP TABLE legal_intake_recordings;
DROP TABLE legal_intake_consents;
DROP TRIGGER intake_events_no_delete;
DROP TRIGGER intake_events_append_only;
DROP TABLE legal_intake_events;
DROP TABLE legal_intake_matches;
DROP TABLE legal_intakes;
DROP TABLE legal_matter_parties;
CREATE TABLE user_roles_old(
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK(role IN ('firm_admin','managing_partner','attorney','paralegal','reviewer','records_admin','read_only')),
  granted_by TEXT,
  granted_at TEXT NOT NULL,
  PRIMARY KEY(user_id, role)
);
INSERT INTO user_roles_old(user_id, role, granted_by, granted_at) SELECT user_id, role, granted_by, granted_at FROM user_roles;
DROP TABLE user_roles;
ALTER TABLE user_roles_old RENAME TO user_roles;
