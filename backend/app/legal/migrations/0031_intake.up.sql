-- S4.1 — intake assistant with conflict screening (OPT-01, D-054). Two firm roles join the list: front-desk intake
-- staff and the conflicts team. Matters keep a register of their parties (adverse, related, the client's group) for
-- conflict searches; removal keeps the history. Intakes are administrative records of a prospective client's
-- enquiry: the screening matches are for the conflicts team only, the trail is append-only, a call recording needs
-- the caller's consent, and replies are approved by a lawyer before the enquirer sees them.
CREATE TABLE user_roles_new(
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK(role IN ('firm_admin','managing_partner','attorney','paralegal','reviewer','records_admin','read_only',
                                          'intake','conflicts')),
  granted_by TEXT,
  granted_at TEXT NOT NULL,
  PRIMARY KEY(user_id, role)
);
INSERT INTO user_roles_new(user_id, role, granted_by, granted_at) SELECT user_id, role, granted_by, granted_at FROM user_roles;
DROP TABLE user_roles;
ALTER TABLE user_roles_new RENAME TO user_roles;

CREATE TABLE legal_matter_parties(
  id             TEXT PRIMARY KEY,
  matter_id      TEXT NOT NULL REFERENCES matters(id),
  name           TEXT NOT NULL,
  role           TEXT NOT NULL CHECK(role IN ('adverse','related','client_group','other')),
  added_by       TEXT NOT NULL,
  added_at       TEXT NOT NULL,
  removed_by     TEXT,
  removed_at     TEXT,
  removal_reason TEXT
);
CREATE INDEX ix_matter_parties_matter ON legal_matter_parties(matter_id);

CREATE TABLE legal_intakes(
  id                     TEXT PRIMARY KEY,
  org_id                 TEXT NOT NULL,
  reference              TEXT NOT NULL,
  channel                TEXT NOT NULL CHECK(channel IN ('phone','in_person','link')),
  status                 TEXT NOT NULL CHECK(status IN ('invited','open','submitted','cleared','conflict','converted','closed','purged')),
  script_version         TEXT NOT NULL,
  disclaimer_version     TEXT,
  disclaimer_accepted_at TEXT,
  answers_json           TEXT NOT NULL DEFAULT '{}',
  screening              TEXT NOT NULL DEFAULT 'not_run' CHECK(screening IN ('not_run','no_match','possible_conflict')),
  screened_at            TEXT,
  decision               TEXT CHECK(decision IN ('cleared','conflict')),
  link_hash              TEXT UNIQUE,
  link_expires_at        TEXT,
  link_submitted_at      TEXT,
  matter_id              TEXT REFERENCES matters(id),
  created_by             TEXT NOT NULL,
  created_at             TEXT NOT NULL,
  updated_at             TEXT,
  version                INTEGER NOT NULL DEFAULT 1,
  UNIQUE(org_id, reference)
);
CREATE INDEX ix_intakes_org ON legal_intakes(org_id, created_at);

CREATE TABLE legal_intake_matches(
  id            TEXT PRIMARY KEY,
  intake_id     TEXT NOT NULL REFERENCES legal_intakes(id),
  searched_name TEXT NOT NULL,
  matched_name  TEXT NOT NULL,
  kind          TEXT NOT NULL CHECK(kind IN ('client','adverse','related','client_group','other')),
  client_id     TEXT,
  matter_id     TEXT,
  score         REAL NOT NULL,
  created_at    TEXT NOT NULL
);
CREATE INDEX ix_intake_matches ON legal_intake_matches(intake_id);

CREATE TABLE legal_intake_events(
  id         TEXT PRIMARY KEY,
  intake_id  TEXT NOT NULL REFERENCES legal_intakes(id),
  action     TEXT NOT NULL,
  actor_id   TEXT,
  at         TEXT NOT NULL,
  note       TEXT
);
CREATE INDEX ix_intake_events ON legal_intake_events(intake_id, at);
-- Append-only; a purge may only blank a note (it can hold a name).
CREATE TRIGGER intake_events_append_only BEFORE UPDATE ON legal_intake_events
  WHEN NEW.note IS NOT NULL OR NEW.action IS NOT OLD.action OR NEW.actor_id IS NOT OLD.actor_id OR NEW.at IS NOT OLD.at
    OR NEW.intake_id IS NOT OLD.intake_id
  BEGIN SELECT RAISE(ABORT, 'intake events are append-only'); END;
CREATE TRIGGER intake_events_no_delete BEFORE DELETE ON legal_intake_events
  BEGIN SELECT RAISE(ABORT, 'intake events are append-only'); END;

CREATE TABLE legal_intake_consents(
  id             TEXT PRIMARY KEY,
  intake_id      TEXT NOT NULL REFERENCES legal_intakes(id),
  method         TEXT NOT NULL CHECK(method IN ('verbal','written')),
  script_version TEXT NOT NULL,
  recorded_by    TEXT NOT NULL,
  recorded_at    TEXT NOT NULL,
  withdrawn_by   TEXT,
  withdrawn_at   TEXT
);

CREATE TABLE legal_intake_recordings(
  id          TEXT PRIMARY KEY,
  intake_id   TEXT NOT NULL REFERENCES legal_intakes(id),
  consent_id  TEXT NOT NULL REFERENCES legal_intake_consents(id),
  wrapped_key TEXT,
  mime        TEXT NOT NULL,
  bytes       INTEGER NOT NULL,
  sha256      TEXT,
  uploaded_by TEXT NOT NULL,
  uploaded_at TEXT NOT NULL,
  removed_at  TEXT,
  removed_why TEXT
);

CREATE TABLE legal_intake_replies(
  id          TEXT PRIMARY KEY,
  intake_id   TEXT NOT NULL REFERENCES legal_intakes(id),
  template    TEXT NOT NULL CHECK(template IN ('acknowledge','can_help','cannot_act','more_info')),
  body        TEXT NOT NULL,
  body_sha256 TEXT NOT NULL,
  status      TEXT NOT NULL CHECK(status IN ('draft','released','withdrawn')),
  drafted_by  TEXT NOT NULL,
  drafted_at  TEXT NOT NULL,
  approved_by TEXT,
  approved_at TEXT
);
CREATE INDEX ix_intake_replies ON legal_intake_replies(intake_id, drafted_at);
