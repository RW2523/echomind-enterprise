-- S3.2 — the clause library and playbook checks (DOC-02a, D-046).

-- A playbook: the firm's positions for one kind of contract and side.
CREATE TABLE legal_playbooks(
  id           TEXT PRIMARY KEY,
  org_id       TEXT NOT NULL,
  name         TEXT NOT NULL,
  description  TEXT,
  jurisdiction TEXT,
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
-- A position on one topic: the standard wording and the fallbacks the firm accepts. Versions are
-- fixed; a managing partner other than the author approves one (by hash), retiring the one before.
CREATE TABLE legal_playbook_positions(
  id             TEXT PRIMARY KEY,
  playbook_id    TEXT NOT NULL REFERENCES legal_playbooks(id),
  topic          TEXT NOT NULL,
  version        INTEGER NOT NULL,
  status         TEXT NOT NULL CHECK(status IN ('draft','approved','retired')),
  title          TEXT NOT NULL,
  standard_text  TEXT NOT NULL,
  fallbacks_json TEXT NOT NULL DEFAULT '[]',
  guidance       TEXT,
  owner          TEXT,
  content_sha256 TEXT NOT NULL,
  created_by     TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  approved_by    TEXT,
  approved_at    TEXT,
  UNIQUE(playbook_id, topic, version)
);
CREATE TRIGGER position_wording_fixed BEFORE UPDATE OF topic, version, title, standard_text, fallbacks_json, guidance, owner,
  content_sha256, created_by ON legal_playbook_positions
  BEGIN SELECT RAISE(ABORT, 'a playbook position is fixed; a change is a new version'); END;

-- A document checked against a playbook: each topic's clause and its class, with the exact words.
-- Advisory: lawyers record whether they agree with each line (append-only; the latest counts).
CREATE TABLE legal_playbook_checks(
  id           TEXT PRIMARY KEY,
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  document_id  TEXT NOT NULL,
  version_id   TEXT NOT NULL,
  playbook_id  TEXT NOT NULL,
  results_json TEXT NOT NULL,
  inputs_json  TEXT NOT NULL DEFAULT '{}',
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX ix_playbook_checks_document ON legal_playbook_checks(matter_id, document_id, created_at);
CREATE TABLE legal_playbook_decisions(
  id        TEXT PRIMARY KEY,
  check_id  TEXT NOT NULL REFERENCES legal_playbook_checks(id),
  topic     TEXT NOT NULL,
  decision  TEXT NOT NULL CHECK(decision IN ('agree','disagree')),
  reason    TEXT,
  by        TEXT NOT NULL,
  at        TEXT NOT NULL
);
CREATE INDEX ix_playbook_decisions ON legal_playbook_decisions(check_id);
CREATE TRIGGER playbook_decisions_no_update BEFORE UPDATE ON legal_playbook_decisions
  BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
