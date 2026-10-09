-- 0005 ethical walls and break-glass (S1.4, SEC-02; IMPLEMENTATION_PLAN §4.2 steps 3 and 6).
-- A wall screens people from protected matters and/or whole clients. It takes effect only once a
-- second managing partner approves it, and holds until a release is approved by another one.
CREATE TABLE ethical_walls(
  id                   TEXT PRIMARY KEY,
  org_id               TEXT NOT NULL,
  name                 TEXT NOT NULL,
  rationale            TEXT NOT NULL,
  status               TEXT NOT NULL DEFAULT 'requested'
                       CHECK(status IN ('requested','active','release_requested','released','rejected')),
  requested_by         TEXT NOT NULL,
  requested_at         TEXT NOT NULL,
  approved_by          TEXT,
  approved_at          TEXT,
  release_requested_by TEXT,
  release_requested_at TEXT,
  released_by          TEXT,
  released_at          TEXT
);
CREATE TABLE wall_screened(
  wall_id TEXT NOT NULL REFERENCES ethical_walls(id),
  user_id TEXT NOT NULL,
  PRIMARY KEY(wall_id, user_id)
);
CREATE INDEX ix_wall_screened_user ON wall_screened(user_id);
CREATE TABLE wall_protected(
  wall_id     TEXT NOT NULL REFERENCES ethical_walls(id),
  object_type TEXT NOT NULL CHECK(object_type IN ('matter','client')),
  object_id   TEXT NOT NULL,
  PRIMARY KEY(wall_id, object_type, object_id)
);
CREATE INDEX ix_wall_protected_obj ON wall_protected(object_type, object_id);

-- A time-limited exception lets one screened person into one protected matter, once approved.
CREATE TABLE wall_exceptions(
  id           TEXT PRIMARY KEY,
  wall_id      TEXT NOT NULL REFERENCES ethical_walls(id),
  user_id      TEXT NOT NULL,
  matter_id    TEXT NOT NULL,
  reason       TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  approved_by  TEXT,
  approved_at  TEXT,
  expires_at   TEXT NOT NULL,
  revoked_by   TEXT,
  revoked_at   TEXT
);
CREATE INDEX ix_wall_exc ON wall_exceptions(wall_id, user_id, matter_id);

-- Emergency read-only access for someone who is not a member. Requested by matter number (the
-- requester cannot see the matter), resolved and approved by a different managing partner.
CREATE TABLE break_glass_grants(
  id               TEXT PRIMARY KEY,
  org_id           TEXT NOT NULL,
  user_id          TEXT NOT NULL,
  requested_number TEXT NOT NULL,
  matter_id        TEXT,
  reason           TEXT NOT NULL,
  minutes          INTEGER NOT NULL,
  requested_at     TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'requested' CHECK(status IN ('requested','active','rejected','ended')),
  decided_by       TEXT,
  decided_at       TEXT,
  decision_note    TEXT,
  expires_at       TEXT,
  ended_by         TEXT,
  ended_at         TEXT
);
CREATE INDEX ix_bg_user_matter ON break_glass_grants(user_id, matter_id);
