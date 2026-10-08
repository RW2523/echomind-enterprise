-- 0002 identity (S1.1, SEC-01; decisions D-006, D-007): server-side sessions, TOTP MFA with
-- recovery codes, lockout, account status, firm roles.
-- `users` is a legacy table normally created by init_db(); it is created here only if absent so
-- this migration also applies to an empty database. The down script leaves `users` in place.
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user', created_at TEXT, tenant TEXT DEFAULT ''
);
ALTER TABLE users ADD COLUMN org_id TEXT;
ALTER TABLE users ADD COLUMN display_name TEXT;
ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active';          -- active | disabled
ALTER TABLE users ADD COLUMN mfa_secret TEXT;                                -- base32; app-level encryption follows D-014
ALTER TABLE users ADD COLUMN mfa_enabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN mfa_last_step INTEGER;                          -- TOTP replay protection
ALTER TABLE users ADD COLUMN failed_logins INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN locked_until REAL;                              -- epoch seconds
ALTER TABLE users ADD COLUMN password_changed_at TEXT;

CREATE TABLE user_roles(
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK(role IN ('firm_admin','managing_partner','attorney','paralegal','reviewer','records_admin','read_only')),
  granted_by TEXT,
  granted_at TEXT NOT NULL,
  PRIMARY KEY(user_id, role)
);

-- Only the SHA-256 of the session token and of the CSRF token are stored.
CREATE TABLE sessions(
  token_hash      TEXT PRIMARY KEY,
  user_id         TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  last_seen_at    TEXT,
  idle_expires_at REAL NOT NULL,
  abs_expires_at  REAL NOT NULL,
  mfa_verified    INTEGER NOT NULL DEFAULT 0,
  csrf_hash       TEXT NOT NULL,
  ip              TEXT,
  user_agent      TEXT,
  revoked_at      TEXT,
  revoke_reason   TEXT
);
CREATE INDEX ix_sessions_user ON sessions(user_id);

CREATE TABLE mfa_recovery_codes(
  user_id    TEXT NOT NULL,
  code_hash  TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at    TEXT,
  PRIMARY KEY(user_id, code_hash)
);
