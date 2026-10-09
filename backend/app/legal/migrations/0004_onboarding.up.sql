-- 0004 onboarding (S1.3, decision D-021): one-time authenticator enrolment links issued by an
-- administrator, and a forced password change after an administrator sets someone's password.
ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0;

-- Only the SHA-256 of a link token is stored. Issuing a new link invalidates older unused ones.
CREATE TABLE mfa_enrollment_tokens(
  token_hash TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL,
  created_by TEXT,
  created_at TEXT NOT NULL,
  expires_at REAL NOT NULL,
  used_at    TEXT,
  revoked_at TEXT
);
CREATE INDEX ix_enrol_user ON mfa_enrollment_tokens(user_id);
