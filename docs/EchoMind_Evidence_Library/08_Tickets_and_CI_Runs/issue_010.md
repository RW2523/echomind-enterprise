# #10 — Authentication, RBAC, audit log and per-tenant scoping (Phase 0b)

| Field | Value |
|---|---|
| State | CLOSED |
| Created | 2026-10-07T01:38:45Z |
| Closed | 2026-10-07T01:38:49Z |
| Labels | enhancement, retrospective, ISO 8.3, area: security |
| URL | https://github.com/RW2523/echomind-enterprise/issues/10 |

**What was delivered:** Local accounts + JWT, login gate, admin user management, activity log, usage metering, CORS lock, WebSocket gating, tenant scoping.

**Commits:**
- `c9bad3d` (2026-06-27) feat(auth): local-accounts + JWT auth core, opt-in via AUTH_ENABLED (Phase 0b, step 1)
- `d12db42` (2026-06-27) feat(auth): frontend login gate, opt-in (Phase 0b, step 2)
- `7a83262` (2026-06-27) feat(auth): RBAC + admin user management (Phase 0b, step 3)
- `ac26d41` (2026-06-27) feat(auth): audit log + usage metering + CORS lock (Phase 0b, step 4)
- `bd428f2` (2026-06-27) feat(auth): WebSocket auth gating (Phase 0b, step 5 — hardening)
- `ef19d68` (2026-06-27) feat(auth): per-tenant scoping (#4 multi-tenant)
- `9810e98` (2026-07-29) feat(security): enable auth; sanitize citation artifacts; reserve 'default' tenant

**Verification:** Login positive/negative and WebSocket gate verified on the reference instance 2026-10-06 (see NC-2026-009 ticket).

**Released in:** `v1.4.0` (tag, 2026-09-22)

---
*Transcribed to this board on 2026-10-06 from the git record and `docs/qms/registers`. Work before that date was not tracked on a board; the dates above come from the commits and registers, not from this ticket's creation time.*
