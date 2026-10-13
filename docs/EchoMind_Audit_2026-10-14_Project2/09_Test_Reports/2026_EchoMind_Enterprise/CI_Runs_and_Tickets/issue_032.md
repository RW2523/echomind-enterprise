# #32 — NC-2026-012: Logging out does not revoke the session token

| Field | Value |
|---|---|
| State | CLOSED |
| Created | 2026-10-07T01:42:00Z |
| Closed | 2026-10-07T01:42:04Z |
| Labels | bug, nonconformity, S3, ISO 8.7, ISO 10.2 |
| URL | https://github.com/RW2523/echomind-enterprise/issues/32 |

**Record:** `REG-04` NC-2026-012 · **Severity:** S3

**Date raised:** 2026-10-06

**Description:** After `POST /api/auth/logout`, replaying the same token returned **HTTP 200** on `/api/docs/list`. A token remains valid until it expires (`AUTH_TOKEN_TTL_MIN` = 720 minutes).

**Root cause (proposed):** Stateless JWT with no server-side revocation; logout only clears the browser cookie.

**Immediate correction:** Logout now revokes the token server-side (hash stored until expiry, rejected by `decode_token`); the voice WebSocket confirms the session with the backend and fails closed — `e0ed4d0`. Re-testing found the logout itself logged as anonymous (the activity log resolved the user after revocation); fixed in `3feb5db`.

**Corrective action:** Regression tests `test_nc012_*` (backend) and `test_auth_session.py` (voice) added to the suites.

**Effectiveness verification:** Acceptance run 4 (2026-10-06 22:28 UTC, v1.4.1): AC-101.7 replayed token → **401**; AC-101.8 activity recorded for the user; AC-101.9 voice 403/101.

**Status:** **Closed** 2026-10-06, effectiveness verified (release v1.4.1)

**Fix commits:**
- `e0ed4d0` (2026-10-06) fix: correct the four defects found by the 2026-10-06 acceptance tests
- `3feb5db` (2026-10-06) fix(audit): attribute activity to the caller before the request runs

---
*Recorded live on 2026-10-06.*
