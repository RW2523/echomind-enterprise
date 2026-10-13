# #29 — NC-2026-009: Public reference instance has no access control

| Field | Value |
|---|---|
| State | OPEN |
| Created | 2026-10-07T01:41:39Z |
| Closed | — |
| Labels | bug, nonconformity, S2, ISO 8.7, ISO 10.2 |
| URL | https://github.com/RW2523/echomind-enterprise/issues/29 |

**Record:** `REG-04` NC-2026-009 · **Severity:** S2

**Date raised:** 2026-10-06 (first observed 2026-09-28)

**Description:** `https://echomind-ajace.com` returns **HTTP 200** with the application, not a 302 to a Cloudflare Access login — checked from the host and from an external network on 2026-09-28, and from the host on 2026-10-06. `/api/auth/config` returns `{"auth_enabled":false}` and `/api/docs/list` returns the knowledge-base document list with no credentials. `.env` sets `AUTH_ENABLED=0` and `VOICE_AUTH_ENABLED=0`. Anyone who finds the URL can read, upload and delete knowledge-base content and use the GPU. This contradicts `docs/PUBLIC_DEPLOYMENT.md` step 4 (Access "MANDATORY"), `SOP-11` §5, and the existing-controls entry of `REG-03` R-04.

**Investigation:** Application auth was enforced on 2026-07-29 (`9810e98`, `73f0b4f`) and was the only gate on the public URL at that time. It has since been switched off in `.env`. **When, by whom and why is not recorded**, because `.env` is untracked and no change record exists. Access logs have not yet been reviewed to establish whether unknown parties used the instance.

**Root cause (proposed):** Security-relevant runtime configuration — `.env` and the Cloudflare dashboard — sits **outside version control and change control**, so a change to it leaves no record and triggers no review. The check that the public URL is gated (`docs/PUBLIC_DEPLOYMENT.md`, a manual `curl`) is not scheduled, so the regression was invisible until an unrelated health check.

**Immediate correction:** 2026-10-06, about 20:00 UTC: `AUTH_ENABLED=1` and `VOICE_AUTH_ENABLED=1` set in `.env` and the backend and voice services recreated. Cloudflare Access was not restored; application auth is the gate.

**Correction verified:** 2026-10-06: from an external network, `/api/docs/list` → **401**; via the public URL, a wrong password → 401, the admin login → 200 and authenticated calls succeed; voice `/ws` → **403** without a session and 101 with one; `/api/auth/config` → `auth_enabled: true`.

**Corrective action (proposed):** (a) Restore a gate and verify both the positive and negative case (`SOP-08` P-8). (b) Schedule an external check that the public URL returns a login challenge, alerting on HTTP 200. (c) Record changes to security-relevant runtime settings in `REG-07`. (d) Review access logs for the exposure window and record the outcome here.

**Effectiveness verification:** Correction verified (above). Effectiveness of the corrective action is pending: actions (b)–(d) must be in place so that switching auth off again would be detected and recorded.

**Evidence:** This entry; `AUDIT_PACK_2026_EchoMind.md` W-8

**Status:** **OPEN** — correction verified 2026-10-06; corrective actions (b)–(d) pending

---
*Recorded live on 2026-10-06.*
