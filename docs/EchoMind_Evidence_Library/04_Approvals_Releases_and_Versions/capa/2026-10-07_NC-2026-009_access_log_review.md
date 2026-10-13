# Access-log review — period the public instance had no login (REG-04 NC-2026-009, action (d))

| Field | Entry |
|---|---|
| Record | CAPA evidence for NC-2026-009, corrective action (d) |
| Performed | 2026-10-07, from the `activity_log` table (SQLite, `echomind_data` volume) |
| Reviewed by | Kishan Haravu Pradeep (Data custodian) — 2026-10-07 |
| Source data | 2,032 logged API calls, 2026-06-27 → 2026-10-07. The log records every POST/PUT/DELETE/PATCH under `/api/` with time, user, path, status and source address. GET requests are not logged. |

## 1. How long the instance was ungated

| Evidence | Time (UTC) |
|---|---|
| Authenticated calls (login enforced, verification after commit `9810e98`) | 2026-07-30 00:42 → 01:59 (109 calls; 401s up to 01:52) |
| **First unauthenticated call that succeeded on a protected endpoint** | **2026-07-30 02:56** (`POST /api/chat/ask` → 200) |
| Login restored (first authenticated call after the fix) | 2026-10-06 20:03 |

**Conclusion: application login was off from 2026-07-30 ~02:56 UTC to 2026-10-06 ~20:00 UTC —
about 68 days — not only from 2026-09-28, when it was first noticed.** Cloudflare Access had been
removed at the same time, so the public URL had no access control for that period. The setting was
changed in `.env`, which is not under version control, so who changed it and why is not recorded.

## 2. Unknown parties reached the instance

64 requests to API paths that do not exist in EchoMind, between 2026-07-31 and 2026-09-22 — the
signature of automated internet scanners:

| Probe | Requests | Dates |
|---|---|---|
| `/api/graphql`, `/api/gql` (GraphQL discovery) | 46 | 07-31, 08-10, 08-12, 08-13, 08-14, 09-11, 09-12, 09-15 |
| `/api/exec`, `/api/run`, `/api/system`, `/api/execute` (command-execution probes) | 15 | 08-03, 08-08, 08-13 |
| `/api/vendor/phpunit/.../eval-stdin.php` (known PHP remote-code-execution probe) | 1 | 09-15 |
| `/api/v1/chat/completions`, `/api/v1/completions` (looking for an open LLM API) | 2 | 09-22 |

All were answered **404** — none of these endpoints exists. Scanners that tried real endpoints would
not stand out, because unauthenticated calls to real endpoints were legitimate during this period.

## 3. Data-changing actions in the period

| Date (UTC) | Actions | Correlation |
|---|---|---|
| 2026-07-30 15:07 | 1 document deleted | Engineering day (commits `2decb99`…`7b04eb9`) |
| 2026-08-07 14:47–18:56 | `delete-all` once, then 33 uploads within the hour | Same day as the paper-evaluation commits (`cbc57d1`, `4e27109`) — consistent with a deliberate re-ingest |
| 2026-08-16 02:24–06:20 | 69 deletions, 78 uploads | Starts ~1 h after commit `7e582e1` "overhaul ingestion" — the documented re-ingest |
| 2026-09-15 03:31–05:47 | 46 deletions, 5 uploads (the Meridian Bank demo set) | Starts minutes after commit `15f3c96` (03:27). **This is when the golden-evaluation corpus left the knowledge base — the probable cause of NC-2026-008.** |

Every data-changing action falls in a window of recorded engineering work. **None can be attributed
with certainty**: the source address is always an internal proxy (Cloudflare Tunnel → nginx →
backend), so internal and external callers look the same.

## 4. Limitations

- Client IP addresses were not recorded — only the proxy's internal address.
- GET requests (reads, including document downloads) are not in the activity log, so reads by unknown
  parties cannot be ruled out. Treat all demonstration content as potentially read.
- Container logs from before 2026-10-06 20:03 UTC (nginx, cloudflared, backend) were lost when the
  containers were recreated; they are not shipped anywhere.

## 5. Outcome and follow-up

- No evidence of malicious data change; all changes align with engineering activity. **To be
  confirmed by the person who performed the 2026-08-07, 08-16 and 09-15 operations.**
- The knowledge base held demonstration material only (no customer deployment).
- Recommended actions (owner to decide): record the real client address (`CF-Connecting-IP`) in the
  activity log; retain proxy logs off the container; check Cloudflare's own analytics for the period;
  log reads of document downloads.
