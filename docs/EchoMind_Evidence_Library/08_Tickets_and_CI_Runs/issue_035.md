# #35 — NC-2026-015: Export gateway does not detect an API-key format

| Field | Value |
|---|---|
| State | CLOSED |
| Created | 2026-10-07T01:42:21Z |
| Closed | 2026-10-07T01:42:25Z |
| Labels | bug, nonconformity, S3, ISO 8.7, ISO 10.2 |
| URL | https://github.com/RW2523/echomind-enterprise/issues/35 |

**Record:** `REG-04` NC-2026-015 · **Severity:** S3

**Date raised:** 2026-10-06

**Description:** With synthetic test values, the gateway detected and redacted the email address, phone number and SSN, but **did not detect the `sk-test-…` API key**, which remained in the redacted copy.

**Root cause (proposed):** The secret patterns do not cover this key format.

**Immediate correction:** Key patterns extended (multi-segment `sk-`/`sk_`, `pk/rk_live/test`, `api_key=` assignments); overlapping matches merged before redaction — `e0ed4d0`.

**Corrective action:** Regression tests `test_nc015_*` added (key formats, overlap, no false positives).

**Effectiveness verification:** Acceptance run 4: EXP-108 → api_key, email, phone, SSN all detected; **no sensitive value** in the redacted copy.

**Status:** **Closed** 2026-10-06, effectiveness verified (release v1.4.1)

**Fix commits:**
- `e0ed4d0` (2026-10-06) fix: correct the four defects found by the 2026-10-06 acceptance tests

---
*Recorded live on 2026-10-06.*
