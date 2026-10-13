# #59 — Make the Ollama healthcheck perform a real embedding

| Field | Value |
|---|---|
| State | CLOSED |
| Created | 2026-10-07T01:44:32Z |
| Closed | 2026-10-07T02:26:52Z |
| Labels | task, ISO 9.1.1 |
| URL | https://github.com/RW2523/echomind-enterprise/issues/59 |

**Source:** NC-2026-016 action (a)

The healthcheck only runs `ollama list`, so the service stayed healthy while every embedding call failed.

---
*Recorded live on 2026-10-06.*
