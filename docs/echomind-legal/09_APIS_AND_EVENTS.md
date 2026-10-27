# 09 — API contracts and event streaming
All `/v1/*` paths require authenticated local session. Every matter operation verifies tenant, user, matter scope and ethical wall. CSRF protection for cookie sessions; rate limits; idempotency on writes; correlation IDs. Contract examples are starting point, not existing EchoMind endpoints.

## Main endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/v1/me` | identity, permitted workspace context |
| GET/POST | `/v1/matters` | list authorized/create with policy |
| GET | `/v1/matters/{id}` | overview, status and authorized stats |
| POST | `/v1/matters/{id}/documents` | upload immutable original; returns job id |
| GET | `/v1/matters/{id}/documents` | list filtered metadata |
| GET | `/v1/documents/{id}/source?citation_id=...` | authorized original/span viewer |
| POST | `/v1/chat/turns` | create streaming turn |
| GET | `/v1/chat/turns/{id}/events` | authenticated SSE stream |
| POST | `/v1/ui/actions` | trusted component actions, CAS version |
| GET | `/v1/matters/{id}/timeline` | provenance-bearing events |
| GET/POST | `/v1/matters/{id}/findings` | list/propose findings |
| POST | `/v1/findings/{id}/review` | attorney resolution + reason |
| POST | `/v1/matters/{id}/drafts` | create source-linked draft job |
| GET/PATCH | `/v1/drafts/{id}` | view and edit with optimistic concurrency |
| POST | `/v1/drafts/{id}/request-approval` | gate external-ready work |
| POST | `/v1/drafts/{id}/export` | local DOCX/PDF, policy controlled |
| POST | `/v1/recordings/sessions` | explicit authorized recording start |
| WS | `/v1/recordings/sessions/{id}/stream` | authenticated PCM/transcript events |
| POST | `/v1/recordings/sessions/{id}/stop` | hard stop + retention policy |
| GET | `/v1/jobs/{id}` | job status and errors |
| POST | `/v1/jobs/{id}/cancel` | cancellation |
| GET | `/v1/audit` | only permitted auditor-visible metadata |
| GET | `/healthz`, `/readyz` | local operational checks, no sensitive data |

## Chat request
```json
{"conversation_id":"conv_demo","matter_id":"matter_demo","text":"Find delivery date conflicts","selected_source_ids":[],"ui_capabilities":["source_answer","evidence_comparison","checklist","timeline"],"client_request_id":"uuid"}
```
SSE events:
```text
event: turn.started
data: {"turn_id":"t1","matter_id":"matter_demo"}

event: answer.delta
data: {"text":"I found one possible date difference."}

event: citation.verified
data: {"citation_id":"c1","document_id":"doc1","page":3}

event: ui.final
data: {"schema_version":"1.0","components":[{"id":"cmp1","type":"evidence_comparison","data_ref":"cmp_42"}]}

event: turn.completed
data: {"turn_id":"t1"}
```
Possible `turn.abstained`, `turn.failed`, `job.queued`, `job.progress`. Client reconnect by event cursor bound to user+matter; do not replay other user's messages.

## Sources and verification
`GET /source` returns safe extracted text + page coordinates if PDF viewer supports, original download link only under user authority. Stable citation IDs resolve to source version and offset. 403 for forbidden, 404 for hidden resources where existence should not be disclosed, 409 for stale source version.

## WebSocket live protocol
Client sends `session.start` after explicit consent state accepted, PCM16 chunks with sequence+timestamp (actual configuration negotiated), `session.pause`, `session.resume`, `session.stop`. Server sends confirmed state, incremental transcript segments, verified/unverified insight cards and backpressure. Unknown or unauthorized session never opens microphone pipeline. Limit streams per appliance; authenticated websocket origin checking.

## Draft mutation / CAS
`PATCH /drafts/{id}` accepts diff or edit ops with `expected_version`; rejects collisions with 409. Approval request includes last version hash so an approved draft cannot silently change before export.

## Errors
All errors `{code,message,request_id,detail?}`; redact document text/tokens. Codes `UNAUTHORIZED`, `FORBIDDEN_MATTER`, `SOURCE_NOT_VERIFIED`, `MODEL_UNAVAILABLE`, `INSUFFICIENT_EVIDENCE`, `JOB_QUEUED`, `RECORDING_NOT_AUTHORIZED`, `VERSION_CONFLICT`, `RESOURCE_LIMIT`.
