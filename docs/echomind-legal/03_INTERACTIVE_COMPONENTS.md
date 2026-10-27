# 03 — Dynamic UI component system
## Why
A chat message can transform into a **trusted interactive tool** while retaining conversation context. The model does not write executable UI. It chooses among versioned, safe component definitions; backend checks the data and action targets.

## Rendering protocol v1
Server response envelope:
```json
{
 "schema_version":"1.0", "message_id":"msg_123", "matter_id":"mat_demo", "summary":"Two items need review.",
 "components":[
   {"id":"a","type":"source_answer","props":{"answer":"The amendment states May 14.","citation_ids":["cit_7"]}},
   {"id":"b","type":"evidence_comparison","data_ref":"cmp_11"}
 ],
 "suggested_actions":[{"id":"view","label":"Inspect sources","action_type":"open_source","target_id":"cit_7"}]
}
```
`data_ref` must be opaque, server-resolved, scoped to request's tenant/user/matter. No raw SQL, arbitrary URLs, HTML, templates, event handler code, CSS or file paths accepted from LLM.

## First-class component registry
| Component | Allowed content | Primary action | Empty state |
|---|---|---|---|
| `source_answer` | short natural language + citation IDs | open source | no evidence found |
| `priority_list` | <=3 prioritized grounded findings | select/expand | no actionable findings |
| `source_chip` | title, page, anchor, verified flag | open document pane | unavailable source |
| `evidence_comparison` | >=2 excerpts and exact refs | inspect / mark reviewed | missing comparison |
| `timeline` | source-supported dated events | open event | no verified dates |
| `checklist` | id, title, status, owner | toggle via API | no tasks |
| `guided_questions` | 1–3 typed questions | submit values | none |
| `document_preview` | draft id, version, citations | edit/review | no draft |
| `approval_request` | action, consequences, target and scope | accept/reject | not authorized |
| `council_summary` | agreements, disputes, uncertainties | expand arguments | insufficient evidence |
| `graph_excerpt` | bounded nodes/edges and source refs | expand neighborhood | no verified links |
| `live_transcript` | consent state, time, text segments | stop/pause/bookmark | inactive |
| `task_progress` | actual job stage, cancellation | cancel/inspect | pending |
| `numeric_chart` | verified numerical data + units + provenance | inspect data | nonnumeric data |

## Selection algorithm (conceptual)
1. Classify user intent: ask_fact, compare, list_tasks, draft, inspect_case, live_session, deliberate, approve.
2. Fetch authorized evidence; verify citations and uncertainty.
3. Select **at most 1 primary interactive component** and up to 2 supporting components initially.
4. For simple fact answer use `source_answer` alone; for comparison show `evidence_comparison`; for draft open editor; for deadlines use confirmed calendar view.
5. Add contextual actions only if implemented, authorized, and reversible where possible. Additional details behind disclosure.
6. Persist component state server-side if it affects matter records; optimistic UI only with rollback on failure.

## Typed frontend contracts (illustrative)
```ts
type Citation = { id: string; documentId: string; page?: number; startOffset?: number; endOffset?: number; status: 'verified'|'unverified' };
type SafeComponent =
 | { id: string; type: 'source_answer'; props: { answer: string; citation_ids: string[] } }
 | { id: string; type: 'evidence_comparison'; data_ref: string }
 | { id: string; type: 'checklist'; data_ref: string }
 | { id: string; type: 'timeline'; data_ref: string }
 | { id: string; type: 'approval_request'; data_ref: string };
```
Use Zod or equivalent client parse **plus** backend Pydantic validation; unknown types render a noninteractive graceful fallback; schema is backwards versioned.

## Action contract
`POST /v1/ui/actions` with `{component_id, action_id, matter_id, expected_version, parameters, idempotency_key}`. Backend revalidates identity, matter access, operation privilege, component origin, server-side target and version; returns updated component state, never trusts client-supplied action destinations. Record audit trail.

## UX dynamics
- Inline card expand/collapse without route change.
- Source drawer works in parallel with conversation; "Use this source" inserts a structured reference in composer.
- Draft opens split pane, references clickable; agent can suggest patches **only**, user accepts edits.
- Answer may stream plaintext first, then finalized verified components; never flash unsupported citations as verified.
- Include `source_version` and `index_version` in evidence output; warn on stale evidence.

## Test cases
Unknown component injection, huge array payload, XSS text, hostile markdown links, inaccessible citation IDs, forged action IDs, component version race, attempted cross-matter card, keyboard navigation, no-JS safe error, narrow screen, slow stream reconnect and cancellation.
