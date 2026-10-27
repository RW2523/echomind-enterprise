# 07 — RAG, GraphRAG, anomaly detection, Council and agents
## Retrieval pipeline (P0)
1. Parse intent+matter context and ensure user can access requested matter.
2. Build query variants (legal phrase exact search + semantically related language); include version and jurisdiction where needed.
3. **Pre-filter** tenant, matter, ACL, document status, legal hold restrictions.
4. BM25 keyword and embedding recall independently; merge (e.g., reciprocal rank fusion); dedupe by source/version.
5. Rerank candidate query/passage pairs using local reranker. Do not treat reranker score as evidence of truth.
6. Neighbor expansion around relevant passage within **same authorized original**; add heading, section, page.
7. Evidence assembler compiles relevant chunks, version dates, contradictions and source metadata.
8. Generate concise answer with explicit claim/source links.
9. Independent verifier maps generated quotations to offsets and checks factual claims against sources. Claims failing support become `unverified` or removed; on no support abstain.
10. Dynamic component renderer chooses minimal UI (answer, comparison, timeline, etc.).

## GraphRAG (P2)
Extraction: names/aliases, roles, contracts, clauses, dates, events, amounts, author/source -> normalize -> human review for ambiguous identities -> versioned graph edges with provenance. Graph is optional; hybrid retrieval remains primary. Begin with query-oriented 1-hop or 2-hop exploration, bounded nodes. Never conflate a person mentioned in a document with a proven participant. Preserve assertion stance, negation, date and document reliability.

## Anomaly detection (P1)
Candidate rules: differing dates for same alleged event; document versions changing duties; conflicting amounts; missing referenced exhibits; signature absence; duplicated exhibits; unsupported authority; OCR uncertainty. Each finding: type, source A/B, why raised, false-positive alternatives, suggested verification, uncertainty, human resolution. Status: `potential`, `in_review`, `explained`, `confirmed`, `dismissed`. No automated dishonesty/fraud declaration.

## LangGraph workflows
Workflow: `authorize -> retrieve -> validate -> plan -> require_approval_if_needed -> execute_allowlisted_tool -> verify -> UI`. Store checkpoint after each node, explicit cancel/resume; retries idempotent. Tool use subject to firm policy and approval; distinguish read-only from mutating tools. Long-running jobs have bounded queue and visible status.

### Matter preparation graph
`user_request -> fetch_matter -> new_record_digest -> deadline_sources_check -> source-backed briefing -> task proposals -> draft agenda -> await_attorney_approval -> save_versions`. Deadline calculations require verified rule engine or human confirmation; never directly from generative model.

## AI Council (P2)
**Question**: user selects matter, issue and constraints. **Independent phase**: strategy, opposing counsel, evidence analyst, procedure/risk analyst review same authorized evidence separately. **Verifier**: cross-check every source and authority. **Debate**: one bounded critique round focused on disagreements; avoid unlimited conversation loops. **Synthesis**: agree/disagree/missing evidence/alternative options; attorney reviews outcome. Different personas with one LLM are not truly independent; diversity may benefit from distinct model families but measure added value before adding heavy models.

Council JSON concept:
```json
{"question":"Should we discuss early settlement?","agreements":["Evidence item X is unresolved"],"disagreements":[{"issue":"Strength of amendment defense","positions":["plausible","uncertain"],"citation_ids":["c1","c2"]}],"missing_evidence":["executed copy"],"options_for_attorney":["review amendment","seek counterpart records"],"final_decision":"attorney_required"}
```
No made-up odds of success, judicial forecasts or independent legal conclusions without validated grounds.

## MCP gateway
Read tools: `matter.search`, `document.read_span`, `timeline.query`, `graph.neighbors`, `calendar.read` (if enabled). Write tools: `draft.create`, `task.propose`, `finding.review`, `export.prepare`; real sends/filings disabled until dedicated approval workflow. Each tool runs under service identity plus **propagated end-user/matter authorization**, resource allowlist, network deny by default, limit I/O size, deadline timeout, structured logging. Never trust tool descriptions from unknown documents or run arbitrary MCP servers from user-uploaded configs.

## Evidence standards
Case facts and outside legal authorities are separate indexes. Primary authority authenticity and currency checked by actual authoritative records; offline imports include origin, license and snapshot date. For opinions, distinguish holding vs dicta and verify quotations. Provide direct source link or local copy, with `last updated` metadata. If only user-uploaded authority exists, label coverage accordingly.

## Evaluation
Golden set with questions requiring negative evidence, superseded contracts, unrelated-but-similar matters, conflicting witnesses, sensitive access, difficult tables and large PDFs. Retrieval Recall@k, MRR, citation precision/recall, answer groundedness, abstention, legal authority validity, discrepancy PPV/recall, extractions and human correction time. Council usefulness tested by blinded attorneys, not model self-evaluation.
