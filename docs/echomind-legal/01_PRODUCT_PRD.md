# 01 — Product requirements (PRD)
## One-liner
EchoMind Legal is a private, on-premises, evidence-grounded AI workspace for small law firms that turns natural-language requests into concise answers, interactive case reviews, drafts and approved actions.

## Problem / user
Target first: 1–10-lawyer U.S. civil or commercial litigation firm with confidential PDFs/DOCX, fragmented documents and email, limited IT. Other verticals (immigration, criminal defense, transactional) require tailored workflows and validation; do not mix presumptive court rules.

Personas: managing partner (matters + oversight); attorney (research, preparation, drafting); paralegal (intake, organize, verify); office admin (users/backup not confidential case content unless authorized); reviewer (approve drafts); compliance/security admin.

## Jobs to be done
- "Tell me what changed in this matter, showing only material issues."
- "Find the clauses and evidence for my claim; let me inspect exactly where they came from."
- "Compare these two accounts; don't decide who is telling the truth."
- "Create a chronology and show unknown dates."
- "Draft a client update from reviewed evidence using our template."
- "Help me prepare for a meeting; listen only after explicit permission."
- "Challenge this argument and show what a reasonable opponent would say."
- "Propose time entries from approved work logs, without fabricated hours."

## Product principles
1. One chat-first entry point; only essential navigation: Chat, Matters, Drafts, Activity; settings hidden under profile.
2. Answer first, details second. Max three findings by default; deeper details on demand.
3. Actions in context, never feature-led module hopping.
4. Sources are first-class, traceable, visible, and objectively checked.
5. Privacy, privilege, consent and approval designed in, not retrofitted.
6. Explicitly distinguish facts, allegations, inferences, law, uncertainties and suggestions.
7. Fast path for low-cost responses; slower background analysis visibly labeled.
8. Human decision-maker always controls legal advice, filing, external communication.

## Scope, prioritization
**P0:** login, firm/matter membership, upload PDF/DOCX/TXT/EML (as supported), local parse, dedup/hash, hybrid indexing, legal chat with verified citations, exact source viewer, supported/unsupported state, responsive UI, interactive source/comparison/checklist, draft editor, audit, restore tests.
**P1:** timelines, entity extraction, discrepancy queue, structured drafting templates, task proposals, consent-based live transcript, offline transcript search, contextual quiet insights.
**P2:** GraphRAG, council, advanced MCP connectors, proposed time capture, policy-backed deadline verification, evidence chain graph.
**P3:** multi-system integrations and richer collaboration; do not claim full automated case-management replacement.

## Functional requirements
FR01 Firm admin can invite/deactivate and assign roles, with explicit per-matter grants and ethical walls.
FR02 Matter creates isolated documents, chat history, embeddings, summaries, graphs and approved outputs.
FR03 Ingest: virus/type check, quotas, preserve original bytes/hash, OCR only when authorized/needed, parse with page/section references, asynchronous indexing status.
FR04 Search: filter tenant/matter ACL in query engine before candidate selection, hybrid BM25/vector, rerank, cite exact spans.
FR05 Chat: stream plaintext plus structured components; user can click sources and continue with selected context.
FR06 Reviewer can reject/confirm evidence anomalies; machine state remains `potential` until human resolved.
FR07 Generate editable draft with citations, source coverage, version history; require approval for external-ready documents.
FR08 Recording requires explicit session start, visible red indicator and clearly implemented stop/pause; retention policy.
FR09 Every state-changing operation emits audit event with actor, matter, action, outcome and correlated request id.
FR10 Private default: no internet outbound calls at runtime; verify via network-deny integration tests.
FR11 AI Council (P2) returns differences, counterarguments, citations and uncertainty, not an autonomous final legal decision.
FR12 All uploads and recordings can be redacted/exported based on actual policy; no blanket forced removal of useful client fields.

## Nonfunctional
P95 target for cached basic UI actions under 300ms local network, chat first token and full report latency **to be benchmarked on actual DGX Spark**, not assumed. Availability/recovery defined and tested per deployment; recoverability > marketing uptime claims. Keyboard and screen reader interaction required. Audio processing should be prioritized only during active sessions.

## Explicit non-goals / guardrails
Not licensed legal advice; not auto-submitting court filings; not automatic trust in retrieved documents or model-generated court citations; not covert always-on listening; not cross-client universal semantic memory; not autonomous email sending; no cloud fallback hidden from user.

## Success
Pilot lawyers can answer sourced case questions, compare evidence, and prepare a reviewable draft with materially less manual navigation while maintaining zero unauthorized access in the test suite. Track task completion, time to find evidence, unsupported citation rate, attorney correction rate, permission violations and restore ability.
