# 10 — Security, confidentiality, privilege and legal safeguards
**Not legal advice or a certification.** Qualified counsel should review professional conduct, privacy, recording laws, confidentiality rules, discovery duties, and jurisdiction-specific court rules for each deployment.

## Threat model
Assets: original confidential matter files, attorney work product, legal advice, client identities, recordings, external credential tokens, private drafts, audit trail. Attackers: other users at firm, outside network, malicious files, unauthorized admins, vendor/update compromise, compromised browser, prompt injection embedded in PDF/EML, rogue MCP tool, model hallucination. Harm: cross-matter leak, privilege loss, inaccurate legal filing, unauthorized recording, missed court deadline, evidence corruption, irreversible external transmission.

## Controls
- TLS, modern authentication, user deactivation, optional MFA and session expiration; matter grants default deny; ethical walls.
- Secret storage protected by OS/key manager; disk encryption at rest; secure backups, independent restore key process.
- Per-matter scoped query prefilters and storage authorization; no global memory. RLS as defense in depth; policy tests at all layers.
- Malicious documents treated as **data only**; never obey embedded instructions, outgoing URLs, or MCP/agent tool directions. Prompt injection defenses include tool boundaries, scopes, allowlists, policy enforcement and red-team tests; prompts alone insufficient.
- PDF/DOCX parser sandbox, resource limits, MIME/extension check, file hash, malware scanning, no macros execution.
- Immutable originals, content hash, evidence chain-of-custody metadata, reviewable derived text and source versioning.
- Default-off recording with clearly active indicator, consent checks, pause/stop actually disable capture, retention and redaction. Different jurisdictions have different consent requirements—policy must be configured and reviewed.
- Human approval for legal filing, external communication, export of sensitive material, document finalization, factual resolution; never AI alone.
- Deadline calculations require jurisdiction-aware vetted rules and attorney verification; do not automatically set critical deadlines based on LLM interpretation.
- Audit actor, action, time, object, result, request ID; redact content while preserving accountability. Restrict audit log access.
- Disable telemetry and cloud fallbacks; audit outbound connections; signed update process; dependency/SBOM scanning on approved connected build machine.

## Existing known gaps from supplied EchoMind description
Prompt-injection defense test success rate still ~5.1% (team-provided, not independently validated): not sufficient as a release assurance; tests and countermeasures required. Backups absent; **must be fixed before real law firm data**. Consent step missing for meeting recordings; block recording capability until implemented. Abstention below team target; establish legal-specific benchmark and gating. These are clear risk flags, not verified current repo findings.

## Human-review matrix
| Action | Require human review? |
|---|---|
| Answer from verified documents | Lawyer should inspect important source |
| Flag potential contradiction | Yes, to resolve |
| Draft document | Yes, prior to client/court use |
| Export confidential client file | User confirmation + policy gate |
| Send email/file court document | Explicit separate approval; out of MVP |
| Convert meeting recording to permanent record | Recording/retention policy and user decision |
| Update binding court deadline | Authoritative rule verification + attorney approval |
| Delete records | Roles, retention and legal hold checks |

## Operational release checks
Offline WAN blockade test; cross-matter & cross-firm adversarial tests; document prompt-injection corpus; secure boot/update and credentials review; backup restore; verified source stability; consent UI/audio behavior; audit tamper checks; penetration test by qualified reviewer; incident response and lost appliance procedure.
