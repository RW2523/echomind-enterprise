"""Scenario profiles: WHO is talking, WHAT to extract, WHICH records to pull, WHICH tags are
allowed (with proof rules), and the domain rules the verifier applies.

A profile is data, not code — adding a vertical means adding a profile here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TagSpec:
    id: str
    label: str            # UI label ("Wrong" for contradicted)
    tone: str             # green|red|yellow|blue|violet|indigo|teal|orange|amber|grey|cyan
    proof: str            # 'quote' | 'record' | 'rule' | 'span' | 'none'
    description: str = ""


@dataclass(frozen=True)
class RecordTarget:
    kind: str                       # customer_file|contract|ticket|previous_call|matter|related_case|account|product|policy|kyc|document
    query_template: str             # semantic query, {name}/{ids} substituted
    include_transcripts: bool = False


@dataclass(frozen=True)
class Rule:
    id: str
    text: str
    trigger: str                    # regex (case-insensitive) on the sentence
    tag: str                        # tag id this rule attaches
    roles: tuple = ()               # restrict to these roles ('' = any)
    severity: str = "medium"


@dataclass(frozen=True)
class ScenarioProfile:
    id: str
    label: str
    description: str
    default_namespace: str
    roles: Dict[str, str]           # {'me': 'agent', 'other': 'caller'} — role ids used in WS
    verify_roles: tuple             # roles whose claims are verified
    lookup_roles: tuple             # roles whose personal details trigger record pulls
    entity_kinds: tuple
    record_targets: tuple           # RecordTarget
    include_transcripts: bool
    tag_vocab: tuple                # TagSpec
    rules: tuple                    # Rule
    prompt_rules: str
    analysis_mode_default: str      # 'flags_only' | 'flags_and_records'
    detect_cues: tuple              # regex fragments used by the scenario detector

    def tag(self, tag_id: str) -> Optional[TagSpec]:
        for t in self.tag_vocab:
            if t.id == tag_id:
                return t
        return None

    def allowed_tag_ids(self) -> List[str]:
        return [t.id for t in self.tag_vocab]

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "default_namespace": self.default_namespace,
            "roles": dict(self.roles),
            "analysis_mode_default": self.analysis_mode_default,
            "tag_vocab": [
                {"id": t.id, "label": t.label, "tone": t.tone, "proof": t.proof, "description": t.description}
                for t in self.tag_vocab
            ],
        }


# ── Shared tag definitions ─────────────────────────────────────────────────────
T = {
    "supported": TagSpec("supported", "Supported", "green", "quote", "Confirmed by a source; the quote is the proof."),
    "contradicted": TagSpec("contradicted", "Wrong", "red", "quote", "Conflicts with a source; the quote shows the correct value."),
    "unverified": TagSpec("unverified", "Unverified", "yellow", "none", "A claim, but no source confirms or denies it."),
    "record-found": TagSpec("record-found", "Record found", "blue", "record", "A record mentioning this person/identifier was pulled."),
    "personal-detail": TagSpec("personal-detail", "Personal detail", "violet", "span", "Name, ID, phone, DOB or similar was spoken; used to look up records."),
    "contract-clause": TagSpec("contract-clause", "Contract", "indigo", "quote", "Matches a clause in a contract/agreement/terms document."),
    "policy": TagSpec("policy", "Policy", "teal", "quote", "Matches a policy / regulation / SOP statement."),
    "risk": TagSpec("risk", "Risk", "orange", "rule", "Risky claim or promise per a source or a domain rule."),
    "violating": TagSpec("violating", "Violation", "red", "rule", "Breaks a rule or policy in the sources."),
    "disclosure-missing": TagSpec("disclosure-missing", "Disclosure missing", "orange", "rule", "A required disclosure/step was not given."),
    "related-case": TagSpec("related-case", "Related case", "indigo", "quote", "A related matter or precedent in the sources."),
    "action-item": TagSpec("action-item", "Action item", "amber", "span", "Something someone must do."),
    "commitment": TagSpec("commitment", "Commitment", "amber", "span", "A promise made to the other party."),
    "question": TagSpec("question", "Question", "grey", "span", "A question that may need a record lookup."),
    "decision": TagSpec("decision", "Decision", "cyan", "span", "A decision taken."),
    "reference": TagSpec("reference", "Reference", "cyan", "quote", "Related material in the knowledge base."),
}


def _tags(*ids: str) -> tuple:
    return tuple(T[i] for i in ids)


_COMMON_TAGS = ("supported", "contradicted", "unverified", "record-found", "personal-detail",
                "action-item", "commitment", "question", "reference")

# ── Profiles ───────────────────────────────────────────────────────────────────
CUSTOMER_CARE = ScenarioProfile(
    id="customer_care",
    label="Customer call",
    description="Support / customer-care agent speaking with a caller. Verifies what the agent says against policy & T&Cs; pulls the caller's records from stated details.",
    default_namespace="",
    roles={"me": "agent", "other": "caller"},
    verify_roles=("agent", "caller"),
    lookup_roles=("caller",),
    entity_kinds=("person", "phone", "email", "dob", "account", "policy", "order", "ticket", "product", "address", "amount", "date"),
    record_targets=(
        RecordTarget("customer_file", "customer record for {name} {ids}"),
        RecordTarget("contract", "contract or agreement for {name} {ids}"),
        RecordTarget("ticket", "support ticket complaint history {name} {ids}"),
        RecordTarget("previous_call", "previous call with {name} {ids}", include_transcripts=True),
    ),
    include_transcripts=True,
    tag_vocab=_tags(*_COMMON_TAGS, "policy", "contract-clause", "risk"),
    rules=(
        Rule("cc-refund-promise", "Refund/compensation promised — must match policy", r"\b(full refund|we will refund|refund you|compensat|waive the fee)\b", "risk", ("agent",)),
        Rule("cc-guarantee", "Guarantee language to a customer", r"\b(guarantee[sd]?|100 ?%|never fail|always works)\b", "risk", ("agent",)),
    ),
    prompt_rules="Verify what the AGENT tells the CALLER against policy, terms and product documents. Flag wrong information, promises not backed by policy, and missing steps. Treat the caller's stated personal details as lookup keys, not claims to verify.",
    analysis_mode_default="flags_and_records",
    detect_cues=(r"thank you for calling", r"how (can|may) i help", r"my (account|order|policy|ticket)", r"customer (service|care|support)", r"order number", r"reference number"),
)

LEGAL = ScenarioProfile(
    id="legal",
    label="Legal consultation",
    description="Lawyer speaking with a client. Pulls the client's matter/previous records and related cases; flags contract clauses, risks and violations with quotes.",
    default_namespace="law",
    roles={"me": "lawyer", "other": "client"},
    verify_roles=("lawyer", "client"),
    lookup_roles=("lawyer", "client"),
    entity_kinds=("person", "org", "case", "contract", "clause", "statute", "date", "amount", "phone", "email"),
    record_targets=(
        RecordTarget("matter", "matter file for client {name} case {ids}"),
        RecordTarget("related_case", "related case precedent similar to {name} {ids}"),
        RecordTarget("contract", "contract agreement clause {name} {ids}"),
        RecordTarget("previous_call", "previous consultation with {name}", include_transcripts=True),
    ),
    include_transcripts=True,
    tag_vocab=_tags(*_COMMON_TAGS, "contract-clause", "related-case", "violating", "risk", "policy", "decision"),
    rules=(
        Rule("law-outcome-guarantee", "Guaranteeing a legal outcome", r"\b(guarantee|certain(ly)? win|you will win|no way (you|we) lose)\b", "risk", ("lawyer",)),
        Rule("law-deadline", "Deadline / limitation period mentioned", r"\b(deadline|limitation period|statute of limitations|within \d+ (days|weeks|months))\b", "action-item"),
    ),
    prompt_rules="Verify statements about the client's matter, contract terms, deadlines and legal positions against the matter files, contracts and case references. Flag clauses that are quoted or paraphrased, related cases, promised outcomes, and anything conflicting with the records.",
    analysis_mode_default="flags_and_records",
    detect_cues=(r"\bcase\b", r"\bhearing\b", r"\bcounsel\b", r"\bclause\b", r"my client", r"\bcourt\b", r"\bplaintiff|defendant\b", r"\bcontract\b"),
)

BANKING = ScenarioProfile(
    id="banking",
    label="Banking",
    description="Bank agent speaking with a client. Pulls KYC/account/product records; flags missing disclosures, mis-selling and policy violations with quotes.",
    default_namespace="bank",
    roles={"me": "banker", "other": "client"},
    verify_roles=("banker", "client"),
    lookup_roles=("client",),
    entity_kinds=("person", "account", "card", "product", "amount", "date", "phone", "email", "dob", "employer"),
    record_targets=(
        RecordTarget("kyc", "KYC customer profile {name} {ids}"),
        RecordTarget("account", "account details statement {name} {ids}"),
        RecordTarget("product", "product terms interest rate fees {ids}"),
        RecordTarget("policy", "bank policy disclosure requirement {ids}"),
        RecordTarget("previous_call", "previous call with {name}", include_transcripts=True),
    ),
    include_transcripts=True,
    tag_vocab=_tags(*_COMMON_TAGS, "disclosure-missing", "violating", "risk", "policy", "contract-clause"),
    rules=(
        Rule("bank-guaranteed-return", "Guaranteed-return language on an investment", r"\b(guaranteed (return|profit|income)|risk[- ]free|cannot lose|no risk)\b", "violating", ("banker",), "high"),
        Rule("bank-recommend-no-risk", "Product recommended — risk disclosure required", r"\b(i (would )?recommend|best option for you|you should (buy|invest|take))\b", "disclosure-missing", ("banker",)),
        Rule("bank-large-transfer", "Large transfer / withdrawal mentioned", r"\b(transfer|withdraw|wire)\b.{0,40}\b(\d{2,3},?\d{3}|\d+ ?(k|thousand|lakh|million))\b", "risk"),
    ),
    prompt_rules="Verify what the BANKER tells the CLIENT against product terms, fee schedules, KYC/AML policy and disclosure rules. Flag mis-selling, missing risk disclosures, guaranteed-return language and anything conflicting with the account or product records.",
    analysis_mode_default="flags_and_records",
    detect_cues=(r"account (balance|number)", r"\bloan\b", r"interest rate", r"\bdeposit\b", r"\bkyc\b", r"\bcredit card\b", r"\bmortgage\b", r"\bbranch\b"),
)

GENERAL = ScenarioProfile(
    id="general",
    label="General conversation",
    description="Two or more people talking (meeting, interview, discussion). Every claim is checked against the knowledge base; action items and decisions are captured.",
    default_namespace="",
    roles={"me": "speaker_a", "other": "speaker_b"},
    verify_roles=("speaker_a", "speaker_b"),
    lookup_roles=("speaker_a", "speaker_b"),
    entity_kinds=("person", "org", "amount", "date", "product", "account", "case", "contract"),
    record_targets=(
        RecordTarget("document", "document about {name} {ids}"),
    ),
    include_transcripts=False,
    tag_vocab=_tags("supported", "contradicted", "unverified", "record-found", "action-item", "commitment", "decision", "risk", "reference", "question"),
    rules=(),
    prompt_rules="Check each factual claim against the sources. Capture action items (who does what by when), decisions and commitments. Do not treat small talk as claims.",
    analysis_mode_default="flags_only",
    detect_cues=(r"\bagenda\b", r"\bnext steps\b", r"\baction item\b", r"let's (decide|move on)", r"\bmeeting\b"),
)

SCENARIOS: Dict[str, ScenarioProfile] = {p.id: p for p in (CUSTOMER_CARE, LEGAL, BANKING, GENERAL)}

_NS_DEFAULT = {"bank": "banking", "law": "legal", "health": "customer_care", "retail": "customer_care", "meetings": "general", "": "general", "default": "general"}


def profile_for(scenario: Optional[str], namespace: Optional[str]) -> ScenarioProfile:
    sid = (scenario or "").strip().lower()
    if sid in SCENARIOS:
        return SCENARIOS[sid]
    return SCENARIOS[_NS_DEFAULT.get((namespace or "").strip().lower(), "general")]


def public_profiles() -> List[dict]:
    return [p.public() for p in SCENARIOS.values()]


_CUE_RES: Dict[str, List[re.Pattern]] = {
    pid: [re.compile(c, re.IGNORECASE) for c in p.detect_cues] for pid, p in SCENARIOS.items()
}


def suggest_scenario(window_text: str, namespace: Optional[str] = None):
    """Cheap scenario detector: cue hits per profile + namespace prior. Returns
    (scenario_id, confidence 0-1, reason). No LLM."""
    txt = (window_text or "").lower()
    if len(txt.split()) < 12:
        return None, 0.0, "too short"
    scores: Dict[str, float] = {}
    for pid, pats in _CUE_RES.items():
        hits = sum(1 for p in pats if p.search(txt))
        scores[pid] = hits / max(3.0, len(pats) * 0.6)
    ns_default = _NS_DEFAULT.get((namespace or "").strip().lower())
    if ns_default and ns_default in scores:
        scores[ns_default] += 0.35
    best = max(scores, key=lambda k: scores[k])
    conf = min(1.0, scores[best])
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.15:
        conf *= 0.6
    return best, round(conf, 2), f"cues={ [k for k,v in ranked if v>0][:3] }"
