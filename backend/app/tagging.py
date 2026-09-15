"""
Tagging + metadata: conversation_type and topic tags for stored transcripts.
Lightweight heuristics now; interface allows LLM upgrade later.
"""
from __future__ import annotations
import re
from typing import List, Tuple
from collections import Counter

# Stopwords for keyphrase extraction (minimal set)
STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
}

CONVERSATION_KEYWORDS = {
    "meeting": ["meeting", "agenda", "minutes", "action items", "follow up", "schedule", "quarterly", "standup", "sync"],
    "lecture": ["lecture", "chapter", "slide", "today we", "today I", "students", "course", "exam", "homework"],
    "interview": ["interview", "candidate", "experience", "role", "team", "salary", "hiring", "position"],
    "brainstorming": ["idea", "brainstorm", "think about", "what if", "option", "concept", "design"],
    "casual": [],
}


def get_conversation_type(text: str) -> str:
    """
    Infer conversation type from text. Returns one of: meeting, lecture, interview, brainstorming, casual.
    """
    if not text or not text.strip():
        return "casual"
    lower = text.lower()
    scores = {}
    for ctype, keywords in CONVERSATION_KEYWORDS.items():
        scores[ctype] = sum(1 for k in keywords if k in lower)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "casual"


# Curated topic taxonomy. Auto-tags are ONLY ever drawn from these labels — never from raw
# transcript words (the old keyphrase extractor produced "everybody knows", "my", "language").
# A tag fires when enough of its cue phrases occur; at most 3 tags; none if nothing is confident.
TOPIC_TAXONOMY = {
    "Contract Review": ["contract", "clause", "agreement", "terms and conditions", "liquidated damages", "arbitration", "breach", "indemnity", "termination clause", "signed agreement", "amendment"],
    "Compliance": ["compliance", "regulation", "regulatory", "disclosure", "conduct policy", "audit", "must not", "policy 4", "suitability", "aml", "anti-money", "sanction"],
    "KYC": ["kyc", "know your customer", "identity verification", "verify your identity", "date of birth", "proof of address", "account number", "id number", "passport"],
    "Refund & Cancellation": ["refund", "cancel", "cancellation", "early termination", "cooling-off", "cooling off", "pro-rated", "money back", "chargeback"],
    "Billing": ["invoice", "billing", "late fee", "charged", "overcharged", "payment due", "autopay", "statement", "balance due", "installment"],
    "Technical Support": ["ticket", "outage", "disconnect", "not working", "technician", "router", "reset", "troubleshoot", "error message", "install"],
    "Complaint": ["complaint", "escalate", "unacceptable", "frustrated", "supervisor", "compensation", "dissatisfied", "formal complaint"],
    "Litigation": ["court", "hearing", "lawsuit", "sue", "litigation", "plaintiff", "defendant", "judgment", "precedent", "file a claim", "limitation period", "statute of limitations"],
    "Investment Advice": ["investment", "fund", "equity", "portfolio", "returns", "risk profile", "market-linked", "guaranteed return", "fixed deposit", "interest rate", "capital protected"],
    "Loan & Credit": ["loan", "mortgage", "credit card", "credit limit", "emi", "repayment", "interest", "collateral", "pre-approved"],
    "Account Services": ["transfer", "wire", "withdrawal", "deposit", "online banking", "transfer limit", "beneficiary", "savings account", "current account"],
    "Onboarding": ["onboarding", "sign up", "activate", "activation", "new customer", "welcome pack", "set up your account", "getting started"],
    "Scheduling": ["schedule", "appointment", "reschedule", "available on", "calendar", "book a", "slot", "next week", "follow-up call"],
    "Budget & Planning": ["budget", "approved", "roadmap", "milestone", "launch date", "headcount", "forecast", "quarter", "q1", "q2", "q3", "q4", "allocation"],
    "Hiring": ["hire", "hiring", "candidate", "interview", "offer letter", "salary", "recruit", "position", "role"],
    "Product Demo": ["demo", "walkthrough", "show you", "feature", "screen share", "let me share", "presentation"],
}
_TOPIC_MIN_HITS = 2          # distinct cue phrases needed
_TOPIC_MIN_WORDS = 30        # do not tag tiny transcripts
_TOPIC_MAX_TAGS = 3


def get_tags(text: str, max_tags: int = _TOPIC_MAX_TAGS) -> List[str]:
    """Topic tags from the curated taxonomy only (never raw transcript words)."""
    if not text or len(text.split()) < _TOPIC_MIN_WORDS:
        return []
    lower = " " + re.sub(r"\s+", " ", text.lower()) + " "
    scored = []
    for topic, cues in TOPIC_TAXONOMY.items():
        hits = {c for c in cues if c in lower}
        if len(hits) >= _TOPIC_MIN_HITS:
            weight = sum(lower.count(c) for c in hits)
            scored.append((topic, len(hits), weight))
    scored.sort(key=lambda x: (-x[1], -x[2]))
    return [t for t, _, _ in scored[:max_tags]]


def topic_for_title(text: str) -> str:
    """Best single topic label for an auto-generated session title, or '' when unsure."""
    tags = get_tags(text, max_tags=1)
    return tags[0] if tags else ""


def get_metadata(text: str) -> Tuple[str, List[str]]:
    """Returns (conversation_type, tags)."""
    return get_conversation_type(text), get_tags(text)
