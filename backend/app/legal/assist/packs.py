"""The legal packs for Silent Assist (S2.5, LIVE-02; decisions D-015 and D-041).

A pack is a set of rules for one kind of meeting:
- **legal_client_call** and **legal_negotiation**: statements about the signed terms, the board's
  authority, commitments and privileged disclosures with a counterparty on the call, and actions
  promised.
- **legal_governance** (board meetings): quorum, a director voting on something they declared an
  interest in, the authority given in the matter's minutes, and actions.

Each rule has a severity and a confidence: what a match of that rule is worth. An alert is shown
only at or above LEGAL_ASSIST_MIN_CONFIDENCE. Every value here is a conservative engineering
default awaiting a lawyer's sign-off (P-008), and the UI says so.

Wording is evidence-relative (D-015): an alert says what was said "matches" or "differs from" a
source in this matter, or "needs review" where there is no source. It never says "true" or "false".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    severity: str                      # critical | notice
    confidence: float                  # what a match is worth (rules); the model's checks are 0.8 when confirmed
    roles: tuple = ()                  # speaker roles it listens to; empty: everyone
    counterparty_present: bool = False  # only when a counterparty is a participant


RULES = {
    "signed_terms": Rule("signed_terms", "A statement about the signed terms", "critical", 0.8),
    "authority": Rule("authority", "A figure outside the authority recorded in the matter", "critical", 0.85),
    "commitment": Rule("commitment", "A commitment made to the other side", "critical", 0.75, ("attorney", "host"), True),
    "disclosure": Rule("disclosure", "Privileged or sensitive information with a counterparty on the call", "critical", 0.8,
                       ("attorney", "host", "client"), True),
    "action": Rule("action", "An action someone took on", "notice", 0.7),
    "quorum": Rule("quorum", "Fewer than half the directors present", "critical", 0.75),
    "conflict": Rule("conflict", "A vote after a declared interest", "critical", 0.85),
}

PACKS = {
    "legal_client_call": ("signed_terms", "authority", "commitment", "disclosure", "action"),
    "legal_negotiation": ("signed_terms", "authority", "commitment", "disclosure", "action"),
    "legal_governance": ("quorum", "conflict", "authority", "action"),
}
LABELS = {"legal_client_call": "Client call", "legal_negotiation": "Negotiation", "legal_governance": "Board or governance"}
_FOR_KIND = {"client_call": "legal_client_call", "negotiation": "legal_negotiation", "board": "legal_governance",
             "interview": "legal_client_call", "internal": "legal_client_call", "other": "legal_client_call", "hearing": None}


def pack_for(meeting_kind: str, chosen: Optional[str]) -> Optional[str]:
    """The pack a meeting uses: the one chosen for it, none if turned off, else its kind's default."""
    if chosen == "off":
        return None
    if chosen in PACKS:
        return chosen
    return _FOR_KIND.get(meeting_kind, "legal_client_call")
