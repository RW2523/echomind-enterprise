"""What a chat message asks for (docs/echomind-legal/03 "selection algorithm", step 1).

Deterministic rules, no model call: the intent only chooses which existing, authorized pipeline answers and which
component shows the result, so it must be fast, predictable and testable. Anything unrecognised is a question about
the matter's documents (`ask_fact`), which is the safe default: research answers only from those documents, with
verified quotes, or abstains.

- `inspect_case`  "What matters today?", "What changed?", "Brief me" → a briefing from records (briefing.py)
- `compare`       "Find conflicting dates", "Are there inconsistencies?" → the conflict scan (discrepancies.py)
- `other_matters` "Use the other matters too" → an explanation that a conversation is limited to its matter
- `draft`         "Draft a client update" → where drafting from firm templates is done
- `deliberate`    "Challenge our position" → where the review panel is (when the firm has it on)
- `smalltalk`     "Hi", "Thanks" → a one-line reminder of what can be asked (no search, no model)
- `ask_fact`      everything else
"""
from __future__ import annotations

import re

INTENTS = ("ask_fact", "compare", "inspect_case", "other_matters", "draft", "deliberate", "smalltalk")

_OTHER_MATTERS = re.compile(
    r"\b(?:other|another|all|every|any|across|different|related)\s+(?:of\s+(?:the|our|my)\s+)?(?:matters?|cases?|clients?|files?)\b"
    r"|\bother\s+case\s+materials?\b|\bacross\s+(?:the\s+)?firm\b|\b(?:outside|beyond)\s+this\s+(?:matter|case)\b", re.I)
_INSPECT = re.compile(
    r"^\s*(?:so\s+|ok(?:ay)?,?\s+)?(?:what\s+(?:matters|changed|has\s+changed|is\s+new|'?s\s+new|needs?\s+(?:my\s+)?attention|should\s+i\s+(?:know|look\s+at|review))"
    r"|brief\s+me|catch\s+me\s+up|give\s+me\s+(?:a|the)\s+(?:briefing|status|summary\s+of\s+what'?s\s+new)"
    r"|anything\s+new|where\s+(?:are|do)\s+we\s+stand|status\s+update)\b"
    r"|\bwhat\s+matters\s+today\b|\bwhat\s+changed\s+(?:in|on|since)\b", re.I)
_COMPARE = re.compile(
    r"\b(?:conflict(?:s|ing)?|contradict\w*|inconsisten\w*|discrepanc\w*|mismatch\w*|disagree\w*|compare|comparison)\b", re.I)
_DRAFT = re.compile(r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:draft|write|prepare|compose)\b"
                    r"|\b(?:draft|write|prepare)\s+(?:a|an|the|me\s+a)\s+(?:\w+\s+){0,3}(?:letter|update|memo|email|reply|notice|agreement|contract)\b", re.I)
_SMALLTALK = re.compile(r"^\s*(?:hi|hello|hey|thanks|thank\s+you|thx|cheers|ok(?:ay)?|good\s+(?:morning|afternoon|evening))"
                        r"(?:\s+(?:there|again|so\s+much|echomind))?\s*[!.?]*\s*$", re.I)
_DELIBERATE = re.compile(r"\b(?:challenge|devil'?s\s+advocate|counter-?arguments?|poke\s+holes|stress[- ]test|argue\s+against"
                         r"|what\s+would\s+(?:opposing|the\s+other\s+side'?s?)\s+counsel)\b", re.I)


def classify(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return "ask_fact"
    if _SMALLTALK.match(t):
        return "smalltalk"
    if _OTHER_MATTERS.search(t):
        return "other_matters"
    if _DRAFT.search(t):
        return "draft"
    if _DELIBERATE.search(t):
        return "deliberate"
    if _COMPARE.search(t):
        return "compare"
    if _INSPECT.search(t):
        return "inspect_case"
    return "ask_fact"
