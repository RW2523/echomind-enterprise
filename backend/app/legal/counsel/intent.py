"""What a chat message asks for (docs/echomind-legal/03 "selection algorithm", step 1).

Deterministic rules, no model call: the intent only chooses which existing, authorized pipeline answers and which
component shows the result, so it must be fast, predictable and testable. Anything unrecognised is a question about
the matter's documents (`ask_fact`), which is the safe default: research answers only from those documents, with
verified quotes, or abstains.

- `inspect_case`  "What matters today?", "What changed?", "Brief me" → a briefing from records (briefing.py)
- `chronology`    "Build a chronology", "Timeline of events" → dated events from the documents (chronology.py)
- `compare`       "Find conflicting dates", "Are there inconsistencies?" → the conflict scan (discrepancies.py)
- `cases_overview` "Compare these cases", "Where do my matters stand?" → the conversation's matters side by side (cases.py)
- `other_matters` "Use the other matters too" → which matters the conversation covers, and how to add one
- `draft`         "Draft a client update" → where drafting from firm templates is done
- `deliberate`    "Challenge our position" → where the review panel is (when the firm has it on)
- `smalltalk`     "Hi", "Thanks" → a one-line reminder of what can be asked (no search, no model)
- `ask_fact`      everything else
"""
from __future__ import annotations

import re

INTENTS = ("ask_fact", "compare", "inspect_case", "chronology", "cases_overview", "other_matters", "draft", "deliberate", "smalltalk")

_CASES = re.compile(
    r"\b(?:compare|comparison\s+of|overview\s+of|status\s+of|summar(?:y|ise|ize)\s+of|side\s+by\s+side)\b[^.?!]{0,30}?"
    r"\b(?:these|the|my|our|both|all(?:\s+(?:these|my|the|our))?)\s+(?:\w+\s+)?(?:cases|matters)\b"
    r"|\b(?:these|both|my|our)\s+(?:cases|matters)\s+side\s+by\s+side\b"
    r"|\bwhere\s+do\s+(?:these|my|our|both|all)\s+(?:the\s+)?(?:cases|matters)\s+stand\b", re.I)

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
_CHRONOLOGY = re.compile(r"\b(?:chronology|chronological|timeline|time\s+line|sequence\s+of\s+events|order\s+of\s+events"
                         r"|what\s+happened\s+when)\b", re.I)
_DELIBERATE = re.compile(r"\b(?:challenge|devil'?s\s+advocate|counter-?arguments?|poke\s+holes|stress[- ]test|argue\s+against"
                         r"|what\s+would\s+(?:opposing|the\s+other\s+side'?s?)\s+counsel)\b", re.I)


def classify(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return "ask_fact"
    if _SMALLTALK.match(t):
        return "smalltalk"
    if _CASES.search(t):
        return "cases_overview"
    if _OTHER_MATTERS.search(t):
        return "other_matters"
    if _DRAFT.search(t):
        return "draft"
    if _DELIBERATE.search(t):
        return "deliberate"
    if _CHRONOLOGY.search(t):
        return "chronology"
    if _COMPARE.search(t):
        return "compare"
    if _INSPECT.search(t):
        return "inspect_case"
    return "ask_fact"
