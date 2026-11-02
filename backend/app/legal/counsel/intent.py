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
- `approvals`     "What is waiting for my approval?" → the review queue's items in these matters, to decide (approvals.py)
- `live`          "Show the live transcript" → the meeting being recorded, or the latest transcript (live.py)
- `amounts`       "Chart the amounts in the documents" → stated amounts as bars, each with its sentence (amounts.py)
- `parties`       "How are the parties connected?" → parties and the documents naming them (parties.py)
- `tasks`         "What are the open obligations?", "List upcoming deadlines" → the obligations register (tasks.py);
                  a question about what a document says about deadlines stays `ask_fact`
- `draft`         "Draft a client update" → three questions, then a draft from a firm template (drafting.py)
- `deliberate`    "Challenge our position" → the review panel, when the firm has it on (deliberation.py)
- `smalltalk`     "Hi", "Thanks" → a one-line reminder of what can be asked (no search, no model)
- `ask_fact`      everything else
"""
from __future__ import annotations

import re

INTENTS = ("ask_fact", "compare", "inspect_case", "chronology", "cases_overview", "other_matters", "documents", "draft", "deliberate",
           "smalltalk", "approvals", "live", "amounts", "parties", "tasks")

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
# The matter's list of documents, read from their records (2026-10-29 QA CHAT-001: "Which documents are available?" went to
# document search and came back "do not answer that"). A question about what documents say ("Which documents mention the
# deposit?") stays a research question.
_DOCUMENTS = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:"
    r"(?:which|what)\s+(?:documents?|files?)\s+(?:are|is|do|does|have|has)\b"
    r"(?![^?]*\b(?:mention|say|said|state|refer|contain|discuss|cover|show|support|prove|deal|address|govern|include|provide|set|"
    r"name|quote|cite|describe|record|confirm|list)\w*)"
    r"|(?:list|show)\s+(?:me\s+)?(?:all\s+|the\s+|my\s+|our\s+)?(?:documents|files)\b(?!\s+(?:that|which|about|on|mentioning|saying|where))"
    r"|what(?:'s|\s+is)\s+(?:in|on)\s+(?:this|the)\s+(?:matter|file|case)(?:\s+file)?\s*\??\s*$"
    r"|how\s+many\s+documents\s+(?:are|do|does|have|has)\b(?![^?]*\b(?:mention|say|state|refer)\w*))", re.I)
_DRAFT = re.compile(r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:draft|write|prepare|compose)\b"
                    r"|\b(?:draft|write|prepare)\s+(?:a|an|the|me\s+a)\s+(?:\w+\s+){0,3}(?:letter|update|memo|email|reply|notice|agreement|contract)\b", re.I)
_SMALLTALK = re.compile(r"^\s*(?:hi|hello|hey|thanks|thank\s+you|thx|cheers|ok(?:ay)?|good\s+(?:morning|afternoon|evening))"
                        r"(?:\s+(?:there|again|so\s+much|echomind))?\s*[!.?]*\s*$", re.I)
_CHRONOLOGY = re.compile(r"\b(?:chronology|chronological|timeline|time\s+line|sequence\s+of\s+events|order\s+of\s+events"
                         r"|what\s+happened\s+when)\b", re.I)
_APPROVALS = re.compile(
    r"\b(?:waiting|pending|awaiting)\s+(?:for\s+)?(?:my\s+|a\s+|your\s+)?(?:approval|sign[- ]?off|decision|review)\b"
    r"|\b(?:my|pending|outstanding)\s+(?:approvals|sign[- ]?offs|reviews)\b"
    r"|\b(?:need|needs|needing)\s+my\s+(?:approval|sign[- ]?off|decision)\b"
    r"|\b(?:anything|what(?:'s|\s+is)?|is\s+there\s+anything)\s+(?:for\s+me\s+)?to\s+(?:approve|sign\s+off)\b", re.I)
_LIVE = re.compile(r"\blive\s+transcript\b|\b(?:current|live|ongoing)\s+meeting\b|\bwhat(?:'s|\s+is)\s+being\s+said\b"
                   r"|\bshow\s+(?:me\s+)?the\s+(?:meeting\s+)?transcript\b|\bis\s+(?:anything|the\s+meeting)\s+being\s+recorded\b", re.I)
_AMOUNTS = re.compile(
    r"\b(?:chart|graph|plot|visuali[sz]e)\b[^.?!]{0,40}\b(?:amounts?|sums?|figures|payments?|fees|money|prices?|costs?|values)\b"
    r"|\b(?:amounts?|sums?|figures|payments?|fees)\b[^.?!]{0,30}\b(?:as\s+a\s+|in\s+a\s+)?(?:chart|graph)\b"
    r"|^\s*(?:please\s+)?(?:list|show)\s+(?:me\s+)?(?:all\s+|the\s+)?(?:amounts|sums|figures|payments)\b"
    r"|^\s*(?:what|which)\s+amounts\b", re.I)
_PARTIES = re.compile(
    r"\b(?:relationships?|connections?|connected|network|graph|map)\b[^.?!]{0,40}\b(?:parties|people|companies|entities|organi[sz]ations)\b"
    r"|\b(?:parties|people|companies|entities|organi[sz]ations)\b[^.?!]{0,40}\b(?:relationships?|connections?|connected|network|graph|map)\b"
    r"|\bwho\s+is\s+connected\s+to\s+whom\b|\bmap\s+(?:out\s+)?the\s+parties\b|\brelationship\s+(?:map|graph)\b", re.I)
# The register's list, not what a document says: "What are the open obligations?" is a list; "What deadlines does the lease
# set?" is a research question.
_TASKS = re.compile(r"\b(?:obligations|deadlines|tasks|to-?dos|to-do\s+list|checklist)\b", re.I)
_TASKS_LIST = re.compile(r"^\s*(?:please\s+)?(?:show|list|give\s+me|what\s+are|which\s+are|are\s+there\s+any|any)\b"
                         r"|\b(?:open|outstanding|upcoming|pending|overdue|my|our|register|next)\b", re.I)
_ABOUT_DOCUMENTS = re.compile(r"\b(?:clause|section|agreement|contract|lease|letter|documents?|says?|stated?|states|sets?|under|mention\w*"
                              r"|requires?|according)\b", re.I)
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
    if _DOCUMENTS.search(t):
        return "documents"
    if _DELIBERATE.search(t):
        return "deliberate"
    if _APPROVALS.search(t):
        return "approvals"
    if _LIVE.search(t):
        return "live"
    if _CHRONOLOGY.search(t):
        return "chronology"
    if _AMOUNTS.search(t):
        return "amounts"
    if _PARTIES.search(t):
        return "parties"
    if _DRAFT.search(t):
        return "draft"
    if _TASKS.search(t) and _TASKS_LIST.search(t) and not _ABOUT_DOCUMENTS.search(t):
        return "tasks"
    if _COMPARE.search(t):
        return "compare"
    if _INSPECT.search(t):
        return "inspect_case"
    return "ask_fact"
