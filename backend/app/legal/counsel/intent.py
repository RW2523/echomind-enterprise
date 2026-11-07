"""What a chat message asks for (docs/echomind-legal/03 "selection algorithm", step 1).

Deterministic rules, no model call: the intent only chooses which existing, authorized pipeline answers and which
component shows the result, so it must be fast, predictable and testable. Anything unrecognised is a question about
the matter's documents (`ask_fact`), which is the safe default: research answers only from those documents, with
verified quotes, or abstains.

- `inspect_case`  "What matters today?", "What changed?", "Brief me" → a briefing from records (briefing.py)
- `chronology`    "Build a chronology", "Timeline of events", "… Give the key dates." → dated events from the documents
                  (chronology.py), narrowed to what the request names ("of the delivery, from dispatch to acceptance")
- `compare`       "Find conflicting dates", "Are there inconsistencies?" → the conflict scan (discrepancies.py)
- `cases_overview` "Compare these cases", "Where do my matters stand?" → the conversation's matters side by side (cases.py)
- `other_matters` "Use the other matters too" → which matters the conversation covers, and how to add one
- `approvals`     "What is waiting for my approval?" → the review queue's items in these matters, to decide (approvals.py)
- `live`          "Show the live transcript" → the meeting being recorded, or the latest transcript (live.py)
- `amounts`       "Chart the amounts in the documents" → stated amounts as bars, each with its sentence (amounts.py)
- `parties`       "How are the parties connected?" → parties and the documents naming them (parties.py)
- `definitions`   "Run a definitions check on the supply agreement", "What drafting problems does it have?" → the rules-only
                  definitions, numbering and cross-reference check of one document (definitions_check.py)
- `tasks`         "What are the open obligations?", "List upcoming deadlines" → the obligations register (tasks.py);
                  a question about what a document says about deadlines stays `ask_fact`
- `draft`         "Draft a client update" → three questions, then a draft from a firm template (drafting.py)
- `deliberate`    "Challenge our position" → the review panel, when the firm has it on (deliberation.py)
- `smalltalk`     "Hi", "Thanks" → a one-line reminder of what can be asked (no search, no model)
- `ask_fact`      everything else
"""
from __future__ import annotations

import re
from typing import Optional

INTENTS = ("ask_fact", "compare", "inspect_case", "chronology", "cases_overview", "other_matters", "documents", "draft", "deliberate",
           "smalltalk", "approvals", "live", "amounts", "parties", "tasks", "definitions")

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
_COMPARE = re.compile(r"\b(?:conflict(?:s|ing)?|contradict\w*|inconsisten\w*|discrepanc\w*|mismatch\w*|disagree\w*)\b", re.I)
# "Compare" asks for the conflict scan when what is compared is the documents' facts ("Compare the May 12 and May 15 claims",
# "compare the witness accounts"). A term in named documents is a question for the documents (2026-11-03 evaluation: "How
# does the liability cap in the reseller agreement compare with the one in the MSA?" was answered with a scan card about
# notice periods).
_COMPARE_FACTS = re.compile(r"\bcompar(?:e|es|ed|ing|ison(?:\s+of)?)\s+(?:the\s+|all\s+(?:the\s+)?|these\s+|both\s+)?(?:[\w-]+\s+){0,6}?"
                            r"(?:dates|claims|accounts|statements|witness(?:es)?|figures|amounts|documents|versions|facts|values)\b", re.I)
# The matter's list of documents, read from their records (2026-10-29 QA CHAT-001: "Which documents are available?" went to
# document search and came back "do not answer that"). A question about what documents say ("Which documents mention the
# deposit?") stays a research question, as does one about documents someone asked for, sent or signed ("What documents has the
# Authority asked Meridian for?", 2026-11-05 end-to-end run).
_DOCUMENTS = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:"
    r"(?:which|what)\s+(?:documents?|files?)\s+(?:are|is|do|does|have|has)\b"
    r"(?![^?]*\b(?:mention|say|said|state|refer|contain|discuss|cover|show|support|prove|deal|address|govern|include|provide|set|"
    r"name|quote|cite|describe|record|confirm|list|ask|request|want|need|seek|require|demand|send|sent|receiv|produc|disclos|"
    r"give|gave|sign|attach|exchang|serve|file)\w*)"
    r"|(?:list|show)\s+(?:me\s+)?(?:all\s+|the\s+|my\s+|our\s+)?(?:documents|files)\b(?!\s+(?:that|which|about|on|mentioning|saying|where))"
    r"|what(?:'s|\s+is)\s+(?:in|on)\s+(?:this|the)\s+(?:matter|file|case)(?:\s+file)?\s*\??\s*$"
    r"|how\s+many\s+documents\s+(?:are|do|does|have|has)\b(?![^?]*\b(?:mention|say|state|refer)\w*))", re.I)
_DRAFT = re.compile(r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:draft|write|prepare|compose)\b"
                    r"|\b(?:draft|write|prepare)\s+(?:a|an|the|me\s+a)\s+(?:\w+\s+){0,3}(?:letter|update|memo|email|reply|notice|agreement|contract)\b", re.I)
# Small talk, spoken or typed: every part of what was said is social ("Hello, can you hear me?", "Thanks, that's all for
# now."); a question in any part ("Hello, when does the term end?") goes to the documents. The kind picks the reply.
_SOCIAL = (
    ("closing", r"(?:that'?s|that\s+is|that'?ll\s+be|that\s+will\s+be)\s+(?:all|it|everything)(?:\s+for\s+(?:now|today))?"
                r"|(?:good\s*)?bye(?:\s+(?:now|then|for\s+now))?|see\s+you(?:\s+(?:later|soon|tomorrow))?"
                r"|(?:talk|speak|chat)\s+(?:to\s+you\s+|with\s+you\s+)?(?:later|soon|tomorrow|next\s+week|on\s+\w+day)"
                r"|catch\s+you\s+(?:later|soon|tomorrow)|until\s+next\s+time|good\s*night"
                r"|have\s+a\s+(?:good|great|nice|lovely)\s+(?:day|evening|weekend|one)"
                r"|nothing\s+else|(?:we'?re|i'?m|we\s+are|i\s+am)\s+done(?:\s+for\s+(?:now|today))?"
                r"|no,?\s+(?:thanks|thank\s+you)|no,?\s+that'?s\s+(?:all|it)"),
    # "What can you do for me on this matter?", "what can you help me with on this case?" (2026-11-03 evaluation)
    ("identity", r"(?:who\s+are\s+you|what\s+are\s+you|what\s+(?:can|do)\s+you\s+do|what\s+can\s+i\s+ask(?:\s+you)?"
                 r"|what\s+(?:can|could)\s+you\s+help\s+(?:me|us)\s+with|how\s+can\s+you\s+help(?:\s+(?:me|us))?"
                 r"|how\s+(?:does\s+this|do\s+you)\s+work|how\s+(?:do|should|can)\s+(?:i|we)\s+use\s+(?:you|this)|help(?:\s+me)?)"
                 r"(?:\s+(?:for|with)\s+(?:me|us))?(?:\s+(?:on|with|in|for)\s+(?:this|the|my|our)\s+(?:matter|case|file))?"),
    ("check", r"(?:can|do)\s+you\s+hear\s+me(?:\s+(?:ok(?:ay)?|now|clearly|well))?|are\s+you\s+(?:there|listening|ready|on)"
              r"|(?:are\s+you\s+)?still\s+(?:there|here|with\s+me|with\s+us)"
              r"|is\s+(?:this|it)\s+(?:working|on)|testing(?:\s+(?:one\s+two(?:\s+three)?|1\s*2(?:\s*3)?))?"),
    ("greeting", r"hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening|day)|how\s+are\s+you(?:\s+(?:doing|today))?"
                 r"|how'?s\s+it\s+going"),
    ("thanks", r"thanks?(?:\s+(?:you|a\s+lot|again))?(?:\s+(?:so|very)\s+much)?|thank\s+you(?:\s+(?:so|very)\s+much)?(?:\s+again)?"
               r"|thx|cheers|much\s+appreciated"
               r"|(?:that'?s|that\s+is|that\s+was|this\s+is|very|really|so|super)?\s*(?:(?:very|really|so|super|extremely)\s+)?(?:helpful|useful)"
               # praise in other words: "that was really clear and quick", "you've been a great help" (2026-11-04)
               r"|(?:that'?s|that\s+is|that\s+was|this\s+is|this\s+was|it\s+was|you'?re|you\s+are|you'?ve\s+been|you\s+have\s+been)\s+"
               r"(?:(?:very|really|so|super|extremely|just|incredibly)\s+)?(?:a\s+(?:great|big|real)\s+help"
               r"|(?:helpful|useful|clear|quick|fast|great|brilliant|excellent|perfect|spot\s+on|thorough|amazing|fantastic)"
               r"(?:\s+and\s+(?:(?:very|really|so)\s+)?(?:helpful|useful|clear|quick|fast|great|thorough|concise))?)"
               r"|(?:exactly|just)\s+what\s+(?:i|we)\s+(?:needed|wanted|was\s+after|were\s+after)|great\s+(?:work|job|stuff)|well\s+done"),
    ("ack", r"ok(?:ay)?|got\s+it|understood|alright|all\s+right|sure|fine|right|great|perfect|lovely|excellent|brilliant|noted"
            r"|(?:that'?s|that\s+is|that\s+was)\s+(?:(?:very|really|so|just)\s+)?(?:great|perfect|fine|clear|brilliant|excellent|lovely)"
            r"|ok(?:ay)?\s+(?:great|perfect|thanks|cool)|cool|nice"),
)
_SOCIAL_RE = [(k, re.compile(rf"^(?:(?:oh|well|um+|uh+|yes|yeah|so|and|ok(?:ay)?)\s+)?(?:{p})(?:\s+(?:there|again|echomind|echo\s+mind))?$", re.I))
              for k, p in _SOCIAL]
_SOCIAL_ORDER = ("closing", "identity", "check", "greeting", "thanks", "ack")


def social(text: str) -> Optional[str]:
    """The kind of small talk when every part of `text` is social, else None."""
    t = " ".join((text or "").split())
    if not t or len(t.split()) > 16:
        return None
    kinds = set()
    for part in re.split(r"\s*[,.!?;:\u2014\u2013]+\s*|\s+-+\s+", t):        # "thank you - that was clear"
        part = part.strip(" '\"").replace("\u2019", "'")
        if not part:
            continue
        kind = next((k for k, rx in _SOCIAL_RE if rx.match(part)), None)
        if kind is None:
            return None
        kinds.add(kind)
    return next((k for k in _SOCIAL_ORDER if k in kinds), None)
_CHRONOLOGY = re.compile(r"\b(?:chronology|chronological|timeline|time\s+line|sequence\s+of\s+events|order\s+of\s+events"
                         r"|what\s+happened\s+when|key\s+(?:dates|events)|dates\s+in\s+(?:date\s+)?order)\b", re.I)
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
# A document's own drafting, checked by rule (2026-11-05 end-to-end run: "Run a definitions check on the Supply Agreement" was
# answered by research from a few passages). What a term means stays a question for the documents ("How is Price defined?").
_DEFINITIONS = re.compile(
    r"\bdefinitions?\s+(?:check|review|problems?|issues?|errors?)\b|\bcheck\s+(?:the\s+|its\s+|their\s+)?(?:definitions|defined\s+terms)\b"
    r"|\bdrafting\s+(?:problems?|issues?|errors?|defects?|mistakes?)\b|\bundefined\s+terms?\b"
    r"|\bdefined\s+(?:twice|more\s+than\s+once)\b|\bdefined\s+but\s+(?:never|not)\s+used\b"
    r"|\b(?:problems?|issues?|errors?|inconsistenc\w*)\s+(?:with|in)\s+(?:the\s+|its\s+)?(?:defined\s+terms|definitions|numbering|cross[- ]references)\b"
    r"|\b(?:broken|wrong|dangling)\s+cross[- ]references?\b|\bnumbering\s+(?:gaps?|errors?|problems?)\b", re.I)
_DELIBERATE = re.compile(r"\b(?:challenge|devil'?s\s+advocate|counter-?arguments?|poke\s+holes|stress[- ]test|argue\s+against"
                         r"|what\s+would\s+(?:opposing|the\s+other\s+side'?s?)\s+counsel)\b", re.I)


def classify(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return "ask_fact"
    if social(t):
        return "smalltalk"
    if _CASES.search(t):
        return "cases_overview"
    if _OTHER_MATTERS.search(t):
        return "other_matters"
    if _DEFINITIONS.search(t):
        return "definitions"
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
    if _COMPARE.search(t) or _COMPARE_FACTS.search(t):
        return "compare"
    if _INSPECT.search(t):
        return "inspect_case"
    return "ask_fact"
