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
- `smalltalk`     "Hi", "Thanks", and the asides of `aside()`: which model, today's date, what was asked before, and requests
                  outside the matter (code, jokes, the weather) → a short reply of their own (no search, no model)
- `ask_fact`      everything else
"""
from __future__ import annotations

import re
from typing import Optional

INTENTS = ("ask_fact", "compare", "inspect_case", "chronology", "cases_overview", "other_matters", "documents", "draft", "deliberate",
           "smalltalk", "approvals", "live", "amounts", "parties", "tasks", "definitions", "case_status")

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
# "Draft" alone asks for a draft; "write", "prepare" and "compose" only with a document to write (2026-11-12, evaluation fix
# list item 5: "Write a Python script…" opened the template-draft card, "Prepare a summary of the payment terms" too).
_DRAFTED = (r"(?:letter|update|memo(?:randum)?|e-?mail|reply|response|notice|agreement|contract|clause|note|minutes|demand|claim|"
            r"statement|submission|report|brief|proposal|offer|amendment|side\s+letter|nda|engagement\s+letter|draft)s?")
_DRAFT = re.compile(r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?draft\b"
                    rf"|^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:write|prepare|compose)(?!\s+(?:down|out)\b)\s+(?:up\s+)?(?:\w+\s+){{0,5}}?{_DRAFTED}\b"
                    rf"|\b(?:draft|write|prepare)\s+(?:a|an|the|me\s+a)\s+(?:\w+\s+){{0,3}}{_DRAFTED}\b", re.I)
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
                 r"|how'?s\s+it\s+going|hola|buenos\s+d[ií]as|bonjour|salut|hallo|guten\s+tag|ciao|buongiorno|ol[áa]|hej"),
    ("thanks", r"thanks?(?:\s+(?:you|a\s+lot|again))?(?:\s+(?:so|very)\s+much)?|thank\s+you(?:\s+(?:so|very)\s+much)?(?:\s+again)?"
               r"|thx|cheers|much\s+appreciated|ta|gracias|merci(?:\s+beaucoup)?|danke(?:\s+sch(?:ö|oe)n)?|grazie|obrigad[oa]|dank\s+je(?:\s+wel)?|tack"
               # "Thanks, that helps!" was searched and then withdrawn (2026-11-11 evaluation)
               r"|(?:that|this|it)\s+(?:really\s+|definitely\s+|certainly\s+)?(?:helps|helped)(?:\s+(?:a\s+lot|loads|a\s+great\s+deal))?"
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


# Asides (2026-11-12, evaluation fix list item 10): questions about the assistant or the conversation, and requests outside a
# matter's documents, each got the decline sentence, and "What is today's date?" a date from a document.
_ASIDES = (
    ("model", re.compile(r"^\s*(?:so\s+|and\s+)?(?:which|what)\s+(?:ai\s+|language\s+)?(?:model|llm|ai|engine)\s+(?:are\s+you|is\s+this|do\s+you\s+use"
                         r"|powers\s+(?:you|this)|runs\s+(?:you|this)|are\s+you\s+(?:running|using|based\s+on))\b"
                         r"|^\s*are\s+you\s+(?:chat\s*gpt|gpt|claude|gemini|llama|qwen|an?\s+(?:ai|llm|bot|robot|language\s+model)|a\s+(?:real\s+)?(?:person|human|lawyer))\b"
                         r"|^\s*who\s+(?:made|built|created|trained)\s+you\b", re.I)),
    ("today", re.compile(r"^\s*(?:so\s+)?(?:what(?:'s|\s+is)\s+(?:today'?s\s+date|the\s+date(?:\s+today)?|today)|what\s+day\s+is\s+(?:it(?:\s+today)?|today)"
                         r"|what\s+(?:date|time)\s+is\s+it(?:\s+(?:today|now))?)\s*\??\s*$", re.I)),
    ("memory", re.compile(r"\bwhat\s+(?:did|have)\s+(?:i|we)\s+(?:first\s+|just\s+|last\s+)?(?:ask|asked|say|said|discuss|discussed|cover|covered)"
                          r"(?:\s+(?:you|about))?(?:\s+(?:first|at\s+the\s+start|earlier|before|so\s+far|last))?\s*\??\s*$"
                          r"|\b(?:my|our)\s+(?:first|last|previous|earlier)\s+question\b|\bwhat\s+have\s+we\s+(?:discussed|covered|talked\s+about)\b"
                          r"|\b(?:summari[sz]e|recap)\s+(?:this|our|the)\s+(?:chat|conversation|discussion)\b", re.I)),
    ("offtopic", re.compile(r"\b(?:write|give\s+me|make|create|generate|code)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(?:\w+\s+){0,2}?"
                            r"(?:python|javascript|java|sql|bash|excel\s+macro|vba|c\+\+|rust|html|css)?\s*(?:script|code|program|function|macro|"
                            r"poem|haiku|song|joke|story|limerick|essay|tweet|recipe)\b"
                            r"|\btell\s+(?:me\s+)?a\s+joke\b|\b(?:what(?:'s|\s+is)\s+the\s+)?weather\b|\b(?:football|soccer|cricket|match)\s+(?:score|result)s?\b"
                            r"|\bstock\s+price\b|\bwho\s+won\s+the\b|\b(?:latest|today'?s)\s+news\b", re.I)),
)


def aside(text: str) -> Optional[str]:
    """"model", "today", "memory" or "offtopic" when `text` is one of those asides, else None. Never a question about the
    documents: one that names a clause, document, party role or figure is left to research."""
    t = " ".join((text or "").split())
    if not t or len(t.split()) > 24 or _ABOUT_DOCUMENTS.search(t) and not re.search(r"\b(?:script|code|program|poem|joke)\b", t, re.I):
        return None
    return next((k for k, rx in _ASIDES if rx.search(t)), None)


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
# Where the case stands, from its updates log, stage and plan (case management, 2026-11-07): "What's the latest on this case?",
# "Where are we with M-101?", "Status update", "What stage is the matter at?". A question about the latest version or date of
# something in the documents stays a research question.
_CASE_STATUS = re.compile(
    r"\bwhat(?:'s|\s+is)\s+the\s+latest\s+(?:on|with|in|for)\b|\blatest\s+(?:news|updates?|developments?)\b"
    r"|\bwhat(?:'s|\s+is|\s+has\s+been)\s+happening\b|\bwhat\s+happened\s+(?:recently|lately|last\s+week|this\s+week)\b"
    r"|\b(?:case|matter)\s+(?:status|updates?|progress|plan)\b|\bstatus\s+of\s+(?:the|this|my|our)\s+(?:case|matter)\b"
    r"|\bstatus\s+update\b|\bwhere\s+are\s+we\s+(?:on|with|in|at)\b|\bwhere\s+(?:is|does)\s+(?:the|this)\s+(?:case|matter)\s+(?:stand|stands|at)\b"
    r"|\bwhat\s+stage\b|\b(?:show|list|any)\s+(?:me\s+)?(?:the\s+)?(?:recent\s+|latest\s+)?updates\b"
    r"|\bprogress\s+(?:on|of|in)\s+(?:the|this)\s+(?:case|matter)\b"
    r"|\bnext\s+steps\s+(?:on|for|in)\s+(?:the|this)\s+(?:case|matter)\b|\bwhat(?:'s|\s+is)\s+the\s+plan\b", re.I)
# The register's list, not what a document says: "What are the open obligations?" is a list; "What deadlines does the lease
# set?" is a research question.
_TASKS = re.compile(r"\b(?:obligations|deadlines|tasks|to-?dos|to-do\s+list|checklist)\b", re.I)
# The register, asked for as such (2026-11-12, evaluation fix list item 5: "What are the payment obligations of the Customer?"
# returned the task list in 0.1 s and never read the documents): a register word, or the bare request for the list. A kind of
# obligation ("payment obligations") or a party or document named ("of the Customer", "Harrowgate's") is a research question.
_REGISTER = re.compile(r"\b(?:open|outstanding|upcoming|pending|overdue|due|my|our|register|next|coming\s+up|this\s+(?:week|month)"
                       r"|today|tomorrow|unconfirmed|confirmed|assigned)\b", re.I)
_TASKS_LIST = re.compile(r"^\s*(?:please\s+)?(?:show|list|give\s+me|what\s+are|which\s+are|are\s+there\s+any|any)\s+(?:me\s+)?(?:all\s+)?"
                         r"(?:the\s+)?(?:obligations|deadlines|tasks|to-?dos)\s*\??\s*$", re.I)
_KIND_OF = re.compile(r"\b(?:payment|confidentiality|insurance|reporting|notice|indemnity|indemnification|repair|maintenance|delivery|"
                      r"data(?:\s+protection)?|audit|exclusivity|non-?compete|warranty|performance|disclosure|tax|security|filing|"
                      r"regulatory|contractual|legal|statutory|financial|post-termination|termination)\s+(?:obligations?|deadlines?|duties)\b", re.I)
_NAMED = re.compile(r"\b(?:of|for|on|owed\s+by|by|to|with)\s+(?:the\s+)?(?!(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
                    r"January|February|March|April|May|June|July|August|September|October|November|December|Today|Tomorrow|This|Next|"
                    r"Me|Us|I)\b)(?![A-Z]{1,4}-\d)[A-Z][\w&'-]+|\b[A-Z][\w&-]+'s\b")
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
    if social(t) or aside(t):
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
    if _CASE_STATUS.search(t):
        return "case_status"
    if _TASKS.search(t) and (_TASKS_LIST.search(t) or _REGISTER.search(t)) and not _ABOUT_DOCUMENTS.search(t) \
            and not _KIND_OF.search(t) and not _NAMED.search(t):
        return "tasks"
    if _COMPARE.search(t) or _COMPARE_FACTS.search(t):
        return "compare"
    if _INSPECT.search(t):
        return "inspect_case"
    return "ask_fact"
