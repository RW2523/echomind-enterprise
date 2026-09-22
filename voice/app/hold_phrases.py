"""
Hold phrases for tool-routed voice turns.

When the LLM decides to call a retrieval tool (``search_knowledge_base``,
``search_transcripts`` or ``lookup_record``) the assistant may need to say
something while retrieval runs.  This module picks that sentence:

* keyed by the tool AND the persona tone (a solicitor "reviews the file",
  a bank officer "confirms the terms", a support agent "looks that up");
* topic-aware: the model's own short tool argument ("refund policy
  cancellations after 14 days", "customer ID 48213", "Ms Patel") is boiled
  down to a spoken noun phrase and slotted in;
* anti-repeat: the caller passes the template ids it used recently and never
  hears the same sentence twice in a row.

The caller only speaks the result if retrieval has not answered within a few
hundred milliseconds, so everything here is pure, synchronous and stdlib-only
(sub-millisecond, no app imports).

Public API
----------
``TONES``, ``TOOLS``, ``HOLD_PHRASE_MAX_WORDS``
``tone_key(persona)``                       -> tone
``topic_from_query(query, max_words=3)``    -> spoken noun phrase or None
``pick_hold_phrase(tool, tone, topic, recent, rng=None)`` -> (text, template_id)
``pick_ack(tone, recent, rng=None)``        -> (text, template_id)

``recent`` is a sequence of template ids, oldest first, newest last.
"""
from __future__ import annotations

import random
import re
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TONES",
    "TOOLS",
    "HOLD_PHRASE_MAX_WORDS",
    "TOPIC_MAX_CHARS",
    "POOLS",
    "ACKS",
    "tone_key",
    "topic_from_query",
    "pick_hold_phrase",
    "pick_ack",
]

TONES = ("legal", "bank", "health", "meetings", "retail", "customer_care", "neutral")
TOOLS = ("search_knowledge_base", "search_transcripts", "lookup_record")

#: Hard cap on the rendered hold phrase (words). Templates that would exceed it
#: once the topic is slotted in are skipped for that call.
HOLD_PHRASE_MAX_WORDS = 9

#: ``topic_from_query`` returns None rather than a phrase longer than this.
TOPIC_MAX_CHARS = 40

_RNG = random.Random()


# ---------------------------------------------------------------------------
# tone_key
# ---------------------------------------------------------------------------

# Ordered: domain tones first, the generic customer_care last so that
# "Banking Customer Support" lands on bank and "Health Care" on health.
# Matching is by word prefix on the lower-cased label, so "Lawyer", "Legal
# Advisor", "law" and "legal" all hit "legal"; "Financial Advisor", "Banking
# Advisor", "bank" and "banking" all hit "bank".
_TONE_STEMS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("legal", ("legal", "law", "solicitor", "attorney", "counsel", "paralegal", "litigat")),
    ("bank", ("bank", "financ", "fiscal", "treasur", "loan", "mortgage", "credit", "wealth")),
    ("health", ("health", "clinic", "medic", "nurse", "doctor", "patient", "hospital", "pharma")),
    ("meetings", ("meeting", "facilitat", "boardroom")),
    ("retail", ("retail", "shop", "store", "sales", "merchant", "commerce", "ecommerce")),
    ("customer_care", ("customer", "support", "care", "helpdesk", "servicedesk")),
)


def tone_key(persona: Optional[str]) -> str:
    """Map any persona label or vertical pack id to one of ``TONES``.

    Case-insensitive and substring-tolerant.  Anything unrecognised
    (Teacher / Professor, Funny & Calming Assistant, AI Expert & Manager,
    General Assistant, EchoMind Guide, None, "") is ``"neutral"``.
    """
    if not persona:
        return "neutral"
    label = str(persona).strip().lower()
    if not label:
        return "neutral"
    if label in TONES:
        return label
    tokens = re.findall(r"[a-z]+", label)
    for tone, stems in _TONE_STEMS:
        if any(tok.startswith(stem) for tok in tokens for stem in stems):
            return tone
    return "neutral"


# ---------------------------------------------------------------------------
# topic_from_query
# ---------------------------------------------------------------------------

# Leading question / verb / filler words removed from the front of the query.
# Never applied to identifiers (tokens with a digit) or acronyms ("US", "IT").
_LEAD_STRIP = frozenset(
    """
    what whats what's who whos who's where wheres where's when whens when's why whys why's
    how hows how's which
    does do did done is are was were be been being am
    can could would should will shall may might must
    have has had having
    i i'm i'd i'll we we're you you're it it's there there's here
    tell me about the a an our my your this that these those
    please kindly just also
    find look up lookup check search get show pull fetch retrieve query see
    regarding on for in of to at from with into re
    any some info information details detail question anything something
    know need needs want wants wanted like help
    say says said mean means explain describe define give list summarize summarise
    remind recall remember hey hi hello echomind echo
    """.split()
)

# Clause starters: everything from here on is cut, always (after >= 1 content word).
_HARD_STOPS = frozenset(
    """
    when whenever if that which who whom whose where because so but while unless than
    whether how why what does do did is are was were be can could should would has have had
    """.split()
)

# Prepositions / coordinators: cut here only when the phrase is longer than
# ``max_words``; a short phrase such as "termination for convenience" or
# "terms and conditions" is kept whole.  Deliberately excludes "of"
# ("terms of service", "statute of limitations").
_SOFT_STOPS = frozenset(
    """
    after before for in on about with regarding at from to by under over during into onto
    via versus vs and or per within without against between across through since until till
    toward towards concerning re plus amid among around near beyond past below above
    """.split()
)

# Dropped when they appear mid-phrase ("what the policy says on refunds" ->
# "policy on refunds").
_MID_DROP = frozenset("the a an please say says said".split())

_TRAILING_DROP = _SOFT_STOPS | _HARD_STOPS | _MID_DROP | frozenset("me us it".split())

_DETERMINERS = frozenset(
    "the a an this that these those his her their its our my your some any each every all no".split()
)

_HONORIFICS = frozenset(
    "mr mrs ms mx dr prof sr jr st rev hon capt sgt lt col gen".split()
)

_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_EDGE_PUNCT = "\"'()[]{}<>,;:!?.…-–—/\\*#&^~`|"


class _Tok:
    __slots__ = ("text", "lower", "is_id", "is_hon", "proper")

    def __init__(self, text: str, is_id: bool = False, is_hon: bool = False, proper: bool = False):
        self.text = text
        self.lower = text.lower()
        self.is_id = is_id
        self.is_hon = is_hon
        self.proper = proper

    @property
    def fixed(self) -> bool:
        """Identifiers, honorifics and acronyms are never treated as stop words."""
        return self.is_id or self.is_hon or _is_acronym(self.text)


def _is_acronym(text: str) -> bool:
    return 2 <= len(text) <= 5 and text.isalpha() and text.isupper()


def _tokenize(raw: str) -> List[_Tok]:
    toks: List[_Tok] = []
    for piece in raw.split():
        core = piece.strip("\"'()[]{}<>")
        if not core:
            continue
        if any(ch.isdigit() for ch in core):
            # Identifier: keep verbatim, only trim sentence punctuation.
            core = core.rstrip("?!.,;:").lstrip(",;:")
            if core:
                toks.append(_Tok(core, is_id=True, proper=True))
            continue
        bare = core.rstrip(".")
        if bare and bare[:1].isupper() and bare.lower() in _HONORIFICS:
            toks.append(_Tok(bare + ".", is_hon=True, proper=True))
            continue
        core = re.sub(r"[^\w'\-]", "", core).replace("_", "").strip("'-")
        if core:
            toks.append(_Tok(core))
    return toks


def _apply_case(toks: List[_Tok]) -> None:
    """Decide the spoken casing of each token in place.

    identifiers / honorifics -> verbatim; short ALL-CAPS -> acronym, kept;
    long ALL-CAPS -> shouting, lowered; MixedCase -> kept; Capitalised -> kept
    as a proper noun unless it is the sentence-initial word of a sentence-cased
    query; everything else -> lowercase.
    """
    n = len(toks)
    for k, t in enumerate(toks):
        if t.is_id or t.is_hon:
            continue
        text = t.text
        if len(text) == 1:
            t.text, t.lower, t.proper = text.lower(), text.lower(), False
            continue
        if text.isupper():
            if len(text) <= 5:
                t.proper = True
            else:
                t.text, t.lower, t.proper = text.lower(), text.lower(), False
            continue
        if any(ch.isupper() for ch in text[1:]):  # DoD, McKinsey, iPhone
            t.proper = True
            continue
        if text[0].isupper():
            nxt = toks[k + 1] if k + 1 < n else None
            nxt_cap = bool(nxt) and not nxt.is_id and (nxt.is_hon or (nxt.text[:1].isupper() and not nxt.text.isupper()))
            if k == 0 and not nxt_cap:
                t.text, t.lower, t.proper = text.lower(), text.lower(), False
            else:
                t.proper = True
            continue
        t.proper = False


def _needs_article(toks: List[_Tok]) -> bool:
    first = toks[0]
    if first.is_id or first.is_hon:
        return False
    if any(t.is_id for t in toks):
        return False
    if first.lower in _DETERMINERS:
        return False
    if all(t.proper for t in toks):
        return False
    return True


def topic_from_query(query: Optional[str], max_words: int = 3) -> Optional[str]:
    """Turn the model's tool argument into a short spoken noun phrase.

    Examples::

        "refund policy cancellations after 14 days" -> "the refund policy cancellations"
        "What is the DoD FMR volume 7A?"            -> "the DoD FMR volume"
        "customer ID 48213"                         -> "customer ID 48213"
        "Ms Patel" / "Ms. Patel"                    -> "Ms. Patel"
        "termination for convenience"               -> "the termination for convenience"
        "check" / "" / None                         -> None

    Returns None when no content word survives or the phrase would be longer
    than ``TOPIC_MAX_CHARS``.
    """
    if not query:
        return None
    try:
        max_words = max(1, int(max_words))
    except (TypeError, ValueError):
        max_words = 3
    raw = str(query).translate(_QUOTE_MAP).strip()
    toks = _tokenize(raw)
    if not toks:
        return None
    _apply_case(toks)

    # 1. Strip leading question / verb / filler words.
    i = 0
    while i < len(toks) and not toks[i].fixed and toks[i].lower in _LEAD_STRIP:
        i += 1
    toks = toks[i:]
    if not toks:
        return None

    # 2. Cut at the first clause starter after >= 1 content word.
    for j in range(1, len(toks)):
        if not toks[j].fixed and toks[j].lower in _HARD_STOPS:
            toks = toks[:j]
            break

    # 3. Drop mid-phrase articles and "says".
    toks = [toks[0]] + [t for t in toks[1:] if t.fixed or t.lower not in _MID_DROP]

    # 4. Too long: cut before the last preposition/conjunction that still fits,
    #    otherwise keep the first max_words.
    if len(toks) > max_words:
        best = 0
        for j in range(1, min(len(toks), max_words + 1)):
            if not toks[j].fixed and toks[j].lower in _SOFT_STOPS:
                best = j
        toks = toks[:best] if best else toks[:max_words]

    # 5. Never end on a preposition / article.
    while toks and not toks[-1].fixed and toks[-1].lower in _TRAILING_DROP:
        toks.pop()
    if not toks:
        return None

    phrase = " ".join(t.text for t in toks)
    if _needs_article(toks):
        phrase = "the " + phrase
    if len(phrase) > TOPIC_MAX_CHARS:
        return None
    return phrase


# ---------------------------------------------------------------------------
# Hold-phrase pools, keyed by (tool, tone)
# ---------------------------------------------------------------------------
# Rules for every template: <= 9 words once rendered with a 3-word topic (so at
# most 6 words around the {topic} slot), ends with a period, no exclamation
# marks, no "Great question", no emoji, never starts with {topic} (the first
# letter of the rendered text is capitalised and the topic keeps its case).
# Each pool mixes commit / restate / acknowledge structures and holds at least
# four slot-less templates so anti-repeat works even with no topic.

_KB_POOLS: Dict[str, Tuple[str, ...]] = {
    "neutral": (
        "Let me pull up {topic}.",
        "Checking what we have on {topic}.",
        "Let me check the documents on {topic}.",
        "Looking into {topic} now.",
        "Seeing what the sources say on {topic}.",
        "Let me look that up.",
        "Give me a second.",
        "Checking the documents now.",
        "Let me see what the sources say.",
    ),
    "legal": (
        "Let me check the clause on {topic}.",
        "Reviewing the file on {topic}.",
        "Checking the agreement on {topic}.",
        "Pulling the relevant provisions on {topic}.",
        "Let me confirm the wording on {topic}.",
        "Let me check the documents.",
        "Give me a moment to review that.",
        "Let me review the file.",
        "Checking the relevant provisions now.",
    ),
    "bank": (
        "Let me confirm the terms on {topic}.",
        "Checking the current details on {topic}.",
        "Let me pull up {topic}.",
        "Checking the conditions on {topic}.",
        "Let me verify {topic} for you.",
        "Let me confirm that for you.",
        "Checking the current terms now.",
        "Give me a second to verify that.",
        "Let me pull up the details.",
    ),
    "health": (
        "Let me check the guidance on {topic}.",
        "Checking the protocol on {topic}.",
        "Seeing what the guidelines say on {topic}.",
        "Pulling the reference on {topic}.",
        "Let me confirm {topic} against the guidance.",
        "Let me check the guidance.",
        "Give me a moment to check that.",
        "Checking the reference material now.",
        "Let me confirm that against the guidance.",
    ),
    "meetings": (
        "Let me find where {topic} came up.",
        "Checking the notes on {topic}.",
        "Seeing what was decided on {topic}.",
        "Looking for {topic} in the meeting notes.",
        "Pulling up the notes on {topic}.",
        "Let me check the meeting notes.",
        "Give me a second to find that.",
        "Checking the notes now.",
        "Let me see what was decided.",
    ),
    "retail": (
        "Let me check {topic} for you.",
        "Checking {topic} now.",
        "Pulling up the details on {topic}.",
        "Looking up {topic} right now.",
        "Seeing what we have on {topic}.",
        "Let me check that for you.",
        "Give me a second.",
        "Checking that now.",
        "Let me pull up the details.",
    ),
    "customer_care": (
        "Let me look that up for you.",
        "Checking {topic} now.",
        "Let me check the policy on {topic}.",
        "Seeing what applies to {topic}.",
        "Pulling up the details on {topic}.",
        "Let me confirm {topic} for you.",
        "Let me find that for you.",
        "Give me a second to check.",
        "Looking into that now.",
        "Checking the policy on that now.",
    ),
}

_TRANSCRIPT_POOLS: Dict[str, Tuple[str, ...]] = {
    "neutral": (
        "Looking back through the transcript for {topic}.",
        "Checking what was said about {topic}.",
        "Checking the transcript for {topic}.",
        "Let me find where {topic} was mentioned.",
        "Searching the conversation for {topic}.",
        "Let me check the transcript.",
        "Give me a second to find that.",
        "Looking back through the transcript.",
        "Let me see what was said.",
    ),
    "legal": (
        "Let me check the record on {topic}.",
        "Reviewing the transcript for {topic}.",
        "Finding what was stated on {topic}.",
        "Checking what was said about {topic}.",
        "Let me search the transcript for {topic}.",
        "Let me locate that in the transcript.",
        "Reviewing the record now.",
        "Give me a moment to check the record.",
        "Let me check the transcript.",
    ),
    "bank": (
        "Checking what was discussed on {topic}.",
        "Let me find where {topic} came up.",
        "Looking back through the conversation for {topic}.",
        "Confirming what was said about {topic}.",
        "Let me check the transcript for {topic}.",
        "Let me check the conversation record.",
        "Let me find that in the transcript.",
        "Checking the transcript now.",
        "Give me a second.",
    ),
    "health": (
        "Checking what was noted on {topic}.",
        "Looking through the transcript for {topic}.",
        "Let me find where {topic} was discussed.",
        "Let me check the transcript for {topic}.",
        "Seeing what was said about {topic}.",
        "Let me check the transcript.",
        "Let me find that in the notes.",
        "Give me a moment.",
        "Searching the session transcript now.",
    ),
    "meetings": (
        "Let me find where {topic} came up.",
        "Checking the discussion on {topic}.",
        "Let me see who raised {topic}.",
        "Looking through the meeting for {topic}.",
        "Finding what was agreed on {topic}.",
        "Let me check the meeting transcript.",
        "Give me a second to find that.",
        "Searching the discussion now.",
        "Let me see what was agreed.",
    ),
    "retail": (
        "Checking what was said about {topic}.",
        "Checking the conversation for {topic}.",
        "Let me find where {topic} came up.",
        "Looking back through the chat for {topic}.",
        "Let me search the conversation for {topic}.",
        "Let me check the conversation.",
        "Let me find that for you.",
        "Give me a second.",
        "Checking the transcript now.",
    ),
    "customer_care": (
        "Checking what was said about {topic}.",
        "Looking back through the conversation for {topic}.",
        "Let me find where {topic} came up.",
        "Checking what we covered on {topic}.",
        "Let me search the conversation for {topic}.",
        "Let me look back through the conversation.",
        "Let me find that for you.",
        "Give me a second to check.",
        "Checking the conversation now.",
    ),
}

_RECORD_POOLS: Dict[str, Tuple[str, ...]] = {
    "neutral": (
        "Pulling up {topic} now.",
        "Let me bring up {topic}.",
        "Opening the record for {topic}.",
        "Pulling up the record on {topic}.",
        "Bringing up the details on {topic}.",
        "Let me pull up that record.",
        "Opening the record now.",
        "Give me a second.",
        "Bringing up the details now.",
    ),
    "legal": (
        "Pulling up the file on {topic}.",
        "Let me bring up {topic}.",
        "Opening the record for {topic}.",
        "Retrieving the case file on {topic}.",
        "Checking the record on {topic}.",
        "Let me bring up the file.",
        "Opening the case record now.",
        "Give me a moment to pull that.",
        "Retrieving the case file now.",
    ),
    "bank": (
        "Pulling up {topic} now.",
        "Let me bring up {topic}.",
        "Opening the record for {topic}.",
        "Retrieving the details on {topic}.",
        "Let me verify {topic} on the account.",
        "Let me pull up that account.",
        "Bringing up the details now.",
        "Give me a second to pull that up.",
        "Retrieving the account details now.",
    ),
    "health": (
        "Pulling up {topic} now.",
        "Bringing up the record on {topic}.",
        "Opening the record for {topic}.",
        "Retrieving the chart for {topic}.",
        "Let me bring up {topic}.",
        "Let me pull up that record.",
        "Opening the record now.",
        "Give me a moment.",
        "Retrieving the chart now.",
    ),
    "meetings": (
        "Pulling up {topic} now.",
        "Bringing up the notes on {topic}.",
        "Opening the record for {topic}.",
        "Let me bring up {topic}.",
        "Retrieving the action items on {topic}.",
        "Let me pull up that record.",
        "Bringing up the notes now.",
        "Give me a second.",
        "Retrieving the action items now.",
    ),
    "retail": (
        "Pulling up {topic} now.",
        "Let me bring up {topic}.",
        "Opening the record for {topic}.",
        "Pulling up the order details on {topic}.",
        "Bringing up {topic} for you.",
        "Let me pull that up for you.",
        "Opening that record now.",
        "Give me a second.",
        "Bringing up the order details now.",
    ),
    "customer_care": (
        "Pulling up {topic} now.",
        "Let me bring up {topic}.",
        "Opening the record for {topic}.",
        "Pulling up the details on {topic}.",
        "Bringing up {topic} for you.",
        "Let me pull that up for you.",
        "Opening the record now.",
        "Give me a second to pull that up.",
        "Bringing up the details now.",
    ),
}

#: Read-only view: ``POOLS[(tool, tone)]`` -> tuple of templates.
POOLS = MappingProxyType(
    {
        **{("search_knowledge_base", tone): tpl for tone, tpl in _KB_POOLS.items()},
        **{("search_transcripts", tone): tpl for tone, tpl in _TRANSCRIPT_POOLS.items()},
        **{("lookup_record", tone): tpl for tone, tpl in _RECORD_POOLS.items()},
    }
)

#: Short acknowledgements for direct (non-tool) answers that are slow to start.
ACKS: Tuple[str, ...] = (
    "Sure.",
    "Okay.",
    "Right.",
    "Let me think.",
    "Well.",
    "So.",
    "Alright.",
    "Let me see.",
)
_ACK_INFORMAL = frozenset({"Well.", "So."})
_FORMAL_TONES = frozenset({"legal", "bank", "health"})


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _tool_key(tool: Optional[str]) -> str:
    t = (tool or "").strip().lower()
    if t in TOOLS:
        return t
    if "transcript" in t or "conversation" in t or "meeting" in t:
        return "search_transcripts"
    if "record" in t or "lookup" in t or "entity" in t or "customer" in t or "case" in t:
        return "lookup_record"
    return "search_knowledge_base"


def _tone_arg(tone: Optional[str]) -> str:
    return tone if tone in TONES else tone_key(tone)


def _clean_topic(topic: Optional[str]) -> Optional[str]:
    if topic is None:
        return None
    s = " ".join(str(topic).split()).rstrip(".").strip()
    if not s or s.lower() in ("none", "null"):
        return None
    if len(s) > 2 * TOPIC_MAX_CHARS:
        return None
    return s


def _render_pool(tool: str, tone: str, topic: Optional[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for n, tmpl in enumerate(POOLS[(tool, tone)], 1):
        if "{topic}" in tmpl:
            if topic is None:
                continue
            text = tmpl.replace("{topic}", topic)
        else:
            text = tmpl
        text = text[:1].upper() + text[1:]
        if len(text.split()) > HOLD_PHRASE_MAX_WORDS:
            continue
        out.append((f"{tool}/{tone}/{n}", text))
    return out


def _choose(
    tiers: Sequence[Sequence[Tuple[str, str]]],
    recent: Sequence[str],
    rng: random.Random,
) -> Tuple[str, str]:
    """Pick from the first tier that still has something not in ``recent``.

    Falls back to "anything but the very last id", then to anything at all.
    """
    recent_ids = [str(r) for r in (recent or ())]
    recent_set = set(recent_ids)
    last = recent_ids[-1] if recent_ids else None
    for tier in tiers:
        fresh = [c for c in tier if c[0] not in recent_set]
        if fresh:
            tid, text = rng.choice(fresh)
            return text, tid
    for tier in tiers:
        not_last = [c for c in tier if c[0] != last]
        if not_last:
            tid, text = rng.choice(not_last)
            return text, tid
    for tier in tiers:
        if tier:
            tid, text = rng.choice(list(tier))
            return text, tid
    raise ValueError("no candidates")


def pick_hold_phrase(
    tool: str,
    tone: str,
    topic: Optional[str],
    recent: Sequence[str],
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """Return ``(text, template_id)`` for a tool call that is taking a while.

    ``tool``   one of ``TOOLS`` (unknown names fall back to search_knowledge_base).
    ``tone``   one of ``TONES`` or any persona label (mapped with ``tone_key``).
    ``topic``  the output of ``topic_from_query`` or None; with None only
               slot-less templates are eligible.
    ``recent`` template ids used recently (oldest first); none of them is
               returned unless that leaves nothing, in which case the
               tool's neutral pool is tried, then anything but the last id.
    ``template_id`` looks like ``"search_knowledge_base/bank/3"``.
    """
    rng = rng or _RNG
    tool_k = _tool_key(tool)
    tone_k = _tone_arg(tone)
    topic_s = _clean_topic(topic)

    primary = _render_pool(tool_k, tone_k, topic_s)
    tiers: List[List[Tuple[str, str]]] = [primary]
    if tone_k != "neutral":
        tiers.append(_render_pool(tool_k, "neutral", topic_s))
    return _choose(tiers, recent, rng)


def pick_ack(
    tone: str,
    recent: Sequence[str],
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """Return a 1-3 word acknowledgement ``(text, "ack/<n>")`` for a direct
    answer that is slow to start.  Formal tones (legal, bank, health) skip the
    chattier "Well." / "So."."""
    rng = rng or _RNG
    tone_k = _tone_arg(tone)
    formal = tone_k in _FORMAL_TONES
    pool = [
        (f"ack/{n}", text)
        for n, text in enumerate(ACKS, 1)
        if not (formal and text in _ACK_INFORMAL)
    ]
    return _choose([pool], recent, rng)
