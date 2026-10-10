"""Silent Assist: alerts on what is said while a meeting is captured (S2.5, LIVE-02; D-015, D-041).

The engine considers each stored transcript segment against its meeting's pack (packs.py):
- **Signed terms and authority.**
  - Amounts, periods, percentages and dates in the segment (compare/facts.py) are paired with
    their closest counterpart in the matter's documents: those the person recording can see, in
    their signed versions.
  - If a figure is being agreed, offered or approved, and a passage of the minutes or another
    authority sets a limit on it ("not be reduced below USD 400,000") that the figure is outside,
    the segment differs from that authority.
  - If the counterpart is a plain term and the figure differs, the model checks that the two are
    about the same thing, and the alert says the segment differs from it.
  - A statement that names the agreement and agrees with it is marked as matching.
- **Commitment** and **disclosure** (only with a counterparty on the call), **action**, **quorum**
  and **conflict**: narrow phrase rules, scoped to the speaker's roster role.
  - A conflict compares a vote with an interest declared earlier in the meeting, or recorded in the
    matter's minutes.

Every alert quotes the exact characters said. Where it is checked against a document, it also
quotes the exact characters of that document. It carries its confidence and the threshold in
force. Nothing is sent to the model but the two passages, fenced, with text addressed to an AI
withheld (T08).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ...core.config import settings
from ..compare import facts as factsmod
from ..compare import scan as scanmod
from ..research import evidence
from ..research import llm as llm_mod
from . import packs

logger = logging.getLogger(__name__)

_COMMITMENT = re.compile(
    r"\b(?:we|i)(?:'ll|\s+will|\s+can|\s+could|\s+are\s+(?:happy|willing|prepared)\s+to|\s+would\s+be\s+(?:happy|willing)\s+to|"
    r"\s+agree\s+to|\s+undertake\s+to|\s+commit\s+to)\s+(?:\w+\s+){0,3}?(?:waive|accept|pay|reduce|extend|agree|offer|give|provide|drop|"
    r"withdraw|concede|forgo|increase|lower|cap|guarantee|absorb|cover)\b", re.I)
_DISCLOSURE = re.compile(
    r"\b(?:our|my)\s+(?:legal\s+)?advice\b|\bwe\s+(?:have\s+)?advised\b|\bprivileged\b|\blegal\s+advice\b|\boff\s+the\s+record\b|"
    r"\bbetween\s+(?:us|ourselves)\b|\bbottom\s+line\b|\b(?:settlement|negotiating)\s+(?:authority|mandate|limit)\b|"
    r"\bour\s+client\s+(?:is|would\s+be)\s+(?:willing|prepared)\s+to\b|\bshould\s+not\s+be\s+relied\s+on\b", re.I)
_ACTION = re.compile(r"\b(?:i|we)(?:'ll|\s+will|\s+shall)\s+(?:\w+\s+)?(?:send|circulate|follow\s+up|confirm|check|draft|prepare|review|"
                     r"call|email|share|update|revert|come\s+back)\b[^.?!]*", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12}
_QUORUM = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:of|out\s+of)\s+(?:the\s+)?"
                     r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+directors\b[^.]{0,40}\bpresent\b", re.I)
_DECLARE = re.compile(r"\b(?:i|he|she)\s+(?:declare[sd]?|ha(?:ve|s|d))\s+an\s+interest\s+in\s+(?:the\s+)?(?P<topic>[^,.;]{3,80})", re.I)
_DECLARED_IN_DOC = re.compile(r"(?P<who>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+declared\s+an\s+interest\s+in\s+(?:the\s+)?(?P<topic>[^,.;]{3,80})")
_VOTE = re.compile(r"\b(?:i\s+vote|i\s+am\s+voting|i(?:'m|\s+am)\s+in\s+favour|i\s+vote\s+in\s+favour|i\s+second|i\s+move|"
                   r"i\s+approve|vote\s+in\s+favour)\b", re.I)
_MIN_WORDS = re.compile(r"\b(?:below|less\s+than|at\s+least|not\s+less\s+than|minimum|no\s+lower\s+than)\b", re.I)
_MAX_WORDS = re.compile(r"\b(?:above|more\s+than|at\s+most|not\s+more\s+than|maximum|no\s+higher\s+than|not\s+exceed)\b", re.I)
_AUTHORITY_WORDS = re.compile(r"\b(?:authori[sz]|resolv|mandate|approv|board)", re.I)
_AGREEMENT_DOC = re.compile(r"\b(?:agreement|contract|lease|deed|order\s+form|amendment|addendum|side\s+letter|terms\s+and\s+conditions)\b",
                            re.I)
_PROPOSES = re.compile(r"\b(?:accept(?:s|ed|ing)?|agree(?:s|d|ing)?|offer(?:s|ed|ing)?|propos(?:e|es|ed|ing)|settle(?:s|d)?|"
                       r"approv(?:e|es|ed|ing)|resolv(?:e|es|ed)|commit(?:s|ted)?|concede(?:s|d)?)\b", re.I)
_NAMES_AGREEMENT = re.compile(r"\b(?:agreement|contract|signed|under\s+the|clause|lease|msa)\b", re.I)
_STOP = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "project", "item", "with"}


@dataclass
class Alert:
    rule: str
    verdict: str                                 # differs | matches | needs_review
    severity: str
    title: str
    said: dict                                   # {start, end, quote} in the segment's text
    confidence: float
    method: str = "rule"
    source: Optional[dict] = None                # a document passage (exact characters) or an earlier segment
    explanation: str = ""
    document_ids: list = field(default_factory=list)


@dataclass
class Context:
    """What the engine knows during one capture."""
    matter_id: str
    meeting_id: str
    uid: str
    pack: str
    roster: dict                                  # participant id → {"name", "role"}
    docs: Optional[list] = None                   # scan.documents(...) for the person, signed versions
    built: Optional[list] = None
    interests: list = field(default_factory=list)  # (participant id, topic words, segment id, quote)

    @property
    def counterparty(self) -> bool:
        return any(p["role"] == "counterparty" for p in self.roster.values())


def _said(text: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "quote": text[start:end]}


def _said_sentence(text: str, start: int, end: int) -> dict:
    """The whole sentence a phrase was found in: what a reviewer needs to judge it."""
    from ..research.prompt import _spans
    for a, b in _spans(text):
        if a <= start < b:
            return _said(text, a, max(b, end))
    return _said(text, start, end)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in _STOP}


def _count(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBER_WORDS.get(word.lower(), 0)


def load_documents(ctx: Context) -> None:
    if ctx.docs is None:
        ctx.docs = scanmod.documents(ctx.matter_id, ctx.uid, "signed")
        ctx.built = [factsmod.document(d["document_id"], d["document_title"], d["version_id"], d["text"], d["spans"]) for d in ctx.docs]


def _constraint(doc_text: str, g: factsmod.Fact) -> Optional[str]:
    """"min" or "max" if the counterpart's sentence sets a limit around its value."""
    sentence = doc_text[g.sentence[0]:g.sentence[1]]
    near = sentence[max(0, g.start - g.sentence[0] - 60):g.end - g.sentence[0] + 20]
    if _MIN_WORDS.search(near):
        return "min"
    if _MAX_WORDS.search(near):
        return "max"
    return None


def _number(f: factsmod.Fact) -> Optional[float]:
    m = re.search(r"[\d.]+$", f.value.replace("%", ""))
    if not m:
        m = re.match(r"([\d.]+)", f.value)
    try:
        return float(m.group(1) if m.lastindex else m.group(0)) if m else None
    except ValueError:
        return None


def _source(ctx: Context, doc_index: int, g: factsmod.Fact) -> dict:
    from ...core.db import get_conn
    from ..retrieval import search as searchmod
    d = ctx.docs[doc_index]
    with get_conn() as conn:
        where = searchmod.locate(conn, d["version_id"], g.sentence[0], g.sentence[1], {})
    return {"document_id": d["document_id"], "document_title": d["document_title"], "version_id": d["version_id"],
            "version_number": d["number"], "version_status": d["status"], "signed": d["status"] == "executed",
            "clause": where.get("clause"), "page": where.get("page_start"), "line": where.get("line"),
            "start": g.sentence[0], "end": g.sentence[1], "quote": d["text"][g.sentence[0]:g.sentence[1]],
            "value": g.shown, "value_start": g.start, "value_end": g.end}


def _what(ctx: Context, k: int, src: dict) -> str:
    """How the alert names its source: "signed agreement" only for a signed agreement."""
    if _AGREEMENT_DOC.search(f"{ctx.built[k].name} {src['document_title']}"):
        return "signed agreement" if src["signed"] else f"{src['document_title']} (not signed)"
    return src["document_title"] + ("" if src["signed"] else " (not signed)")


def _where(src: dict) -> str:
    clause = f"cl. {src['clause']}" if src.get("clause") else src["document_title"]
    return clause if src.get("clause") else f"“{clause}”"


async def _terms(ctx: Context, seg: dict, rules: set) -> list[Alert]:
    """Figures said, checked against the matter's documents (signed versions)."""
    text = seg["text"]
    said = factsmod.extract(text, [(0, len(text))])
    if not said:
        return []
    load_documents(ctx)
    out: list[Alert] = []
    for f in said:
        if f.kind not in ("money", "period", "percent", "date"):
            continue
        # Closest counterparts: one among limits (an authority), one among plain terms.
        limits, terms = [], []
        for k, doc in enumerate(ctx.built or []):
            for g in doc.facts:
                if g.kind != f.kind or g.unit != f.unit:
                    continue
                score = factsmod._about_the_same(f, g)
                if score < settings.LEGAL_ASSIST_NEAR_MIN:
                    continue
                limit = _constraint(ctx.docs[k]["text"], g)
                sentence = ctx.docs[k]["text"][g.sentence[0]:g.sentence[1]]
                (limits if limit and _AUTHORITY_WORDS.search(sentence) else terms).append((score, k, g, limit))
        if limits and "authority" in rules and _PROPOSES.search(text):
            score, k, g, limit = max(limits, key=lambda x: x[0])
            a, b = _number(f), _number(g)
            if a is not None and b is not None and ((limit == "min" and a < b) or (limit == "max" and a > b)):
                src = _source(ctx, k, g)
                out.append(Alert("authority", "differs", "critical", f"Differs from the authority in {src['document_title']}, {_where(src)}",
                                 _said(text, f.start, f.end), packs.RULES["authority"].confidence, source=src,
                                 explanation=f"{f.shown} is {'below' if limit == 'min' else 'above'} the limit of {g.shown}.",
                                 document_ids=[src["document_id"]]))
        # A statement about "the agreement" is checked against agreements; a letter, notice or minutes
        # only when no agreement in the matter speaks to it (and then named, never "the agreement").
        agreements = [t for t in terms if _AGREEMENT_DOC.search(f"{ctx.built[t[1]].name} {ctx.docs[t[1]]['document_title']}")]
        terms = agreements or terms
        if not terms or "signed_terms" not in rules:
            continue
        # The two closest plain terms. Context words alone can tie (a fee clause and a cap that mentions
        # "the Fees paid"), so the model judges them in order. If either has the same value, the
        # statement agrees with the documents and nothing is raised against them.
        top = sorted(terms, key=lambda x: -x[0])[:2]
        same = next((t for t in top if t[2].value == f.value), None)
        if same:
            src = _source(ctx, same[1], same[2])
            what = _what(ctx, same[1], src)
            if _NAMES_AGREEMENT.search(text):
                out.append(Alert("signed_terms", "matches", "notice", f"Matches {what}, {_where(src)}", _said(text, f.start, f.end),
                                 packs.RULES["signed_terms"].confidence, source=src, document_ids=[src["document_id"]]))
            continue
        for score, k, g, _ in top:
            src = _source(ctx, k, g)
            checked = await _model_check(ctx, seg, f, src)
            if checked is not None:
                what = _what(ctx, k, src)
                out.append(Alert("signed_terms", "differs", "critical", f"Differs from {what}, {_where(src)}", _said(text, f.start, f.end),
                                 checked[0], method="model", source=src, explanation=checked[1], document_ids=[src["document_id"]]))
                break
    return out


async def _model_check(ctx: Context, seg: dict, f: factsmod.Fact, src: dict) -> Optional[tuple[float, str]]:
    """The model confirms the statement and the passage are about the same thing. None when it says
    they are not, or cannot say (a conservative alert is no alert)."""
    speaker = ctx.roster.get(seg.get("speaker_participant_id") or "", {})
    c = {"kind": f.kind,
         "a": {"document_title": "What was said in the meeting", "version_number": "", "version_status": speaker.get("role", ""),
               "clause": None, "parties": [], "context": seg["text"], "value": f.shown, "context_start": 0},
         "b": {"document_title": src["document_title"], "version_number": src["version_number"], "version_status": src["version_status"],
               "clause": src.get("clause"), "parties": [], "context": src["quote"], "value": src["value"], "context_start": src["start"]}}
    try:
        raw = await llm_mod.get().chat(scanmod.messages(c), temperature=0.0, max_tokens=240)
    except Exception as e:
        logger.warning("Assist: the model could not check a statement (%s)", type(e).__name__)
        return None
    checked = scanmod.check(c, raw)
    if checked["verdict"] != "conflict":
        return None
    verified = [q for q in checked["evidence"] if q["verification"] in ("verbatim", "normalized")]
    return (packs.RULES["signed_terms"].confidence if verified else 0.6), checked["explanation"]


def _phrases(ctx: Context, seg: dict, rules: set) -> list[Alert]:
    text = seg["text"]
    speaker = ctx.roster.get(seg.get("speaker_participant_id") or "")
    role = speaker["role"] if speaker else None
    out: list[Alert] = []

    def allowed(rule_id: str) -> bool:
        r = packs.RULES[rule_id]
        return rule_id in rules and (not r.roles or role in r.roles) and (not r.counterparty_present or ctx.counterparty)
    if allowed("commitment"):
        m = _COMMITMENT.search(text)
        if m:
            out.append(Alert("commitment", "needs_review", "critical", "Needs review: a commitment to the other side",
                             _said_sentence(text, m.start(), m.end()), packs.RULES["commitment"].confidence))
    if allowed("disclosure"):
        m = _DISCLOSURE.search(text)
        if m:
            out.append(Alert("disclosure", "needs_review", "critical", "Needs review: privileged or sensitive information with a counterparty on the call",
                             _said_sentence(text, m.start(), m.end()), packs.RULES["disclosure"].confidence))
    if allowed("action"):
        m = _ACTION.search(text)
        if m:
            out.append(Alert("action", "needs_review", "notice", "Action taken on", _said_sentence(text, m.start(), m.end()),
                             packs.RULES["action"].confidence))
    if allowed("quorum"):
        m = _QUORUM.search(text)
        if m and 0 < _count(m.group(1)) * 2 <= _count(m.group(2)):
            out.append(Alert("quorum", "needs_review", "critical", "Needs review: fewer than half the directors present — check the quorum",
                             _said_sentence(text, m.start(), m.end()), packs.RULES["quorum"].confidence,
                             explanation="No quorum rule is cited from this matter's documents; check the articles."))
    if "conflict" in rules and speaker:
        m = _DECLARE.search(text)
        if m:
            ctx.interests.append((seg.get("speaker_participant_id"), _words(m.group("topic")), seg["id"], text[m.start():m.end()]))
        vote = _VOTE.search(text)
        if vote:
            mine = [x for x in ctx.interests if x[0] == seg.get("speaker_participant_id") and x[1] & _words(text)]
            source = {"segment_id": mine[0][2], "quote": mine[0][3]} if mine else _declared_in_documents(ctx, speaker["name"], text)
            if source:
                out.append(Alert("conflict", "differs", "critical", "Differs from a declaration of interest" +
                                 (" earlier in this meeting" if mine else f" in {source['document_title']}"),
                                 _said_sentence(text, vote.start(), vote.end()), packs.RULES["conflict"].confidence, source=source,
                                 document_ids=[source["document_id"]] if source.get("document_id") else []))
    return out


def _declared_in_documents(ctx: Context, name: str, text: str) -> Optional[dict]:
    load_documents(ctx)
    for d in ctx.docs or []:
        for m in _DECLARED_IN_DOC.finditer(d["text"]):
            if m.group("who").lower() == name.lower() and _words(m.group("topic")) & _words(text):
                return {"document_id": d["document_id"], "document_title": d["document_title"], "version_id": d["version_id"],
                        "version_number": d["number"], "start": m.start(), "end": m.end(), "quote": d["text"][m.start():m.end()]}
    return None


async def consider(ctx: Context, seg: dict) -> list[Alert]:
    """Alerts for one stored segment, at or above the threshold in force."""
    rules = set(packs.PACKS.get(ctx.pack, ()))
    if not rules or not seg.get("text", "").strip():
        return []
    found = _phrases(ctx, seg, rules)
    if rules & {"signed_terms", "authority"}:
        try:
            found += await _terms(ctx, seg, rules)
        except Exception as e:
            logger.warning("Assist: checking the terms failed (%s)", type(e).__name__)
    return [a for a in found if a.confidence >= settings.LEGAL_ASSIST_MIN_CONFIDENCE]


def verify_source(a: Alert, texts: dict[str, str]) -> bool:
    """A document source must be exact characters of its version's text (by construction; checked)."""
    s = a.source or {}
    if not s.get("version_id"):
        return True
    t = texts.get(s["version_id"])
    return bool(t) and t[s["start"]:s["end"]] == s["quote"] and evidence.verify(s["quote"], t).verification == "verbatim"
