"""A chronology of the matter, from its documents (docs/echomind-legal/03 `timeline`, 04, 11 LEGAL-011).

Nothing here is generated or inferred by the model. Each event is a date a document states, with its exact characters,
sentence, clause and version, so it opens in the source panel:

- **dated events**: every full date the conflict scan's reader finds (compare/facts.py), labelled by what the sentence does
  with it: the document's own date ("dated April 1, 2026", "Sent: May 9, 2026"), a deadline ("by May 15, 2026"), or an event;
  a recollection ("Witness A recalls …") is marked as reported, and a date the same passage hedges ("was not sure",
  "on or about", "believes") as uncertain;
- **dates not stated**: a partial or relative date about something that happened ("move to the 20th", "went out yesterday")
  is listed with its words and no date; it is never turned into one;
- **deadlines from the obligations register**, with their review state: a date is only "confirmed" when a lawyer confirmed it;
- **what people decided**: an event that is one side of a potential inconsistency carries its state and the reviewer's
  reason ("Explained by alice: dispatch and arrival"), so an explained pair reads as the two events it is;
- **what was replaced**: an event in a clause another readable document replaces is marked, with the amendment's name.

Only documents the person may read, in the version answers come from (signed by default), in this one matter. A sentence
addressed to an AI ("ignore previous instructions …") is never shown as an event.

A chronology of something ("of the delivery, from dispatch to acceptance", "How did the delivery date come to be changed?
Give the key dates") is the matter's chronology narrowed to it (2026-11-05 end-to-end run: the whole matter's 174 events,
the first 60 shown, never reached the June deliveries): the events in the passages a search for it finds, and, when a range
is asked for, those from the first event that names its start to the last that names its end; of a thread of messages one
of those passages is in, every message's date and subject. Nothing is added: an event is still a date a document states,
with its words.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import quote as urlquote

from ...core.db import get_conn
from ..compare import facts, scan
from ..documents import repo as doc_repo
from ..research import amendments
from ..research import prompt as research_prompt
from ..retrieval import search as searchmod
from . import discrepancies

MAX_EVENTS = 60
CONTEXT = 420

_HEDGE = re.compile(r"\b(?:not\s+sure|unsure|uncertain|believes?|approximately|on\s+or\s+about|may\s+have|might\s+have|possibly|"
                    r"perhaps|estimated|(?:does|did)\s+not\s+(?:recall|remember)|cannot\s+(?:recall|remember))\b", re.I)
_REPORTED = re.compile(r"\b(?:recalls?|remembers?|believes?|said|says|stated|told|testified|according\s+to)\b", re.I)
_DEADLINE = re.compile(r"\b(?:by|no\s+later\s+than|on\s+or\s+before|not\s+later\s+than|before|until|due(?:\s+on)?)\s*(?:the\s+)?$", re.I)
_SENT = re.compile(r"\b(?:sent|date|dated)\s*:?\s*$", re.I)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_PARTIAL = re.compile(rf"\bthe\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?(?:{_MONTHS}))")
_RELATIVE = re.compile(r"\b(?:yesterday|today|tomorrow|tonight|last\s+(?:week|month|night)|next\s+(?:week|month)|earlier\s+this\s+week)\b",
                       re.I)
_EVENT_WORDS = (frozenset().union(*facts._ANCHORS.values())
                | {"went", "sent", "signed", "arrived", "meeting", "called", "met", "shipped", "moved", "delivered"})


_LABEL_LINE = re.compile(r"^\s*(?:sent|date|dated|received|signed|effective|from|to|subject|cc)\b[^:\n]{0,20}:", re.I)
_SUBJECT_LINE = re.compile(r"^\s*(?:subject|re)\s*:", re.I)
_RANGE_JOIN = re.compile(r"^\s*(?:to|until|through|–|—|-)\s*$", re.I)


def _event_span(text: str, sentence: tuple[int, int], start: int) -> tuple[int, int]:
    """The words an event shows. A date in a header field ("Sent: May 9, 2026") is that line, with the message's subject
    line when it comes next; a date in a table is its row; heading lines before the dated line ("1. Amendment of clause
    4.2") are left out; otherwise the whole sentence (a wrapped line is kept whole)."""
    a, b = sentence
    lines, pos = [], a
    while pos < b:
        nl = text.find("\n", pos, b)
        e = b if nl < 0 else nl
        if text[pos:e].strip():
            lines.append((pos, e))
        pos = e + 1
    i = next((k for k, (s, e) in enumerate(lines) if s <= start <= e), None)
    if len(lines) <= 1 or i is None:
        return (a, _subject_end(text, b)) if _LABEL_LINE.match(text[a:b]) else (a, b)
    ls, le = lines[i]
    if _LABEL_LINE.match(text[ls:le]):
        return ls, _subject_end(text, le)
    if text[ls:le].lstrip().startswith("|"):             # a table row: that row, not the table from its header
        return ls, le
    if all((e - s) < 60 and not text[s:e].rstrip().endswith(".") for s, e in lines[:i]):
        return ls, b
    return a, b


FOCUS_PASSAGES = 24                # for a subject ("of the delivery"); a question narrows more ("How did … come to be changed?")
FOCUS_QUESTION_PASSAGES = 12
_REQUEST = re.compile(r"\b(?:chronolog\w*|timeline|time\s+line|sequence\s+of\s+events|order\s+of\s+events|key\s+dates|key\s+events"
                      r"|dates\s+in\s+order|what\s+happened\s+when)\b", re.I)
_OF = re.compile(r"\b(?:chronolog\w*|timeline|time\s+line|sequence\s+of\s+events|order\s+of\s+events|key\s+dates|key\s+events)"
                 r"\s+(?:of|for|about|on|covering|regarding|concerning|relating\s+to|in)\s+(?P<about>[^.?!;]+)", re.I)
_RANGE = re.compile(r"\b(?:from|since)\s+(?:the\s+)?(?P<a>[a-z][\w' -]{2,40}?)\s+(?:to|until|till|through|up\s+to)\s+(?:the\s+)?"
                    r"(?P<b>[a-z][\w' -]{2,40}?)\s*(?=[.,;?!]|$)"
                    r"|\bbetween\s+(?:the\s+)?(?P<c>[a-z][\w' -]{2,40}?)\s+and\s+(?:the\s+)?(?P<d>[a-z][\w' -]{2,40}?)\s*(?=[.,;?!]|$)", re.I)
_ASKING = re.compile(r"^\W*(?:(?:please|can\s+you|could\s+you|would\s+you|i\s+need|i\s+want|let\s+me\s+see|now)\s+)*"
                     r"(?:give|show|build|make|create|prepare|draw\s+up|produce|put\s+together|list|set\s+out|do|run|what\s+(?:is|are))?"
                     r"\s*(?:me|us)?\s*(?:a|an|the|all\s+the)?\s*(?:full|short|brief|detailed|quick)?\s*", re.I)
_EVERYTHING = re.compile(r"^(?:the\s+|this\s+|our\s+|my\s+)?(?:matter|case|file|documents?|events?|everything|all|it|dispute|"
                         r"whole\s+(?:matter|case|file)|what\s+happened)$", re.I)


def topic(text: str) -> Optional[dict]:
    """What a chronology request is about, or None for the whole matter: {"about", "query", "from", "to"}.
    "Give me a chronology of the delivery, from dispatch to acceptance." -> the delivery, from dispatch to acceptance;
    "How did the delivery date come to be changed? Give the key dates." -> that question; "Build a chronology" -> None."""
    t = " ".join((text or "").split())
    rng = _RANGE.search(t)
    start, end = ((rng["a"] or rng["c"]).strip(), (rng["b"] or rng["d"]).strip()) if rng else (None, None)
    m = _OF.search(t)
    about = m["about"] if m else ""
    if rng and m and rng.start() >= m.start("about"):
        about = t[m.start("about"):rng.start()]
    about = re.sub(r"^\s*(?:the\s+)?(?:matter|case)\s+of\s+", "", about).strip(" ,.;:-")
    if not about:                                 # the question asked, without the request for dates ("Give the key dates.")
        kept = [x for x in re.split(r"(?<=[.?!])\s+", t) if not _REQUEST.search(x) or len(_REQUEST.sub("", _ASKING.sub("", x)).split()) > 3]
        about = " ".join(kept).strip()
        if about == t and _REQUEST.search(t):
            about = _REQUEST.sub(" ", _ASKING.sub("", t))
            about = re.sub(r"^\W*(?:of|for|about|on)\s+", "", " ".join(about.split())).strip(" ,.;:-?")
    if rng and about:
        about = about.replace(rng.group(0), "").strip(" ,.;:-")
    if (not about or _EVERYTHING.match(about) or len(about.split()) < 1) and not rng:
        return None
    if about and _EVERYTHING.match(about):
        about = ""
    query = " ".join(x for x in (about, start, end) if x)
    return ({"about": about[:120], "query": query[:300], "from": start, "to": end,
             "k": FOCUS_QUESTION_PASSAGES if about.endswith("?") else FOCUS_PASSAGES} if query.strip() else None)


_SUFFIX = re.compile(r"(?:ances?|ences?|ments?|ings?|ions?|ery|ies|ed|es|s|al)$")


def _stem(word: str) -> str:
    w = word.lower()
    cut = _SUFFIX.sub("", w)
    return cut if len(cut) >= 4 else w


def _names(text: str, term: Optional[str]) -> bool:
    """The event's words name the range's term ("dispatch" in "dispatched May 10, 2026")."""
    if not term:
        return False
    stems = [_stem(w) for w in re.findall(r"[a-z]{3,}", term.lower()) if w not in ("the", "and", "of")]
    words = re.findall(r"[a-z]+", (text or "").lower())
    return bool(stems) and all(any(w.startswith(st) for w in words) for st in stems)


def _subject_end(text: str, le: int) -> int:
    """A message's date line is shown with its subject line when that comes next ("Date: May 4, 2026 … Subject: request to
    move the delivery date"): the date alone says nothing of what happened."""
    pos = le
    for _ in range(2):
        nl = text.find("\n", pos)
        if nl < 0:
            break
        end = text.find("\n", nl + 1)
        end = len(text) if end < 0 else end
        line = text[nl + 1:end]
        if _SUBJECT_LINE.match(line):
            return end
        if not _LABEL_LINE.match(line):
            break
        pos = end
    return le


def _clean(s: str) -> str:
    return " ".join((s or "").split())


def _context(text: str, sentence: tuple[int, int], start: int, end: int) -> tuple[str, int, int]:
    """The sentence (trimmed around the value when long) and the value's place in it."""
    a, b = sentence
    if b - a > CONTEXT:
        a = max(a, start - CONTEXT // 2)
        b = min(b, a + CONTEXT)
    return text[a:b], start - a, end - a


def _source(conn, number: str, doc: dict, cache: dict, sentence: tuple[int, int], start: int, end: int, value: str) -> dict:
    ctx, vs, ve = _context(doc["text"], sentence, start, end)
    where = searchmod.locate(conn, doc["version_id"], start, end, cache)
    page = where.get("page_start")
    return {"document_id": doc["document_id"], "document_title": doc["document_title"], "version_id": doc["version_id"],
            "version_number": doc["number"], "version_status": doc["status"], "clause": where.get("clause"),
            "page": page if isinstance(page, int) and page >= 1 else None, "value": value[:300], "context": ctx[:1500],
            "value_start": max(0, vs), "value_end": max(vs, ve),
            "reader_path": f"/m/{urlquote(number, safe='')}/documents/{doc['document_id']}/read?v={doc['version_id']}"
                           f"&at={start}-{max(end, start + 1)}",
            "stale": False}


def _replaced(docs: list[dict]) -> dict[str, list[tuple[int, int, str]]]:
    """Per version: character ranges of clauses another readable document replaces or changes, with a note saying so."""
    targets = [{"document_id": d["document_id"], "title": d["document_title"], "head": d["text"][:1500]} for d in docs]
    by_doc = {d["document_id"]: d for d in docs}
    out: dict[str, list[tuple[int, int, str]]] = {}
    with get_conn() as conn:
        for d in docs:
            for ch in amendments.changes(d["text"]):
                t = amendments._target_for(ch, d["text"], [x for x in targets if x["document_id"] != d["document_id"]])
                if not t:
                    continue
                vid = by_doc[t["document_id"]]["version_id"]
                rows = conn.execute("SELECT char_start, char_end FROM legal_clauses WHERE version_id=? AND (number=? OR number LIKE ?)",
                                    (vid, ch.clause, ch.clause + ".%")).fetchall()
                if not rows:
                    continue
                signed = d["status"] == "executed"
                note = (f"Clause {ch.clause} was {ch.how} by {d['document_title']}" + (" (signed)." if signed else ", which is not signed."))
                out.setdefault(vid, []).append((min(r[0] for r in rows), max(r[1] for r in rows), note + ("|signed" if signed else "")))
    return out


def _discrepancy_notes(principal: dict, matter_id: str) -> dict[tuple[str, int], str]:
    """(version, value start) of each side of a potential inconsistency the person may see -> what was decided about it."""
    out = {}
    for r in discrepancies.listed(principal, matter_id, discrepancies.OPEN + discrepancies.RESOLVED):
        hist = discrepancies.history(r["id"])
        for side, other in (("a", "b"), ("b", "a")):
            s, o = r["detail"][side], r["detail"][other]
            if r["status"] in discrepancies.OPEN:
                text = f"Potential inconsistency with {o.get('document_title')} (“{o.get('value')}”), not resolved."
            else:
                last = hist[-1] if hist else {}
                text = (f"{r['status'].capitalize()} by {last.get('by', 'a reviewer')}"
                        + (f": “{last['note']}”" if last.get("note") else "") + f" (with {o.get('document_title')}).")
            out[(s["version_id"], s.get("value_start"))] = text[:300]
    return out


def build(principal: dict, matter: dict, focus: Optional[dict] = None) -> tuple[dict, list[str]]:
    """The timeline props and the document ids it names. `focus`: what the chronology is of (topic()), with "ranges", the
    characters ({version id: [(start, end)]}) of the passages a search for it found in this matter for this person."""
    uid, mid, number = principal["sub"], matter["id"], matter["number"]
    docs = scan.documents(mid, uid, "signed")
    replaced = _replaced(docs)
    decided = _discrepancy_notes(principal, mid)
    dated, undated, seen = [], [], set()
    cache: dict = {}
    with get_conn() as conn:
        for d in docs:
            text = d["text"]
            own = facts._OWN_DATE.search(text[:1500])
            sentences = list(facts._sentences(text, d["spans"]))
            dated_sentences = set()
            dates = sorted((f for f in facts.extract(text, d["spans"])
                            if f.kind == "date" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.value or "")), key=lambda f: f.start)
            ends = {}                                # "May 13, 2026 to May 14, 2026": one event, from the first date
            for f1, f2 in zip(dates, dates[1:]):
                if f1.sentence == f2.sentence and _RANGE_JOIN.match(text[f1.end:f2.start]):
                    ends[f1.start] = f2
            joined = {f2.start for f2 in ends.values()}
            for f in dates:
                if f.start in joined:
                    continue
                f_end, shown = (ends[f.start].end, text[f.start:ends[f.start].end]) if f.start in ends else (f.end, f.shown)
                a, b = _event_span(text, f.sentence, f.start)
                sent = text[a:b]
                if research_prompt.addressed_to_ai(sent):
                    continue
                key = (d["version_id"], f.start)
                if key in seen:
                    continue
                seen.add(key)
                dated_sentences.add(f.sentence)
                nxt = next((text[s2:e2] for s2, e2 in sentences if 0 <= s2 - b < 4), "")     # a hedge may follow ("was not sure")
                before = text[max(a, f.start - 30):f.start]
                kind = ("document_date" if (own and own.start() <= f.start < own.end()) or _SENT.search(before[-12:])
                        else "deadline" if _DEADLINE.search(before) else "event")
                ev = {"date": f.value, "date_text": shown[:80],
                      "certainty": "uncertain" if _HEDGE.search(sent) or _HEDGE.search(nxt) else "stated",
                      "kind": kind, "reported": bool(_REPORTED.search(sent)), "superseded": False, "text": _clean(sent)[:400],
                      "notes": [], "source": _source(conn, number, d, cache, (a, b), f.start, f_end, shown)}
                for k in (key, (d["version_id"], ends[f.start].start) if f.start in ends else None):   # a range: either end
                    if k in decided:
                        ev["notes"].append(decided[k])
                for ra, rb, note in replaced.get(d["version_id"], []):
                    if ra <= f.start < rb:
                        text_note, _, signed = note.partition("|")
                        ev["notes"].append(text_note[:300])
                        ev["superseded"] = ev["superseded"] or signed == "signed"
                dated.append(ev)
            for s, e in sentences:
                if (s, e) in dated_sentences:
                    continue
                sent = text[s:e]
                m = _PARTIAL.search(sent) or _RELATIVE.search(sent)
                words = set(re.findall(r"[a-z]+", sent.lower()))
                if not m or not (words & _EVENT_WORDS) or research_prompt.addressed_to_ai(sent):
                    continue
                undated.append({"date_text": m.group(0)[:80], "certainty": "undated", "kind": "event",
                                "reported": bool(_REPORTED.search(sent)), "superseded": False, "text": _clean(sent)[:400], "notes": [],
                                "source": _source(conn, number, d, cache, (s, e), s + m.start(), s + m.end(), m.group(0))})
        by_version = {d["version_id"]: d for d in docs}
        rows = conn.execute(
            "SELECT o.document_id, o.version_id, o.kind, o.action_text, o.when_text, o.due_date, o.status, o.evidence_json "
            "FROM legal_obligations o JOIN legal_documents d ON d.id = o.document_id "
            f"WHERE o.matter_id = ? AND o.status IN ('proposed','confirmed','done') AND o.due_date IS NOT NULL AND {doc_repo.VISIBLE_TO}",
            (mid, uid)).fetchall()
        for did, vid, okind, action, when, due, status, ej in rows:
            d = by_version.get(vid)
            evidence = json.loads(ej or "{}")
            if not d or evidence.get("start") is None:
                continue
            state = {"proposed": "Proposed from the document, not confirmed", "confirmed": "Confirmed by a lawyer",
                     "done": "Done (confirmed by a lawyer)"}[status]
            a, b = evidence["start"], evidence["end"]
            # The register's deadline is usually a date this chronology already has (the same words): give it the state.
            same = next((ev for ev in dated if ev["source"]["version_id"] == vid and ev["date"] == due[:10]
                         and a - 200 <= _at(ev) < b + 200), None)
            if same:
                same["kind"], same["status"] = "deadline", state
                continue
            sentence = next(((s, e) for s, e in facts._sentences(d["text"], d["spans"]) if s <= a < e), (a, b))
            dated.append({"date": due[:10], "date_text": (when or due)[:80], "certainty": "stated", "kind": "obligation",
                          "reported": False, "superseded": False, "text": _clean(action)[:400], "status": state, "notes": [],
                          "source": _source(conn, number, d, cache, sentence, a, b, d["text"][a:b])})
    dated.sort(key=lambda e: (e["date"], e["source"]["document_title"], _at(e)))
    undated.sort(key=lambda e: (e["source"]["document_title"], _at(e)))
    whole = len(dated)
    narrowed = None
    if focus and focus.get("ranges"):
        narrowed = _narrow(dated, undated, focus)
        if narrowed is not None:
            dated, undated = narrowed
    events = dated + undated
    shown, more = events[:MAX_EVENTS], max(0, len(events) - MAX_EVENTS)
    for i, ev in enumerate(shown):
        ev["id"] = f"ev{i + 1}"
        ev["notes"] = ev["notes"][:3]
    n_docs = len({e["source"]["document_id"] for e in shown})
    if not shown:
        summary = f"No dates found in the {len(docs)} document{'s' if len(docs) != 1 else ''} of {number} you can read."
    elif narrowed is not None:
        about = focus.get("about") or "what was asked"
        about = f"“{about}”" if about.endswith("?") else about
        span = ""
        if focus.get("from") or focus.get("to"):
            span = (f", from {focus['from'] or 'the start'} ({dated[0]['date_text']})" if dated else "") + \
                   (f" to {focus['to'] or 'the end'} ({dated[-1]['date_text']})" if dated else "")
        summary = (f"{len(dated)} dated event{'s' if len(dated) != 1 else ''} about {about}{span}, from {n_docs} "
                   f"document{'s' if n_docs != 1 else ''} in {number}. The whole chronology has {whole}.")
    elif focus:
        summary = (f"Nothing dated was found about {focus.get('about') or 'what was asked'}, so the whole chronology is shown: "
                   f"{len(dated)} dated event{'s' if len(dated) != 1 else ''} from {n_docs} document{'s' if n_docs != 1 else ''} in {number}.")
    else:
        summary = (f"{len(dated)} dated event{'s' if len(dated) != 1 else ''} from {n_docs} document{'s' if n_docs != 1 else ''}"
                   + (f", and {len(undated)} whose date is not stated" if undated else "") + f" in {number}.")
    props = {"summary": summary[:400], "events": shown, "more": more,
             "caution": ("Dates are shown as the documents state them; nothing is inferred. Recollections and uncertain or partial "
                         "dates are marked, and a deadline counts only once a lawyer confirms it.")}
    return props, sorted({e["source"]["document_id"] for e in shown})


def _narrow(dated: list[dict], undated: list[dict], focus: dict) -> Optional[tuple[list[dict], list[dict]]]:
    """The events in the passages found for the focus, and the date and subject of every message of a thread one of them
    is in (the message that started it is often not among them), within its range when it has one; None when none is in
    them."""
    ranges = focus["ranges"]

    def inside(ev: dict) -> bool:
        vid, at = ev["source"]["version_id"], _at(ev)
        if vid in ranges and ev["kind"] == "document_date" and _LABEL_LINE.match(ev["text"]) \
                and re.search(r"\bsubject\s*:", ev["text"], re.I):
            return True
        return any(a <= at < b for a, b in ranges.get(vid, []))
    near = [e for e in dated if inside(e)]
    if not near:
        return None
    first = [e["date"] for e in near if _names(e["text"], focus.get("from"))]
    last = [e["date"] for e in near if _names(e["text"], focus.get("to"))]
    lo = min(first) if first else None
    hi = max([d for d in last if lo is None or d >= lo], default=None)
    near = [e for e in near if (lo is None or e["date"] >= lo) and (hi is None or e["date"] <= hi)]
    return near, [e for e in undated if inside(e)] if not (lo or hi) else []


def _at(ev: dict) -> int:
    """An event's character position in its version (from its reader link)."""
    m = re.search(r"&at=(\d+)-", ev["source"].get("reader_path") or "")
    return int(m.group(1)) if m else 0
