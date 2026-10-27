"""Amendments: when a clause an answer relies on was replaced or changed by another document of the matter.

Found on the DGX Spark (docs/echomind-legal/IMPLEMENTATION_STATUS.md, live finding 1): asked "Which contractual deadline
applied?", research cited the agreement's clause 4.2 ("delivery by May 15, 2026") and the amendment's clause 2.1 ("all other
terms remain in force") and concluded the amendment changed nothing, because the amendment's clause 1.1 — "Clause 4.2 of the
Services Agreement is deleted and replaced with: … May 20, 2026" — never reached the model. Every quote was verified; the
conclusion was wrong.

The link between the two passages is taken from the documents' own words, never from the model:
- an amending sentence names a clause of another agreement and says it is replaced, amended, deleted, varied or substituted
  ("Clause 4.2 of the Services Agreement is deleted and replaced with …", "In clause 9.1 of the Agreement, … is replaced by …");
- the agreement is the matter document whose title or heading carries those words (and, when the amending document refers to
  it by date — "the Services Agreement dated April 1, 2026" — the one that gives that date as its own); "the Agreement" means
  the one agreement the amending document refers to by name and date;
- whether the amendment is signed comes from its version's status (executed), and from when it applies only from its own
  words ("takes effect on …", "signed it on …", "made on …"), quoted; nothing is inferred.

Only documents the person may read, in the version the answer is drawn from (signed by default), are considered, all within
the one matter of the search scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ...core.db import get_conn
from ..compare import facts
from ..compare.align import fold
from ..documents import repo as doc_repo
from ..documents import versions as doc_versions
from ..retrieval import search as searchmod

MAX_ADDED = 3                      # amending passages added to one answer's sources
MAX_PASSAGE = 1500                 # characters of an amending clause given to the model

_TITLE = (r"(?:(?:[A-Z][\w'’-]*\s+){0,5}?(?:Agreement|Contract|Lease|Deed|Licen[cs]e|Terms|Order\s+Form|Statement\s+of\s+Work|"
          r"Side\s+Letter|Addendum)|Agreement)")
_VERB = (r"(?P<how>deleted\s+and\s+replaced|deleted\s+and\s+substituted|replaced|substituted|amended(?:\s+and\s+restated)?|"
         r"deleted|varied|restated)")
# "Clause 4.2 of the Services Agreement is (hereby) deleted and replaced with …"
_DIRECT = re.compile(r"\b(?i:clause|section|paragraph|article)s?\s+(?P<clause>\d{1,3}(?:\.\d{1,3}){0,3}[a-z]?)\s+of\s+(?:the\s+)?"
                     rf"(?P<title>{_TITLE})\s+(?:is|are|shall\s+be)\s+(?:hereby\s+)?{_VERB}\b")
# "In clause 9.1 of the Services Agreement, “USD 240,000” is replaced by “USD 250,000”."
_WITHIN = re.compile(r"\b(?i:in)\s+(?i:clause|section|paragraph|article)\s+(?P<clause>\d{1,3}(?:\.\d{1,3}){0,3}[a-z]?)\s+of\s+(?:the\s+)?"
                     rf"(?P<title>{_TITLE})\s*,[^.;]{{0,240}}?\b(?:is|are)\s+(?:hereby\s+)?{_VERB}\b")
_EFFECT = re.compile(rf"(?i:takes?\s+effect|effective|comes?\s+into\s+(?:force|effect)|with\s+effect\s+from)[^.\n]{{0,160}}?{facts._DATE_RE}")
_SIGNED = re.compile(rf"(?i:signed|executed)[^.\n]{{0,80}}?\b(?i:on)\s+(?:the\s+)?{facts._DATE_RE}")
_CANDIDATE_WORDS = ("replaced", "amended", "deleted", "substituted", "varied", "restated")


@dataclass(frozen=True)
class Change:
    clause: str          # the clause of the other agreement that is changed
    title: str           # how the amending document names that agreement
    how: str             # deleted and replaced, amended, …
    start: int           # the amending sentence, in the amending version's text
    end: int


def changes(text: str) -> list[Change]:
    """Every sentence of `text` that names a clause of another agreement and replaces or changes it."""
    out, seen = [], set()
    for rx in (_DIRECT, _WITHIN):
        for m in rx.finditer(text or ""):
            key = (m.group("clause"), m.start())
            if key in seen:
                continue
            seen.add(key)
            out.append(Change(m.group("clause"), " ".join(m.group("title").split()), " ".join(m.group("how").lower().split()),
                              m.start(), m.end()))
    return sorted(out, key=lambda c: c.start)


def _sentence(text: str, start: int, end: int) -> tuple[int, int, str]:
    a = max(text.rfind(". ", 0, start) + 2 if text.rfind(". ", 0, start) >= 0 else 0, text.rfind("\n", 0, start) + 1)
    b = text.find(".", end)
    b = len(text) if b < 0 else b + 1
    return a, b, " ".join(text[a:b].split())


def effective(text: str) -> Optional[dict]:
    """When the amending document says it applies, in its own words: an effective date, else a signing date, else the date it
    says it was made. None when it says none of these."""
    for kind, rx in (("effective", _EFFECT), ("signed", _SIGNED), ("made", facts._OWN_DATE)):
        m = rx.search(text or "")
        if m:
            a, b, words = _sentence(text, m.start(), m.end())
            words = re.sub(r"^\(?\d{1,3}(?:\.\d{1,3})*\)?\s+", "", words)        # without its own clause number
            return {"kind": kind, "date": facts._date(m), "words": words[:240], "char_start": a, "char_end": b}
    return None


def _refers(text: str) -> list[tuple[str, Optional[str]]]:
    """The agreements an amending document refers to by name, with the date it gives for each."""
    return [(" ".join(m.group("title").split()), facts._date(m)) for m in facts._REFERENCE.finditer(text[:3000])]


def _named(title_words: str, target: dict) -> bool:
    words = fold(title_words).split()
    words = [w for w in words if w not in ("the", "this")]
    if not words:
        return False
    hay = " ".join(fold(target["title"] + " " + target["head"][:400]).split())
    return " ".join(words) in hay


def _own_date(head: str) -> Optional[str]:
    m = facts._OWN_DATE.search(head)
    return facts._date(m) if m else None


def _target_for(change: Change, amending_text: str, targets: list[dict]) -> Optional[dict]:
    refs = _refers(amending_text)
    words = change.title
    if fold(words).split() == ["agreement"]:                      # "the Agreement": the one agreement it refers to by name
        named = {t for t, _ in refs}
        if len(named) != 1:
            return None
        words = named.pop()
    found = [t for t in targets if _named(words, t)]
    dated = {d for t, d in refs if d and _named(t, {"title": words, "head": ""})}
    if dated:
        found = [t for t in found if _own_date(t["head"]) in dated] or found
    return found[0] if len(found) == 1 else None


def _clause_covers(passage_clause: Optional[str], changed: str) -> bool:
    return bool(passage_clause) and (passage_clause == changed or passage_clause.startswith(changed + "."))


def for_passages(matter_id: str, uid: str, passages: list[dict], policy: str = "signed") -> list[dict]:
    """Links from passages whose clause another readable document of the matter changes, to that document's amending clause.
    Each link: the passage index, what is changed, the amending passage (in search's passage shape), whether the amendment is
    signed and when it says it applies."""
    wanted = {}
    for i, p in enumerate(passages):
        c = p["citation"]
        if c.get("clause"):
            wanted.setdefault(c["document_id"], []).append(i)
    if not wanted:
        return []
    pick = ("d.current_version_id" if policy == "current"
            else f"COALESCE({doc_versions._LATEST_SIGNED.format(doc='d.id')}, d.current_version_id)")
    like = " OR ".join("t.text LIKE ?" for _ in _CANDIDATE_WORDS)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT d.id, d.title, v.id, v.number, v.status, t.text FROM legal_documents d "
            f"JOIN legal_document_versions v ON v.id = {pick} JOIN legal_version_text t ON t.version_id = v.id "
            f"WHERE d.matter_id = ? AND {doc_repo.VISIBLE_TO} AND ({like}) ORDER BY d.created_at LIMIT 50",
            (matter_id, uid, *[f"%{w}%" for w in _CANDIDATE_WORDS])).fetchall()
        targets = []
        for did in wanted:
            vid = passages[wanted[did][0]]["citation"]["version_id"]
            head = conn.execute("SELECT substr(text, 1, 1500) FROM legal_version_text WHERE version_id=?", (vid,)).fetchone()
            targets.append({"document_id": did, "title": passages[wanted[did][0]]["citation"]["document_title"],
                            "head": head[0] if head else ""})
        links, cache = [], {}

        def passage_at(did, title, vid, number, status, text, a, b, focus):
            if b - a > MAX_PASSAGE:
                a, b = max(a, focus[0] - 200), min(b, focus[0] + MAX_PASSAGE - 200)
            span = searchmod.locate(conn, vid, a, b, cache)
            where = searchmod.locate(conn, vid, focus[0], focus[1], cache)
            return {"chunk_id": f"amendment:{vid}:{a}", "text": text[a:b], "score": None, "score_kind": None,
                    "citation": {"matter_id": matter_id, "document_id": did, "document_title": title, "version_id": vid,
                                 "version_number": number, "version_status": status, "signed": status == "executed",
                                 "page_start": span.get("page_start"), "page_end": span.get("page_end"),
                                 "char_start": a, "char_end": b, "clause": where.get("clause"), "line": where.get("line"),
                                 "focus_start": focus[0], "focus_end": focus[1]}}

        def clause_span(vid, text, start, end):
            where = searchmod.locate(conn, vid, start, end, cache)
            own = conn.execute("SELECT char_start, char_end FROM legal_clauses WHERE version_id=? AND number=? "
                               "ORDER BY level DESC LIMIT 1", (vid, where.get("clause"))).fetchone() if where.get("clause") else None
            return (own[0], own[1]) if own else _sentence(text, start, end)[:2]

        for did, title, vid, number, status, text in rows:
            for ch in changes(text):
                target = _target_for(ch, text, [t for t in targets if t["document_id"] != did])
                if not target:
                    continue
                hit = [i for i in wanted[target["document_id"]] if _clause_covers(passages[i]["citation"].get("clause"), ch.clause)]
                if not hit:
                    continue
                a, b = clause_span(vid, text, ch.start, ch.end)
                passage = passage_at(did, title, vid, number, status, text, a, b, (ch.start, ch.end))
                eff = effective(text)
                when = None                          # the clause saying when it applies, when it is not the amending clause
                if eff and not (a <= eff["char_start"] < b):
                    ea, eb = clause_span(vid, text, eff["char_start"], eff["char_end"])
                    when = passage_at(did, title, vid, number, status, text, ea, eb, (eff["char_start"], eff["char_end"]))
                for i in hit:
                    links.append({"index": i, "clause": ch.clause, "how": ch.how, "passage": passage, "when": when,
                                  "signed": status == "executed", "effective": eff})
    return links
