"""/api/legal/matters/{id}/research (S1.7, SEC-03 / RAG-01).

A question is answered only from the matter's own documents: the passages come from the matter's
partition through an AuthorizedScope, and the model sees nothing else. The answer streams as
NDJSON — `sources`, then `chunk`s, then `done` — and the scope is re-checked while it streams: if
the person loses access, the stream ends with `scope_revoked` and nothing more is sent. When no
passage is relevant the answer is an explicit abstention and the model is not called.

The same stream serves the voice service, which calls it with an on-behalf-of token (D-013).
History is private to its owner within the matter.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator, Literal, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note, record
from ..documents import versions as doc_versions
from ..matters.routes import principal_of, require, require_law_edition
from ..retrieval import scope as scopemod
from ..retrieval import search as searchmod
from . import llm as llm_mod
from . import amendments, attribution, dates, evidence, followup, prompt
from . import kinds
from . import stated as statedmod
from . import support as supportmod
from . import working

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


class Turn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(default="", max_length=8000)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: Optional[list[Turn]] = Field(default=None, max_length=6)   # S5.1b: the conversation so far (D-063)
    versions: Literal["signed", "current"] = "signed"     # S2.2a: answer from signed versions (default) or latest drafts
    documents: Optional[list[str]] = Field(default=None, max_length=50)   # S2.6: only these documents (a board pack)


def _event(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _source(n: int, p: dict) -> dict:
    c = p["citation"]
    return {"n": n, "document_id": c["document_id"], "document_title": c["document_title"], "version_id": c["version_id"],
            "version_number": c["version_number"], "page_start": c["page_start"], "page_end": c["page_end"],
            "char_start": c["char_start"], "char_end": c["char_end"], "text": p["text"],
            "clause": c.get("clause"), "line": c.get("line"), "focus_start": c.get("focus_start"), "focus_end": c.get("focus_end"),
            "version_status": c.get("version_status"), "signed": c.get("signed"),
            "instructions": prompt.addressed_to_ai(p["text"]),       # withheld from the model's copy (T08)
            **({"amends": p["amends"]} if p.get("amends") else {}),
            **({"matter_id": p["matter"]["id"], "matter_number": p["matter"]["number"]} if p.get("matter") else {})}


def _save(turn: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_research_turns (id, org_id, matter_id, owner_id, question, answer, citations_json, status, "
                     "auth, created_at, finished_at, evidence_json, conflicts_json, versions, abstention_reason, amendments_json) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (turn["id"], turn["org_id"], turn["matter_id"], turn["owner_id"], turn["question"], turn["answer"],
                      json.dumps(turn["citations"], ensure_ascii=False), turn["status"], turn["auth"], turn["created_at"],
                      policy.now_iso(), json.dumps(turn.get("evidence") or [], ensure_ascii=False),
                      json.dumps(turn.get("conflicts") or [], ensure_ascii=False), turn.get("versions"), turn.get("abstention_reason"),
                      json.dumps(turn.get("amendments") or [], ensure_ascii=False)))
        conn.commit()


def _check(raw: str, sources: list[dict]) -> tuple[dict, list]:
    """One finished paragraph: its display text and its claims with every quote verified (S2.2b). A quote given
    without a citation is cited to the first source, in rank order, that contains it (D-073), never to a passage
    flagged for text addressed to an AI (T08)."""
    def where(quote: str):
        return next((s["n"] for s in sources if not s.get("instructions")
                     and evidence.verify(quote, s["text"]).verification != "failed"), None)

    def relocate(n: int, quote: str):
        # 2026-10-29 QA RAG-002: "[3: "Either party may terminate … 60 days' written notice"]" with one source — the model wrote
        # the clause number (§3) as the source number, and every sentence went unverified. A quote that is not in the source it
        # names is cited where it is, but only when one document holds it; anywhere else, or in several, it stays failed.
        if 1 <= n <= len(sources) and evidence.verify(quote, sources[n - 1]["text"]).verification != "failed":
            return None
        found = [s for s in sources if not s.get("instructions") and evidence.verify(quote, s["text"]).verification != "failed"]
        return found[0]["n"] if found and len({s["document_id"] for s in found}) == 1 else None
    para = evidence.parse(raw, match=where, relocate=relocate)
    claims = []
    for claim in para.claims:
        for c in claim.citations:
            src = sources[c.n - 1] if 1 <= c.n <= len(sources) else None
            if src is None:
                c.verification = "failed"
            elif c.quote is None:
                c.verification = "not_quoted"
            else:
                found = evidence.verify(c.quote, src["text"])
                c.verification, c.start, c.end = found.verification, found.start, found.end
                if c.verification != "failed" and _heading_only(c.quote, claim.sentence):
                    c.verification, c.start, c.end = "failed", None, None
        claims.append({"sentence": claim.sentence, "state": claim.state,
                       "citations": [{"n": c.n, "quote": c.quote, "verification": c.verification, **({"matched": True} if c.matched else {}),
                                      **({"moved_from": c.moved_from} if c.moved_from else {})}
                                     for c in claim.citations]})
    return ({"text": para.display, "claims": claims, "unsupported": prompt.unsupported(para.display, len(sources)),
             "failed": [c["sentence"] for c in claims if c["state"] == "failed"]}, para.claims)


# A sentence that only says where something is ("These details are found in Section 2 of the agreement"), and a statement
# followed by an explicit attribution ("… is 60 days. This is stated in the source as: "… 60 days' written notice" [1]")
# (2026-10-29 QA RAG-001 and RAG-002: both were listed as unsupported beside a verified quote that carries them).
_LOCATION = re.compile(r"\b(?:section|clause|§|paragraph|article|cl\.)\s*(\d+(?:\.\d+)*)", re.I)
_POINTER_WORDS = frozenset("""these this that those the a an all both details detail information terms term it they are is be
can may found set out stated state specified specify provided provide contained contain described describe given listed in of
under within at on and agreement document documents contract source sources excerpt as above below""".split())
_ATTRIBUTES = re.compile(r"^\W*(?:(?:this|that|these|it|they)\s+(?:is|are|was|were)\s+(?:stated|set\s+out|specified|provided|"
                         r"found|confirmed|described|written|recorded|shown)|as\s+(?:stated|set\s+out|provided)\s+in)\b", re.I)
_NUMBER_WORDS = frozenset("""one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty forty forty-five
fifty sixty seventy eighty ninety hundred thousand million""".split())


def _figures(text: str) -> set:
    t = text.lower()
    return set(re.findall(r"\d[\d,.]*\d|\d", t)) | {w for w in re.findall(r"[a-z-]+", t) if w in _NUMBER_WORDS}


def _pointed(text: str, unsupported: list, items: list) -> list:
    """`unsupported` without the sentences a verified quote of the answer demonstrably carries: a pointer to the clause the
    quote is in, and a statement whose every figure is in the quote of the attribution that follows it. Anything else
    stays unsupported. `items` is the answer's evidence, so a saved answer reads the same when it is opened again."""
    verified = [e for e in items if e.get("verification") in ("verbatim", "normalized")]
    clauses = set()
    for e in verified:
        parts = str(e.get("clause") or "").split(".")
        clauses |= {".".join(parts[:k]) for k in range(1, len(parts) + 1) if parts[0]}
    sentences = [x.strip() for x in prompt._sentences(text)]
    out = []
    for s in unsupported:
        if _CITED_MARK.search(s):
            out.append(s)
            continue
        m = _LOCATION.search(s)
        if m and m.group(1) in clauses:
            rest = (s[:m.start()] + " " + s[m.end():]).lower()
            if all(w in _POINTER_WORDS for w in re.findall(r"[a-z]+", rest)):
                continue
        i = sentences.index(s) if s in sentences else -1
        nxt = sentences[i + 1] if 0 <= i < len(sentences) - 1 and _ATTRIBUTES.search(sentences[i + 1]) else ""
        ns = {int(n) for n in re.findall(r"\[(\d{1,2})\]", nxt)}
        quotes = " ".join(e.get("quote") or "" for e in verified if e.get("n") in ns)
        if quotes and _figures(s) and _figures(s) <= _figures(quotes):
            said = {w[:5] for w in re.findall(r"[a-z]{4,}", s.lower())} - {w[:5] for w in _POINTER_WORDS}
            if said & {w[:5] for w in re.findall(r"[a-z]{4,}", quotes.lower())}:
                continue
        out.append(s)
    return out


_CITED_MARK = re.compile(r"\[\d{1,2}\]")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Z &/',-]{1,58}$")           # capitals only: no figures, no sentence


def _heading_only(quote: str, sentence: str) -> bool:
    """A quote that is only a heading ("GOVERNING LAW") is the clause's name, not a statement: it supports a sentence only
    when the sentence speaks of what it names (2026-10-29 QA RAG-003: it was the one verified quote of an answer about
    renewal, and the source showed "cl. 5")."""
    q = quote.strip(" .:;\"'")
    if not _HEADING.match(q) or len(q.split()) > 6:
        return False
    named = {w[:5] for w in re.findall(r"[a-z]{4,}", q.lower())}
    said = {w[:5] for w in re.findall(r"[a-z]{4,}", re.sub(r"\[[^\]]*\]", " ", sentence).lower())}
    return bool(named) and not (named & said)


def _stands(claim) -> bool:
    """A claim an answer stands on: a verified quote, or a cited sentence with words of its own. Bare markers left after
    "The documents do not state …" ("[1] [1] [1]", seen 2026-10-27) support nothing."""
    words = re.search(r"[^\W\d_]{2}", re.sub(r"\[[^\]]*\]", " ", claim.sentence))
    return claim.state == "supported" or (claim.state == "unquoted" and words is not None)


def _carried(claims: list, text: str, unsupported: list) -> tuple[list, list]:
    """A bare marker in a paragraph of its own cites the sentence before it ("… is governed by the laws of Sweden.\n\n[2]",
    2026-11-03 evaluation): the two are one claim, so what the sentence says is checked against the source it points at, and
    a sentence that names the consultancy agreement and points at the reseller agreement does not stand. Before, the sentence
    was unsupported, the marker supported nothing, and nothing was left for the attribution checks to see. Returns the
    claims and, for each joined claim, the sentence it carries."""
    sentences = [x.strip() for x in prompt._sentences(text)]
    out: list = []
    joined: list = []
    at = 0
    for c in claims:
        bare = not re.search(r"[^\W\d_]{2}", re.sub(r"\[[^\]]*\]", " ", c.sentence))
        i = next((k for k in range(at, len(sentences)) if sentences[k] == c.sentence.strip()), None)
        if i is not None:
            at = i + 1
        before = sentences[i - 1] if i else None
        if bare and c.citations and before and before in unsupported:
            c = evidence.Claim(sentence=f"{before} {c.sentence.strip()}", citations=c.citations)
            joined.append((before, c))
        out.append(c)
    return out, joined


def _numbers(scopes) -> list[str]:
    """The numbers of the matters asked about."""
    with get_conn() as conn:
        return [conn.execute("SELECT number FROM matters WHERE id=?", (sc.matter_id,)).fetchone()[0] for sc in scopes]


def _titles(scopes, uid: str) -> dict[str, str]:
    """The titles of the documents the person may read in the matters asked about (document id → title)."""
    from ..documents import repo as doc_repo
    return {d["id"]: d["title"] or "" for sc in scopes for d in doc_repo.list_documents(sc.matter_id, uid)}


_HEADS: dict[tuple[str, str], tuple[set, list, list]] = {}       # (version id, title) → (parties, roles); a version's text is fixed


def _heads(scopes, uid: str) -> dict[str, tuple[set, list, list, set]]:
    """For each document the person may read in the matters asked about (document id →): its parties (the companies of its
    title and preamble, and those it defines by a role), the roles it defines ("Provider", "Client") and the roles whichever
    party acts takes ("the Recipient", with the sentence defining them), and its parties' full names. Each version is read once."""
    from ..documents import repo as doc_repo
    out = {}
    for sc in scopes:
        for d in doc_repo.list_documents(sc.matter_id, uid):
            v = d.get("current_version") or {}
            key = (v.get("id") or "", d.get("title") or "")
            if key not in _HEADS:
                body = doc_repo.reader(v["id"]) if v.get("id") else None
                text = (body or {}).get("text") or ""
                roles = attribution.roles_of(text)
                if len(_HEADS) > 20000:
                    _HEADS.clear()
                _HEADS[key] = (attribution.parties_of(key[1], text) | ({attribution.party_key(n) for n, _ in roles} - {""}), roles,
                               attribution.taken_roles(text), attribution.party_names(key[1], text))
            out[d["id"]] = _HEADS[key]
    return out


MAX_OPENINGS = 2
MAX_PARTY_DOCUMENTS = 3          # the documents the parties a question names share, each searched for its best passage

# Context around what was found (2026-11-07 review point 3): the passage before or after one whose sentence sits at its edge,
# in the same clause; the definitions of the defined terms the best passages use; and the clause a passage is "subject to".
# Each is the document's own passage, numbered and cited like any other source; at most MAX_CONTEXT are added, and none once
# the sources reach MAX_CONTEXT_CHARS, so what the search found always comes first.
MAX_CONTEXT = 4
MAX_CONTEXT_CHARS = 14000
MAX_DEFINITIONS = 2
_EDGE = 150
_SUBJECT = re.compile(r"\b(?:subject\s+to|except\s+as\s+(?:provided|set\s+out|stated)\s+in|notwithstanding|save\s+as\s+(?:provided|set\s+out)"
                      r"\s+in|in\s+accordance\s+with)\s+(?:clause|section|paragraph)s?\s+(\d{1,2}(?:\.\d{1,2}){0,2})", re.I)


def _context(scope, passages: list[dict], versions: str) -> list[dict]:
    if settings.LEGAL_RESEARCH_CONTEXT != "on" or not passages:
        return []
    have = {p["chunk_id"] for p in passages}
    total = sum(len(p["text"]) for p in passages)
    out: list[dict] = []

    def add(found: list[dict], why: str) -> None:
        nonlocal total
        for q in found:
            if len(out) >= MAX_CONTEXT or total + len(q["text"]) > MAX_CONTEXT_CHARS or q["chunk_id"] in have:
                continue
            have.add(q["chunk_id"])
            total += len(q["text"])
            out.append({**q, "context": why})
    top = passages[:3]
    for p in top:                                       # a clause the passage's edge cuts
        c = p["citation"]
        fs, fe, cs, ce = c.get("focus_start"), c.get("focus_end"), c.get("char_start"), c.get("char_end")
        head = (c.get("clause") or "").split(".")[0]
        if None in (fs, fe, cs, ce) or not head:
            continue
        places = ([(c["version_id"], cs - 1)] if fs - cs < _EDGE and cs > 0 else []) + ([(c["version_id"], ce + 1)] if ce - fe < _EDGE else [])
        if places:
            near = searchmod.passages_at(scope, places, versions)
            add([q for q in near if (q["citation"].get("clause") or "").split(".")[0] == head], "continues")
    versions_of = list(dict.fromkeys(p["citation"]["version_id"] for p in top))
    if versions_of:                                     # the definitions of terms the best passages use
        with get_conn() as conn:
            terms = conn.execute(f"SELECT version_id, term, char_start, char_end FROM legal_defined_terms WHERE matter_id=? AND version_id IN "
                                 f"({','.join('?' * len(versions_of))})", (scope.matter_id, *versions_of)).fetchall()
        places = []
        for p in top:
            c = p["citation"]
            for vid, term, a, b in terms:
                if vid != c["version_id"] or len(term) < 3 or not term[:1].isupper() or len(places) >= MAX_DEFINITIONS:
                    continue
                inside = c.get("char_start") is not None and c["char_start"] <= a < (c.get("char_end") or 0)
                if not inside and re.search(rf"\b{re.escape(term)}\b", p["text"]) and (vid, a) not in places:
                    places.append((vid, a))
        if places:
            add(searchmod.passages_at(scope, places, versions), "definition")
    for p in passages[:6]:                              # the clause a passage is subject to
        numbers = list(dict.fromkeys(_SUBJECT.findall(p["text"])))[:2]
        if numbers:
            add(searchmod.clause_passages(scope, numbers, versions, [p["citation"]["document_id"]], limit=2), "subject to")
    return out


def _openings(scope, question: str, passages: list[dict], roles: dict, versions: str) -> list[dict]:
    """The opening passage of each source document (at most two, best ranked first) that defines by a role a party the
    question names, when the sources do not already include it, else that source is marked as the opening. Only documents
    already among the sources: nothing new is brought in."""
    named = attribution.named_in(question, roles)
    have, docs = {p["chunk_id"] for p in passages}, []
    for p in passages:
        c = p["citation"]
        rs = roles.get(c["document_id"]) or []
        if (c["document_id"], c["version_id"]) not in docs and any(attribution.party_key(n) in named for n, _ in rs):
            docs.append((c["document_id"], c["version_id"]))
    out = []
    for did, vid in docs[:MAX_OPENINGS]:
        op = searchmod.opening(scope, vid, [n for n, _ in roles[did]], versions)
        if op and op["chunk_id"] not in have:
            have.add(op["chunk_id"])
            out.append({**op, "opening": True})
        elif op:                                        # found by a search already: it is still the opening (2026-11-05)
            for p in passages:
                if p["chunk_id"] == op["chunk_id"]:
                    p["opening"] = True
    return out


_CLAUSE_CITED = re.compile(r"\b[Cc]lauses?\s+\d")


def _misquotes(passages: list[dict], named: list[str], titles: dict) -> list[dict]:
    """Words another document puts in quotation marks as a named document's, in a sentence that names it, that the named
    document's own text (the version among the sources) does not contain (2026-11-05 brief: the advice memo quoting clause
    4.2 of the MSA as "net sixty (60) days" where the MSA says thirty). Rules only; nothing is asked of the model."""
    from ..documents import repo as doc_repo
    from ..documents import versions as doc_versions
    own, version = {}, {p["citation"]["document_id"]: p["citation"]["version_id"] for p in passages}
    with get_conn() as conn:                            # a named document none of whose passages was retrieved: its signed text
        for did in named:
            if did not in version:
                r = conn.execute(f"SELECT COALESCE({doc_versions._LATEST_SIGNED.format(doc='d.id')}, d.current_version_id) "
                                 "FROM legal_documents d WHERE d.id=? AND d.deleted_at IS NULL", (did,)).fetchone()
                if r and r[0]:
                    version[did] = r[0]
    for did in named:
        body = doc_repo.reader(version[did]) if version.get(did) else None
        if body and body.get("text"):
            own[did] = body["text"]
    out, seen = [], set()
    for p in passages:                                  # any other document's sentence that quotes a named one
        # … or that names none, in a document whose title says it is about one ("Advice memo on the MSA": "Clause 4.2
        # provides that … \"Invoices are payable net sixty (60) days\"", 2026-11-05 end-to-end run)
        of_title = attribution.subject(p["citation"]["document_title"], titles)
        for a, b in prompt.document_sentences(p["text"]):
            sentence = p["text"][a:b]
            named_here = attribution.referenced(sentence, titles) or (of_title if _CLAUSE_CITED.search(sentence) else set())
            if p["citation"]["document_id"] in named_here:       # "This Supply Agreement (the …)": the document speaks of itself
                continue
            about = [d for d in named_here if d in own]
            for q in attribution.quoted_words(sentence):
                for d in about:
                    key = (p["citation"]["document_id"], d, q)
                    if key in seen or evidence.verify(q, own[d]).verification != "failed":
                        continue
                    seen.add(key)
                    out.append({"by": p["citation"]["document_title"], "by_id": p["citation"]["document_id"], "of": titles.get(d, ""),
                                "of_id": d, "quote": q[:300]})
    return out[:3]


_NOT_TERMS = attribution._TYPES | {"party", "parties", "company"}


def _defined_twice(question: str, passages: list[dict], named: list[str], roles: dict) -> list[dict]:
    """Terms the question asks about ("Which price applies to Products?") that a document answering it defines more than once,
    with different meanings: "Price" as the Schedule 2 price in 1.1 and as the Order's price in 5.1 (2026-11-05 end-to-end run;
    the definitions check's own reader, rules only). The documents named, else those of the first passages; at most three."""
    from ..definitions import check as defcheck
    from ..documents import repo as doc_repo
    asked = set(re.findall(r"[a-z]+", (question or "").lower()))
    first: dict[str, dict] = {}
    for p in passages:
        first.setdefault(p["citation"]["document_id"], p["citation"])
    order = list(dict.fromkeys([d for d in named if d in first] + [p["citation"]["document_id"] for p in passages[:4]]))[:3]
    out = []
    with get_conn() as conn:
        for did in order:
            c = first[did]
            text = (doc_repo.reader(c["version_id"]) or {}).get("text") or ""
            own_roles = {r.lower() for _, r in roles.get(did, [])}
            groups: dict[str, list[dict]] = {}
            for d in defcheck.definitions(text):
                if not d.get("signpost"):
                    groups.setdefault(d["term"], []).append(d)
            for term, ds in groups.items():
                words = re.findall(r"[a-z]+", term.lower())
                if len(ds) < 2 or len({d["meaning"] for d in ds}) < 2 or not words or term.lower() in own_roles \
                        or set(words) & _NOT_TERMS or not all(w in asked or f"{w}s" in asked for w in words):
                    continue
                out.append({"document_id": did, "document_title": c["document_title"], "version_id": c["version_id"], "term": term,
                            "definitions": [{"start": d["sentence"][0], "end": d["sentence"][1],
                                             "text": " ".join(text[d["sentence"][0]:d["sentence"][1]].split())[:300],
                                             "clause": searchmod.locate(conn, c["version_id"], d["start"], d["end"]).get("clause")}
                                            for d in ds[:3]]})
    return out[:2]


def _twice_answer(twice: list[dict], sources: list[dict]) -> str:
    """The definitions of a term the document defines twice, each quoted from its own source, as an answer (the model declined).
    Each quote is the longest run of the definition without quotation marks, so the marker reads like any other."""
    def run_of(words: str) -> str:
        return max((p.strip(" ,;.") for p in re.split(r"[\"\u201c\u201d]", words)), key=len, default="")

    out = []
    for t in twice:
        quoted = []
        for d in t["definitions"]:
            q = run_of(d["text"])
            n = next((x["n"] for x in sources if x["version_id"] == t["version_id"]
                      and evidence.verify(q, x["text"]).verification != "failed"), None) if len(q) >= 12 else None
            if n:
                quoted.append((d.get("clause"), n, q))
        if len(quoted) < 2:
            continue
        (c1, n1, q1), rest = quoted[0], quoted[1:]
        lines = [f'{t["document_title"]} defines "{t["term"]}"{f" in clause {c1}" if c1 else ""}: \u201c{q1}\u201d [{n1}: "{q1}"].']
        lines += [f'It defines "{t["term"]}" again{f" in clause {c}" if c else ""}, differently: \u201c{q}\u201d [{n}: "{q}"].'
                  for c, n, q in rest]
        lines.append(f'The document does not say which definition applies, so it is ambiguous on this point '
                     f'[{n1}: "{q1}"] [{rest[0][1]}: "{rest[0][2]}"].')
        out.append(" ".join(lines))
    return "\n\n".join(out)


def _version_texts(version_ids) -> list[str]:
    """The stored text of each cited version: the whole document, not only the passage."""
    from ..documents import repo as doc_repo
    out = []
    for vid in sorted(v for v in version_ids if v):
        body = doc_repo.reader(vid)
        if body and body.get("text"):
            out.append(body["text"])
    return out


def _calculation(sources: list[dict], question: str, titles: dict) -> Optional[dict]:
    """Term and notice dates calculated from one document's stated words (2026-10-29 QA RAG-003): the document the question
    names, or else the best-ranked one among the sources that states a start and a term. Every fact keeps its exact words,
    clause and characters; every derived date says it is calculated and on what assumptions. Only sources already
    authorized for this answer are read."""
    from ..documents import repo as doc_repo
    named = attribution.referenced(question, titles)
    order = []
    for s in sources:
        if (s["document_id"], s["version_id"]) not in order and (not named or s["document_id"] in named) and not s.get("instructions"):
            order.append((s["document_id"], s["version_id"]))
    with get_conn() as conn:
        for document_id, version_id in order:
            body = doc_repo.reader(version_id)
            text = (body or {}).get("text") or ""
            calc = dates.calculate(text) if text else None
            if not calc or not (calc.derived or calc.not_calculated):
                continue
            src = next(s for s in sources if s["version_id"] == version_id)
            facts = []
            for f in calc.facts:
                where = searchmod.locate(conn, version_id, f.start, f.end)
                facts.append({"label": f.label, "value": f.value, "quote": f.words(text), "clause": where.get("clause"),
                              "char_start": f.start, "char_end": f.end})
            return {"document_id": document_id, "document_title": src["document_title"], "version_id": version_id,
                    "version_number": src.get("version_number"), **({"matter_number": src["matter_number"]} if src.get("matter_number") else {}),
                    "facts": facts, "derived": calc.derived, "assumptions": calc.assumptions, "not_calculated": calc.not_calculated}
    return None


def _evidence(claims: list, sources: list[dict]) -> list[dict]:
    """Every quote the answer relies on, with where it is: document, version and its status, clause,
    page, line and exact characters of the version's text."""
    items, seen, cache = [], {}, {}
    with get_conn() as conn:
        for claim in claims:
            for c in claim.citations:
                if not 1 <= c.n <= len(sources):
                    continue
                if (c.n, c.quote) in seen:                 # the same quote for another sentence: it supports that one too
                    said = items[seen[(c.n, c.quote)]]["sentences"]
                    if claim.sentence not in said:
                        said.append(claim.sentence)
                    continue
                seen[(c.n, c.quote)] = len(items)
                src, base = sources[c.n - 1], sources[c.n - 1]["char_start"]
                if c.verification in ("verbatim", "normalized") and base is not None:
                    a, b, quote = base + c.start, base + c.end, src["text"][c.start:c.end]
                elif c.verification == "not_quoted" and src.get("focus_start") is not None and base is not None:
                    a, b = src["focus_start"], src["focus_end"]
                    quote = src["text"][a - base:b - base]
                else:
                    a = b = None
                    quote = c.quote or ""
                where = searchmod.locate(conn, src["version_id"], a, b, cache) if a is not None else {}
                items.append({"n": c.n, "quote": quote, "verification": c.verification, "document_id": src["document_id"],
                              "document_title": src["document_title"], "version_id": src["version_id"],
                              "version_number": src["version_number"], "version_status": src.get("version_status"),
                              "signed": src.get("signed"), "clause": where.get("clause") or src.get("clause"),
                              "page_start": where.get("page_start", src["page_start"]), "page_end": where.get("page_end", src["page_end"]),
                              "line": where.get("line"), "char_start": a, "char_end": b, "sentences": [claim.sentence],
                              **({"matched": True} if c.matched else {}), **({"moved_from": c.moved_from} if c.moved_from else {})})
    return items


def _amended(passages: list[dict], links: list[dict]) -> list[dict]:
    """Add each amending passage (at most amendments.MAX_ADDED) to the sources, marked with what it changes; returns the links
    with the source index of the amending passage."""
    out, added = [], 0
    for link in links:
        p, target = link["passage"], passages[link["index"]]["citation"]
        at = next((i for i, q in enumerate(passages) if q["citation"]["version_id"] == p["citation"]["version_id"]
                   and q["citation"].get("char_start") is not None and q["citation"]["char_start"] < p["citation"]["char_end"]
                   and p["citation"]["char_start"] < q["citation"]["char_end"]), None)
        if at is None:
            if added >= amendments.MAX_ADDED:
                continue
            passages.append(p)
            at, added = len(passages) - 1, added + 1
        passages[at]["amends"] = {"document_title": target["document_title"], "clause": link["clause"], "how": link["how"],
                                  "signed": link["signed"], "effective": (link["effective"] or {}).get("words")}
        w = link.get("when")                       # the amendment's own words on when it applies, citable too
        if w and added < amendments.MAX_ADDED and not any(
                q["citation"]["version_id"] == w["citation"]["version_id"] and q["citation"].get("char_start") is not None
                and q["citation"]["char_start"] < w["citation"]["char_end"] and w["citation"]["char_start"] < q["citation"]["char_end"]
                for q in passages):
            passages.append(w)
            added += 1
        out.append({**link, "by_index": at})
    return out


def _relies_on(n: int, rng: tuple[int, int], cited: list[int], evidence: list[dict]) -> bool:
    """The answer relies on characters [rng) of source n: a quote it gives from n lands there, or it cites n without a
    located quote (then it cannot be told apart, so it counts)."""
    located = [e for e in evidence if e.get("n") == n and e.get("char_start") is not None and e.get("char_end") is not None]
    if located:
        return any(e["char_start"] < rng[1] and rng[0] < e["char_end"] for e in located)
    return n in cited


def _changed(links: list[dict], sources: list[dict], cited: list[int], evidence: list[dict]) -> list[dict]:
    """What the answer should know: each changed clause it relies on (or each amendment it cites), whether the amendment is
    signed, when it says it applies, and whether the answer cites the amendment too."""
    out, seen = [], set()
    for link in links:
        t, a = sources[link["index"]], sources[link["by_index"]]
        tn, an = link["index"] + 1, link["by_index"] + 1
        relies = _relies_on(tn, link["target_range"], cited, evidence)
        if (tn, an) in seen or (not relies and an not in cited):
            continue
        seen.add((tn, an))
        eff = link["effective"]
        own = link["passage"]["citation"]          # the amending clause itself, even when a wider retrieved passage holds it
        out.append({"document_id": t["document_id"], "document_title": t["document_title"], "version_id": t["version_id"],
                    **({"matter_number": t["matter_number"]} if t.get("matter_number") else {}),
                    "clause": link["clause"], "n": tn, "how": link["how"], "cited": relies, "amendment_cited": an in cited,
                    "by": {"document_id": own["document_id"], "document_title": own["document_title"], "version_id": own["version_id"],
                           "version_number": own["version_number"], "clause": own.get("clause"), "signed": link["signed"], "n": an,
                           "char_start": own["char_start"], "char_end": own["char_end"],
                           "effective": ({k: eff[k] for k in ("kind", "date", "words")} if eff else None)}})
    return out


def _choices(kind: Optional[tuple], unclear: Optional[tuple], passages: list[dict], titles: dict[str, str]) -> Optional[dict]:
    """What the question could mean (2026-11-07 review point 10): the documents of the kind it names, the most relevant
    first, and the parties a name it gives could be, in the documents' own spelling."""
    out = {}
    if kind:
        seen = list(dict.fromkeys(p["citation"]["document_id"] for p in passages if p["citation"]["document_id"] in kind[1]))
        ordered = seen + [d for d in kind[1] if d not in seen]
        out.update(kind=kind[0], count=len(kind[1]), documents=[{"id": d, "title": titles.get(d) or ""} for d in ordered[:12]])
    if unclear:
        texts = list(titles.values()) + [p["text"] for p in passages]

        def spelled(name: str) -> str:
            for t in texts:
                m = re.search(rf"\b{re.escape(name)}\b(?:,?[ \t]+(?:{attribution._FORMS})\b\.?)?", t, re.I)
                if m:
                    return m.group(0)
            return name.title()
        out["parties"] = {"said": unclear[0], "names": [spelled(n) for n in unclear[1][:4]]}
    return out or None


def _status(abstained: bool, claims: list, unsupported: list, failed: list) -> str:
    if abstained:
        return "abstained"
    if not any(_stands(c) for c in claims):            # a bare marker supports nothing (2026-11-03)
        return "unverified"
    return "partial" if failed or unsupported else "answered"


async def _search_several(scopes: list, principal: dict, question: str, versions: str, documents: Optional[list[str]]):
    """A conversation about several matters (2026-10-28): each matter is searched under its own authorized scope, its
    amendment links are found inside that matter, and the groups are put together with every passage naming its matter.
    A matter with nothing relevant adds nothing."""
    k = max(3, settings.LEGAL_RESEARCH_PASSAGES // len(scopes) + 1)
    groups = []
    with get_conn() as conn:
        numbers = {sc.matter_id: conn.execute("SELECT number FROM matters WHERE id=?", (sc.matter_id,)).fetchone()[0] for sc in scopes}
    for sc in scopes:
        r = await searchmod.search(sc, question, k=k, versions=versions, documents=documents, wide=True)
        ps = r["passages"]
        if r["abstained"] or not ps:
            continue
        ls = _amended(ps, await run_in_threadpool(amendments.for_passages, sc.matter_id, principal["sub"], ps, versions))
        for p in ps:
            p["matter"] = {"id": sc.matter_id, "number": numbers[sc.matter_id]}
        best = max((float(p["score"]) for p in ps if p.get("score") is not None), default=float("-inf"))
        groups.append((best, ps, ls))
    passages, links = [], []
    for _, ps, ls in sorted(groups, key=lambda g: -g[0]):      # the most relevant matter first
        offset = len(passages)
        passages.extend(ps)
        links.extend({**l, "index": l["index"] + offset, "by_index": l["by_index"] + offset} for l in ls)
    return passages, links, not passages


async def _answer(scope, principal: dict, question: str, turn: dict, versions: str = "signed",
                  documents: Optional[list[str]] = None, understood: Optional[dict] = None,
                  extra_scopes: tuple = ()) -> AsyncIterator[str]:
    status, answer, citations, items, found, reason, changed = "error", None, [], [], [], None, []
    scopes = [scope, *extra_scopes]
    several = len(scopes) > 1
    try:
        if understood:                                      # S5.1b (D-063): what a follow-up was taken to ask
            yield _event({"type": "understood", **understood})
        # 2026-11-03 evaluation: the documents' parties and defined roles. The search reads "Harrowgate (Provider)" where the
        # question says "Harrowgate", and the model is given the opening passage saying who "the Provider" is; a question
        # about "the supply agreement with Alder Retail" also searches the documents Alder Retail is party to.
        heads = await run_in_threadpool(_heads, scopes, principal["sub"])
        roles = {did: h[1] for did, h in heads.items() if h[1]}
        taken = {did: h[2] for did, h in heads.items() if h[2]}
        known = set().union(*(h[0] for h in heads.values())) if heads else set()
        search_q = attribution.with_roles(question, roles)
        asked_parties = attribution.named_parties(question, known)
        target = [] if documents or not asked_parties else [did for did, h in heads.items() if asked_parties <= h[0]]
        titles = await run_in_threadpool(_titles, scopes, principal["sub"])
        same = None if several or documents else kinds.same_kind(question, titles, target if asked_parties else None)
        # More of that kind than are answered for each, or a name several parties' names begin with: the person is offered
        # the choice (2026-11-07 review point 10), never asked to choose before an answer
        many = None if several or documents or same else kinds.many_of_kind(question, titles)
        unclear = None if several or documents else attribution.ambiguous_party(question, set().union(*(h[3] for h in heads.values()))
                                                                                     if heads else set())
        # A document the question names ("the payment terms in the Master Services Agreement"): its own passages are searched
        # as well and come first, so another document that describes it (an advice memo quoting it) cannot take its place;
        # that one stays among the sources, marked secondary (2026-11-05 brief: the memo's misquote answered for the MSA).
        # Not when the person chose the documents (their choice stands) or several matters are asked about.
        named_docs = [] if several or documents else sorted(attribution.referenced(question, titles))
        if several:
            passages, links, nothing_found = await _search_several(scopes, principal, search_q, versions, documents)
        else:
            result = await searchmod.search(scope, search_q, k=settings.LEGAL_RESEARCH_PASSAGES, versions=versions, documents=documents,
                                            wide=True)
            passages, nothing_found = result["passages"], result["abstained"]
            if target and len(target) < len(heads):
                named = await searchmod.search(scope, search_q, k=2, versions=versions, documents=target, wide=True)
                have = {p["chunk_id"] for p in named["passages"]}
                passages = named["passages"] + [p for p in passages if p["chunk_id"] not in have]
                nothing_found = nothing_found and named["abstained"]
            if (said := attribution.mentioned(question, known)):     # the clause's own words, without the names (2026-11-05)
                among = documents or target or [did for did, h in heads.items() if said <= h[0]] or None
                few = among if among and len(among) <= MAX_PARTY_DOCUMENTS else None
                got = await searchmod.search(scope, attribution.without_parties(question, said), k=20 if few else 2,  # all reranked anyway
                                             versions=versions, documents=among, wide=True)
                # the best two, and the best of each of a few documents the parties share, so one cannot crowd out the rest
                # ("What fee arrangement did Voss-Lindqvist ask Calder & Wynne for?": two memo passages, never the engagement
                # letter's "We would like a fixed fee of USD 28,000", end-to-end run 2026-11-05)
                best = got["passages"][:2]
                if few:
                    shown = {p["citation"]["document_id"] for p in best}
                    for p in got["passages"][2:]:
                        if p["citation"]["document_id"] not in shown:
                            best.append(p)
                            shown.add(p["citation"]["document_id"])
                have = {p["chunk_id"] for p in passages}
                passages = passages + [p for p in best if p["chunk_id"] not in have]
                nothing_found = nothing_found and got["abstained"]
            if named_docs and len(named_docs) < len(titles):
                first = []                                  # each named document on its own, so one cannot crowd out another
                inside = attribution.without_titles(search_q, [titles[d] for d in named_docs]) or search_q
                for did in named_docs[:3]:
                    own = await searchmod.search(scope, inside, k=settings.LEGAL_RESEARCH_NAMED_PASSAGES if len(named_docs) == 1 else 2,
                                                 versions=versions, documents=[did], wide=True)
                    first += [p for p in own["passages"] if p["chunk_id"] not in {q["chunk_id"] for q in first}]
                    nothing_found = nothing_found and own["abstained"]
                have = {p["chunk_id"] for p in first}
                passages = first + [p for p in passages if p["chunk_id"] not in have]
            # A clause the question names by number ("What does clause 14 cover?", "clauses 4.1, 4.2 and 4.3"): its own passage,
            # from the named document or else the matter's contracts, is read first (2026-11-05 end-to-end run).
            if (numbers := searchmod.clause_numbers(question)):
                held = await run_in_threadpool(searchmod.clause_passages, scope, numbers, versions, documents or named_docs or None)
                if not held and named_docs and not documents:          # the named document has no such clause: the contracts
                    held = await run_in_threadpool(searchmod.clause_passages, scope, numbers, versions, None)
                have = {p["chunk_id"] for p in held}
                passages = held + [p for p in passages if p["chunk_id"] not in have]
            for side in prompt.compare_sides(question):     # each side of a comparison on its own (2026-11-04)
                got = await searchmod.search(scope, attribution.with_roles(side, roles), k=2, versions=versions, documents=documents,
                                             wide=True)
                have = {p["chunk_id"] for p in passages}
                passages = passages + [p for p in got["passages"] if p["chunk_id"] not in have]
                nothing_found = nothing_found and got["abstained"]
            for did in (same[1] if same else []):           # each document of a kind several share (research/kinds.py)
                got = await searchmod.search(scope, search_q, k=1, versions=versions, documents=[did], wide=True)
                have = {p["chunk_id"] for p in passages}
                passages = passages + [p for p in got["passages"] if p["chunk_id"] not in have]
                nothing_found = nothing_found and got["abstained"]
            passages = passages + await run_in_threadpool(_openings, scope, question, passages, roles, versions)
            passages = passages + await run_in_threadpool(_context, scope, passages, versions)
            links = _amended(passages, await run_in_threadpool(amendments.for_passages, scope.matter_id, principal["sub"], passages, versions))
        # A named document (above): another document's words it puts in quotation marks as the named one's are checked against
        # the named document's own text, by rule; where they are not there, the named document's passage nearest to them is read
        # too, and the model is told, so the answer can set the two side by side (2026-11-05 brief).
        misquoted: list[dict] = []
        named_note = ""
        if named_docs and any(p["citation"]["document_id"] in named_docs for p in passages):
            misquoted = await run_in_threadpool(_misquotes, passages, named_docs, titles)
            for m in misquoted:
                near = await searchmod.search(scope, m["quote"], k=1, versions=versions, documents=[m["of_id"]], wide=True)
                passages += [q for q in near["passages"] if q["chunk_id"] not in {x["chunk_id"] for x in passages}]
                named_note += (f'(Checked against the documents: "{m["by"]}" quotes "{m["of"]}" as "{m["quote"]}"; those words are '
                               f'not in "{m["of"]}".)\n')
            # an amendment of the named document is part of its terms, not another document's description of them
            amending = {link["passage"]["citation"]["document_id"] for link in links
                        if passages[link["index"]]["citation"]["document_id"] in named_docs}
            for p in passages:
                if p["citation"]["document_id"] not in named_docs and p["citation"]["document_id"] not in amending and not p.get("amends"):
                    p["secondary"] = True
            named_note += prompt.named_note([titles[d] for d in named_docs if d in titles])
        # A term the question asks about that the answering document defines twice, with different meanings: both definitions
        # are read and the model is told the document is ambiguous on it (2026-11-05 end-to-end run: "Which price applies?").
        twice = [] if several else await run_in_threadpool(_defined_twice, question, passages, named_docs, roles)
        if twice:
            held = await run_in_threadpool(searchmod.passages_at, scope, [(t["version_id"], d["start"]) for t in twice
                                                                           for d in t["definitions"]], versions)
            passages += [q for q in held if q["chunk_id"] not in {x["chunk_id"] for x in passages}]
            named_note += "".join(prompt.twice_note(t["document_title"], t["term"], [d["text"] for d in t["definitions"]]) for t in twice)
        # Figures worked out by rule (research/working.py, 2026-11-05 end-to-end run): a planning call writes the arithmetic
        # over the numbers the sources state, each number is found in a source with its unit, and the arithmetic is done
        # there; the answer model is given the results. The provisions the sources say the figure is worked out under, a short
        # document the question names and the agreements' best passages are read first; what the plan says is missing is
        # searched for in the documents asked about. One matter at a time.
        worked = None
        if not several and working.asks(question):
            inside = documents or named_docs or None
            # the provisions the sources say the figure is worked out under ("calculated in accordance with paragraph 3 of
            # Schedule 2"), read too: the rule is often in a schedule no search for the question finds
            if (refs := working.references(question, passages)):
                held = await run_in_threadpool(searchmod.provision_passages, scope, refs, versions, documents)
                passages = passages + [q for q in held if q["chunk_id"] not in {x["chunk_id"] for x in passages}]
            asked_docs = documents or named_docs or working.named_in(question, titles)
            for did in asked_docs[:2]:
                held = await run_in_threadpool(searchmod.document_passages, scope, did, versions)   # a short one read whole
                passages = passages + [q for q in held if q["chunk_id"] not in {x["chunk_id"] for x in passages}]
            # the rule a calculation applies is an agreement's: their best passages are read too, so letters working out
            # their own figures cannot crowd it out (end-to-end run: the service credit clause, never retrieved)
            contracts = [] if documents else await run_in_threadpool(searchmod.kind_ids, scope)
            if contracts:
                got = await searchmod.search(scope, search_q, k=2, versions=versions, documents=contracts, wide=True)
                passages = passages + [q for q in got["passages"] if q["chunk_id"] not in {x["chunk_id"] for x in passages}]

            async def more(need: str) -> list[dict]:
                got = await searchmod.search(scope, attribution.with_roles(need, roles), k=2, versions=versions, documents=inside,
                                             wide=True)
                return got["passages"]
            worked, passages = await working.work(question, passages, more, frozenset(asked_docs))
            if worked:
                named_note += working.note(worked)
        sources = [_source(n, p) for n, p in enumerate(passages, start=1)]
        choices = _choices(same or many, unclear, passages, titles)
        yield _event({"type": "sources", "citations": sources})
        if nothing_found or not passages:
            status, answer, reason = "abstained", prompt.ABSTAIN, "no_passages"
            yield _event({"type": "chunk", "text": prompt.ABSTAIN})
            yield _event({"type": "done", "answer": prompt.ABSTAIN, "status": status, "abstained": True, "abstention_reason": reason,
                          "citations": [], "evidence": [], "conflicts": [], "unsupported": [], "failed": [], "uncited": False,
                          "turn_id": turn["id"], "versions": versions, "absence": None})
            return
        # A question about another matter by its number is not answered from this one's documents (research/attribution.py)
        numbers = await run_in_threadpool(_numbers, scopes)
        foreign = attribution.foreign_matters(question, numbers, [p["text"] for p in passages] + list(titles.values()))
        if foreign:
            own = ", ".join(numbers)
            status, answer, reason = "abstained", (
                f"The question refers to {foreign[0][:20]}, which is not the matter asked about ({own}), and no document found "
                f"mentions it, so nothing is answered for it. Answers here come only from {own}; to work on another matter, "
                "start a conversation with it."), "other_matter"
            yield _event({"type": "chunk", "text": answer})
            yield _event({"type": "done", "answer": answer, "status": status, "abstained": True, "abstention_reason": reason,
                          "citations": [], "evidence": [], "conflicts": [], "unsupported": [], "failed": [], "uncited": False,
                          "turn_id": turn["id"], "versions": versions, "absence": answer, "stated": None})
            return
        buf, shown, claims, unsupported, failed, last_check = "", [], [], [], [], time.monotonic()
        several_of = kinds.note(same[0], [titles[d] for d in same[1]]) if same else ""
        several_of += named_note
        stream = llm_mod.get().chat_stream(prompt.build(question, passages, several_of, calculate=False if worked else None),
                                           temperature=settings.LEGAL_RESEARCH_TEMPERATURE,
                                           max_tokens=settings.LEGAL_RESEARCH_MAX_TOKENS)

        def finish(raw: str):
            para, para_claims = _check(prompt.clean(raw), sources)
            claims.extend(para_claims)
            unsupported.extend(para["unsupported"])
            failed.extend(para["failed"])
            shown.append(para["text"])
            return para

        async for piece in stream:
            if time.monotonic() - last_check >= settings.LEGAL_STREAM_RECHECK_SEC:
                for sc in scopes:
                    await run_in_threadpool(scopemod.recheck, sc)       # raises ScopeRevoked
                last_check = time.monotonic()
            buf += piece
            while "\n\n" in buf:                                       # a paragraph is shown only once it is checked
                raw, buf = buf.split("\n\n", 1)
                if prompt.clean(raw).strip():
                    para = finish(raw)
                    yield _event({"type": "paragraph", "index": len(shown) - 1, **para})
                    yield _event({"type": "chunk", "text": ("\n\n" if len(shown) > 1 else "") + para["text"]})
        for sc in scopes:
            await run_in_threadpool(scopemod.recheck, sc)
        if prompt.clean(buf).strip():
            para = finish(buf)
            yield _event({"type": "paragraph", "index": len(shown) - 1, **para})
            yield _event({"type": "chunk", "text": ("\n\n" if len(shown) > 1 else "") + para["text"]})
        text = "\n\n".join(shown)
        claims, joined = _carried(claims, text, unsupported)
        # Abstention in other words (S2.2, D-038):
        # - nothing is supported, and the text says the documents do not answer; or
        # - the answer opens by saying the documents do not cover the question, and every passage it
        #   cites is only weakly related to the question; or
        # - the answer opens by saying the documents do not cover the question, every cited sentence after it only says what
        #   the documents lack or merely mention, and a word of the question is in none of the documents it cites
        #   (2026-10-28): that is the model declining, with context. When every word of the question is in those documents,
        #   the answer stands as before (a literal "the documents do not say when he said it" that then gives the time; "no
        #   deposit for a warehouse … the only mention of a deposit is GBP 21,000", in the warehouse lease).
        # A supported answer that also says the abstention sentence is still an answer.
        # Attribution (2026-10-28, held-out 2 h2-abstain-storage-law; research/attribution.py): a cited sentence that names a
        # document of the matter must cite that document, or it does not stand; and a question that names a document is not
        # answered from other documents alone.
        # A citation speaks of a document when it is from that document, or its passage itself names it (the board minutes
        # authorising "the renewal of the Harrowgate Master Services Agreement" speak of the MSA).
        speaks_of: dict[int, set] = {}

        def spoken(claim) -> set:
            out = set()
            for c in claim.citations:
                if 1 <= c.n <= len(sources):
                    if c.n not in speaks_of:
                        speaks_of[c.n] = {sources[c.n - 1]["document_id"]} | attribution.referenced(sources[c.n - 1]["text"], titles)
                    out |= speaks_of[c.n]
            return out
        # A party's own record (its board, officers, offices, finances, advisers) rests only on a document that party is party
        # to (2026-10-28, held-out 3: Voss-Lindqvist's next board meeting given as Kestrel's). Parties are read only when the
        # question or a sentence speaks of one's record.
        parties: dict[str, set] = {did: h[0] for did, h in heads.items()}

        def of_parties(claim) -> set:
            return set().union(*(parties.get(sources[c.n - 1]["document_id"], set()) for c in claim.citations
                                 if 1 <= c.n <= len(sources))) if claim.citations else set()

        def party_to(claim, wanted: set) -> bool:
            """Some document the claim cites has every one of `wanted` as a party."""
            return any(wanted <= parties.get(sources[c.n - 1]["document_id"], set()) for c in claim.citations if 1 <= c.n <= len(sources))
        def names_all(claim, wanted: set) -> bool:
            """Some passage the claim cites names every one of `wanted` itself."""
            return any(all(re.search(rf"\b{re.escape(p)}", sources[c.n - 1]["text"], re.I) for p in wanted)
                       for c in claim.citations if 1 <= c.n <= len(sources))
        maybe_party = attribution.mentions_record(question) or any(attribution.mentions_record(c.sentence) for c in claims)
        misattributed = []
        for claim in claims:
            named = attribution.referenced(claim.sentence, titles)
            if named and claim.citations and not (spoken(claim) & named):
                misattributed.append(claim)
            elif maybe_party and claim.citations and (owners := attribution.party_records(claim.sentence, known)) \
                    and not (owners & of_parties(claim)):
                misattributed.append(claim)
            # "The price under the supply agreement with Alder Retail is GBP 120,000", resting on an agreement Alder Retail
            # is not party to (2026-11-03 evaluation)
            # A passage that itself names them all still speaks of them (minutes approving "the supply agreement between
            # Tidewater and Northwind").
            elif claim.citations and (said := attribution.named_parties(claim.sentence, known)) and \
                    any(said <= h[0] for h in heads.values()) and not party_to(claim, said) and not names_all(claim, said):
                misattributed.append(claim)
        unsupported.extend(c.sentence for c in misattributed if c.sentence not in unsupported)
        standing = [c for c in claims if _stands(c) and c not in misattributed]
        for said, claim in joined:                     # a sentence its marker now carries is no longer unsupported
            if claim in standing and said in unsupported:
                unsupported.remove(said)
        asked = attribution.referenced(question, titles)
        # The named document's own passages (and its amendments) speak for it, and so does a passage of another document that
        # names it (minutes approving "the renewal of the MSA", a draft amendment changing its clause 4.1) — unless that
        # document misquotes it: an advice memo whose quote of the MSA the MSA does not hold is not the MSA (2026-11-05 brief;
        # requiring the own text always withheld "Which clauses would the draft amendment change?", end-to-end run).
        of_asked = set(asked) | {link["passage"]["citation"]["document_id"] for link in links       # and its amendments
                                 if link["index"] < len(sources) and sources[link["index"]]["document_id"] in asked}
        misquoting = {m["by_id"] for m in misquoted}

        def for_asked(claim) -> bool:
            cited = {sources[c.n - 1]["document_id"] for c in claim.citations if 1 <= c.n <= len(sources)}
            return bool(cited & of_asked) or (not cited & misquoting and bool(spoken(claim) & asked))
        owner = attribution.party_records(question, known) if maybe_party else set()
        answering = [c for c in standing if not prompt.only_context(c.sentence)]
        off_target = (
            (bool(asked) and bool(standing) and not any(for_asked(c) for c in standing))
            # every sentence that would have stood named one document and cited another, and nothing else is left (held-out 3)
            or (bool(misattributed) and not standing)
            # "not stated" for the document asked about, then only another document's terms (held-out 3: the MSA's 48 hours
            # for a question about the Kestrel storage agreement)
            or (bool(asked) and prompt.opens_not_stated(text) and bool(answering)
                and not any(for_asked(c) for c in answering))
            # a question about the document a party is party to ("the supply agreement with Alder Retail"), answered only
            # from documents that party is not party to (2026-11-03 evaluation)
            or (bool(target) and bool(answering) and not any(party_to(c, asked_parties) for c in answering)))
        off_party = bool(owner) and bool(standing) and not any(owner & of_parties(c) for c in standing)
        nothing = not standing
        # A named legal concept the cited documents never deal with, while another retrieved document does (held-out 4: a
        # break notice given as the notice of non-renewal).
        elsewhere = []
        if standing and attribution.asks_concept(question):
            cited_docs = {sources[c.n - 1]["document_id"] for claim in standing for c in claim.citations if 1 <= c.n <= len(sources)}
            cited_versions = {s["version_id"] for s in sources if s["document_id"] in cited_docs}
            others = [s["text"] for s in sources if s["document_id"] not in cited_docs]
            if others:
                elsewhere = attribution.concepts_elsewhere(question, await run_in_threadpool(_version_texts, cited_versions), others)
        declined = False
        # The result worked out by rule is in the answer: "the documents do not state the credit; worked out, it is USD 1,850"
        # says the result is not stated, which is true, and is not a decline.
        given = bool(worked) and bool(answering) and working.given(worked, text)
        if not given and (prompt.opens_not_stated(text) or (not answering and prompt.opens_absent(text))):
            cited = {sources[c.n - 1]["version_id"] for claim in claims for c in claim.citations if 1 <= c.n <= len(sources)}
            texts = await run_in_threadpool(_version_texts, cited)
            missing = prompt.missing_terms(question, texts) if texts else []
            # "Not stated", then only context; or "not stated", then a true but unrelated cited fact (EV14, 2026-10-28: "Who
            # chairs Harrowgate's board?" — "Thomas Reyes is General Counsel [1]"), when the documents cited hold none of
            # several of the question's own words. One missing word is often the model's paraphrase ("responsible" for
            # "shall keep … in repair"), so a fact that stands is set aside only on two.
            declined = bool(missing) and (not answering or len(missing) >= prompt.MISSING_TO_SET_ASIDE)
        cited_scores = [passages[c.n - 1].get("score") for claim in claims for c in claim.citations
                        if 1 <= c.n <= len(passages) and passages[c.n - 1].get("score") is not None]
        weak = not cited_scores or max(float(x) for x in cited_scores) < settings.LEGAL_ABSTAIN_WEAK_SCORE
        # "Not stated", then cited facts: a decline with context or a hedged answer, which the words cannot tell apart; the
        # passages those facts cite are asked whether they state what was asked (research/stated.py, 2026-11-04). The checks
        # of which document and party a sentence rests on come first and are never undone by it.
        mode = settings.LEGAL_RESEARCH_STATED_CHECK
        hedged = prompt.opens_not_stated(text) or prompt.opens_silent(text)
        stated = shadow = None
        # A term's dates are worked out and labelled by research/dates.py (QA RAG-003), never declined for not being stated.
        if mode in ("hedged", "all", "shadow") and answering and (hedged or mode != "hedged") \
                and not (off_target or off_party or elsewhere) and not dates.asks_term_dates(question) and not given:
            cited_n = dict.fromkeys(c.n for claim in answering for c in claim.citations if 1 <= c.n <= len(passages))
            cited_docs = dict.fromkeys(passages[n - 1]["citation"]["document_id"] for n in cited_n)
            shown = [passages[n - 1] for n in cited_n] + [p for n, p in enumerate(passages, start=1)
                                                          if p.get("opening") and n not in cited_n]
            defined = [f'in "{titles.get(did) or did}", ' + ", ".join(f'"the {role}" is {name}' for name, role in roles[did])
                       for did in cited_docs if roles.get(did)]
            defined += [f'in "{titles.get(did) or did}", either party may be ' + " or ".join(f'"the {r}"' for r in rs)
                        + f', as it defines: "{sentence}"' for did in cited_docs for rs, sentence in taken.get(did, [])]
            verdict = await statedmod.answers(question, shown, defined)
            if hedged or mode == "all":
                stated = verdict
                if stated is not None:
                    declined = not stated
            else:
                shadow = verdict
        abstained = (nothing and (prompt.is_abstention(text) or prompt.says_not_stated(text))) or \
            (prompt.says_not_stated(text) and weak and stated is not True and not given) or declined or off_target or off_party \
            or bool(elsewhere)
        # The model declined a question about a term the answering document defines twice, with different meanings: the
        # definitions, in the document's own words and verified like any quote, are the answer (2026-11-05 end-to-end run:
        # "Which price applies to Products under the Tidewater supply agreement?" declined with both definitions in its sources).
        # So with figures worked out by rule: the working, each step quoting the figures it rests on, is the answer.
        for ruled in ((_twice_answer(twice, sources) if twice else ""), (working.answer(worked) if worked else "")):
            if not (abstained and ruled) or off_target or off_party or elsewhere:
                continue
            para, para_claims = _check(ruled, sources)
            if para_claims and not para["failed"] and all(c.verification in ("verbatim", "normalized")
                                                          for claim in para_claims for c in claim.citations):
                text, claims, unsupported, failed, abstained = para["text"], para_claims, list(para["unsupported"]), [], False
                yield _event({"type": "paragraph", "index": 0, **para})
        numbers = prompt.cited_numbers(text, len(sources))
        citations = [] if abstained else [sources[n - 1] for n in numbers]
        items = [] if abstained else await run_in_threadpool(_evidence, claims, sources)
        if not abstained:
            unsupported = _pointed(text, unsupported, items)
        # Each cited sentence against the passage it quotes, and conditions it leaves out (research/support.py, 2026-11-07):
        # a sentence its passage does not support joins the sentences without a verified source; its quote stays verified.
        support = None
        if not abstained and settings.LEGAL_RESEARCH_SUPPORT_CHECK in ("on", "shadow"):
            support = await supportmod.check([c for c in claims if _stands(c)], sources)
            if support and settings.LEGAL_RESEARCH_SUPPORT_CHECK == "on":
                unsupported = list(unsupported) + [x["sentence"] for x in support["not_supported"] if x["sentence"] not in unsupported]
        status, answer = _status(abstained, claims, unsupported, failed), text
        reason = (None if not abstained else "other_documents" if off_target else "other_party" if off_party
                  else "not_in_cited" if elsewhere else "model")
        found = [] if abstained else await run_in_threadpool(doc_versions.conflicts, citations or sources, versions)
        calculation = (await run_in_threadpool(_calculation, sources, question, titles)
                       if not abstained and dates.asks_term_dates(question) else None)
        changed = [] if abstained else _changed(links, sources, numbers, items)
        if status == "answered" and any(a["by"]["signed"] and a["cited"] and not a["amendment_cited"] for a in changed):
            status = "partial"                     # it relies on wording a signed amendment replaced, without the amendment
        yield _event({"type": "done", "answer": text, "status": status, "abstained": abstained, "abstention_reason": reason,
                      "citations": citations, "evidence": items, "conflicts": found, "unsupported": unsupported, "failed": failed,
                      "uncited": not abstained and not numbers, "turn_id": turn["id"], "versions": versions, "amendments": changed,
                      "absence": prompt.absence(text) if abstained else None, "calculation": calculation,
                      "working": working.public(worked) if worked and not abstained else None,
                      "stated": stated, "misquotes": misquoted, "support": support, "choices": choices,
                      "defined_twice": [{"document": t["document_title"], "term": t["term"],
                                         "clauses": [d["clause"] for d in t["definitions"] if d["clause"]]} for t in twice], **({"stated_shadow": shadow} if mode == "shadow" else {})})
    except scopemod.ScopeRevoked:
        status, answer, citations, items, found = "scope_revoked", None, [], [], []
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        logger.error("Research: answer failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The answer could not be completed. Try again."})
    finally:
        try:
            if not several:      # an answer about several matters lives only in its conversation, never in one matter's history
                _save({**turn, "answer": answer, "citations": citations, "status": status, "evidence": items, "conflicts": found,
                       "versions": versions, "abstention_reason": reason, "amendments": changed})
            outcome = "deny" if status == "scope_revoked" else "error" if status == "error" else "allow"
            record("research.answer", outcome, principal=principal, object_type="research_turn", object_id=turn["id"],
                   matter_id=turn["matter_id"], reason=status,
                   details={"citations": len(citations), "verified": sum(1 for e in items if e["verification"] in ("verbatim", "normalized")),
                            **({"matters": [sc.matter_id for sc in scopes]} if several else {})})
        except Exception as e:
            logger.error("Research: could not record the outcome (%s)", type(e).__name__)


@router.post("/matters/{matter_id}/research/query")
@audited("research.query", object_type="matter", object_param="matter_id")
async def research_query(matter_id: str, inp: QueryIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    question = inp.question.strip()
    try:
        scope = await run_in_threadpool(scopemod.authorize, p, matter_id)
    except scopemod.ScopeDenied:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    understood, extra = None, {}
    if inp.history:                                         # S5.1b (D-063): a follow-up is made to stand alone first
        question, rewritten = await followup.standalone(question, [t.model_dump() for t in inp.history])
        understood, extra = {"question": question, "rewritten": rewritten}, {"history_turns": len(inp.history), "rewritten": rewritten}
    note(details={"query": {"chars": len(question), "sha256": hashlib.sha256(question.encode()).hexdigest()}, **extra})
    turn = {"id": str(uuid.uuid4()), "org_id": p["org_id"], "matter_id": matter_id, "owner_id": p["sub"],
            "question": question, "auth": p["auth"], "created_at": policy.now_iso()}
    return StreamingResponse(_answer(scope, p, question, turn, inp.versions, inp.documents or None, understood),
                             media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/matters/{matter_id}/research")
def research_history(matter_id: str, request: Request):
    """The caller's own questions in this matter, newest first."""
    p = principal_of(request)
    require(p, "research.query", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT id, question, answer, citations_json, status, created_at, evidence_json, conflicts_json, versions, "
                            "abstention_reason, amendments_json FROM legal_research_turns "
                            "WHERE matter_id=? AND owner_id=? ORDER BY created_at DESC LIMIT 50", (matter_id, p["sub"])).fetchall()
    from ..documents import repo as doc_repo
    cited = {c.get("document_id") for r in rows for c in json.loads(r[3] or "[]")}
    withheld = doc_repo.withheld_from(cited, p["sub"])
    return {"turns": [_turn(r, withheld) for r in rows]}


def _turn(r, withheld: frozenset | set = frozenset()) -> dict:
    citations = json.loads(r[3] or "[]")
    if withheld and any(c.get("document_id") in withheld for c in citations):
        # The answer drew on a document the person may no longer read (its access list narrowed): neither its wording nor its
        # passages are shown again (commercial-readiness review F3, 2026-10-28).
        return {"id": r[0], "question": r[1], "answer": "", "citations": [], "status": "abstained", "created_at": r[5],
                "unsupported": [], "evidence": [], "conflicts": [], "versions": r[8], "abstention_reason": "access_changed",
                "amendments": []}
    # Every valid number the answer cites was saved, so the highest saved one bounds the valid range.
    evidence = json.loads(r[6] or "[]")
    unsupported = (_pointed(r[2], prompt.unsupported(r[2], max((c["n"] for c in citations), default=0)), evidence)
                   if r[2] and r[4] in ("answered", "partial", "unverified") else [])
    return {"id": r[0], "question": r[1], "answer": r[2], "citations": citations, "status": r[4], "created_at": r[5],
            "unsupported": unsupported, "evidence": evidence, "conflicts": json.loads(r[7] or "[]"),
            "versions": r[8], "abstention_reason": r[9], "amendments": json.loads(r[10] or "[]") if len(r) > 10 else [],
            "absence": prompt.absence(r[2]) if r[4] == "abstained" and r[2] else None}


@router.post("/matters/{matter_id}/research/{turn_id}/review")
@audited("research.note.send", object_type="research_turn", object_param="turn_id")
def send_for_review(matter_id: str, turn_id: str, request: Request):
    """The person who asked sends an answer for a second lawyer's review (S2.7, D-043). The answer is
    fixed as it was given, as a research note; sending it again opens the same note."""
    from fastapi import HTTPException
    from .. import artifacts
    from . import notes
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    with get_conn() as conn:
        r = conn.execute("SELECT id, question, answer, citations_json, status, created_at, evidence_json, conflicts_json, versions, "
                         "abstention_reason FROM legal_research_turns WHERE id=? AND matter_id=? AND owner_id=?",
                         (turn_id, matter_id, p["sub"])).fetchone()
        name = (conn.execute("SELECT username FROM users WHERE id=?", (p["sub"],)).fetchone() or ["someone"])[0]
    if not r:
        raise HTTPException(status_code=404, detail="Not Found")       # turns are private to the person who asked
    turn = _turn(r)
    if turn["status"] not in notes.KINDS:
        raise HTTPException(status_code=409, detail="This question could not be answered, so there is nothing to review. Ask it again.")
    existing = artifacts.latest(matter_id, "research_turn", turn_id, "research_note")
    if existing:
        return {"artifact": existing, "changed": False}
    a = artifacts.create(matter_id, "research_note", "research_turn", turn_id, notes.content(turn, name), notes.inputs(turn),
                         p["sub"], "model")
    note(details={"artifact": a["id"], "status": turn["status"], "evidence": len(turn["evidence"])})
    return {"artifact": a, "changed": True}
