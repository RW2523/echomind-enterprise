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
from . import amendments, evidence, followup, prompt

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
    para = evidence.parse(raw, match=where)
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
        claims.append({"sentence": claim.sentence, "state": claim.state,
                       "citations": [{"n": c.n, "quote": c.quote, "verification": c.verification, **({"matched": True} if c.matched else {})}
                                     for c in claim.citations]})
    return ({"text": para.display, "claims": claims, "unsupported": prompt.unsupported(para.display, len(sources)),
             "failed": [c["sentence"] for c in claims if c["state"] == "failed"]}, para.claims)


def _stands(claim) -> bool:
    """A claim an answer stands on: a verified quote, or a cited sentence with words of its own. Bare markers left after
    "The documents do not state …" ("[1] [1] [1]", seen 2026-10-27) support nothing."""
    words = re.search(r"[^\W\d_]{2}", re.sub(r"\[[^\]]*\]", " ", claim.sentence))
    return claim.state == "supported" or (claim.state == "unquoted" and words is not None)


def _version_texts(version_ids) -> list[str]:
    """The stored text of each cited version: the whole document, not only the passage."""
    from ..documents import repo as doc_repo
    out = []
    for vid in sorted(v for v in version_ids if v):
        body = doc_repo.reader(vid)
        if body and body.get("text"):
            out.append(body["text"])
    return out


def _evidence(claims: list, sources: list[dict]) -> list[dict]:
    """Every quote the answer relies on, with where it is: document, version and its status, clause,
    page, line and exact characters of the version's text."""
    items, seen, cache = [], set(), {}
    with get_conn() as conn:
        for claim in claims:
            for c in claim.citations:
                if not 1 <= c.n <= len(sources) or (c.n, c.quote) in seen:
                    continue
                seen.add((c.n, c.quote))
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
                              "line": where.get("line"), "char_start": a, "char_end": b, **({"matched": True} if c.matched else {})})
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


def _status(abstained: bool, claims: list, unsupported: list, failed: list) -> str:
    if abstained:
        return "abstained"
    states = [c.state for c in claims]
    if not any(s in ("supported", "unquoted") for s in states):
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
        r = await searchmod.search(sc, question, k=k, versions=versions, documents=documents)
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
        if several:
            passages, links, nothing_found = await _search_several(scopes, principal, question, versions, documents)
        else:
            result = await searchmod.search(scope, question, k=settings.LEGAL_RESEARCH_PASSAGES, versions=versions, documents=documents)
            passages = result["passages"]
            links = _amended(passages, await run_in_threadpool(amendments.for_passages, scope.matter_id, principal["sub"], passages, versions))
            nothing_found = result["abstained"]
        sources = [_source(n, p) for n, p in enumerate(passages, start=1)]
        yield _event({"type": "sources", "citations": sources})
        if nothing_found or not passages:
            status, answer, reason = "abstained", prompt.ABSTAIN, "no_passages"
            yield _event({"type": "chunk", "text": prompt.ABSTAIN})
            yield _event({"type": "done", "answer": prompt.ABSTAIN, "status": status, "abstained": True, "abstention_reason": reason,
                          "citations": [], "evidence": [], "conflicts": [], "unsupported": [], "failed": [], "uncited": False,
                          "turn_id": turn["id"], "versions": versions})
            return
        buf, shown, claims, unsupported, failed, last_check = "", [], [], [], [], time.monotonic()
        stream = llm_mod.get().chat_stream(prompt.build(question, passages), temperature=0.1,
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
        nothing = not any(_stands(c) for c in claims)
        answering = [c for c in claims if _stands(c) and not prompt.only_context(c.sentence)]
        declined = False
        if not answering and prompt.opens_not_stated(text):
            cited = {sources[c.n - 1]["version_id"] for claim in claims for c in claim.citations if 1 <= c.n <= len(sources)}
            texts = await run_in_threadpool(_version_texts, cited)
            declined = bool(texts) and bool(prompt.missing_terms(question, texts))
        cited_scores = [passages[c.n - 1].get("score") for claim in claims for c in claim.citations
                        if 1 <= c.n <= len(passages) and passages[c.n - 1].get("score") is not None]
        weak = not cited_scores or max(float(x) for x in cited_scores) < settings.LEGAL_ABSTAIN_WEAK_SCORE
        abstained = (nothing and (prompt.is_abstention(text) or prompt.says_not_stated(text))) or \
            (prompt.says_not_stated(text) and weak) or declined
        numbers = prompt.cited_numbers(text, len(sources))
        citations = [] if abstained else [sources[n - 1] for n in numbers]
        items = [] if abstained else await run_in_threadpool(_evidence, claims, sources)
        status, answer, reason = _status(abstained, claims, unsupported, failed), text, ("model" if abstained else None)
        found = [] if abstained else await run_in_threadpool(doc_versions.conflicts, citations or sources, versions)
        changed = [] if abstained else _changed(links, sources, numbers, items)
        if status == "answered" and any(a["by"]["signed"] and a["cited"] and not a["amendment_cited"] for a in changed):
            status = "partial"                     # it relies on wording a signed amendment replaced, without the amendment
        yield _event({"type": "done", "answer": text, "status": status, "abstained": abstained, "abstention_reason": reason,
                      "citations": citations, "evidence": items, "conflicts": found, "unsupported": unsupported, "failed": failed,
                      "uncited": not abstained and not numbers, "turn_id": turn["id"], "versions": versions, "amendments": changed})
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
    unsupported = (prompt.unsupported(r[2], max((c["n"] for c in citations), default=0))
                   if r[2] and r[4] in ("answered", "partial", "unverified") else [])
    return {"id": r[0], "question": r[1], "answer": r[2], "citations": citations, "status": r[4], "created_at": r[5],
            "unsupported": unsupported, "evidence": json.loads(r[6] or "[]"), "conflicts": json.loads(r[7] or "[]"),
            "versions": r[8], "abstention_reason": r[9], "amendments": json.loads(r[10] or "[]") if len(r) > 10 else []}


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
