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
from . import evidence, prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
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
            "instructions": prompt.addressed_to_ai(p["text"])}       # withheld from the model's copy (T08)


def _save(turn: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_research_turns (id, org_id, matter_id, owner_id, question, answer, citations_json, status, "
                     "auth, created_at, finished_at, evidence_json, conflicts_json, versions, abstention_reason) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (turn["id"], turn["org_id"], turn["matter_id"], turn["owner_id"], turn["question"], turn["answer"],
                      json.dumps(turn["citations"], ensure_ascii=False), turn["status"], turn["auth"], turn["created_at"],
                      policy.now_iso(), json.dumps(turn.get("evidence") or [], ensure_ascii=False),
                      json.dumps(turn.get("conflicts") or [], ensure_ascii=False), turn.get("versions"), turn.get("abstention_reason")))
        conn.commit()


def _check(raw: str, sources: list[dict]) -> tuple[dict, list]:
    """One finished paragraph: its display text and its claims with every quote verified (S2.2b)."""
    para = evidence.parse(raw)
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
                       "citations": [{"n": c.n, "quote": c.quote, "verification": c.verification} for c in claim.citations]})
    return ({"text": para.display, "claims": claims, "unsupported": prompt.unsupported(para.display, len(sources)),
             "failed": [c["sentence"] for c in claims if c["state"] == "failed"]}, para.claims)


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
                              "line": where.get("line"), "char_start": a, "char_end": b})
    return items


def _status(abstained: bool, claims: list, unsupported: list, failed: list) -> str:
    if abstained:
        return "abstained"
    states = [c.state for c in claims]
    if not any(s in ("supported", "unquoted") for s in states):
        return "unverified"
    return "partial" if failed or unsupported else "answered"


async def _answer(scope, principal: dict, question: str, turn: dict, versions: str = "signed",
                  documents: Optional[list[str]] = None) -> AsyncIterator[str]:
    status, answer, citations, items, found, reason = "error", None, [], [], [], None
    try:
        result = await searchmod.search(scope, question, k=settings.LEGAL_RESEARCH_PASSAGES, versions=versions, documents=documents)
        passages = result["passages"]
        sources = [_source(n, p) for n, p in enumerate(passages, start=1)]
        yield _event({"type": "sources", "citations": sources})
        if result["abstained"] or not passages:
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
                await run_in_threadpool(scopemod.recheck, scope)        # raises ScopeRevoked
                last_check = time.monotonic()
            buf += piece
            while "\n\n" in buf:                                       # a paragraph is shown only once it is checked
                raw, buf = buf.split("\n\n", 1)
                if prompt.clean(raw).strip():
                    para = finish(raw)
                    yield _event({"type": "paragraph", "index": len(shown) - 1, **para})
                    yield _event({"type": "chunk", "text": ("\n\n" if len(shown) > 1 else "") + para["text"]})
        await run_in_threadpool(scopemod.recheck, scope)
        if prompt.clean(buf).strip():
            para = finish(buf)
            yield _event({"type": "paragraph", "index": len(shown) - 1, **para})
            yield _event({"type": "chunk", "text": ("\n\n" if len(shown) > 1 else "") + para["text"]})
        text = "\n\n".join(shown)
        # Abstention in other words (S2.2, D-038):
        # - nothing is supported, and the text says the documents do not answer; or
        # - the answer opens by saying the documents do not cover the question, and every passage it
        #   cites is only weakly related to the question.
        # A supported answer that also says the abstention sentence is still an answer.
        nothing = not any(c.state in ("supported", "unquoted") for c in claims)
        cited_scores = [passages[c.n - 1].get("score") for claim in claims for c in claim.citations
                        if 1 <= c.n <= len(passages) and passages[c.n - 1].get("score") is not None]
        weak = not cited_scores or max(float(x) for x in cited_scores) < settings.LEGAL_ABSTAIN_WEAK_SCORE
        abstained = (nothing and (prompt.is_abstention(text) or prompt.says_not_stated(text))) or \
            (prompt.says_not_stated(text) and weak)
        numbers = prompt.cited_numbers(text, len(sources))
        citations = [] if abstained else [sources[n - 1] for n in numbers]
        items = [] if abstained else await run_in_threadpool(_evidence, claims, sources)
        status, answer, reason = _status(abstained, claims, unsupported, failed), text, ("model" if abstained else None)
        found = [] if abstained else await run_in_threadpool(doc_versions.conflicts, citations or sources, versions)
        yield _event({"type": "done", "answer": text, "status": status, "abstained": abstained, "abstention_reason": reason,
                      "citations": citations, "evidence": items, "conflicts": found, "unsupported": unsupported, "failed": failed,
                      "uncited": not abstained and not numbers, "turn_id": turn["id"], "versions": versions})
    except scopemod.ScopeRevoked:
        status, answer, citations, items, found = "scope_revoked", None, [], [], []
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        logger.error("Research: answer failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The answer could not be completed. Try again."})
    finally:
        try:
            _save({**turn, "answer": answer, "citations": citations, "status": status, "evidence": items, "conflicts": found,
                   "versions": versions, "abstention_reason": reason})
            outcome = "deny" if status == "scope_revoked" else "error" if status == "error" else "allow"
            record("research.answer", outcome, principal=principal, object_type="research_turn", object_id=turn["id"],
                   matter_id=turn["matter_id"], reason=status,
                   details={"citations": len(citations), "verified": sum(1 for e in items if e["verification"] in ("verbatim", "normalized"))})
        except Exception as e:
            logger.error("Research: could not record the outcome (%s)", type(e).__name__)


@router.post("/matters/{matter_id}/research/query")
@audited("research.query", object_type="matter", object_param="matter_id")
async def research_query(matter_id: str, inp: QueryIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    question = inp.question.strip()
    note(details={"query": {"chars": len(question), "sha256": hashlib.sha256(question.encode()).hexdigest()}})
    try:
        scope = await run_in_threadpool(scopemod.authorize, p, matter_id)
    except scopemod.ScopeDenied:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    turn = {"id": str(uuid.uuid4()), "org_id": p["org_id"], "matter_id": matter_id, "owner_id": p["sub"],
            "question": question, "auth": p["auth"], "created_at": policy.now_iso()}
    return StreamingResponse(_answer(scope, p, question, turn, inp.versions, inp.documents or None), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/matters/{matter_id}/research")
def research_history(matter_id: str, request: Request):
    """The caller's own questions in this matter, newest first."""
    p = principal_of(request)
    require(p, "research.query", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT id, question, answer, citations_json, status, created_at, evidence_json, conflicts_json, versions, "
                            "abstention_reason FROM legal_research_turns "
                            "WHERE matter_id=? AND owner_id=? ORDER BY created_at DESC LIMIT 50", (matter_id, p["sub"])).fetchall()
    return {"turns": [_turn(r) for r in rows]}


def _turn(r) -> dict:
    citations = json.loads(r[3] or "[]")
    # Every valid number the answer cites was saved, so the highest saved one bounds the valid range.
    unsupported = (prompt.unsupported(r[2], max((c["n"] for c in citations), default=0))
                   if r[2] and r[4] in ("answered", "partial", "unverified") else [])
    return {"id": r[0], "question": r[1], "answer": r[2], "citations": citations, "status": r[4], "created_at": r[5],
            "unsupported": unsupported, "evidence": json.loads(r[6] or "[]"), "conflicts": json.loads(r[7] or "[]"),
            "versions": r[8], "abstention_reason": r[9]}


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
