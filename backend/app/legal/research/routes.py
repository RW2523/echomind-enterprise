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
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note, record
from ..matters.routes import principal_of, require, require_law_edition
from ..retrieval import scope as scopemod
from ..retrieval import search as searchmod
from . import llm as llm_mod
from . import prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _event(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _source(n: int, p: dict) -> dict:
    c = p["citation"]
    return {"n": n, "document_id": c["document_id"], "document_title": c["document_title"], "version_id": c["version_id"],
            "version_number": c["version_number"], "page_start": c["page_start"], "page_end": c["page_end"],
            "char_start": c["char_start"], "char_end": c["char_end"], "text": p["text"],
            "clause": c.get("clause"), "line": c.get("line"), "focus_start": c.get("focus_start"), "focus_end": c.get("focus_end"),
            "instructions": prompt.addressed_to_ai(p["text"])}       # withheld from the model's copy (T08)


def _save(turn: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_research_turns (id, org_id, matter_id, owner_id, question, answer, citations_json, status, "
                     "auth, created_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (turn["id"], turn["org_id"], turn["matter_id"], turn["owner_id"], turn["question"], turn["answer"],
                      json.dumps(turn["citations"], ensure_ascii=False), turn["status"], turn["auth"], turn["created_at"],
                      policy.now_iso()))
        conn.commit()


async def _answer(scope, principal: dict, question: str, turn: dict) -> AsyncIterator[str]:
    status, answer, citations = "error", None, []
    try:
        result = await searchmod.search(scope, question, k=settings.LEGAL_RESEARCH_PASSAGES)
        passages = result["passages"]
        sources = [_source(n, p) for n, p in enumerate(passages, start=1)]
        yield _event({"type": "sources", "citations": sources})
        if result["abstained"] or not passages:
            status, answer = "abstained", prompt.ABSTAIN
            yield _event({"type": "chunk", "text": prompt.ABSTAIN})
            yield _event({"type": "done", "answer": prompt.ABSTAIN, "citations": [], "abstained": True, "turn_id": turn["id"]})
            return
        text, last_check = "", time.monotonic()
        stream = llm_mod.get().chat_stream(prompt.build(question, passages), temperature=0.1,
                                           max_tokens=settings.LEGAL_RESEARCH_MAX_TOKENS)
        async for piece in stream:
            if time.monotonic() - last_check >= settings.LEGAL_STREAM_RECHECK_SEC:
                await run_in_threadpool(scopemod.recheck, scope)        # raises ScopeRevoked
                last_check = time.monotonic()
            text += piece
            shown = piece.replace("<think>", "").replace("</think>", "")
            if shown:
                yield _event({"type": "chunk", "text": shown})
        await run_in_threadpool(scopemod.recheck, scope)
        text = prompt.clean(text)
        numbers = prompt.cited_numbers(text, len(sources))
        abstained = prompt.is_abstention(text)
        status, answer = ("abstained" if abstained else "answered"), text
        citations = [] if abstained else [sources[n - 1] for n in numbers]
        yield _event({"type": "done", "answer": text, "citations": citations, "abstained": abstained,
                      "uncited": not abstained and not numbers, "unsupported": prompt.unsupported(text, len(sources)),
                      "turn_id": turn["id"]})
    except scopemod.ScopeRevoked:
        status, answer, citations = "scope_revoked", None, []
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        logger.error("Research: answer failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The answer could not be completed. Try again."})
    finally:
        try:
            _save({**turn, "answer": answer, "citations": citations, "status": status})
            record("research.answer", "allow" if status in ("answered", "abstained") else ("deny" if status == "scope_revoked" else "error"),
                   principal=principal, object_type="research_turn", object_id=turn["id"], matter_id=turn["matter_id"],
                   reason=status, details={"citations": len(citations)})
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
    return StreamingResponse(_answer(scope, p, question, turn), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/matters/{matter_id}/research")
def research_history(matter_id: str, request: Request):
    """The caller's own questions in this matter, newest first."""
    p = principal_of(request)
    require(p, "research.query", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT id, question, answer, citations_json, status, created_at FROM legal_research_turns "
                            "WHERE matter_id=? AND owner_id=? ORDER BY created_at DESC LIMIT 50", (matter_id, p["sub"])).fetchall()
    return {"turns": [_turn(r) for r in rows]}


def _turn(r) -> dict:
    citations = json.loads(r[3] or "[]")
    # Every valid number the answer cites was saved, so the highest saved one bounds the valid range.
    unsupported = prompt.unsupported(r[2], max((c["n"] for c in citations), default=0)) if r[2] and r[4] == "answered" else []
    return {"id": r[0], "question": r[1], "answer": r[2], "citations": citations, "status": r[4], "created_at": r[5],
            "unsupported": unsupported}
