"""/api/legal/counsel — the chat-first workspace (docs/echomind-legal/03 and 09).

- `GET  /counsel/conversations[?matter_id=&q=]`  the caller's conversations in matters they can still open (`q` searches
                                                 titles and questions)
- `POST /counsel/conversations`                  start one in a matter (`research.query` on it)
- `GET  /counsel/conversations/{id}`             its turns (a tree: each names the turn it follows), each with its validated
                                                 component envelope and the caller's rating
- `PATCH  /counsel/conversations/{id}`           rename it
- `DELETE /counsel/conversations/{id}`           take it off the caller's list (kept as work product)
- `POST /counsel/conversations/{id}/turns`       ask (after a given turn, or again as another version of one); NDJSON
                                                 events (orchestrator.py); idempotent on client_request_id
- `PUT  /counsel/turns/{id}/feedback`            the asker's rating of one answer
- `GET  /counsel/components/{id}`                one card as its records stand now (a checklist, a draft, a panel run,
                                                 a meeting refresh themselves from here)
- `GET  /counsel/citations/{id}`                 the cited passage, re-authorized and re-verified (citations.py)
- `POST /counsel/actions`                        a component action: identity, matter right, component origin,
                                                 version (CAS) and idempotency key are all checked here; resolving an
                                                 inconsistency, ticking an obligation, deciding on a draft
- `GET  /counsel/discrepancies?matter_id=`       potential inconsistencies the caller may see, with their state
- `GET  /counsel/activity`                       the caller's own recent actions, and documents still being read in
                                                 their matters (the Activity destination)

A conversation, turn, component or citation of someone else, in a matter the caller cannot open, or that does not
exist, answers the same 404. Every route writes one audit event; the stream writes `counsel.answer` when it ends.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from . import citations, discrepancies, orchestrator, schemas, store

router = APIRouter(prefix="/legal/counsel", tags=["legal"], dependencies=[Depends(require_law_edition)])
_NO_STORE = {"Cache-Control": "no-store"}
_Id = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _own_conversation(p: dict, conversation_id: str) -> dict:
    """The caller's conversation, in matters they may all still research; anything else is 404."""
    conv = store.conversation(conversation_id, p["sub"])
    if not conv or conv["org_id"] != p["org_id"]:
        raise _not_found()
    for mid in conv["matter_ids"]:
        require(p, "research.query", mid)
    return conv


# ── Conversations ────────────────────────────────────────────────────────────────
_MatterId = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")]


class ConversationIn(BaseModel):
    """One matter (`matter_id`), or several (`matter_ids`, the first being the conversation's own; owner's request 2026-10-28)."""
    matter_id: Optional[_MatterId] = None
    matter_ids: list[_MatterId] = Field(default_factory=list, max_length=schemas.MAX_MATTERS)

    @model_validator(mode="after")
    def _one_or_more(self):
        ids = list(dict.fromkeys(self.matter_ids or ([self.matter_id] if self.matter_id else [])))
        if not ids:
            raise ValueError("choose a matter")
        self.matter_ids = ids
        return self


@router.get("/conversations")
def list_conversations(request: Request, matter_id: Optional[str] = None, q: Optional[str] = Query(None, max_length=100)):
    p = principal_of(request)
    if matter_id:
        require(p, "research.query", matter_id)
    q = " ".join((q or "").split()) or None
    return JSONResponse({"conversations": store.list_conversations(p, matter_id, q=q)}, headers=_NO_STORE)


@router.post("/conversations", status_code=201)
@audited("counsel.conversation.create", object_type="matter", object_param=None)
def create_conversation(inp: ConversationIn, request: Request):
    p = principal_of(request)
    for mid in inp.matter_ids:                                   # each one is the caller's to research (404/403 as anywhere)
        require(p, "research.query", mid)
    if len(inp.matter_ids) > 1:
        from ...core.db import get_conn
        with get_conn() as conn:
            marks = {r[0]: r[1] for r in conn.execute(
                f"SELECT id, sensitivity FROM matters WHERE id IN ({','.join('?' * len(inp.matter_ids))})", inp.matter_ids)}
        if any(marks.get(m) == "highly_restricted" for m in inp.matter_ids):
            raise HTTPException(status_code=409, detail="A highly restricted matter can only be discussed on its own.")
    conv = store.create_conversation(p["org_id"], inp.matter_ids, p["sub"])
    note(object_type="conversation", object_id=conv["id"], matter_id=inp.matter_ids[0],
         details={"matters": inp.matter_ids} if len(inp.matter_ids) > 1 else None)
    return {"conversation": conv}


@router.get("/conversations/{conversation_id}")
@audited("counsel.conversation.read", object_type="conversation", object_param="conversation_id")
def read_conversation(conversation_id: str, request: Request):
    p = principal_of(request)
    conv = _own_conversation(p, conversation_id)
    out = []
    rated = store.feedback(conv["id"], p["sub"])
    for t in store.turns(conv["id"], p["sub"]):
        item = {"id": t["id"], "seq": t["seq"], "parent_turn_id": t["parent_turn_id"], "question": t["question"],
                "understood": t["understood"], "intent": t["intent"], "status": t["status"], "created_at": t["created_at"],
                "finished_at": t["finished_at"]}
        if t["status"] in ("completed", "abstained"):
            item["envelope"] = orchestrator.envelope_for(p, t, conv)
        if t["id"] in rated:
            item["feedback"] = rated[t["id"]]
        out.append(item)
    note(matter_id=conv["matter_id"], details={"turns": len(out), **({"matters": conv["matter_ids"]} if len(conv["matter_ids"]) > 1 else {})})
    return JSONResponse({"conversation": {**{k: conv[k] for k in ("id", "matter_id", "title", "created_at", "updated_at")},
                                          "matter_ids": conv["matter_ids"]}, "turns": out},
                        headers=_NO_STORE)


class ConversationPatch(BaseModel):
    model_config = {"extra": "forbid"}
    title: str = Field(min_length=1, max_length=store.TITLE_MAX)


@router.patch("/conversations/{conversation_id}")
@audited("counsel.conversation.rename", object_type="conversation", object_param="conversation_id")
def rename_conversation(conversation_id: str, inp: ConversationPatch, request: Request):
    p = principal_of(request)
    conv = _own_conversation(p, conversation_id)
    title = " ".join(inp.title.split())
    if not title:
        raise HTTPException(status_code=422, detail="Give the conversation a name.")
    store.rename(conv["id"], p["sub"], title)
    note(matter_id=conv["matter_id"], details={"chars": len(title)})
    return JSONResponse({"conversation": {"id": conv["id"], "title": store.conversation(conv["id"], p["sub"])["title"]}}, headers=_NO_STORE)


@router.delete("/conversations/{conversation_id}", status_code=204)
@audited("counsel.conversation.delete", object_type="conversation", object_param="conversation_id")
def delete_conversation(conversation_id: str, request: Request):
    """Off the caller's list. Allowed while they can still open its matters (otherwise it is not theirs to see either);
    the records stay as the matter's work product."""
    p = principal_of(request)
    conv = _own_conversation(p, conversation_id)
    store.delete(conv["id"], p["sub"])
    note(matter_id=conv["matter_id"])
    return Response(status_code=204, headers=_NO_STORE)


_TurnId = Field(default=None, pattern=r"^turn_[0-9a-f]{32}$")


class FormIn(BaseModel):
    """The answers to a guided form the server asked in this conversation (guided_questions)."""
    model_config = {"extra": "forbid"}
    component_id: str = Field(pattern=r"^cmp_[0-9a-f]{32}$")
    values: dict[Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")],
                 Annotated[str, StringConstraints(max_length=1000)]] = Field(default_factory=dict, max_length=3)


class TurnIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    client_request_id: Optional[str] = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    document_ids: list[Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,80}$")]] = Field(default_factory=list, max_length=10)
    focus_discrepancy_id: Optional[str] = Field(default=None, pattern=r"^dsc_[0-9a-f]{32}$")
    after_turn_id: Optional[str] = _TurnId                # the turn this question follows (default: the newest)
    retry_of: Optional[str] = _TurnId                     # ask again, or edited: another version of that turn
    form: Optional[FormIn] = None                         # answers to a form the server asked

    @model_validator(mode="after")
    def _one_place(self):
        if self.after_turn_id and self.retry_of:
            raise ValueError("a question either follows a turn or is another version of one")
        return self


def _ndjson(events):
    async def body():
        async for e in events:
            yield json.dumps(e, ensure_ascii=False) + "\n"
    return StreamingResponse(body(), media_type="application/x-ndjson", headers={**_NO_STORE, "X-Accel-Buffering": "no"})


async def _replay(p: dict, t: dict):
    yield orchestrator.event("turn.started", {"turn_id": t["id"], "conversation_id": t["conversation_id"], "matter_id": t["matter_id"],
                                              "intent": t["intent"], "replayed": True})
    if t["status"] in ("completed", "abstained"):
        yield orchestrator.event("ui.final", await run_in_threadpool(orchestrator.envelope_for, p, t))
        yield orchestrator.event(orchestrator._STATE_EVENT[t["status"]], {"turn_id": t["id"], "status": t["status"]})
    else:
        yield orchestrator.event("turn.failed", {"turn_id": t["id"], "message": orchestrator.FAILED})


@router.post("/conversations/{conversation_id}/turns")
@audited("counsel.turn", object_type="conversation", object_param="conversation_id")
async def ask(conversation_id: str, inp: TurnIn, request: Request):
    p = principal_of(request)
    conv = await run_in_threadpool(_own_conversation, p, conversation_id)
    text = " ".join(inp.text.split())
    if not text:
        raise HTTPException(status_code=422, detail="Type a question.")
    note(matter_id=conv["matter_id"], details={"chars": len(text), "documents": len(inp.document_ids)})
    if inp.client_request_id:
        prior = await run_in_threadpool(store.turn_by_request, p["sub"], inp.client_request_id)
        if prior:
            if prior["conversation_id"] != conv["id"]:
                raise HTTPException(status_code=409, detail="That request id was already used.")
            if prior["status"] == "running":
                raise HTTPException(status_code=409, detail="This question is still being answered.")
            return _ndjson(_replay(p, prior))
    if await run_in_threadpool(store.running_turn, conv["id"]):
        raise HTTPException(status_code=409, detail="Wait for the answer in progress, or stop it, before asking again.")
    parent = store._FOLLOW_NEWEST
    for tid in (inp.after_turn_id, inp.retry_of):
        if tid:
            t = await run_in_threadpool(store.turn, tid, p["sub"])
            if not t or t["conversation_id"] != conv["id"]:
                raise _not_found()
            parent = t["id"] if tid == inp.after_turn_id else t["parent_turn_id"]
    if inp.retry_of:
        note(details={"chars": len(text), "documents": len(inp.document_ids), "version_of": inp.retry_of})
    form = None
    if inp.form:                                            # a form this server asked, in this conversation, of this person
        comp = await run_in_threadpool(store.component, inp.form.component_id, p["sub"])
        if not comp or comp["conversation_id"] != conv["id"] or comp["type"] != "guided_questions" or comp["matter_id"] not in conv["matter_ids"]:
            raise _not_found()
        form = {"component": comp, "values": inp.form.values}
        note(details={"form": (comp["props"].get("_form") or {}).get("purpose")})
    return _ndjson(orchestrator.run(p, conv, text, inp.client_request_id, inp.document_ids, inp.focus_discrepancy_id, parent, form))


class FeedbackIn(BaseModel):
    model_config = {"extra": "forbid"}
    rating: Optional[Literal["up", "down"]]
    reason: Optional[Literal[store.FEEDBACK_REASONS]] = None
    note: Optional[str] = Field(default=None, max_length=500)


@router.get("/components/{component_id}")
@audited("counsel.component.read", object_type="component", object_param="component_id")
def read_component(component_id: str, request: Request):
    """One card of the caller's own conversation, rebuilt from its records as they may see them now."""
    p = principal_of(request)
    comp = store.component(component_id, p["sub"])
    if not comp:
        raise _not_found()
    conv = _own_conversation(p, comp["conversation_id"])
    note(matter_id=comp["matter_id"], details={"type": comp["type"]})
    return JSONResponse({"component": schemas.validate_component(orchestrator.materialize(p, comp, conv["matter_id"]))}, headers=_NO_STORE)


@router.put("/turns/{turn_id}/feedback")
@audited("counsel.feedback", object_type="counsel_turn", object_param="turn_id")
def rate_answer(turn_id: str, inp: FeedbackIn, request: Request):
    """Whether an answer helped, from the person who asked it, while they can still open its conversation."""
    p = principal_of(request)
    t = store.turn(turn_id, p["sub"])
    if not t:
        raise _not_found()
    conv = _own_conversation(p, t["conversation_id"])
    if t["status"] not in ("completed", "abstained"):
        raise HTTPException(status_code=409, detail="Only a finished answer can be rated.")
    if inp.rating != "down" and (inp.reason or inp.note):
        raise HTTPException(status_code=422, detail="Say what was wrong only with a thumbs down.")
    note_text = " ".join((inp.note or "").split()) or None
    out = store.set_feedback(t, p["org_id"], inp.rating, inp.reason, note_text)
    note(matter_id=conv["matter_id"], details={"rating": inp.rating, "reason": inp.reason, "note_chars": len(note_text or "")})
    return JSONResponse({"feedback": out}, headers=_NO_STORE)


# ── Sources ──────────────────────────────────────────────────────────────────────
@router.get("/citations/{citation_id}")
@audited("counsel.citation.open", object_type="citation", object_param="citation_id")
def open_citation(citation_id: str, request: Request):
    p = principal_of(request)
    try:
        out = citations.resolve(p, citation_id)
    except citations.NotFound:
        raise _not_found()
    except citations.Mismatch:
        note(outcome="deny", reason="source_mismatch")
        raise HTTPException(status_code=409, detail="The stored document no longer matches this citation, so it is not shown. "
                                                    "Ask the question again to cite the current text.")
    note(matter_id=out["matter_id"], details={"verified": out["verified"], "status": out["verification"]})
    return JSONResponse({"citation": out}, headers=_NO_STORE)


# ── Discrepancies ────────────────────────────────────────────────────────────────
@router.get("/discrepancies")
def list_discrepancies(request: Request, matter_id: str, state: Literal["open", "resolved", "all"] = "open"):
    p = principal_of(request)
    require(p, "research.query", matter_id)
    statuses = {"open": discrepancies.OPEN, "resolved": discrepancies.RESOLVED,
                "all": discrepancies.OPEN + discrepancies.RESOLVED}[state]
    rows = discrepancies.listed(p, matter_id, statuses)
    return JSONResponse({"discrepancies": [{"id": r["id"], "title": discrepancies.title(r), "status": r["status"], "kind": r["kind"],
                                            "version": r["version"], "updated_at": r["updated_at"]} for r in rows]}, headers=_NO_STORE)


# ── Actions ──────────────────────────────────────────────────────────────────────
class ActionIn(BaseModel):
    component_id: str = Field(pattern=r"^cmp_[0-9a-f]{32}$")
    action_id: Literal["review", "obligation", "decide"]
    matter_id: str = _Id
    expected_version: int = Field(ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ReviewParams(BaseModel):
    model_config = {"extra": "forbid"}
    resolution: Literal["in_review", "explained", "confirmed", "dismissed"]
    note: Optional[str] = Field(default=None, max_length=1000)


class ObligationParams(BaseModel):
    model_config = {"extra": "forbid"}
    obligation_id: str = _Id
    action: Literal["done", "reopen", "confirm"]


class DecideParams(BaseModel):
    model_config = {"extra": "forbid"}
    artifact_id: str = _Id
    decision: Literal["approved", "changes_requested"]
    reason: Optional[str] = Field(default=None, max_length=500)
    read_it: bool = False                                  # the person says they read the version before approving it


_ACTIONS = {"evidence_comparison": {"review"}, "checklist": {"obligation"}, "approval_request": {"decide"}}


def _conflict(e: HTTPException, current: Optional[dict]) -> JSONResponse:
    return JSONResponse({"detail": e.detail, "code": "VERSION_CONFLICT", "component": current}, status_code=409, headers=_NO_STORE)


@router.post("/actions")
@audited("counsel.action", object_type="component", object_param=None)
def component_action(inp: ActionIn, request: Request):
    p = principal_of(request)
    note(object_id=inp.component_id, matter_id=inp.matter_id, details={"action": inp.action_id})
    comp = store.component(inp.component_id, p["sub"])           # the component must be one this server gave this person
    if not comp or comp["matter_id"] != inp.matter_id:            # its own matter, also in a conversation about several
        raise _not_found()
    conv = store.conversation(comp["conversation_id"], p["sub"])
    if not conv:                                                  # its conversation was deleted from the person's list
        raise _not_found()
    primary = conv["matter_id"]
    require(p, "research.query", inp.matter_id)                  # the matter is still theirs to open
    done = store.replay(p["sub"], inp.idempotency_key)            # a repeat: after the checks, and the card as it is now (F9)
    if done:
        if done["component_id"] != inp.component_id or done["action_id"] != inp.action_id:
            raise HTTPException(status_code=409, detail="That idempotency key was used for another action.")
        return JSONResponse({"component": schemas.validate_component(orchestrator.materialize(p, comp, primary)), "replayed": True},
                            headers=_NO_STORE)
    if inp.action_id not in _ACTIONS.get(comp["type"], set()):
        raise HTTPException(status_code=400, detail="This card has no such action.")
    if inp.action_id == "obligation":
        return _obligation(p, inp, comp, primary)
    if inp.action_id == "decide":
        return _decide(p, inp, comp, conv, primary)
    try:
        params = ReviewParams.model_validate(inp.parameters)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Choose a resolution, and say briefly why.")
    require(p, "findings.review", inp.matter_id)
    did = comp["props"].get("discrepancy_id")
    try:
        row = discrepancies.review(p, inp.matter_id, did, params.resolution, params.note, inp.expected_version)
    except discrepancies.NotFound:
        raise _not_found()
    except discrepancies.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except discrepancies.Conflict as e:
        current = orchestrator.materialize(p, comp, primary) if e.current else None
        return JSONResponse({"detail": str(e), "code": "VERSION_CONFLICT", "component": current}, status_code=409, headers=_NO_STORE)
    from ..audit.record import record
    record("discrepancy.review", "allow", principal=p, object_type="discrepancy", object_id=did, matter_id=inp.matter_id,
           details={"status": params.resolution, "previous": {"status": row["from_status"]}, "note_chars": len(params.note or "")})
    result = {"component": schemas.validate_component(orchestrator.materialize(p, comp, primary))}
    store.remember(p["sub"], inp.idempotency_key, inp.component_id, inp.action_id, result)
    return JSONResponse(result, headers=_NO_STORE)


def _current(p: dict, comp: dict, primary: Optional[str]) -> dict:
    return schemas.validate_component(orchestrator.materialize(p, comp, primary))


def _obligation(p: dict, inp: ActionIn, comp: dict, primary: Optional[str]) -> JSONResponse:
    """Tick done, reopen or confirm an obligation the checklist showed, through the register's own change."""
    from ..audit.record import record
    from . import tasks
    try:
        params = ObligationParams.model_validate(inp.parameters)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Choose done, reopen or confirm.")
    if params.obligation_id not in (comp["props"].get("obligation_ids") or []):
        raise _not_found()                                   # only what this card showed
    try:
        out = tasks.change(p, inp.matter_id, params.obligation_id, params.action, inp.expected_version)
    except HTTPException as e:
        if e.status_code == 409:
            return _conflict(e, _current(p, comp, primary))
        raise
    record("obligation.update", "allow", principal=p, object_type="obligation", object_id=params.obligation_id, matter_id=inp.matter_id,
           details={"action": params.action, "version": out["obligation"]["version"], "mode": "chat"})
    result = {"component": _current(p, comp, primary)}
    store.remember(p["sub"], inp.idempotency_key, inp.component_id, inp.action_id, result)
    return JSONResponse(result, headers=_NO_STORE)


def _decide(p: dict, inp: ActionIn, comp: dict, conv: Optional[dict], primary: Optional[str]) -> JSONResponse:
    """Approve, or ask for changes to, a version the approval card showed, through the review route's own checks."""
    from ..audit.record import record
    from . import approvals
    try:
        params = DecideParams.model_validate(inp.parameters)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Choose approve or ask for changes.")
    ref = next((r for r in comp["props"].get("items") or [] if r["artifact_id"] == params.artifact_id), None)
    if not ref or ref["matter_id"] not in ((conv or {}).get("matter_ids") or []):
        raise _not_found()
    require(p, "research.query", ref["matter_id"])
    if params.decision == "approved" and not params.read_it:
        raise HTTPException(status_code=422, detail="Confirm that you read this version before approving it.")
    try:
        out = approvals.decide(p, ref, params.decision, (params.reason or "").strip() or None)
    except HTTPException as e:
        if e.status_code == 409:
            return _conflict(e, _current(p, comp, primary))
        raise
    record("artifact.review", "allow", principal=p, object_type="ai_artifact", object_id=params.artifact_id, matter_id=ref["matter_id"],
           details={"decision": params.decision, "kind": out["artifact"]["kind"], "version": out["artifact"]["version"], "mode": "chat"})
    result = {"component": _current(p, comp, primary)}
    store.remember(p["sub"], inp.idempotency_key, inp.component_id, inp.action_id, result)
    return JSONResponse(result, headers=_NO_STORE)


# ── Activity ─────────────────────────────────────────────────────────────────────
@router.get("/activity")
@audited("counsel.activity.read")
def my_activity(request: Request, before: Optional[int] = None, limit: int = Query(50, ge=1, le=100)):
    """What the caller did (their own audit events, never anyone else's), with matter numbers only for matters they
    can still open, and how many documents are still being read in each of their matters."""
    from ...core.db import get_conn
    from .. import policy
    from ..audit import routes as audit_routes
    p = principal_of(request)
    where, params = ["org_id = ?", "actor_user_id = ?", "action <> 'counsel.activity.read'"], [p["org_id"], p["sub"]]
    if before:
        where.append("seq < ?")
        params.append(before)
    out = audit_routes._events(" AND ".join(where), params, limit, with_numbers=False)
    args = {"uid": p["sub"], "org": p["org_id"], "now": policy.now_iso()}
    with get_conn() as conn:
        visible = {r[0]: {"id": r[0], "number": r[1], "title": r[2]} for r in conn.execute(
            f"SELECT m.id, m.number, m.title FROM matters m WHERE {policy.visible_matters_sql()}", args).fetchall()}
        reading = conn.execute(
            "SELECT v.matter_id, COUNT(*) FROM legal_document_versions v JOIN matters m ON m.id = v.matter_id "
            f"WHERE v.index_status IN ('queued','indexing') AND {policy.visible_matters_sql()} GROUP BY v.matter_id", args).fetchall()
    for e in out["events"]:
        e.pop("ip", None)
        e.pop("session_ref", None)
        if e.get("matter"):
            e["matter"] = visible.get(e["matter"]["id"])
    note(details={"events": len(out["events"])})
    return JSONResponse({**out, "in_progress": [{"matter": visible[mid], "documents": n} for mid, n in reading if mid in visible]},
                        headers=_NO_STORE)
