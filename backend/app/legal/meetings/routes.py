"""/api/legal/matters/{id}/meetings (S1.8, SEC-05).

Contributors plan meetings, list who takes part, and record each person's consent with the
method used; anyone on the matter may record a withdrawal, which stops a running capture at once.
Participant names are matter content: they appear here, never in the audit trail.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field

from ..audit.record import audited, note
from ..matters import repo as matters_repo
from ..matters.routes import principal_of, require, require_law_edition
from . import agenda as agenda_mod
from . import capture, consent, repo, vocabulary
from . import transcript as transcript_mod

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
ws_router = APIRouter(prefix="/legal", tags=["legal"])      # sockets: the edition is checked by the gate itself

Kind = Literal["client_call", "internal", "negotiation", "interview", "board", "hearing", "other"]
Role = Literal["host", "attorney", "client", "counterparty", "witness", "expert", "other"]
Method = Literal["verbal_on_record", "written", "host_attestation"]
_BASE = "/matters/{matter_id}/meetings"


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class MeetingIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    kind: Kind = "other"
    scheduled_at: Optional[datetime] = None


class ParticipantIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: Role = "other"


class ConsentIn(BaseModel):
    method: Method
    statement_read: bool


def _meeting(matter_id: str, meeting_id: str) -> dict:
    m = repo.get_meeting_row(matter_id, meeting_id)
    if not m:
        raise _not_found()
    return m


def _number(matter_id: str) -> str:
    return matters_repo.get_matter(matter_id)["number"]


@router.post(_BASE)
@audited("meeting.create", object_type="meeting")
def create_meeting(matter_id: str, inp: MeetingIn, request: Request):
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    when = inp.scheduled_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z") if inp.scheduled_at else None
    mid = repo.create_meeting(p["org_id"], matter_id, inp.title.strip(), inp.kind, when, p["sub"])
    note(object_id=mid, details={"kind": inp.kind})
    return {"meeting": repo.meeting_detail(matter_id, mid, _number(matter_id))}


@router.get(_BASE)
def list_meetings(matter_id: str, request: Request):
    require(principal_of(request), "meetings.read", matter_id)
    return {"meetings": repo.list_meetings(matter_id)}


@router.get(_BASE + "/{meeting_id}")
@audited("meeting.read", object_type="meeting", object_param="meeting_id")
def get_meeting(matter_id: str, meeting_id: str, request: Request):
    require(principal_of(request), "meetings.read", matter_id)
    d = repo.meeting_detail(matter_id, meeting_id, _number(matter_id))
    if not d:
        raise _not_found()
    return {"meeting": {**d, "assist": _assist_state(d), "agenda": agenda_mod.items(matter_id, meeting_id, principal_of(request)["sub"])}}


@router.post(_BASE + "/{meeting_id}/end")
@audited("meeting.end", object_type="meeting", object_param="meeting_id")
def end_meeting(matter_id: str, meeting_id: str, request: Request):
    require(principal_of(request), "meetings.create", matter_id)
    _meeting(matter_id, meeting_id)
    with capture.meeting_lock(meeting_id):
        repo.end_meeting(meeting_id)
        capture.stop_meeting(meeting_id, "meeting_ended")
    return {"meeting": repo.meeting_detail(matter_id, meeting_id, _number(matter_id))}


@router.post(_BASE + "/{meeting_id}/participants")
@audited("participant.add", object_type="participant")
def add_participant(matter_id: str, meeting_id: str, inp: ParticipantIn, request: Request):
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    m = _meeting(matter_id, meeting_id)
    if m["status"] != "planned":
        raise HTTPException(status_code=409, detail="This meeting has ended.")
    pid = repo.add_participant(meeting_id, matter_id, inp.name.strip(), inp.role, None, p["sub"])
    note(object_id=pid, details={"role": inp.role})
    return {"participant": repo.participant(meeting_id, pid)}


@router.delete(_BASE + "/{meeting_id}/participants/{participant_id}")
@audited("participant.remove", object_type="participant", object_param="participant_id")
def remove_participant(matter_id: str, meeting_id: str, participant_id: str, request: Request):
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    _meeting(matter_id, meeting_id)
    if not repo.participant(meeting_id, participant_id):
        raise _not_found()
    repo.remove_participant(meeting_id, participant_id, p["sub"])
    return {"ok": True}


@router.post(_BASE + "/{meeting_id}/participants/{participant_id}/consent")
@audited("consent.grant", object_type="participant", object_param="participant_id")
def grant_consent(matter_id: str, meeting_id: str, participant_id: str, inp: ConsentIn, request: Request):
    p = principal_of(request)
    require(p, "consent.capture", matter_id)
    m = _meeting(matter_id, meeting_id)
    person = repo.participant(meeting_id, participant_id)
    if not person or person["removed_at"]:
        raise _not_found()
    if m["status"] != "planned":
        raise HTTPException(status_code=409, detail="This meeting has ended.")
    if not inp.statement_read:
        raise HTTPException(status_code=422, detail="Read the consent statement to the participant first.")
    statement = consent.statement_for(_number(matter_id))
    note(details={"method": inp.method})
    repo.add_consent_event(meeting_id, participant_id, "granted", inp.method,
                           hashlib.sha256(statement.encode()).hexdigest(), p["sub"])
    return {"ok": True}


@router.post(_BASE + "/{meeting_id}/participants/{participant_id}/consent/withdraw")
@audited("consent.withdraw", object_type="participant", object_param="participant_id")
def withdraw_consent(matter_id: str, meeting_id: str, participant_id: str, request: Request):
    """Anyone on the matter may record a withdrawal (reading rights suffice): it must never be hard
    to stop a recording. Running captures of the meeting stop before this returns."""
    p = principal_of(request)
    require(p, "meetings.read", matter_id)
    _meeting(matter_id, meeting_id)
    if not repo.participant(meeting_id, participant_id):
        raise _not_found()
    with capture.meeting_lock(meeting_id):
        repo.add_consent_event(meeting_id, participant_id, "withdrawn", None, None, p["sub"])
        stopped = capture.stop_meeting(meeting_id, "consent_withdrawn")
    note(details={"captures_stopped": stopped})
    return {"ok": True, "captures_stopped": stopped}


@router.get(_BASE + "/{meeting_id}/transcript")
@audited("meeting.transcript.read", object_type="meeting", object_param="meeting_id")
def transcript(matter_id: str, meeting_id: str, request: Request):
    """The transcript as corrected (S2.4): each segment's text and speaker after its revisions, its
    raw text and full revision history, wall-clock times, and the capture's marks."""
    require(principal_of(request), "meetings.read", matter_id)
    _meeting(matter_id, meeting_id)
    return transcript_mod.view(matter_id, meeting_id)


class RevisionIn(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    speaker_participant_id: Optional[str] = Field(default=None, max_length=64)
    reason: str = Field(min_length=3, max_length=500)
    suggestion_id: Optional[str] = Field(default=None, max_length=32)


def _open_for_corrections(matter_id: str, meeting_id: str, p: dict) -> dict:
    require(p, "meetings.create", matter_id)
    return _meeting(matter_id, meeting_id)


@router.post(_BASE + "/{meeting_id}/transcript/segments/{segment_id}/revisions")
@audited("transcript.revise", object_type="transcript_segment", object_param="segment_id")
def revise_segment(matter_id: str, meeting_id: str, segment_id: str, inp: RevisionIn, request: Request):
    """Correct a segment's text or speaker, with a reason. The raw text is kept; a correction taken
    from the matter's vocabulary carries that suggestion's provenance."""
    p = principal_of(request)
    _open_for_corrections(matter_id, meeting_id, p)
    if inp.text is None and inp.speaker_participant_id is None and not inp.suggestion_id:
        raise HTTPException(status_code=422, detail="Change the text or the speaker.")
    source, provenance = "person", {}
    if inp.suggestion_id:
        seg = next((s for s in transcript_mod.view(matter_id, meeting_id)["segments"] if s["id"] == segment_id), None)
        found = next((x for x in vocabulary.suggest(seg, vocabulary.terms(matter_id, meeting_id, p["sub"]))
                      if x["id"] == inp.suggestion_id), None) if seg else None
        if not found:
            raise HTTPException(status_code=409, detail="That suggestion no longer applies. Refresh the transcript.")
        source = "vocabulary"
        provenance = {"heard": found["heard"], "suggested": found["suggested"], "kind": found["kind"], **found["source"]}
        inp = RevisionIn(text=found["text"], reason=inp.reason)
    try:
        made = transcript_mod.revise(matter_id, meeting_id, segment_id, p["sub"], inp.reason, inp.text, inp.speaker_participant_id,
                                     source, provenance)
    except transcript_mod.NotFound:
        raise _not_found()
    except transcript_mod.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"kinds": [m["kind"] for m in made], "source": source})
    return {"revisions": made}


@router.get(_BASE + "/{meeting_id}/transcript/suggestions")
@audited("meeting.transcript.suggestions", object_type="meeting", object_param="meeting_id")
def transcript_suggestions(matter_id: str, meeting_id: str, request: Request):
    """Corrections from the matter's vocabulary (participants, companies, defined terms), not applied."""
    p = principal_of(request)
    require(p, "meetings.read", matter_id)
    _meeting(matter_id, meeting_id)
    vocab = vocabulary.terms(matter_id, meeting_id, p["sub"])
    found = [x for s in transcript_mod.view(matter_id, meeting_id)["segments"] for x in vocabulary.suggest(s, vocab)]
    note(details={"suggestions": len(found), "vocabulary": len(vocab)})
    return {"suggestions": found}


@router.post(_BASE + "/{meeting_id}/transcript/finalise")
@audited("transcript.finalise", object_type="meeting", object_param="meeting_id")
def finalise_transcript(matter_id: str, meeting_id: str, request: Request):
    """The transcript as corrected becomes a matter document (or a new version of it): research can
    cite it, and holds, purges and exports apply."""
    p = principal_of(request)
    m = _open_for_corrections(matter_id, meeting_id, p)
    require(p, "documents.import", matter_id)
    try:
        out = transcript_mod.finalise(p["org_id"], matter_id, {**m, "number": _number(matter_id)}, p["sub"])
    except transcript_mod.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"changed": out["changed"], "version_number": out.get("version_number")})
    return out


@ws_router.websocket(_BASE + "/{meeting_id}/capture")
async def capture_socket(ws: WebSocket, matter_id: str, meeting_id: str):
    from ...core.edition import is_law
    if not is_law():
        await ws.close(code=4404)
        return
    await capture.handle(ws, matter_id, meeting_id)


# ── Silent Assist (S2.5; D-041) ──────────────────────────────────────────────────
class AssistIn(BaseModel):
    pack: Optional[Literal["legal_client_call", "legal_negotiation", "legal_governance", "off"]] = None   # None: the kind's default


class DecisionIn(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


def _assist_state(m: dict) -> dict:
    from ...core.db import get_conn
    from ..assist import packs
    with get_conn() as conn:
        chosen = conn.execute("SELECT assist_pack FROM legal_meetings WHERE id=?", (m["id"],)).fetchone()[0]
    pack = packs.pack_for(m["kind"], chosen)
    return {"pack": pack, "chosen": chosen, "label": packs.LABELS.get(pack) if pack else None,
            "rules": [{"id": r, "label": packs.RULES[r].label, "severity": packs.RULES[r].severity} for r in packs.PACKS.get(pack, ())]}


@router.get(_BASE + "/{meeting_id}/alerts")
@audited("meeting.alerts.read", object_type="meeting", object_param="meeting_id")
def list_alerts(matter_id: str, meeting_id: str, request: Request):
    """The meeting's Silent Assist alerts, with the pack in use and how the alerts fared."""
    from ..assist import repo as assist_repo
    require(principal_of(request), "meetings.read", matter_id)
    m = _meeting(matter_id, meeting_id)
    alerts = assist_repo.list_for(matter_id, meeting_id)
    note(details={"alerts": len(alerts)})
    return {"alerts": alerts, "assist": _assist_state(m), "telemetry": assist_repo.telemetry(matter_id, meeting_id)}


@router.post(_BASE + "/{meeting_id}/alerts/{alert_id}/{decision}")
@audited("assist.decide", object_type="assist_alert", object_param="alert_id")
def decide_alert(matter_id: str, meeting_id: str, alert_id: str, decision: Literal["confirm", "dismiss"], inp: DecisionIn,
                 request: Request):
    """Confirm an alert, or dismiss it with a reason. Decided once."""
    from ..assist import repo as assist_repo
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    _meeting(matter_id, meeting_id)
    reason = (inp.reason or "").strip() or None
    if decision == "dismiss" and (not reason or len(reason) < 3):
        raise HTTPException(status_code=422, detail="Say why the alert is dismissed.")
    try:
        out = assist_repo.decide(matter_id, meeting_id, alert_id, "confirmed" if decision == "confirm" else "dismissed", p["sub"], reason)
    except assist_repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if out is None:
        raise _not_found()
    note(details={"decision": decision, "rule": out["rule"], "verdict": out["verdict"]})
    return {"alert": out}


@router.patch(_BASE + "/{meeting_id}/assist")
@audited("meeting.assist", object_type="meeting", object_param="meeting_id")
def set_assist(matter_id: str, meeting_id: str, inp: AssistIn, request: Request):
    """Choose the meeting's legal pack, go back to its kind's default, or turn alerts off."""
    from ...core.db import get_conn
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    m = _meeting(matter_id, meeting_id)
    with get_conn() as conn:
        conn.execute("UPDATE legal_meetings SET assist_pack=? WHERE id=?", (inp.pack, meeting_id))
        conn.commit()
    note(details={"pack": inp.pack or "default"})
    return {"assist": _assist_state(m)}


# ── Agenda and board pack (S2.6; D-042) ──────────────────────────────────────────
class AgendaIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    document_ids: list[str] = Field(default_factory=list, max_length=50)


@router.post(_BASE + "/{meeting_id}/agenda")
@audited("meeting.agenda.add", object_type="meeting", object_param="meeting_id")
def add_agenda_item(matter_id: str, meeting_id: str, inp: AgendaIn, request: Request):
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    _meeting(matter_id, meeting_id)
    try:
        item = agenda_mod.add(matter_id, meeting_id, inp.title.strip(), inp.document_ids, p["sub"])
    except agenda_mod.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    note(details={"documents": len(item["documents"])})
    return {"item": item}


@router.delete(_BASE + "/{meeting_id}/agenda/{item_id}")
@audited("meeting.agenda.remove", object_type="agenda_item", object_param="item_id")
def remove_agenda_item(matter_id: str, meeting_id: str, item_id: str, request: Request):
    p = principal_of(request)
    require(p, "meetings.create", matter_id)
    _meeting(matter_id, meeting_id)
    if not agenda_mod.remove(matter_id, meeting_id, item_id):
        raise _not_found()
    return {"ok": True}


# ── Minutes (S2.6; D-042): drafted from the full transcript, approved by a second person, then filed ──
class ItemEdit(BaseModel):
    id: str = Field(min_length=1, max_length=16)
    text: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    owner: Optional[str] = Field(default=None, max_length=200)
    deadline: Optional[str] = Field(default=None, max_length=200)
    remove: bool = False


class ItemAdd(BaseModel):
    agenda_item: Optional[str] = Field(default=None, max_length=64)
    kind: Literal["decision", "action", "unresolved", "discussion"]
    text: str = Field(min_length=1, max_length=2000)
    owner: Optional[str] = Field(default=None, max_length=200)
    deadline: Optional[str] = Field(default=None, max_length=200)


class MinutesEdit(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)
    items: list[ItemEdit] = Field(default_factory=list, max_length=200)
    add: list[ItemAdd] = Field(default_factory=list, max_length=50)
    reason: str = Field(min_length=3, max_length=500)


class ReviewIn(BaseModel):
    decision: Literal["approved", "changes_requested"]
    content_sha256: str = Field(min_length=64, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=500)


def _username(uid: str) -> str:
    from ...core.db import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    return r[0] if r else "someone"


def _transcript_live(matter_id: str, meeting_id: str) -> bool:
    from ...core.db import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT transcript_deleted_at FROM legal_meetings WHERE id=? AND matter_id=?", (meeting_id, matter_id)).fetchone()
    return bool(r) and r[0] is None


def _minutes(matter_id: str, meeting_id: str) -> list[dict]:
    from .. import artifacts
    return artifacts.versions(matter_id, "meeting", meeting_id, "minutes")


@router.post(_BASE + "/{meeting_id}/minutes/draft")
@audited("minutes.draft", object_type="meeting", object_param="meeting_id")
async def draft_minutes(matter_id: str, meeting_id: str, request: Request):
    """Minutes drafted by the model from the whole transcript as corrected, every item citing its lines."""
    from starlette.concurrency import run_in_threadpool
    from .. import artifacts
    from . import minutes as minutes_mod
    p = principal_of(request)
    await run_in_threadpool(require, p, "ai.draft", matter_id)
    m = await run_in_threadpool(_meeting, matter_id, meeting_id)
    if m["status"] != "ended":
        raise HTTPException(status_code=409, detail="End the meeting before drafting its minutes.")
    if not await run_in_threadpool(_transcript_live, matter_id, meeting_id) or not await run_in_threadpool(repo.segments, matter_id, meeting_id):
        raise HTTPException(status_code=409, detail="There is no transcript to draft minutes from.")
    agenda = await run_in_threadpool(agenda_mod.items, matter_id, meeting_id, p["sub"])
    content, inputs = await minutes_mod.draft(m, matter_id, agenda)
    out = await run_in_threadpool(artifacts.create, matter_id, "minutes", "meeting", meeting_id, content, inputs, p["sub"], "model")
    note(object_id=meeting_id, details={"version": out["version"], **content["summary"], "batches": inputs["batches"]})
    return {"minutes": out}


@router.get(_BASE + "/{meeting_id}/minutes")
@audited("minutes.read", object_type="meeting", object_param="meeting_id")
def read_minutes(matter_id: str, meeting_id: str, request: Request):
    require(principal_of(request), "meetings.read", matter_id)
    _meeting(matter_id, meeting_id)
    versions = _minutes(matter_id, meeting_id) if _transcript_live(matter_id, meeting_id) else []
    note(details={"versions": len(versions)})
    return {"versions": versions}


@router.post(_BASE + "/{meeting_id}/minutes/{artifact_id}/edit")
@audited("minutes.edit", object_type="ai_artifact", object_param="artifact_id")
def edit_minutes(matter_id: str, meeting_id: str, artifact_id: str, inp: MinutesEdit, request: Request):
    """A person's changes make a new version. Items keep their cited lines; an edited item says who
    edited it; an added item cites nothing and says so."""
    import copy
    from .. import artifacts
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    _meeting(matter_id, meeting_id)
    cur = artifacts.get(matter_id, artifact_id)
    if not cur or cur["subject_id"] != meeting_id or cur["kind"] != "minutes":
        raise _not_found()
    if cur["content_sha256"] != inp.content_sha256 or cur["status"] not in ("draft", "approved"):
        raise HTTPException(status_code=409, detail="These minutes changed since you opened them. Reload and try again.")
    content = copy.deepcopy(cur["content"])
    editor = _username(p["sub"])
    by_id = {it["id"]: it for sec in content["sections"] for it in sec["items"]}
    for e in inp.items:
        it = by_id.get(e.id)
        if not it:
            raise HTTPException(status_code=422, detail=f"There is no item {e.id}.")
        if e.remove:
            for sec in content["sections"]:
                sec["items"] = [x for x in sec["items"] if x["id"] != e.id]
            continue
        for field in ("text", "owner", "deadline"):
            value = getattr(e, field)
            if value is not None and value != it.get(field):
                it[field] = value.strip() or None
                it["edited_by"] = editor
    sections = {sec["agenda_item"]: sec for sec in content["sections"]}
    next_id = 1 + max((int(i["id"][1:]) for sec in content["sections"] for i in sec["items"] if i["id"][1:].isdigit()), default=0)
    for a in inp.add:
        sec = sections.get(a.agenda_item) or sections.get(None)
        if sec is None:
            sec = {"agenda_item": None, "title": "Other matters", "items": []}
            content["sections"].append(sec)
            sections[None] = sec
        sec["items"].append({"id": f"m{next_id}", "kind": a.kind, "text": a.text.strip(), "owner": a.owner if a.kind == "action" else None,
                             "deadline": a.deadline if a.kind == "action" else None, "sources": [], "status": "added_by_person",
                             "edited_by": editor})
        next_id += 1
    items = [i for sec in content["sections"] for i in sec["items"]]
    content["summary"] = {k + "s": sum(1 for i in items if i["kind"] == k) for k in ("decision", "action", "unresolved", "discussion")} | \
        {"not_verified": sum(1 for i in items if i["status"] != "verified")}
    out = artifacts.create(matter_id, "minutes", "meeting", meeting_id, content, {**cur["inputs"], "edited_from": cur["version"],
                                                                                 "reason": inp.reason}, p["sub"], "person")
    note(details={"version": out["version"], "edited": len(inp.items), "added": len(inp.add)})
    return {"minutes": out}


@router.post(_BASE + "/{meeting_id}/minutes/{artifact_id}/review")
@audited("minutes.review", object_type="ai_artifact", object_param="artifact_id")
def review_minutes(matter_id: str, meeting_id: str, artifact_id: str, inp: ReviewIn, request: Request):
    """Approve the draft you read (by its content hash), or ask for changes. Not by its author."""
    from .. import artifacts
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)
    _meeting(matter_id, meeting_id)
    cur = artifacts.get(matter_id, artifact_id)
    if not cur or cur["subject_id"] != meeting_id:
        raise _not_found()
    if inp.decision == "changes_requested" and not (inp.reason or "").strip():
        raise HTTPException(status_code=422, detail="Say what should change.")
    try:
        out = artifacts.decide(matter_id, artifact_id, inp.decision, p["sub"], inp.content_sha256, (inp.reason or "").strip() or None)
    except artifacts.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"decision": inp.decision, "version": out["version"]})
    return {"minutes": out}


@router.post(_BASE + "/{meeting_id}/minutes/{artifact_id}/file")
@audited("minutes.file", object_type="ai_artifact", object_param="artifact_id")
def file_minutes(matter_id: str, meeting_id: str, artifact_id: str, request: Request):
    """Approved minutes become a matter document, carrying the approval and the hash it bound."""
    import hashlib
    from ...core.db import get_conn
    from .. import artifacts
    from ..documents import blobs
    from ..documents import repo as docs
    from ..jobs import worker
    from . import minutes as minutes_mod
    p = principal_of(request)
    require(p, "documents.import", matter_id)
    m = _meeting(matter_id, meeting_id)
    cur = artifacts.get(matter_id, artifact_id)
    if not cur or cur["subject_id"] != meeting_id:
        raise _not_found()
    bound = artifacts.approved_hash(cur)
    if not bound:
        raise HTTPException(status_code=409, detail="Only approved minutes can be filed.")
    if cur.get("document_id") and docs.get_document(matter_id, cur["document_id"]):
        return {"document_id": cur["document_id"], "changed": False}
    approval = next(d for d in reversed(cur["decisions"]) if d["decision"] == "approved")
    body = minutes_mod.as_text(cur["content"], approval).encode("utf-8")
    doc_id = docs.create_document(p["org_id"], matter_id, f"Minutes — {m['title']}"[:300], "note", "final", p["sub"])
    version_id = docs.new_id()
    wrapped = blobs.store(matter_id, version_id, body)
    try:
        docs.add_version(version_id, doc_id, matter_id, f"minutes-{meeting_id[:8]}-v{cur['version']}.md", "text/markdown", len(body),
                         hashlib.sha256(body).hexdigest(), None, wrapped, p["sub"], status="final")
    except BaseException:
        blobs.remove(matter_id, version_id)
        raise
    with get_conn() as conn:
        conn.execute("UPDATE legal_ai_artifacts SET document_id=? WHERE id=?", (doc_id, artifact_id))
        conn.commit()
    worker.wake()
    note(details={"version": cur["version"], "content_sha256": bound})
    return {"document_id": doc_id, "changed": True}
