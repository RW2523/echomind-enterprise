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
from . import capture, consent, repo

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
    return {"meeting": d}


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
    require(principal_of(request), "meetings.read", matter_id)
    _meeting(matter_id, meeting_id)
    return {"segments": repo.segments(matter_id, meeting_id)}


@ws_router.websocket(_BASE + "/{meeting_id}/capture")
async def capture_socket(ws: WebSocket, matter_id: str, meeting_id: str):
    from ...core.edition import is_law
    if not is_law():
        await ws.close(code=4404)
        return
    await capture.handle(ws, matter_id, meeting_id)
