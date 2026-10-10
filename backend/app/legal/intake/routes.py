"""Intake assistant with conflict screening (S4.1, OPT-01; decision D-054).

Matter parties (always on; conflict searches read them):
- `GET  /api/legal/matters/{id}/parties` (members): the matter's live party register.
- `POST /api/legal/matters/{id}/parties` (`parties.manage`: lead, attorney): add a party (adverse, related, the client's group, other).
- `POST …/parties/{pid}/remove` (`parties.manage`): remove it with a reason; the history stays.

Intake (LEGAL_INTAKE; 404 when off):
- `GET  /api/legal/intake`, `GET …/intake/{iid}`: intakes for intake staff, lawyers, the conflicts team and records
  administrators. Screening matches and decision notes only for `conflicts.review`.
- `POST /api/legal/intake` (`intake.manage`): a phone or in-person intake taken by staff.
- `POST /api/legal/intake/links` (`intake.manage`; LEGAL_INTAKE_LINKS): an intake the enquirer answers through a single-use link.
- `POST …/{iid}/disclaimer`, `…/submit` (`intake.manage`; staff intakes): the disclaimer, then the scripted answers.
- `POST …/{iid}/decision` (`conflicts.review`): cleared or conflict, with a note.
- `POST …/{iid}/consent`, `…/consent/withdraw`, `…/recording`, `GET …/recording` (`intake.manage`): a call
  recording, only under the caller's consent.
- `POST …/{iid}/replies` (`intake.manage`), `…/replies/{rid}/approve` (`intake.approve`, not the author): replies.
- `POST …/{iid}/convert` (`matters.create`): a cleared intake becomes a client and matter.
- `POST …/{iid}/close` (`intake.manage`), `DELETE …/{iid}` (`intake.purge`): close, or purge the personal data.

Enquirer links (LEGAL_INTAKE and LEGAL_INTAKE_LINKS; no session, the token is the only key, rate-limited):
- `GET /api/intake-link/{token}`, `POST …/accept`, `POST …/submit`.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from . import repo
from .replies import TEMPLATES
from .script import DISCLAIMER, SCRIPT

parties_router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def intake_on() -> None:
    if not settings.LEGAL_INTAKE:
        raise _not_found()


def links_on() -> None:
    if not (settings.LEGAL_INTAKE and settings.LEGAL_INTAKE_LINKS):
        raise _not_found()


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(intake_on)])
public_router = APIRouter(prefix="/intake-link", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(links_on)])


# ── Matter parties ───────────────────────────────────────────────────────────────
class PartyIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    role: Literal["adverse", "related", "client_group", "other"]

    @field_validator("name")
    @classmethod
    def _named(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Give the party's name.")
        return v


class RemoveIn(BaseModel):
    reason: str = Field(min_length=1, max_length=300)


def _parties(matter_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT p.id, p.name, p.role, u.username, p.added_at FROM legal_matter_parties p LEFT JOIN users u ON u.id = p.added_by "
                            "WHERE p.matter_id=? AND p.removed_at IS NULL ORDER BY p.added_at, p.rowid", (matter_id,)).fetchall()
    return [{"id": i, "name": n, "role": r, "added_by": u, "added_at": a} for i, n, r, u, a in rows]


@parties_router.get("/matters/{matter_id}/parties")
def list_parties(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    return {"parties": _parties(matter_id), "can_manage": policy.authorize(p, "parties.manage", matter_id).allowed}


@parties_router.post("/matters/{matter_id}/parties")
@audited("matter.party.add", object_type="matter_party", object_param="matter_id")
def add_party(matter_id: str, inp: PartyIn, request: Request):
    p = principal_of(request)
    require(p, "parties.manage", matter_id)
    pid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_matter_parties (id, matter_id, name, role, added_by, added_at) VALUES (?,?,?,?,?,?)",
                     (pid, matter_id, inp.name, inp.role, p["sub"], policy.now_iso()))
        conn.commit()
    note(matter_id=matter_id, details={"role": inp.role})
    return {"party": next(x for x in _parties(matter_id) if x["id"] == pid)}


@parties_router.post("/matters/{matter_id}/parties/{party_id}/remove")
@audited("matter.party.remove", object_type="matter_party", object_param="party_id")
def remove_party(matter_id: str, party_id: str, inp: RemoveIn, request: Request):
    p = principal_of(request)
    require(p, "parties.manage", matter_id)
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_matter_parties SET removed_at=?, removed_by=?, removal_reason=? WHERE id=? AND matter_id=? AND removed_at IS NULL",
                         (policy.now_iso(), p["sub"], inp.reason.strip(), party_id, matter_id)).rowcount
        conn.commit()
    if not n:
        raise _not_found()
    note(matter_id=matter_id)
    return {"parties": _parties(matter_id)}


# ── Intake (firm staff) ──────────────────────────────────────────────────────────
class NewIn(BaseModel):
    channel: Literal["phone", "in_person"]


class DisclaimerIn(BaseModel):
    version: str = Field(min_length=1, max_length=60)


class SubmitIn(BaseModel):
    answers: dict


class DecisionIn(BaseModel):
    decision: Literal["cleared", "conflict"]
    note: Optional[str] = Field(default=None, max_length=1000)


class ConsentIn(BaseModel):
    method: Literal["verbal", "written"]


class ReplyIn(BaseModel):
    template: str = Field(min_length=1, max_length=40)
    request: Optional[str] = Field(default=None, max_length=600)


class ApproveIn(BaseModel):
    body_sha256: str = Field(min_length=64, max_length=64)


class ConvertIn(BaseModel):
    client_id: Optional[str] = Field(default=None, max_length=64)
    client_name: Optional[str] = Field(default=None, max_length=200)
    client_code: Optional[str] = Field(default=None, max_length=40)
    matter_number: str = Field(min_length=1, max_length=40)
    matter_title: str = Field(min_length=1, max_length=200)


def _can(p: dict) -> dict:
    a = lambda action: policy.authorize(p, action).allowed
    return {"manage": a("intake.manage"), "approve": a("intake.approve"), "conflicts": a("conflicts.review"), "purge": a("intake.purge"),
            "convert": a("matters.create"), "links": bool(settings.LEGAL_INTAKE_LINKS)}


def _reader(p: dict) -> dict:
    """Intake staff, lawyers, the conflicts team and records administrators read intakes; others get 403."""
    can = _can(p)
    if not (can["manage"] or can["conflicts"] or can["purge"]):
        require(p, "intake.manage")
    return can


def _view(p: dict, iid: str) -> dict:
    v = repo.view(iid, p["org_id"], conflicts=policy.authorize(p, "conflicts.review").allowed)
    if v is None:
        raise _not_found()
    return v


def _intake(p: dict, iid: str) -> dict:
    d = repo.get_row(iid, p["org_id"])
    if d is None:
        raise _not_found()
    return d


def _staff_only(d: dict) -> None:
    if d["channel"] == "link":
        raise HTTPException(status_code=409, detail="The enquirer answers this intake through their link.")


def _run(fn, *args):
    try:
        return fn(*args)
    except repo.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except LookupError:
        raise _not_found()


@router.get("/intake")
def list_intakes(request: Request):
    p = principal_of(request)
    can = _reader(p)
    return {"intakes": repo.listing(p["org_id"]), "can": can, "script": SCRIPT, "disclaimer": DISCLAIMER, "templates": list(TEMPLATES)}


@router.post("/intake")
@audited("intake.create", object_type="intake")
def create_intake(inp: NewIn, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    iid = repo.create(p["org_id"], inp.channel, p["sub"])
    note(object_id=iid, details={"kind": inp.channel})
    return _view(p, iid)


@router.post("/intake/links", dependencies=[Depends(links_on)])
@audited("intake.link.create", object_type="intake")
def create_link(request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    iid, token = repo.create_link(p["org_id"], p["sub"])
    note(object_id=iid)
    v = _view(p, iid)
    return {**v, "link": f"/intake/{token}", "expires_at": v["intake"]["link"]["expires_at"]}


@router.get("/intake/{iid}")
@audited("intake.read", object_type="intake", object_param="iid")
def read_intake(iid: str, request: Request):
    p = principal_of(request)
    _reader(p)
    return _view(p, iid)


@router.post("/intake/{iid}/disclaimer")
@audited("intake.disclaimer", object_type="intake", object_param="iid")
def accept_disclaimer(iid: str, inp: DisclaimerIn, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _staff_only(_intake(p, iid))
    _run(repo.accept_disclaimer, iid, inp.version, p["sub"])
    return _view(p, iid)


@router.post("/intake/{iid}/submit")
@audited("intake.submit", object_type="intake", object_param="iid")
def submit_intake(iid: str, inp: SubmitIn, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _staff_only(_intake(p, iid))
    _run(repo.submit, iid, inp.answers, p["sub"])
    v = _view(p, iid)
    note(details={"possible_conflict": v["intake"]["screening"]["result"] == "possible_conflict",
                  "names": 2 + len(v["intake"]["answers"].get("other_parties") or [])})
    return v


@router.post("/intake/{iid}/decision")
@audited("intake.decision", object_type="intake", object_param="iid")
def decide(iid: str, inp: DecisionIn, request: Request):
    p = principal_of(request)
    require(p, "conflicts.review")
    _intake(p, iid)
    _run(repo.decide, iid, inp.decision, inp.note, p["sub"])
    note(details={"status": inp.decision, "noted": bool((inp.note or "").strip())})
    return _view(p, iid)


@router.post("/intake/{iid}/consent")
@audited("intake.consent", object_type="intake", object_param="iid")
def record_consent(iid: str, inp: ConsentIn, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _staff_only(_intake(p, iid))
    _run(repo.record_consent, iid, inp.method, p["sub"])
    note(details={"method": inp.method})
    return _view(p, iid)


@router.post("/intake/{iid}/consent/withdraw")
@audited("intake.consent.withdraw", object_type="intake", object_param="iid")
def withdraw_consent(iid: str, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _intake(p, iid)
    _run(repo.withdraw_consent, iid, p["sub"])
    return _view(p, iid)


_AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".oga", ".webm", ".flac", ".aac")
_MAX_RECORDING = 50 * 1024 * 1024


@router.post("/intake/{iid}/recording")
@audited("intake.recording.add", object_type="intake", object_param="iid")
def add_recording(iid: str, request: Request, file: UploadFile = File(...)):
    p = principal_of(request)
    require(p, "intake.manage")
    _staff_only(_intake(p, iid))
    mime = (file.content_type or "").lower()
    if not mime.startswith("audio/") or not (file.filename or "").lower().endswith(_AUDIO_EXT):
        raise HTTPException(status_code=422, detail="Attach an audio recording (WAV, MP3, M4A, OGG, WebM, FLAC or AAC).")
    data = file.file.read(_MAX_RECORDING + 1)
    if not data or len(data) > _MAX_RECORDING:
        raise HTTPException(status_code=422, detail="A recording must be under 50 MB.")
    _run(repo.add_recording, iid, data, mime, p["sub"])
    note(details={"mime": mime, "bytes": len(data)})
    return _view(p, iid)


@router.get("/intake/{iid}/recording")
@audited("intake.recording.read", object_type="intake", object_param="iid")
def get_recording(iid: str, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _intake(p, iid)
    got = repo.recording(iid)
    if not got:
        raise _not_found()
    data, mime = got
    return Response(content=data, media_type=mime, headers={"Cache-Control": "no-store", "Content-Disposition": "inline"})


@router.post("/intake/{iid}/replies")
@audited("intake.reply.draft", object_type="intake", object_param="iid")
def draft_reply(iid: str, inp: ReplyIn, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _intake(p, iid)
    rid = _run(repo.draft_reply, iid, inp.template, inp.request, p["sub"])
    note(details={"kind": inp.template})
    v = _view(p, iid)
    return {**v, "reply": next(r for r in v["intake"]["replies"] if r["id"] == rid)}


@router.post("/intake/{iid}/replies/{rid}/approve")
@audited("intake.reply.approve", object_type="intake", object_param="iid")
def approve_reply(iid: str, rid: str, inp: ApproveIn, request: Request):
    p = principal_of(request)
    require(p, "intake.approve")
    _intake(p, iid)
    _run(repo.approve_reply, iid, rid, inp.body_sha256, p["sub"])
    v = _view(p, iid)
    return {**v, "reply": next(r for r in v["intake"]["replies"] if r["id"] == rid)}


@router.post("/intake/{iid}/convert")
@audited("intake.convert", object_type="intake", object_param="iid")
def convert(iid: str, inp: ConvertIn, request: Request):
    p = principal_of(request)
    require(p, "matters.create")
    _intake(p, iid)
    m = _run(repo.convert, iid, p["org_id"], p["sub"], inp.client_id, inp.client_name, inp.client_code, inp.matter_number, inp.matter_title)
    note(matter_id=m["id"], details={"new_client": not inp.client_id})
    return {**_view(p, iid), "matter": m}


@router.post("/intake/{iid}/close")
@audited("intake.close", object_type="intake", object_param="iid")
def close(iid: str, request: Request):
    p = principal_of(request)
    require(p, "intake.manage")
    _intake(p, iid)
    _run(repo.close, iid, p["sub"])
    return _view(p, iid)


@router.delete("/intake/{iid}")
@audited("intake.purge", object_type="intake", object_param="iid")
def purge(iid: str, request: Request):
    p = principal_of(request)
    require(p, "intake.purge")
    _intake(p, iid)
    _run(repo.purge, iid, p["sub"])
    return _view(p, iid)


# ── Enquirer links (no session) ──────────────────────────────────────────────────
_WINDOW, _PER_LINK, _PER_ADDRESS = 600.0, 30, 300
_seen: dict[str, deque] = {}
_lock = threading.Lock()


def _limited(key: str, limit: int) -> bool:
    now = time.monotonic()
    with _lock:
        q = _seen.setdefault(key, deque())
        while q and now - q[0] > _WINDOW:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False


def _link(token: str, request: Request) -> dict:
    address = request.client.host if request.client else "unknown"
    if _limited("link:" + repo.token_hash(token), _PER_LINK) or _limited("addr:" + address, _PER_ADDRESS):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a few minutes and try again.")
    d = repo.by_token(token) if 20 <= len(token) <= 100 else None
    if d is None:
        raise _not_found()
    note(object_id=d["id"], actor_username="intake-link")
    return d


class AnswersIn(BaseModel):
    answers: dict


@public_router.get("/{token}")
@audited("intake.link.read", object_type="intake", session_actor=False)
def link_page(token: str, request: Request):
    return repo.public_view(_link(token, request))


@public_router.post("/{token}/accept")
@audited("intake.link.accept", object_type="intake", session_actor=False)
def link_accept(token: str, inp: DisclaimerIn, request: Request):
    d = _link(token, request)
    _run(repo.accept_disclaimer, d["id"], inp.version, None)
    return repo.public_view(repo.by_token(token))


@public_router.post("/{token}/submit")
@audited("intake.link.submit", object_type="intake", session_actor=False)
def link_submit(token: str, inp: AnswersIn, request: Request):
    d = _link(token, request)
    _run(repo.submit, d["id"], inp.answers, None)
    return repo.public_view(repo.by_token(token))
