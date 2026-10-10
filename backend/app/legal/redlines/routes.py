"""Redlines (S3.3, DOC-02b; decision D-047).

- `POST /api/legal/matters/{id}/playbook-checks/{cid}/redline`: revise the lines found outside the
  playbook (or the topics named) to the firm's positions (`ai.draft`).
- `GET  …/documents/{did}/redlines`: the latest version of each redline of a document.
- `POST …/redlines/{aid}/edit`: a lawyer's changes (a new version): wording, a fallback, or a change removed.
- `GET  …/redlines/{aid}/word`: the redline as Word tracked changes, checked before it is sent (no-store).
Review and filing go through the S2.7 routes; filing adds the Word file as the document's next version.
"""
from __future__ import annotations

import copy
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.db import get_conn
from .. import artifacts
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from ..playbooks import library
from . import service

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _username(uid: str) -> str:
    with get_conn() as conn:
        r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    return r[0] if r else "someone"


class RedlineIn(BaseModel):
    topics: Optional[list[str]] = Field(default=None, max_length=20)
    choices: dict[str, str] = Field(default_factory=dict)


class ChangeEdit(BaseModel):
    id: str = Field(max_length=20)
    revised: Optional[str] = Field(default=None, max_length=8000)
    remove: bool = False
    choice: Optional[str] = Field(default=None, max_length=20)


class RedlineEdit(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)
    changes: list[ChangeEdit] = Field(default_factory=list, max_length=30)
    reason: Optional[str] = Field(default=None, max_length=500)


def _redline(p: dict, matter_id: str, artifact_id: str) -> dict:
    a = artifacts.get(matter_id, artifact_id)
    if not a or a["kind"] != "redline" or not artifacts.visible_to(a, p["sub"]):
        raise _not_found()
    return a


@router.post("/matters/{matter_id}/playbook-checks/{check_id}/redline")
@audited("redline.create", object_type="playbook_check", object_param="check_id")
async def create_redline(matter_id: str, check_id: str, inp: RedlineIn, request: Request):
    from ..playbooks.routes import _check
    p = principal_of(request)
    await run_in_threadpool(require, p, "ai.draft", matter_id)
    check = await run_in_threadpool(_check, p, matter_id, check_id)
    positions = {x["topic"]: x for x in await run_in_threadpool(library.approved_positions, p["org_id"], check["playbook"]["id"])}
    content, inputs = await service.propose(check, positions, check["playbook"]["name"] or "the playbook", inp.choices, inp.topics, matter_id)
    if not content["changes"]:
        raise HTTPException(status_code=409, detail="Nothing to revise: no line chosen is outside the playbook with a clause to change.")
    a = await run_in_threadpool(artifacts.create, matter_id, "redline", "redline", str(uuid.uuid4()), content, inputs, p["sub"], "model")
    note(details={"artifact": a["id"], **content["summary"]})
    return {"redline": a}


@router.get("/matters/{matter_id}/documents/{document_id}/redlines")
def list_redlines(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT a.id FROM legal_ai_artifacts a WHERE a.matter_id=? AND a.kind='redline' AND a.version = "
                            "(SELECT MAX(b.version) FROM legal_ai_artifacts b WHERE b.subject_id=a.subject_id AND b.kind='redline') "
                            "AND json_extract(a.content_json, '$.document.id')=? ORDER BY a.created_at DESC", (matter_id, document_id)).fetchall()
    out = [a for a in (artifacts.get(matter_id, r[0]) for r in rows) if a and artifacts.visible_to(a, p["sub"])]
    return {"redlines": [{"id": a["id"], "title": a["title"], "version": a["version"], "status": a["status"], "changes": len(a["content"]["changes"]),
                          "created_by_name": a["created_by_name"], "created_at": a["created_at"], "document_id": a.get("document_id")} for a in out]}


@router.post("/matters/{matter_id}/redlines/{artifact_id}/edit")
@audited("redline.edit", object_type="ai_artifact", object_param="artifact_id")
def edit_redline(matter_id: str, artifact_id: str, inp: RedlineEdit, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    cur = _redline(p, matter_id, artifact_id)
    latest = artifacts.latest(matter_id, "redline", cur["subject_id"], "redline")
    if cur["content_sha256"] != inp.content_sha256 or latest["id"] != cur["id"]:
        raise HTTPException(status_code=409, detail="This redline changed since you opened it. Reload and try again.")
    content = copy.deepcopy(cur["content"])
    editor = _username(p["sub"])
    positions = ({x["topic"]: x for x in library.approved_positions(p["org_id"], content["playbook"]["id"])}
                 if content.get("playbook") else {})              # S3.5 corrections have no playbook
    by_id = {c["id"]: c for c in content["changes"]}
    for e in inp.changes:
        c = by_id.get(e.id)
        if not c:
            raise HTTPException(status_code=422, detail=f"There is no change {e.id}.")
        if e.remove:
            content["changes"] = [x for x in content["changes"] if x["id"] != e.id]
            continue
        if e.choice:
            pos = positions.get(c["topic"])
            n = int(e.choice.split()[-1]) if e.choice.startswith("fallback") and e.choice.split()[-1].isdigit() else None
            if not pos or (n and not 1 <= n <= len(pos["fallbacks"])):
                raise HTTPException(status_code=422, detail=f"There is no {e.choice} for {c['title']}.")
            firm = pos["fallbacks"][n - 1]["text"] if n else pos["standard"]
            c.update(revised=service._number(c["original"]) + firm, fitted=False, fit_note=None,
                     **{"from": {"kind": "fallback" if n else "standard", "n": n, "position_id": pos["id"], "position_version": pos["version"], "text": firm}})
        if e.revised is not None and e.revised.strip():
            c["revised"] = e.revised.strip()
        c.update(ops=service.ops_of(c["original"], c["revised"]), edited_by=editor)
    if not content["changes"]:
        raise HTTPException(status_code=422, detail="A redline needs at least one change.")
    content["summary"] = {"changes": len(content["changes"]), "fitted": sum(bool(c.get("fitted")) for c in content["changes"])}
    out = artifacts.create(matter_id, "redline", "redline", cur["subject_id"], content,
                           {**cur["inputs"], "edited_from": cur["version"], "reason": inp.reason}, p["sub"], "person")
    note(details={"version": out["version"], "changes": len(inp.changes)})
    return {"redline": out}


@router.get("/matters/{matter_id}/redlines/{artifact_id}/word")
@audited("redline.word", object_type="ai_artifact", object_param="artifact_id")
def redline_word(matter_id: str, artifact_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    a = _redline(p, matter_id, artifact_id)
    try:
        data, info = service.to_word(matter_id, a, a["created_by_name"] or "EchoMind Law")
    except service.Unverified as e:
        raise HTTPException(status_code=409, detail="The Word file failed its checks and was not released: "
                                                    + ", ".join(k for k, ok in e.checks.items() if not ok))
    note(details={"bytes": len(data), "missed": len(info["missed"])})
    return Response(data, media_type=DOCX, headers={"Cache-Control": "no-store", "X-Redline-Missed": str(len(info["missed"])),
                                                    "Content-Disposition": f'attachment; filename="redline-{artifact_id[:8]}.docx"'})


def filed(p: dict, a: dict) -> tuple[str, str]:
    """Filing an approved redline: the Word file with tracked changes as the document's next version
    (status draft). Returns (document id, version id)."""
    data, _ = service.to_word(a["matter_id"], a, a["created_by_name"] or "EchoMind Law")
    stem = a["content"]["document"]["filename"].rsplit(".", 1)[0]
    vid = artifacts.file_version(p, a, a["content"]["document"]["id"], data, f"{stem} — redline v{a['version']}.docx", DOCX, "draft")
    return a["content"]["document"]["id"], vid
