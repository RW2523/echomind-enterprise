"""Defined-term and cross-reference checks (S3.5, DOC-04; decision D-049).

- `POST /api/legal/matters/{id}/documents/{did}/definitions-check`: check a document (`research.query`: any member
  who can see it). Rules only, no model.
- `GET  …/documents/{did}/definitions-checks`, `GET …/definitions-checks/{cid}`: for those who can see the document.
- `POST …/definitions-checks/{cid}/notes`: a lawyer notes a finding (lead, attorney, reviewer). Append-only.
- `POST …/definitions-checks/{cid}/redline`: the fixes a lawyer picks become a redline (S3.3): a second lawyer
  approves it before it is filed as the document's next version, with tracked changes. Nothing changes here.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.db import get_conn
from .. import artifacts, policy
from ..audit.record import audited, note
from ..documents import numbering
from ..documents import repo as documents
from ..matters.routes import principal_of, require, require_law_edition
from ..redlines import service as redlines
from . import check as checkmod

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class CheckIn(BaseModel):
    version_id: Optional[str] = Field(default=None, max_length=64)


class NoteIn(BaseModel):
    finding: str = Field(min_length=1, max_length=16)
    note: str = Field(min_length=1, max_length=1000)


class Pick(BaseModel):
    finding: str = Field(min_length=1, max_length=16)
    fix: int = Field(ge=0, le=9)


class RedlineIn(BaseModel):
    fixes: list[Pick] = Field(default_factory=list, max_length=60)


_COLS = "id, matter_id, document_id, version_id, results_json, created_by, created_at"


def _version(matter_id: str, document_id: str, version_id: Optional[str]) -> tuple[str, str, int]:
    with get_conn() as conn:
        if version_id:
            r = conn.execute("SELECT v.id, t.text, v.number FROM legal_document_versions v JOIN legal_version_text t ON t.version_id = v.id "
                             "WHERE v.id=? AND v.document_id=? AND v.matter_id=?", (version_id, document_id, matter_id)).fetchone()
        else:
            r = conn.execute("SELECT v.id, t.text, v.number FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                             "JOIN legal_version_text t ON t.version_id = v.id WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
    if not r:
        raise LookupError(document_id)
    return r[0], r[1], r[2]


def _rows(version_id: str) -> list[dict]:
    with get_conn() as conn:
        return [{"number": n, "heading": h, "level": lv, "parent": p, "start": s, "end": e} for n, h, lv, p, s, e in conn.execute(
            "SELECT number, heading, level, parent, char_start, char_end FROM legal_clauses WHERE version_id=? ORDER BY char_start, level",
            (version_id,))]


def run(matter_id: str, document_id: str, version_id: Optional[str] = None) -> dict:
    vid, text, number = _version(matter_id, document_id, version_id)
    findings = checkmod.analyse(text, _rows(vid))
    for i, f in enumerate(findings, 1):
        f["id"] = f"f{i}"
    terms = len({d["term"] for d in checkmod.definitions(text)})
    return {"version_id": vid, "version_number": number, "findings": findings, "definitions": terms, "engine": checkmod.VERSION}


def _out(conn, r) -> dict:
    cid, mid, did, vid, rj, by, at = r
    stored = json.loads(rj)
    notes: dict[str, list] = {}
    for fid, text, who, when in conn.execute(
            "SELECT finding_id, note, (SELECT username FROM users u WHERE u.id = n.by), at FROM legal_definition_notes n WHERE check_id=? "
            "ORDER BY at, rowid", (cid,)):
        notes.setdefault(fid, []).append({"note": text, "by": who, "at": when})
    findings = stored["findings"]
    for f in findings:
        f["notes"] = notes.get(f["id"], [])
    return {"id": cid, "document_id": did, "version_id": vid, "version_number": stored.get("version_number"), "findings": findings,
            "summary": checkmod.summary(findings, stored.get("definitions")), "engine": stored.get("engine"), "advisory": True,
            "created_by": (conn.execute("SELECT username FROM users WHERE id=?", (by,)).fetchone() or [None])[0], "created_at": at}


def _check(p: dict, matter_id: str, check_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {_COLS} FROM legal_definition_checks WHERE id=? AND matter_id=?", (check_id, matter_id)).fetchone()
        if not r or not documents.visible(matter_id, r[2], p["sub"]):
            raise _not_found()
        return _out(conn, r)


@router.post("/matters/{matter_id}/documents/{document_id}/definitions-check")
@audited("check.definitions", object_type="document", object_param="document_id")
async def run_check(matter_id: str, document_id: str, request: Request, inp: Optional[CheckIn] = None):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    if not await run_in_threadpool(documents.visible, matter_id, document_id, p["sub"]):
        raise _not_found()
    try:
        result = await run_in_threadpool(run, matter_id, document_id, inp.version_id if inp else None)
    except LookupError:
        raise _not_found()
    cid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_definition_checks (id, matter_id, document_id, version_id, results_json, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?)", (cid, matter_id, document_id, result["version_id"], json.dumps(result, ensure_ascii=False),
                                                p["sub"], policy.now_iso()))
        conn.commit()
    out = await run_in_threadpool(_check, p, matter_id, cid)
    note(details={"check": cid, **out["summary"]})
    return {"check": out}


@router.get("/matters/{matter_id}/documents/{document_id}/definitions-checks")
def list_checks(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    if not documents.visible(matter_id, document_id, p["sub"]):
        raise _not_found()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_COLS} FROM legal_definition_checks WHERE matter_id=? AND document_id=? ORDER BY created_at DESC",
                            (matter_id, document_id)).fetchall()
        return {"checks": [_out(conn, r) for r in rows]}


@router.get("/matters/{matter_id}/definitions-checks/{check_id}")
def read_check(matter_id: str, check_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    return {"check": _check(p, matter_id, check_id)}


@router.post("/matters/{matter_id}/definitions-checks/{check_id}/notes")
@audited("check.definitions.note", object_type="definition_check", object_param="check_id")
def add_note(matter_id: str, check_id: str, inp: NoteIn, request: Request):
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)                 # lead, attorney or reviewer
    current = _check(p, matter_id, check_id)
    if not any(f["id"] == inp.finding for f in current["findings"]):
        raise HTTPException(status_code=422, detail=f"This check has no finding {inp.finding}.")
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_definition_notes (id, check_id, finding_id, note, by, at) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4()), check_id, inp.finding, inp.note.strip(), p["sub"], policy.now_iso()))
        conn.commit()
    note(details={"finding": inp.finding})
    return {"check": _check(p, matter_id, check_id)}


def proposal(matter_id: str, check: dict, picks: list[Pick]) -> tuple[dict, dict]:
    """The picked fixes as a redline's content: one change per clause (or paragraph), every fix in it applied."""
    by_id = {f["id"]: f for f in check["findings"]}
    chosen, seen = [], set()
    for pk in picks:
        f = by_id.get(pk.finding)
        if not f or pk.fix >= len(f["fixes"]):
            raise ValueError(f"There is no fix {pk.fix + 1} for finding {pk.finding}.")
        if pk.finding in seen:
            raise ValueError(f"Pick one fix for finding {pk.finding}.")
        seen.add(pk.finding)
        chosen.append((f, f["fixes"][pk.fix]))
    if not chosen:
        raise ValueError("Pick at least one fix.")
    doc = redlines._document(matter_id, check["document_id"], check["version_id"])
    text = doc.pop("text")
    rows = numbering.complete(text, _rows(check["version_id"]))
    units: dict[tuple[int, int], dict] = {}
    for f, fx in chosen:
        c = next((r for r in rows if r["number"] == fx.get("clause")), None)
        if c and c["start"] <= fx["start"] and fx["end"] <= c["end"]:
            key, info = (c["start"], c["end"]), {"number": c["number"], "heading": c.get("heading"), "start": c["start"], "end": c["end"]}
        else:                                              # outside any clause (a preamble): its paragraph
            a = text.rfind("\n", 0, fx["start"]) + 1
            b = text.find("\n", fx["end"])
            b = len(text) if b < 0 else b
            key, info = (a, b), {"number": None, "heading": None, "start": a, "end": b}
        units.setdefault(key, {"clause": info, "fixes": []})["fixes"].append((f, fx))
    changes = []
    for (a, b), u in sorted(units.items()):
        raw = text[a:b]
        spans = sorted(u["fixes"], key=lambda x: x[1]["start"])
        for (_, x), (_, y) in zip(spans, spans[1:]):
            if y["start"] < x["end"]:
                raise ValueError("Two of the fixes picked change the same words; pick one.")
        for f, fx in reversed(spans):
            raw = raw[:fx["start"] - a] + fx["revised"] + raw[fx["end"] - a:]
        number = u["clause"]["number"]
        original = redlines.own_text(number, text[a:b])
        revised = redlines.own_text(number, raw) if raw.strip() else ""
        reasons = " ".join(f["message"] for f, _ in spans)
        labels = "; ".join(fx["label"] for _, fx in spans)
        changes.append({"id": f"c{len(changes) + 1}", "topic": spans[0][0]["type"], "title": labels, "clause": u["clause"],
                        "original": original, "revised": revised, "ops": redlines.ops_of(original, revised), "reason": reasons,
                        "comment": f"{reasons} Correction: {labels}.", "fitted": False, "fit_note": None,
                        "from": {"kind": "fix", "n": None, "findings": [f["id"] for f, _ in spans], "position_id": None,
                                 "position_version": None, "text": labels}})
    content = {"title": f"Corrections — {doc['title']}", "document": doc, "origin": "definitions", "check_id": check["id"],
               "playbook": None, "changes": changes, "summary": {"changes": len(changes), "fitted": 0}}
    inputs = {"definitions_check_id": check["id"], "engine": checkmod.VERSION,
              "fixes": [{"finding": pk.finding, "fix": pk.fix} for pk in picks]}
    return content, inputs


@router.post("/matters/{matter_id}/definitions-checks/{check_id}/redline")
@audited("redline.create", object_type="definition_check", object_param="check_id")
def create_redline(matter_id: str, check_id: str, inp: RedlineIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    check = _check(p, matter_id, check_id)
    try:
        content, inputs = proposal(matter_id, check, inp.fixes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError:
        raise _not_found()
    a = artifacts.create(matter_id, "redline", "redline", str(uuid.uuid4()), content, inputs, p["sub"], "person")
    note(details={"artifact": a["id"], "changes": len(content["changes"]), "fixes": len(inp.fixes)})
    return {"redline": a}
