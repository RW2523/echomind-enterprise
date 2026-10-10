"""Due diligence (S3.7, DOC-06; decision D-051).

Firm:
- `GET  /api/legal/diligence/checklists`: approved versions for every lawyer; drafts too for those who manage them.
- `POST /api/legal/diligence/checklists` (`diligence.manage`): a new checklist, or a new version of one (`family_id`).
- `POST /api/legal/diligence/checklists/{cid}/approve` (`diligence.approve`): a managing partner other than the author, for that hash.

Matter:
- `POST /api/legal/matters/{id}/diligence/runs` (`ai.draft`): run an approved checklist against the matter's documents (a job).
- `GET  …/diligence/runs`, `GET …/diligence/runs/{rid}`: runs, their items, missing documents and issues, for what the viewer can see.
- `GET  …/diligence/issues/{iid}`, `PATCH …/diligence/issues/{iid}` (lead, attorney, reviewer): verify, dismiss (with a reason) or reopen.
"""
from __future__ import annotations

import json
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..documents import repo as documents
from ..jobs import worker
from ..matters.routes import principal_of, require, require_law_edition
from . import checklists

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class ItemIn(BaseModel):
    id: str = Field(min_length=1, max_length=60)
    section: Optional[str] = Field(default=None, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    expect: list[str] = Field(min_length=1, max_length=12)
    questions: list[str] = Field(default_factory=list, max_length=6)


class ChecklistIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    expected_law: Optional[str] = Field(default=None, max_length=100)
    items: list[ItemIn] = Field(min_length=1, max_length=120)
    family_id: Optional[str] = Field(default=None, max_length=64)


class ApproveIn(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)


class RunIn(BaseModel):
    checklist_id: str = Field(min_length=1, max_length=64)


class DecideIn(BaseModel):
    expected_version: int = Field(ge=1)
    action: Literal["verify", "dismiss", "reopen"]
    reason: Optional[str] = Field(default=None, max_length=500)


def _manager(p: dict) -> bool:
    return policy.authorize(p, "diligence.manage").allowed or policy.authorize(p, "diligence.approve").allowed


@router.get("/diligence/checklists")
def list_checklists(request: Request):
    p = principal_of(request)
    return {"checklists": checklists.listing(p["org_id"], drafts=_manager(p)),
            "can": {"manage": policy.authorize(p, "diligence.manage").allowed, "approve": policy.authorize(p, "diligence.approve").allowed},
            "questions": list(checklists.QUESTIONS)}


@router.post("/diligence/checklists")
@audited("diligence.checklist.create", object_type="dd_checklist")
def create_checklist(inp: ChecklistIn, request: Request):
    p = principal_of(request)
    require(p, "diligence.manage")
    try:
        c = checklists.create(p["org_id"], inp.name, [i.model_dump() for i in inp.items], inp.expected_law, p["sub"], inp.family_id)
    except checklists.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError:
        raise _not_found()
    note(object_id=c["id"], details={"items": len(c["items"]), "version": c["version"]})
    return {"checklist": c}


@router.post("/diligence/checklists/{checklist_id}/approve")
@audited("diligence.checklist.approve", object_type="dd_checklist", object_param="checklist_id")
def approve_checklist(checklist_id: str, inp: ApproveIn, request: Request):
    p = principal_of(request)
    require(p, "diligence.approve")
    try:
        c = checklists.approve(p["org_id"], checklist_id, p["sub"], inp.content_sha256)
    except checklists.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"version": c["version"]})
    return {"checklist": c}


# ── Runs ────────────────────────────────────────────────────────────────────────────
_RCOLS = "r.id, r.matter_id, r.checklist_id, r.status, r.results_json, r.error, r.created_by, r.created_at, r.finished_at"
_ICOLS = ("id", "run_id", "matter_id", "item_id", "document_id", "version_id", "kind", "severity", "title", "detail", "evidence_json",
          "status", "reviewer_id", "decided_at", "version", "created_at")


def _username(conn, uid: Optional[str]) -> Optional[str]:
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


class _Seen:
    """Which documents this viewer can see, and their current titles (asked once each)."""

    def __init__(self, matter_id: str, uid: str):
        self.matter_id, self.uid, self.cache = matter_id, uid, {}

    def title(self, did: str) -> Optional[str]:
        if did not in self.cache:
            d = documents.get_document(self.matter_id, did, self.uid)
            self.cache[did] = d["title"] if d else None
        return self.cache[did]


def _issue_out(conn, r, seen: _Seen) -> Optional[dict]:
    d = dict(zip(_ICOLS, r))
    title = seen.title(d["document_id"])
    if title is None:
        return None
    return {"id": d["id"], "run_id": d["run_id"], "item_id": d["item_id"], "document_id": d["document_id"], "document_title": title,
            "version_id": d["version_id"], "kind": d["kind"], "severity": d["severity"], "title": d["title"], "detail": d["detail"],
            "evidence": json.loads(d["evidence_json"]), "status": d["status"], "reviewer": _username(conn, d["reviewer_id"]),
            "decided_at": d["decided_at"], "version": d["version"], "created_at": d["created_at"]}


def _run_view(conn, r, uid: str) -> dict:
    rid, mid, cid, status, rj, err, by, at, fin = r
    seen = _Seen(mid, uid)
    res = json.loads(rj or "{}")
    c = conn.execute("SELECT name, version FROM legal_dd_checklists WHERE id=?", (cid,)).fetchone()
    items = []
    for it in res.get("items", []):
        docs = [{"id": d["id"], "title": seen.title(d["id"])} for d in it["documents"] if seen.title(d["id"]) is not None]
        items.append({"item_id": it["item_id"], "title": it["title"], "section": it["section"], "documents": docs, "missing": it["missing"]})
    issues = [x for x in (_issue_out(conn, row, seen) for row in conn.execute(
        f"SELECT {', '.join(_ICOLS)} FROM legal_dd_issues WHERE run_id=? ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
        "created_at, rowid", (rid,))) if x]
    missing = [{"item_id": i["item_id"], "title": i["title"], "section": i["section"]} for i in items if i["missing"]]
    summary = {"items": len(items), "missing": len(missing), "issues": len(issues),
               **{s: sum(1 for i in issues if i["severity"] == s) for s in ("high", "medium", "low")},
               **{s: sum(1 for i in issues if i["status"] == s) for s in ("preliminary", "verified", "dismissed")}}
    return {"run": {"id": rid, "status": status, "error": err, "checklist": {"id": cid, "name": c[0] if c else None, "version": c[1] if c else None},
                    "created_by": _username(conn, by), "created_at": at, "finished_at": fin, "documents_read": res.get("documents_read")},
            "items": items, "missing": missing, "issues": issues, "summary": summary}


@router.post("/matters/{matter_id}/diligence/runs")
@audited("diligence.run.start", object_type="matter", object_param="matter_id")
def start_run(matter_id: str, inp: RunIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    c = checklists.get(p["org_id"], inp.checklist_id)
    if not c:
        raise _not_found()
    if c["status"] != "approved":
        raise HTTPException(status_code=409, detail="Only an approved checklist can be run; a managing partner approves it first.")
    rid, now = str(uuid.uuid4()), policy.now_iso()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_dd_runs (id, matter_id, checklist_id, status, results_json, error, created_by, created_at, finished_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (rid, matter_id, c["id"], "queued", "{}", None, p["sub"], now, None))
        conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) VALUES (?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), "diligence.run", p["org_id"], matter_id, rid, p["sub"], now))
        conn.commit()
    worker.wake()
    note(details={"run": rid, "items": len(c["items"])})
    return {"run": {"id": rid, "status": "queued", "checklist": {"id": c["id"], "name": c["name"], "version": c["version"]}}}


@router.get("/matters/{matter_id}/diligence/runs")
def list_runs(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_RCOLS} FROM legal_dd_runs r WHERE r.matter_id=? ORDER BY r.created_at DESC", (matter_id,)).fetchall()
        out = []
        for r in rows:
            v = _run_view(conn, r, p["sub"])
            out.append({**v["run"], "summary": v["summary"]})
        return {"runs": out}


@router.get("/matters/{matter_id}/diligence/runs/{run_id}")
def read_run(matter_id: str, run_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        r = conn.execute(f"SELECT {_RCOLS} FROM legal_dd_runs r WHERE r.id=? AND r.matter_id=?", (run_id, matter_id)).fetchone()
        if not r:
            raise _not_found()
        return _run_view(conn, r, p["sub"])


def _issue(p: dict, matter_id: str, issue_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_ICOLS)} FROM legal_dd_issues WHERE id=? AND matter_id=?", (issue_id, matter_id)).fetchone()
        out = _issue_out(conn, r, _Seen(matter_id, p["sub"])) if r else None
    if not out:
        raise _not_found()
    return out


@router.get("/matters/{matter_id}/diligence/issues/{issue_id}")
def read_issue(matter_id: str, issue_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    i = _issue(p, matter_id, issue_id)
    with get_conn() as conn:
        trail = [{"action": a, "reason": r, "by": _username(conn, by), "at": at} for a, r, by, at in conn.execute(
            "SELECT action, reason, by, at FROM legal_dd_issue_events WHERE issue_id=? ORDER BY at, rowid", (issue_id,))]
    return {"issue": i, "trail": trail}


@router.patch("/matters/{matter_id}/diligence/issues/{issue_id}")
@audited("diligence.issue.update", object_type="dd_issue", object_param="issue_id")
def decide_issue(matter_id: str, issue_id: str, inp: DecideIn, request: Request):
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)                 # lead, attorney or reviewer
    cur = _issue(p, matter_id, issue_id)
    if cur["version"] != inp.expected_version:
        raise HTTPException(status_code=409, detail="This issue changed since you opened it. Reload and try again.")
    reason = (inp.reason or "").strip() or None
    if inp.action == "dismiss" and (not reason or len(reason) < 3):
        raise HTTPException(status_code=422, detail="Say why it is not an issue.")
    to = {"verify": "verified", "dismiss": "dismissed", "reopen": "preliminary"}[inp.action]
    if (inp.action in ("verify", "dismiss") and cur["status"] != "preliminary") or (inp.action == "reopen" and cur["status"] == "preliminary"):
        raise HTTPException(status_code=409, detail=f"This issue is {cur['status']}.")
    now = policy.now_iso()
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_dd_issues SET status=?, reviewer_id=?, decided_at=?, version=version+1 WHERE id=? AND version=?",
                         (to, None if to == "preliminary" else p["sub"], None if to == "preliminary" else now, issue_id, cur["version"])).rowcount
        if n != 1:
            raise HTTPException(status_code=409, detail="This issue changed since you opened it. Reload and try again.")
        conn.execute("INSERT INTO legal_dd_issue_events (id, issue_id, action, reason, by, at) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4()), issue_id, inp.action, reason, p["sub"], now))
        conn.commit()
    note(details={"action": inp.action, "severity_high": cur["severity"] == "high"})
    return {"issue": _issue(p, matter_id, issue_id)}
