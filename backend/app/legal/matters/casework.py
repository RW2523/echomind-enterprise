"""Case management (owner's request 2026-11-07): what is happening on a matter, its plan, and where it stands.

- Updates: what happened (a call, a filing, a hearing, the client's instructions), posted by the matter's people. They
  are never changed or removed; a correction is a new update naming the one it corrects (migration 0042 triggers).
- Plan: tasks, key dates, hearings and filing deadlines, entered by a person (never from what a model read), each change
  kept as an event. A stage's standard tasks are added only when a person asks.
- Stage: where the matter stands in its track (pipeline.py); a move is also an update, so the history shows it.
- Pipeline: the open matters a person can see, by stage.
- Import: open many matters from a spreadsheet saved as CSV, previewed row by row first.

Every route asks the policy engine first; audit details carry kinds, counts and identifiers, never what was written.
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...core.db import get_conn
from ..audit.record import audited, note, record
from ..policy import _LIVE_MEMBERSHIP, now_iso
from . import pipeline, repo
from .routes import principal_of, require, require_law_edition

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

UpdateKind = Literal["note", "call", "meeting", "filing", "hearing", "order", "instruction", "milestone"]
TaskKind = Literal["task", "key_date", "hearing", "filing"]
TaskStatus = Literal["open", "done", "cancelled"]
Track = Literal["disputes", "transactions", "advisory"]
Sensitivity = Literal["standard", "restricted", "highly_restricted"]

MAX_UPDATES = 200
MAX_IMPORT_ROWS = 200
MAX_IMPORT_BYTES = 1024 * 1024
_EXCERPT = 160


def _new_id() -> str:
    return str(uuid.uuid4())


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _names(conn, ids) -> dict[str, str]:
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    qs = ",".join("?" * len(ids))
    return {r[0]: r[1] for r in conn.execute(
        f"SELECT id, COALESCE(display_name, username) FROM users WHERE id IN ({qs})", ids).fetchall()}


def _matter_or_404(matter_id: str) -> dict:
    m = repo.get_matter(matter_id)
    if not m:
        raise _not_found()
    return m


# ── Schemas ──────────────────────────────────────────────────────────────────────
class UpdateIn(BaseModel):
    kind: UpdateKind = "note"
    body: str = Field(min_length=1, max_length=4000)
    occurred_on: Optional[date] = None
    document_id: Optional[str] = Field(default=None, max_length=64)


class CorrectionIn(BaseModel):
    body: str = Field(min_length=1, max_length=4000)
    kind: Optional[UpdateKind] = None
    occurred_on: Optional[date] = None


class TaskIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    kind: TaskKind = "task"
    due_on: Optional[date] = None
    assignee_id: Optional[str] = Field(default=None, max_length=64)
    stage: Optional[str] = Field(default=None, max_length=40)


class TaskPatch(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    kind: Optional[TaskKind] = None
    due_on: Optional[date] = None
    clear_due: bool = False
    assignee_id: Optional[str] = Field(default=None, max_length=64)
    clear_assignee: bool = False
    stage: Optional[str] = Field(default=None, max_length=40)
    status: Optional[TaskStatus] = None
    version: int


class StandardTasksIn(BaseModel):
    stage: Optional[str] = Field(default=None, max_length=40)
    titles: Optional[list[str]] = Field(default=None, max_length=12)


class StageIn(BaseModel):
    stage: str = Field(min_length=1, max_length=40)
    track: Optional[Track] = None
    note: Optional[str] = Field(default=None, max_length=1000)
    effective_on: Optional[date] = None                  # when it moved, if earlier than today (a case brought in with its history)
    version: int


# ── Reading the records ──────────────────────────────────────────────────────────
_U_COLS = "id, matter_id, author_id, kind, body, occurred_on, document_id, corrects_id, stage_from, stage_to, created_at"
_U_KEYS = tuple(c.strip() for c in _U_COLS.split(","))
_T_COLS = "id, matter_id, title, kind, due_on, assignee_id, stage, status, source, created_by, created_at, updated_at, version"
_T_KEYS = tuple(c.strip() for c in _T_COLS.split(","))


def list_updates(matter_id: str, limit: int = MAX_UPDATES) -> list[dict]:
    """Newest first; each with its author's name, the update correcting it (if any) and its linked document's title."""
    with get_conn() as conn:
        rows = [dict(zip(_U_KEYS, r)) for r in conn.execute(
            f"SELECT {_U_COLS} FROM legal_matter_updates WHERE matter_id=? ORDER BY occurred_on DESC, created_at DESC LIMIT ?",
            (matter_id, limit)).fetchall()]
        names = _names(conn, [r["author_id"] for r in rows])
        linked = sorted({r["document_id"] for r in rows if r["document_id"]})
        docs = {r[0]: r[1] for r in conn.execute(
            f"SELECT id, title FROM legal_documents WHERE matter_id=? AND id IN ({','.join('?' * len(linked))})",
            (matter_id, *linked)).fetchall()} if linked else {}
    corrected = {r["corrects_id"]: r["id"] for r in rows if r["corrects_id"]}
    out = []
    for r in rows:
        out.append({**r, "author": names.get(r["author_id"], ""), "corrected_by": corrected.get(r["id"]),
                    "document_title": docs.get(r["document_id"]) if r["document_id"] else None})
    return out


def list_tasks(matter_id: str, include_closed: bool = True) -> list[dict]:
    with get_conn() as conn:
        rows = [dict(zip(_T_KEYS, r)) for r in conn.execute(
            f"SELECT {_T_COLS} FROM legal_matter_tasks WHERE matter_id=? {'' if include_closed else 'AND status=' + repr('open')} "
            "ORDER BY status='open' DESC, due_on IS NULL, due_on, created_at", (matter_id,)).fetchall()]
        names = _names(conn, [r["assignee_id"] for r in rows] + [r["created_by"] for r in rows])
    today = _today().isoformat()
    return [{**r, "assignee": names.get(r["assignee_id"]) if r["assignee_id"] else None, "created_by_name": names.get(r["created_by"], ""),
             "overdue": bool(r["status"] == "open" and r["due_on"] and r["due_on"] < today)} for r in rows]


def _task(conn, matter_id: str, task_id: str) -> Optional[dict]:
    row = conn.execute(f"SELECT {_T_COLS} FROM legal_matter_tasks WHERE id=? AND matter_id=?", (task_id, matter_id)).fetchone()
    return dict(zip(_T_KEYS, row)) if row else None


def _live_member(conn, matter_id: str, user_id: str) -> bool:
    return bool(conn.execute(f"SELECT 1 FROM matter_memberships mm WHERE mm.matter_id=:mid AND mm.user_id=:uid AND {_LIVE_MEMBERSHIP}",
                             {"mid": matter_id, "uid": user_id, "now": now_iso()}).fetchone())


def status_summary(matter_id: str, updates: int = 5, tasks: int = 5) -> dict:
    """The matter's stage, latest updates and next open tasks, for the chat's case-status card (no model involved)."""
    m = _matter_or_404(matter_id)
    ups = [u for u in list_updates(matter_id, limit=40) if not u["corrected_by"]][:updates]
    open_tasks = [t for t in list_tasks(matter_id) if t["status"] == "open"]
    return {"matter": m, "updates": ups, "tasks": open_tasks[:tasks], "open_tasks": len(open_tasks),
            "overdue": sum(1 for t in open_tasks if t["overdue"])}


# ── Updates ──────────────────────────────────────────────────────────────────────
def _check_date(d: Optional[date]) -> str:
    d = d or _today()
    if d > _today() + timedelta(days=1):
        raise HTTPException(status_code=422, detail="An update records what happened; plan future dates as tasks or key dates.")
    if d.year < 1900:
        raise HTTPException(status_code=422, detail="That date is not plausible.")
    return d.isoformat()


@router.get("/matters/{matter_id}/updates")
def get_updates(matter_id: str, request: Request):
    require(principal_of(request), "matter.read", matter_id)
    return {"updates": list_updates(matter_id)}


def _insert_update(conn, p: dict, matter_id: str, kind: str, body: str, occurred_on: str, *, document_id=None, corrects_id=None,
                   stage_from=None, stage_to=None) -> str:
    uid = _new_id()
    conn.execute("INSERT INTO legal_matter_updates (id, org_id, matter_id, author_id, kind, body, occurred_on, document_id, corrects_id, "
                 "stage_from, stage_to, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (uid, p["org_id"], matter_id, p["sub"], kind, body.strip(), occurred_on, document_id, corrects_id, stage_from, stage_to,
                  now_iso()))
    return uid


@router.post("/matters/{matter_id}/updates")
@audited("matter.update_posted", object_type="matter_update")
def post_update(matter_id: str, inp: UpdateIn, request: Request):
    p = principal_of(request)
    require(p, "updates.post", matter_id)
    occurred = _check_date(inp.occurred_on)
    if inp.document_id:
        from ..documents import repo as docs
        if not docs.get_document(matter_id, inp.document_id, p["sub"]):
            raise HTTPException(status_code=422, detail="That document is not in this matter.")
    with get_conn() as conn:
        uid = _insert_update(conn, p, matter_id, inp.kind, inp.body, occurred, document_id=inp.document_id)
        conn.commit()
    note(object_id=uid, matter_id=matter_id, details={"kind": inp.kind, "linked_document": bool(inp.document_id)})
    return {"update": next(u for u in list_updates(matter_id) if u["id"] == uid)}


@router.post("/matters/{matter_id}/updates/{update_id}/correct")
@audited("matter.update_corrected", object_type="matter_update", object_param="update_id")
def correct_update(matter_id: str, update_id: str, inp: CorrectionIn, request: Request):
    p = principal_of(request)
    require(p, "updates.post", matter_id)
    with get_conn() as conn:
        row = conn.execute("SELECT kind, occurred_on, document_id FROM legal_matter_updates WHERE id=? AND matter_id=?",
                           (update_id, matter_id)).fetchone()
        if not row:
            raise _not_found()
        if row[0] == "stage":
            raise HTTPException(status_code=409, detail="A stage change is corrected by moving the matter to the right stage.")
        if conn.execute("SELECT 1 FROM legal_matter_updates WHERE corrects_id=?", (update_id,)).fetchone():
            raise HTTPException(status_code=409, detail="This update has already been corrected; correct the correction instead.")
        occurred = _check_date(inp.occurred_on) if inp.occurred_on else row[1]
        uid = _insert_update(conn, p, matter_id, inp.kind or row[0], inp.body, occurred, document_id=row[2], corrects_id=update_id)
        conn.commit()
    note(matter_id=matter_id, details={"kind": inp.kind or row[0]})
    return {"update": next(u for u in list_updates(matter_id) if u["id"] == uid)}


# ── Plan ─────────────────────────────────────────────────────────────────────────
def _task_event(conn, task_id: str, matter_id: str, action: str, changes: dict, by: str) -> None:
    conn.execute("INSERT INTO legal_matter_task_events (id, task_id, matter_id, action, changes_json, by, at) VALUES (?,?,?,?,?,?,?)",
                 (_new_id(), task_id, matter_id, action, json.dumps(changes, sort_keys=True, default=str), by, now_iso()))


def _valid_stage(matter: dict, key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    if not pipeline.stage(matter["track"], key):
        raise HTTPException(status_code=422, detail="That stage is not part of this matter's track.")
    return key


@router.get("/matters/{matter_id}/tasks")
def get_tasks(matter_id: str, request: Request):
    require(principal_of(request), "matter.read", matter_id)
    return {"tasks": list_tasks(matter_id)}


@router.post("/matters/{matter_id}/tasks")
@audited("matter.task_added", object_type="matter_task")
def add_task(matter_id: str, inp: TaskIn, request: Request):
    p = principal_of(request)
    require(p, "plan.manage", matter_id)
    m = _matter_or_404(matter_id)
    stage = _valid_stage(m, inp.stage) or m["stage"]
    tid, now = _new_id(), now_iso()
    with get_conn() as conn:
        if inp.assignee_id and not _live_member(conn, matter_id, inp.assignee_id):
            raise HTTPException(status_code=422, detail="Only someone working on this matter can be given a task.")
        conn.execute("INSERT INTO legal_matter_tasks (id, org_id, matter_id, title, kind, due_on, assignee_id, stage, status, source, "
                     "created_by, created_at, updated_at, version) VALUES (?,?,?,?,?,?,?,?,'open','person',?,?,?,1)",
                     (tid, p["org_id"], matter_id, inp.title.strip(), inp.kind, inp.due_on.isoformat() if inp.due_on else None,
                      inp.assignee_id, stage, p["sub"], now, now))
        _task_event(conn, tid, matter_id, "created", {"kind": inp.kind, "due_on": inp.due_on, "assigned": bool(inp.assignee_id)}, p["sub"])
        conn.commit()
    note(object_id=tid, matter_id=matter_id, details={"kind": inp.kind, "due": bool(inp.due_on), "assigned": bool(inp.assignee_id)})
    return {"task": next(t for t in list_tasks(matter_id) if t["id"] == tid)}


@router.patch("/matters/{matter_id}/tasks/{task_id}")
@audited("matter.task_changed", object_type="matter_task", object_param="task_id")
def change_task(matter_id: str, task_id: str, inp: TaskPatch, request: Request):
    p = principal_of(request)
    require(p, "plan.manage", matter_id)
    m = _matter_or_404(matter_id)
    with get_conn() as conn:
        cur = _task(conn, matter_id, task_id)
        if not cur:
            raise _not_found()
        sets: dict = {}
        if inp.title is not None:
            sets["title"] = inp.title.strip()
        if inp.kind is not None:
            sets["kind"] = inp.kind
        if inp.clear_due:
            sets["due_on"] = None
        elif inp.due_on is not None:
            sets["due_on"] = inp.due_on.isoformat()
        if inp.clear_assignee:
            sets["assignee_id"] = None
        elif inp.assignee_id is not None:
            if not _live_member(conn, matter_id, inp.assignee_id):
                raise HTTPException(status_code=422, detail="Only someone working on this matter can be given a task.")
            sets["assignee_id"] = inp.assignee_id
        if inp.stage is not None:
            sets["stage"] = _valid_stage(m, inp.stage)
        if inp.status is not None:
            sets["status"] = inp.status
        sets = {k: v for k, v in sets.items() if cur[k] != v}
        if not sets:
            return {"task": next(t for t in list_tasks(matter_id) if t["id"] == task_id)}
        assigns = ", ".join(f"{k} = :{k}" for k in sets)
        res = conn.execute(f"UPDATE legal_matter_tasks SET {assigns}, version = version + 1, updated_at = :now "
                           f"WHERE id = :id AND matter_id = :mid AND version = :v",
                           {**sets, "now": now_iso(), "id": task_id, "mid": matter_id, "v": inp.version})
        if res.rowcount != 1:
            raise HTTPException(status_code=409, detail="This task was changed by someone else. Reload and try again.")
        status = sets.get("status")
        action = {"done": "done", "cancelled": "cancelled", "open": "reopened"}.get(status, "changed") if status else "changed"
        _task_event(conn, task_id, matter_id, action,
                    {k: (v if k in ("status", "kind", "due_on", "stage") else bool(v)) for k, v in sets.items()}, p["sub"])
        conn.commit()
    note(matter_id=matter_id, details={"fields": sorted(sets), "status": sets.get("status")})
    return {"task": next(t for t in list_tasks(matter_id) if t["id"] == task_id)}


@router.post("/matters/{matter_id}/plan/standard-tasks")
@audited("matter.standard_tasks_added", object_type="matter", object_param="matter_id")
def add_standard_tasks(matter_id: str, inp: StandardTasksIn, request: Request):
    """Add a stage's standard tasks (all of them, or the titles chosen from them), leaving out any already planned."""
    p = principal_of(request)
    require(p, "plan.manage", matter_id)
    m = _matter_or_404(matter_id)
    key = _valid_stage(m, inp.stage) or m["stage"]
    st = pipeline.stage(m["track"], key)
    wanted = list(st.tasks) if inp.titles is None else [t for t in st.tasks if t in set(inp.titles)]
    added, now = [], now_iso()
    with get_conn() as conn:
        have = {r[0].casefold() for r in conn.execute(
            "SELECT title FROM legal_matter_tasks WHERE matter_id=? AND status <> 'cancelled'", (matter_id,)).fetchall()}
        for title in wanted:
            if title.casefold() in have:
                continue
            tid = _new_id()
            conn.execute("INSERT INTO legal_matter_tasks (id, org_id, matter_id, title, kind, due_on, assignee_id, stage, status, source, "
                         "created_by, created_at, updated_at, version) VALUES (?,?,?,?,'task',NULL,NULL,?,'open','standard',?,?,?,1)",
                         (tid, p["org_id"], matter_id, title, key, p["sub"], now, now))
            _task_event(conn, tid, matter_id, "created", {"kind": "task", "standard": True}, p["sub"])
            added.append(tid)
        conn.commit()
    note(details={"stage": key, "added": len(added)})
    return {"added": len(added), "tasks": list_tasks(matter_id)}


# ── Stage ────────────────────────────────────────────────────────────────────────
@router.post("/matters/{matter_id}/stage")
@audited("matter.stage_moved", object_type="matter", object_param="matter_id")
def move_stage(matter_id: str, inp: StageIn, request: Request):
    p = principal_of(request)
    d = require(p, "stage.move", matter_id)
    m = _matter_or_404(matter_id)
    track = inp.track or m["track"]
    if not pipeline.stage(track, inp.stage):
        raise HTTPException(status_code=422, detail="That stage is not part of the track.")
    if track == m["track"] and inp.stage == m["stage"]:
        raise HTTPException(status_code=409, detail="The matter is already at that stage.")
    before = pipeline.stage_label(m["track"], m["stage"])
    after = pipeline.stage_label(track, inp.stage)
    body = (inp.note or "").strip() or (f"Moved from {before} to {after}." if before else f"Moved to {after}.")
    now = now_iso()
    on = _check_date(inp.effective_on)
    since = now if on == _today().isoformat() else f"{on}T09:00:00.000Z"
    with get_conn() as conn:
        res = conn.execute("UPDATE matters SET track=:t, stage=:s, stage_since=:since, version=version+1, updated_at=:now "
                           "WHERE id=:id AND version=:v", {"t": track, "s": inp.stage, "since": since, "now": now, "id": matter_id,
                                                          "v": inp.version})
        if res.rowcount != 1:
            raise HTTPException(status_code=409, detail="This matter was changed by someone else. Reload and try again.")
        _insert_update(conn, p, matter_id, "stage", body, on, stage_from=m["stage"], stage_to=inp.stage)
        conn.commit()
    note(details={"track": track, "stage_from": m["stage"], "stage_to": inp.stage})
    from .routes import _with_hold
    return {"matter": _with_hold([{**repo.get_matter(matter_id), "my_role": d.role}])[0]}


# ── Pipeline ─────────────────────────────────────────────────────────────────────
@router.get("/pipeline/tracks")
def get_tracks(request: Request):
    principal_of(request)
    return {"tracks": pipeline.public()}


@router.get("/pipeline")
@audited("pipeline.read", object_type="pipeline")
def get_pipeline(request: Request, closed: bool = Query(False)):
    """The matters this person may see, with their stage, lead, last update and next task (the listing already applies
    memberships, walls and break-glass)."""
    p = principal_of(request)
    matters = [m for m in repo.list_visible_matters(p) if closed or m["status"] == "open"]
    ids = [m["id"] for m in matters]
    if not ids:
        note(details={"matters": 0})
        return {"matters": [], "tracks": pipeline.public()}
    qs = ",".join("?" * len(ids))
    today = _today()
    with get_conn() as conn:
        leads: dict[str, list[str]] = {}
        for mid, name in conn.execute(
                f"SELECT mm.matter_id, COALESCE(u.display_name, u.username) FROM matter_memberships mm JOIN users u ON u.id = mm.user_id "
                f"WHERE mm.matter_id IN ({qs}) AND mm.role='lead' AND mm.revoked_at IS NULL AND mm.valid_from <= ? "
                f"AND (mm.valid_to IS NULL OR mm.valid_to > ?) ORDER BY mm.granted_at", (*ids, now_iso(), now_iso())).fetchall():
            leads.setdefault(mid, []).append(name)
        last: dict[str, dict] = {}
        for mid, kind, body, occurred in conn.execute(
                f"SELECT u.matter_id, u.kind, u.body, u.occurred_on FROM legal_matter_updates u WHERE u.matter_id IN ({qs}) "
                f"AND NOT EXISTS (SELECT 1 FROM legal_matter_updates c WHERE c.corrects_id = u.id) "
                f"ORDER BY u.occurred_on DESC, u.created_at DESC", ids).fetchall():
            last.setdefault(mid, {"kind": kind, "excerpt": body[:_EXCERPT] + ("…" if len(body) > _EXCERPT else ""), "occurred_on": occurred})
        nxt: dict[str, dict] = {}
        counts: dict[str, list[int]] = {}
        for mid, title, kind, due in conn.execute(
                f"SELECT matter_id, title, kind, due_on FROM legal_matter_tasks WHERE matter_id IN ({qs}) AND status='open' "
                f"ORDER BY due_on IS NULL, due_on, created_at", ids).fetchall():
            c = counts.setdefault(mid, [0, 0])
            c[0] += 1
            c[1] += bool(due and due < today.isoformat())
            nxt.setdefault(mid, {"title": title, "kind": kind, "due_on": due})
    out = []
    for m in matters:
        since = (m.get("stage_since") or m["created_at"] or "")[:10]
        try:
            days = (today - date.fromisoformat(since)).days
        except ValueError:
            days = None
        out.append({"id": m["id"], "number": m["number"], "title": m["title"], "client": m["client"]["name"], "status": m["status"],
                    "practice_area": m["practice_area"], "track": m["track"], "stage": m["stage"],
                    "stage_label": pipeline.stage_label(m["track"], m["stage"]), "days_in_stage": days, "my_role": m.get("my_role"),
                    "leads": leads.get(m["id"], []), "last_update": last.get(m["id"]), "next_task": nxt.get(m["id"]),
                    "open_tasks": counts.get(m["id"], [0, 0])[0], "overdue": counts.get(m["id"], [0, 0])[1]})
    note(details={"matters": len(out)})
    return {"matters": out, "tracks": pipeline.public()}


# ── For a new matter: the colleagues who can join it, and a free number ────────────
_LAWYERS = frozenset({"managing_partner", "attorney", "paralegal"})


@router.get("/colleagues")
def colleagues(request: Request):
    """The firm's active people (username and name), for choosing a new matter's team. Only for those who open matters."""
    p = principal_of(request)
    require(p, "matters.create")
    from .. import policy
    with get_conn() as conn:
        rows = conn.execute("SELECT id, username, COALESCE(display_name, username) FROM users WHERE org_id=? "
                            "AND COALESCE(status,'active')='active' ORDER BY COALESCE(display_name, username)", (p["org_id"],)).fetchall()
    return {"colleagues": [{"username": r[1], "display_name": r[2], "lawyer": bool(policy.firm_roles_of(r[0]) & _LAWYERS)}
                           for r in rows if r[0] != p["sub"]]}


@router.get("/matter-numbers/next")
def next_number(request: Request, prefix: str = Query("M-", max_length=16, pattern=r"^[A-Za-z0-9._ -]{0,16}$")):
    """The next number after the highest one with this prefix in the firm (numbers are unique firm-wide)."""
    p = principal_of(request)
    require(p, "matters.create")
    with get_conn() as conn:
        taken = [r[0] for r in conn.execute("SELECT number FROM matters WHERE org_id=? AND number LIKE ? ESCAPE '\\'",
                                            (p["org_id"], prefix.replace("%", "\\%").replace("_", "\\_") + "%")).fetchall()]
    nums = [int(t[len(prefix):]) for t in taken if t[len(prefix):].isdigit()]
    width = max([len(t[len(prefix):]) for t in taken if t[len(prefix):].isdigit()] or [3])
    return {"number": f"{prefix}{str((max(nums) if nums else 100) + 1).zfill(width)}"}


# ── Import from a spreadsheet ────────────────────────────────────────────────────
IMPORT_COLUMNS = ("number", "title", "client", "client_code", "practice_area", "jurisdiction", "sensitivity", "track", "stage",
                  "description", "other_parties")
_ALIASES = {"matter number": "number", "matter": "number", "case number": "number", "case": "number", "name": "title",
            "matter title": "title", "case title": "title", "client name": "client", "practice area": "practice_area",
            "area": "practice_area", "client code": "client_code", "other side": "other_parties", "other parties": "other_parties",
            "parties": "other_parties", "opposing party": "other_parties", "summary": "description"}
_SENSITIVITY = {"standard": "standard", "restricted": "restricted", "highly restricted": "highly_restricted",
                "highly_restricted": "highly_restricted", "": "restricted"}


def _header(h: str) -> str:
    k = " ".join((h or "").strip().casefold().replace("_", " ").split())
    return _ALIASES.get(k, k.replace(" ", "_"))


def _stage_key(track: str, text: str) -> Optional[str]:
    t = " ".join(text.casefold().split())
    for s in pipeline.TRACKS[track].stages:
        if t in (s.key.replace("_", " "), s.key, s.label.casefold()):
            return s.key
    return None


def _read_rows(data: bytes) -> list[dict]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = data.decode("cp1252")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="The file could not be read as text. Save the spreadsheet as CSV (UTF-8).")
    if "\x00" in text:
        raise HTTPException(status_code=422, detail="This is not a CSV file. Save the spreadsheet as CSV (UTF-8).")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise HTTPException(status_code=422, detail="The file has no rows.")
    head = [_header(h) for h in rows[0]]
    if "number" not in head or "title" not in head or "client" not in head:
        raise HTTPException(status_code=422, detail="The first row must name the columns, including number, title and client.")
    body = rows[1:]
    if len(body) > MAX_IMPORT_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_IMPORT_ROWS} matters can be imported at once.")
    return [{h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(head) if h in IMPORT_COLUMNS} for r in body]


def _plan_import(p: dict, rows: list[dict]) -> list[dict]:
    clients = repo.list_visible_clients(p)
    by_name = {c["name"].casefold(): c for c in clients}
    by_code = {(c["code"] or "").casefold(): c for c in clients if c["code"]}
    with get_conn() as conn:
        taken = {r[0].casefold() for r in conn.execute("SELECT number FROM matters WHERE org_id=?", (p["org_id"],)).fetchall()}
    seen: set[str] = set()
    out = []
    for i, r in enumerate(rows, start=2):
        errors = []
        number, title, client = r.get("number", ""), r.get("title", ""), r.get("client", "")
        if not number:
            errors.append("The matter number is missing.")
        elif len(number) > 64:
            errors.append("The matter number is longer than 64 characters.")
        elif number.casefold() in taken:
            errors.append("A matter with this number already exists.")
        elif number.casefold() in seen:
            errors.append("This number appears twice in the file.")
        seen.add(number.casefold())
        if not title:
            errors.append("The title is missing.")
        elif len(title) > 300:
            errors.append("The title is longer than 300 characters.")
        if not client:
            errors.append("The client is missing.")
        match = by_code.get(r.get("client_code", "").casefold()) if r.get("client_code") else None
        match = match or by_name.get(client.casefold())
        sens = _SENSITIVITY.get(r.get("sensitivity", "").casefold().replace("-", " "))
        if sens is None:
            errors.append("Sensitivity must be standard, restricted or highly restricted.")
        track_text = r.get("track", "").casefold()
        track = track_text if track_text in pipeline.TRACKS else (pipeline.suggest_track(r.get("practice_area")) if not track_text else None)
        if track is None:
            errors.append("Track must be disputes, transactions or advisory.")
        stage = None
        if track:
            stage = _stage_key(track, r["stage"]) if r.get("stage") else pipeline.first_stage(track)
            if stage is None:
                errors.append(f"“{r['stage'][:40]}” is not a stage of the {pipeline.TRACKS[track].label.lower()} track.")
        parties = [x.strip() for x in r.get("other_parties", "").replace("|", ";").split(";") if x.strip()][:20]
        for field, limit in (("practice_area", 120), ("jurisdiction", 120), ("description", 2000)):
            if len(r.get(field, "")) > limit:
                errors.append(f"The {field.replace('_', ' ')} is longer than {limit} characters.")
        if any(len(x) > 160 for x in parties):
            errors.append("An other party's name is longer than 160 characters.")
        out.append({"row": i, "number": number, "title": title, "client": client, "client_id": match["id"] if match else None,
                    "new_client": not match and bool(client), "practice_area": r.get("practice_area") or None,
                    "jurisdiction": r.get("jurisdiction") or None, "sensitivity": sens or "restricted", "track": track,
                    "stage": stage, "stage_label": pipeline.stage_label(track, stage) if track else "",
                    "description": r.get("description") or None, "other_parties": parties, "errors": errors})
    return out


@router.get("/matters-import/template")
def import_template(request: Request):
    principal_of(request)
    sample = ("number,title,client,client_code,practice_area,jurisdiction,sensitivity,track,stage,description,other_parties\r\n"
              "M-900,Example supply dispute (DEMO),Example Client Ltd (DEMO),EXAMPLE,Commercial litigation,England & Wales,"
              "restricted,disputes,Pleadings,Late delivery under the supply agreement,Example Supplier plc (DEMO)\r\n")
    return Response(sample, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="matters-import-template.csv"', "Cache-Control": "no-store"})


@router.post("/matters-import")
@audited("matters.import", object_type="matter")
def import_matters(request: Request, file: UploadFile = File(...), preview: bool = Query(True)):
    """Preview (the default) shows every row with its problems; importing opens each row that has none, the importer as
    its lead, and reports the rest. New clients are created only for rows that need one."""
    p = principal_of(request)
    require(p, "matters.create")
    data = file.file.read(MAX_IMPORT_BYTES + 1)
    if len(data) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="The file can be at most 1 MB.")
    rows = _plan_import(p, _read_rows(data))
    ok = [r for r in rows if not r["errors"]]
    if preview:
        note(details={"rows": len(rows), "ready": len(ok), "preview": True})
        return {"rows": rows, "ready": len(ok), "created": []}
    if any(r["new_client"] for r in ok):
        require(p, "clients.create")
    created, new_clients = [], {}
    for r in ok:
        cid = r["client_id"]
        if not cid:
            key = r["client"].casefold()
            if key not in new_clients:
                try:
                    c = repo.create_client(p["org_id"], r["client"][:200], None, p["sub"])
                except repo.Conflict:
                    r["errors"].append("The client could not be created.")
                    continue
                record("client.create", principal=p, object_type="client", object_id=c["id"], details={"method": "import"})
                new_clients[key] = c["id"]
            cid = new_clients[key]
        try:
            m = repo.create_matter(p["org_id"], cid, r["number"], r["title"], p["sub"], practice_area=r["practice_area"],
                                   jurisdiction=r["jurisdiction"], sensitivity=r["sensitivity"], description=r["description"],
                                   track=r["track"], stage=r["stage"], parties=tuple((n, "adverse") for n in r["other_parties"]))
        except repo.Conflict as e:
            r["errors"].append(str(e))
            continue
        record("matter.create", principal=p, object_type="matter", object_id=m["id"], matter_id=m["id"],
               details={"sensitivity": m["sensitivity"], "track": m["track"], "stage": m["stage"], "method": "import"})
        created.append({"row": r["row"], "id": m["id"], "number": m["number"]})
    note(details={"rows": len(rows), "created": len(created), "clients_created": len(new_clients), "preview": False})
    return {"rows": rows, "ready": len(ok), "created": created}
