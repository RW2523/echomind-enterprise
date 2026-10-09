"""/api/legal/connectors and /api/legal/matters/{id}/connector-links (S1.11, SEC-08).

Firm administrators configure connectors (connectors.manage) but never see which matters use
them or what they hold: only counts. A matter's lead links a folder of a connector to the matter.
"""
from __future__ import annotations

import json
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.db import get_conn
from ..audit.record import audited, note
from ..documents import blobs
from ..matters.routes import principal_of, require, require_law_edition
from ..policy import now_iso
from . import sync as syncmod
from .local_folder import clean_folder, root_allowed

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


class ConnectorIn(BaseModel):
    kind: Literal["local_folder"]
    name: str = Field(min_length=1, max_length=120)
    root: str = Field(min_length=1, max_length=1000)
    sync_minutes: int = Field(default=15, ge=1, le=1440)
    stale_after_minutes: int = Field(default=60, ge=1, le=10080)


class LinkIn(BaseModel):
    connector_id: str = Field(min_length=1, max_length=64)
    remote_path: str = Field(min_length=1, max_length=500)


def _public(conn, c: dict) -> dict:
    links = conn.execute("SELECT COUNT(*) FROM legal_connector_links WHERE connector_id=? AND removed_at IS NULL", (c["id"],)).fetchone()[0]
    docs = conn.execute("SELECT COUNT(*) FROM legal_documents WHERE connector_id=? AND deleted_at IS NULL", (c["id"],)).fetchone()[0]
    runs = conn.execute("SELECT started_at, finished_at, status, report_json FROM legal_connector_runs WHERE connector_id=? "
                        "ORDER BY started_at DESC LIMIT 5", (c["id"],)).fetchall()
    fresh = conn.execute("SELECT last_ok_at IS NOT NULL AND (julianday('now') - julianday(last_ok_at)) * 1440 <= stale_after_minutes "
                         "FROM legal_connectors WHERE id=?", (c["id"],)).fetchone()[0]
    return {"id": c["id"], "kind": c["kind"], "name": c["name"], "status": c["status"], "sync_minutes": c["sync_minutes"],
            "stale_after_minutes": c["stale_after_minutes"], "last_sync_at": c["last_sync_at"], "last_ok_at": c["last_ok_at"],
            "last_status": c["last_status"], "last_error": c["last_error"], "fresh": bool(fresh) and c["status"] == "active",
            "links": links, "documents": docs,
            "runs": [{"started_at": r[0], "finished_at": r[1], "status": r[2], "report": json.loads(r[3] or "{}")} for r in runs]}


def _all(org_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(syncmod._C_COLS)} FROM legal_connectors WHERE org_id=? ORDER BY created_at", (org_id,)).fetchall()
        return [_public(conn, dict(zip(syncmod._C_COLS, r))) for r in rows]


def _one(org_id: str, cid: str) -> dict:
    c = syncmod.connector_row(cid, org_id)
    if not c:
        raise HTTPException(status_code=404, detail="Not Found")
    return c


@router.get("/connectors")
def list_connectors(request: Request):
    p = principal_of(request)
    require(p, "connectors.manage")
    return {"connectors": _all(p["org_id"]), "providers": [
        {"kind": "local_folder", "name": "Folder on this server", "available": True},
        {"kind": "dms", "name": "Firm document management system", "available": False,
         "note": "Needs the firm's choice of provider, credentials and network approval (P-010)."}]}


@router.post("/connectors")
@audited("connector.create", object_type="connector")
def create_connector(inp: ConnectorIn, request: Request):
    p = principal_of(request)
    require(p, "connectors.manage")
    if not root_allowed(inp.root):
        raise HTTPException(status_code=422, detail="That folder is not one EchoMind may read (LEGAL_CONNECTOR_LOCAL_ROOTS).")
    cid = str(uuid.uuid4())
    sealed = blobs.seal(json.dumps({"root": inp.root}).encode(), cid)
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_connectors (id, org_id, kind, name, config_sealed, status, sync_minutes, stale_after_minutes, "
                     "created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (cid, p["org_id"], inp.kind, inp.name.strip(), sealed, "active", inp.sync_minutes, inp.stale_after_minutes,
                      p["sub"], now_iso()))
        conn.commit()
    note(object_id=cid, details={"kind": inp.kind})
    return {"connector": next(c for c in _all(p["org_id"]) if c["id"] == cid)}


@router.post("/connectors/{connector_id}/sync")
@audited("connector.sync", object_type="connector", object_param="connector_id")
def sync_now(connector_id: str, request: Request):
    p = principal_of(request)
    require(p, "connectors.manage")
    _one(p["org_id"], connector_id)
    run = syncmod.sync(connector_id, p["sub"])
    note(details={"status": run["status"]})
    return {"run": run}


def _set_status(connector_id: str, request: Request, status: str):
    p = principal_of(request)
    require(p, "connectors.manage")
    _one(p["org_id"], connector_id)
    with get_conn() as conn:
        conn.execute("UPDATE legal_connectors SET status=? WHERE id=?", (status, connector_id))
        conn.commit()
    return {"connector": next(c for c in _all(p["org_id"]) if c["id"] == connector_id)}


@router.post("/connectors/{connector_id}/disable")
@audited("connector.disable", object_type="connector", object_param="connector_id")
def disable(connector_id: str, request: Request):
    return _set_status(connector_id, request, "disabled")


@router.post("/connectors/{connector_id}/enable")
@audited("connector.enable", object_type="connector", object_param="connector_id")
def enable(connector_id: str, request: Request):
    return _set_status(connector_id, request, "active")


# ── Links (a matter's lead) ──────────────────────────────────────────────────────
@router.get("/matters/{matter_id}/connector-links")
def list_links(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        links = conn.execute("SELECT l.id, l.connector_id, k.name, l.remote_path, (SELECT username FROM users WHERE id=l.created_by), "
                             "l.created_at, k.status, k.last_ok_at, k.last_status FROM legal_connector_links l "
                             "JOIN legal_connectors k ON k.id = l.connector_id WHERE l.matter_id=? AND l.removed_at IS NULL "
                             "ORDER BY l.created_at", (matter_id,)).fetchall()
        connectors = conn.execute("SELECT id, name, kind FROM legal_connectors WHERE org_id=? AND status='active' ORDER BY name",
                                  (p["org_id"],)).fetchall()
    return {"links": [dict(zip(("id", "connector_id", "connector_name", "remote_path", "created_by", "created_at", "connector_status",
                                "last_ok_at", "last_status"), r)) for r in links],
            "connectors": [{"id": r[0], "name": r[1], "kind": r[2]} for r in connectors]}


@router.post("/matters/{matter_id}/connector-links")
@audited("connector.link", object_type="connector_link")
def add_link(matter_id: str, inp: LinkIn, request: Request):
    p = principal_of(request)
    require(p, "matter.update", matter_id)
    folder = clean_folder(inp.remote_path)
    if folder is None:
        raise HTTPException(status_code=422, detail="Give a folder inside the connector, such as Clients/Harrowgate.")
    c = syncmod.connector_row(inp.connector_id, p["org_id"])
    if not c or c["status"] != "active":
        raise HTTPException(status_code=404, detail="Not Found")
    lid = str(uuid.uuid4())
    with get_conn() as conn:
        if conn.execute("SELECT 1 FROM legal_connector_links WHERE connector_id=? AND matter_id=? AND remote_path=? AND removed_at IS NULL",
                        (c["id"], matter_id, folder)).fetchone():
            raise HTTPException(status_code=409, detail="This folder is already linked to the matter.")
        conn.execute("INSERT INTO legal_connector_links (id, connector_id, matter_id, remote_path, created_by, created_at) VALUES (?,?,?,?,?,?)",
                     (lid, c["id"], matter_id, folder, p["sub"], now_iso()))
        conn.commit()
    note(object_id=lid, details={"connector": c["id"]})
    return {"link": {"id": lid, "connector_id": c["id"], "connector_name": c["name"], "remote_path": folder}}


@router.delete("/matters/{matter_id}/connector-links/{link_id}")
@audited("connector.unlink", object_type="connector_link", object_param="link_id")
def remove_link(matter_id: str, link_id: str, request: Request):
    """Stop syncing a folder. Documents already synced stay (with their permissions) until deleted."""
    p = principal_of(request)
    require(p, "matter.update", matter_id)
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_connector_links SET removed_at=? WHERE id=? AND matter_id=? AND removed_at IS NULL",
                         (now_iso(), link_id, matter_id)).rowcount
        conn.commit()
    if not n:
        raise HTTPException(status_code=404, detail="Not Found")
    return {"ok": True}
