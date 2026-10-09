"""Synchronising connector links into matters (S1.11, SEC-08; spec test T15).

For each folder a matter's lead linked:
- an item without usable permissions is not imported (`acl_missing`), and a document already
  synced from it becomes visible to nobody until its permissions are fixed;
- a new item is imported through the normal path (type checks, encryption, indexing job, audit),
  visible only to the people its source names (and, at every read, only if they are on the matter);
- a changed item becomes a new version; permissions are replaced each time (`acl_changed`);
- an item gone from the source is hidden at once and queued for the usual approved purge — unless
  the matter is under a legal hold, which keeps it (`held`).
A run that cannot read the source is `failed` and changes nothing; documents then hide by
themselves once the connector's freshness window has passed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Optional

from ...core.db import get_conn
from ..audit.record import record
from ..documents import blobs, repo as documents, validate
from ..jobs import worker
from ..policy import now_iso
from ..records import deletion, holds
from .base import ConnectorError
from .local_folder import LocalFolderConnector

logger = logging.getLogger(__name__)
SYSTEM = "system:connector"
_C_COLS = ("id", "org_id", "kind", "name", "config_sealed", "status", "sync_minutes", "stale_after_minutes", "last_sync_at",
           "last_ok_at", "last_status", "last_error", "created_by", "created_at")


def connector_row(connector_id: str, org_id: Optional[str] = None) -> Optional[dict]:
    q = f"SELECT {', '.join(_C_COLS)} FROM legal_connectors WHERE id=?"
    args: list = [connector_id]
    if org_id:
        q += " AND org_id=?"
        args.append(org_id)
    with get_conn() as conn:
        r = conn.execute(q, args).fetchone()
    return dict(zip(_C_COLS, r)) if r else None


def config_of(c: dict) -> dict:
    return json.loads(blobs.unseal(c["config_sealed"], c["id"]))


def instance(c: dict):
    if c["kind"] == "local_folder":
        return LocalFolderConnector(config_of(c)["root"])
    raise ConnectorError("unknown_kind")


def _links(connector_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT l.id, l.matter_id, l.remote_path, l.created_by, m.org_id FROM legal_connector_links l "
                            "JOIN matters m ON m.id = l.matter_id WHERE l.connector_id=? AND l.removed_at IS NULL", (connector_id,)).fetchall()
    return [dict(zip(("id", "matter_id", "remote_path", "created_by", "org_id"), r)) for r in rows]


def _user_ids(org_id: str, usernames: list[str]) -> set[str]:
    if not usernames:
        return set()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT id FROM users WHERE org_id=? AND COALESCE(status,'active')='active' "
                            f"AND username IN ({','.join('?' * len(usernames))})", (org_id, *usernames)).fetchall()
    return {r[0] for r in rows}


def _import(c: dict, link: dict, item, data: bytes, document_id: Optional[str]) -> str:
    checked = validate.check(item.name, None, data)
    if len(data) > validate.max_bytes():
        raise validate.Rejected(413, "too_large")
    if document_id is None:
        title = os.path.splitext(checked.filename)[0] or checked.filename
        document_id = documents.create_synced_document(link["org_id"], link["matter_id"], title, link["created_by"], c["id"],
                                                       link["id"], item.remote_id)
    vid = documents.new_id()
    wrapped = blobs.store(link["matter_id"], vid, data)
    try:
        documents.add_version(vid, document_id, link["matter_id"], checked.filename, checked.mime, len(data),
                              hashlib.sha256(data).hexdigest(), checked.pages, wrapped, link["created_by"])
    except BaseException:
        blobs.remove(link["matter_id"], vid)
        raise
    return document_id


def _sync_link(c: dict, conn_impl, link: dict, report: dict) -> None:
    known = documents.find_remote(link["id"])
    seen = set()
    for item in conn_impl.list(link["remote_path"]):
        if item.unsafe:
            report["skipped_unsafe"] += 1
            continue
        seen.add(item.remote_id)
        existing = known.get(item.remote_id)
        if existing and existing["deleted"]:
            continue                                         # removed earlier; awaiting or after purge
        if item.acl is None:
            report["acl_missing"] += 1
            if existing:
                documents.set_acl(existing["id"], set())     # permissions lost at the source: nobody
            continue
        allowed = _user_ids(link["org_id"], item.acl)
        if not allowed and not existing:
            report["acl_denied"] += 1                        # nobody it names has an account here
            continue
        try:
            if existing is None:
                did = _import(c, link, item, conn_impl.fetch(link["remote_path"], item.remote_id), None)
                documents.set_acl(did, allowed)
                documents.set_remote_version(did, item.version)
                report["imported"] += 1
                continue
            if existing["remote_version"] != item.version:
                _import(c, link, item, conn_impl.fetch(link["remote_path"], item.remote_id), existing["id"])
                documents.set_remote_version(existing["id"], item.version)
                report["updated"] += 1
            else:
                report["unchanged"] += 1
                if existing["state"] != "current":
                    documents.set_source_state(existing["id"], "current")
            if documents.set_acl(existing["id"], allowed):
                report["acl_changed"] += 1
        except validate.Rejected:
            report["rejected"] += 1
        except ConnectorError:
            report["errors"] += 1
    for remote_id, doc in known.items():
        if remote_id in seen or doc["deleted"]:
            continue
        try:
            deletion.request(link["org_id"], link["matter_id"], "document", doc["id"], f"Removed at the source ({c['name']})",
                             SYSTEM, origin="connector")
            report["removed"] += 1
        except holds.Held:
            documents.set_source_state(doc["id"], "removed_in_source")
            report["held"] += 1
        except (deletion.Conflict, deletion.NotFound):
            pass


def sync(connector_id: str, by: Optional[str] = None) -> dict:
    c = connector_row(connector_id)
    rid, started = str(uuid.uuid4()), now_iso()
    report = {k: 0 for k in ("links", "imported", "updated", "unchanged", "removed", "held", "acl_missing", "acl_denied",
                             "acl_changed", "rejected", "skipped_unsafe", "errors")}
    status, error = "ok", None
    try:
        if c is None or c["status"] != "active":
            raise ConnectorError("disabled")
        impl = instance(c)
        impl.check()
        for link in _links(connector_id):
            report["links"] += 1
            _sync_link(c, impl, link, report)
        if report["acl_missing"] or report["acl_denied"] or report["rejected"] or report["skipped_unsafe"] or report["errors"]:
            status = "partial"
    except ConnectorError as e:
        status, error = "failed", str(e)
    except Exception as e:                                   # never leave a connector looking healthy by accident
        logger.error("Connectors: sync of %s failed (%s)", connector_id, type(e).__name__)
        status, error = "failed", type(e).__name__
    finished = now_iso()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_connector_runs (id, connector_id, started_at, finished_at, status, report_json) VALUES (?,?,?,?,?,?)",
                     (rid, connector_id, started, finished, status, json.dumps({**report, "error": error})))
        if status == "failed":
            conn.execute("UPDATE legal_connectors SET last_sync_at=?, last_status=?, last_error=? WHERE id=?", (finished, status, error, connector_id))
        else:
            conn.execute("UPDATE legal_connectors SET last_sync_at=?, last_ok_at=?, last_status=?, last_error=NULL WHERE id=?",
                         (finished, finished, status, connector_id))
        conn.commit()
    if report["imported"] or report["updated"]:
        worker.wake()
    record("connector.sync.run", "allow" if status != "failed" else "error", actor_user_id=by, object_type="connector",
           object_id=connector_id, reason=status, details={k: v for k, v in report.items() if v})
    return {"id": rid, "started_at": started, "finished_at": finished, "status": status, "report": report, "error": error}


def due() -> list[str]:
    """Active connectors whose next scheduled sync is due."""
    with get_conn() as conn:
        rows = conn.execute("SELECT id FROM legal_connectors WHERE status='active' AND (last_sync_at IS NULL "
                            "OR (julianday('now') - julianday(last_sync_at)) * 1440 >= sync_minutes)").fetchall()
    return [r[0] for r in rows]
