"""The export gateway (S1.10, SEC-06; policy P-006 safe defaults).

request (export.request + the membership's can_export) → redaction plan, adjustable by the
requester → submit → approval by a different lead or attorney (export.approve), who must
acknowledge it when the document is labelled privileged or work product → render → verify →
watermark "CONFIDENTIAL — <matter> — <requester> — <time>" → stored encrypted for
LEGAL_EXPORT_TTL_HOURS → downloaded only by the requester or the approver, never cached.
An export whose verification fails is not released.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..documents import blobs
from ..policy import now_iso
from ..records import labels
from . import detect, render, source, verify

FOLDER = "exports"
_COLS = ("id", "org_id", "matter_id", "object_type", "object_id", "source_version_id", "title", "format", "purpose", "status",
         "plan_json", "requested_by", "requested_at", "submitted_at", "decided_by", "decided_at", "decision_note", "privileged_ack",
         "watermark", "wrapped_key", "output_sha256", "output_bytes", "verification_json", "expires_at", "downloads", "source_sha256")
PROTECTED_LABELS = ("privileged", "work_product")


class Refused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _now_iso() -> str:
    return now_iso()


def _row(conn, matter_id: str, export_id: str) -> Optional[dict]:
    r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_exports WHERE id=? AND matter_id=?", (export_id, matter_id)).fetchone()
    return dict(zip(_COLS, r)) if r else None


def _label(e: dict) -> Optional[str]:
    if e["object_type"] != "document":
        return None
    cur = labels.state("document", e["object_id"])["current"]
    return cur["label"] if cur else None


def _name(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _out(conn, e: dict) -> dict:
    plan = json.loads(e["plan_json"])
    return {"id": e["id"], "object_type": e["object_type"], "object_id": e["object_id"], "title": e["title"], "format": e["format"],
            "purpose": e["purpose"], "status": e["status"], "plan": plan, "label": _label(e),
            "requested_by": _name(conn, e["requested_by"]), "requested_at": e["requested_at"], "submitted_at": e["submitted_at"],
            "decided_by": _name(conn, e["decided_by"]), "decided_at": e["decided_at"], "decision_note": e["decision_note"],
            "watermark": e["watermark"], "output_bytes": e["output_bytes"], "output_sha256": e["output_sha256"],
            "verification": json.loads(e["verification_json"]) if e["verification_json"] else None,
            "expires_at": e["expires_at"], "downloads": e["downloads"], "source_sha256": e["source_sha256"]}


def _expire_if_due(conn, e: dict) -> dict:
    if e["status"] == "ready" and e["expires_at"] and e["expires_at"] <= _now_iso():
        blobs.remove(e["matter_id"], e["id"], FOLDER)
        conn.execute("UPDATE legal_exports SET status='expired', wrapped_key=NULL WHERE id=?", (e["id"],))
        conn.commit()
        e = {**e, "status": "expired", "wrapped_key": None}
    return e


def get(matter_id: str, export_id: str) -> Optional[dict]:
    with get_conn() as conn:
        e = _row(conn, matter_id, export_id)
        return _out(conn, _expire_if_due(conn, e)) if e else None


def listing(matter_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_exports WHERE matter_id=? ORDER BY requested_at DESC LIMIT 100",
                            (matter_id,)).fetchall()
        return [_out(conn, _expire_if_due(conn, dict(zip(_COLS, r)))) for r in rows]


def _bound(src: "source.Source") -> str:
    """S2.7 (D-043): the hash of exactly what the redactions are planned on — the version and the text a
    reader sees. An approval renders only the same text; a new version or a corrected transcript differs."""
    return hashlib.sha256(f"{src.kind}\n{src.version_id or ''}\n{src.text}".encode("utf-8")).hexdigest()


def _approval(object_type: str, object_id: str, src: "source.Source") -> Optional[dict]:
    """A document filed from an approved AI draft carries that approval; a copy changed since is refused."""
    if object_type != "document":
        return None
    from .. import artifacts
    with get_conn() as conn:
        filing = artifacts.filing_for_document(conn, object_id)
        if not filing:
            return None
        sha = conn.execute("SELECT sha256 FROM legal_document_versions WHERE id=?", (src.version_id,)).fetchone()
    if filing["version_id"] != src.version_id or not sha or sha[0] != filing["body_sha256"]:
        what = artifacts.KINDS[filing["kind"]][1].lower()
        raise Refused(409, f"This document was filed from approved {what} and has changed since it was approved. "
                           "Have the changes approved as a new version before exporting it.")
    return {"kind": filing["kind"], "version": filing["version"], "approved_by": filing["approved_by"],
            "approved_at": filing["approved_at"], "content_sha256": filing["content_sha256"]}


def create(p: dict, matter_id: str, object_type: str, object_id: str, fmt: str, purpose: str) -> dict:
    if object_type == "document":
        from ..documents import repo as documents
        if not documents.visible(matter_id, object_id, p["sub"]):
            raise Refused(404, "Not Found")
    try:
        src = source.load(matter_id, object_type, object_id)
    except source.NotFound:
        raise Refused(404, "Not Found")
    approval = _approval(object_type, object_id, src)
    items = detect.detect(src.text, matter_id, p["org_id"], detect.parties_for(matter_id, object_type, object_id))
    plan = {"items": items, "source": src.kind, "warnings": {"image_pages": src.image_pages}}
    if approval:
        plan["approval"] = approval
    eid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_exports (id, org_id, matter_id, object_type, object_id, source_version_id, title, format, purpose, "
                     "status, plan_json, requested_by, requested_at, source_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (eid, p["org_id"], matter_id, object_type, object_id, src.version_id, src.title, fmt, purpose, "draft",
                      json.dumps(plan), p["sub"], _now_iso(), _bound(src)))
        conn.commit()
    return get(matter_id, eid)


def update_plan(p: dict, matter_id: str, export_id: str, toggles: list[dict], add: list[str]) -> dict:
    with get_conn() as conn:
        e = _row(conn, matter_id, export_id)
        if not e:
            raise Refused(404, "Not Found")
        if e["requested_by"] != p["sub"]:
            raise Refused(403, "Only the person who asked for this export can change its redactions.")
        if e["status"] != "draft":
            raise Refused(409, "This export was already submitted; its redactions can no longer change.")
        plan = json.loads(e["plan_json"])
        by_id = {i["id"]: i for i in plan["items"]}
        for t in toggles:
            if t.get("id") in by_id:
                by_id[t["id"]]["redact"] = bool(t.get("redact"))
        for text in add:
            text = (text or "").strip()
            if len(text) >= 2 and not any(i["kind"] == "manual" and i["text"] == text for i in plan["items"]):
                plan["items"].append(detect.manual(text))
        conn.execute("UPDATE legal_exports SET plan_json=? WHERE id=?", (json.dumps(plan), export_id))
        conn.commit()
    return get(matter_id, export_id)


def submit(p: dict, matter_id: str, export_id: str) -> dict:
    with get_conn() as conn:
        e = _row(conn, matter_id, export_id)
        if not e:
            raise Refused(404, "Not Found")
        if e["requested_by"] != p["sub"]:
            raise Refused(403, "Only the person who asked for this export can submit it.")
        if e["status"] != "draft":
            raise Refused(409, f"This export is {e['status'].replace('_', ' ')}.")
        conn.execute("UPDATE legal_exports SET status='pending_approval', submitted_at=? WHERE id=?", (_now_iso(), export_id))
        conn.commit()
    return get(matter_id, export_id)


def _decide_checks(p: dict, e: Optional[dict]) -> dict:
    if not e:
        raise Refused(404, "Not Found")
    if e["status"] != "pending_approval":
        raise Refused(409, f"This export is {e['status'].replace('_', ' ')}, not waiting for approval.")
    if e["requested_by"] == p["sub"]:
        raise Refused(403, "A different lawyer on the matter must approve this export.")
    return e


def approve(p: dict, matter_id: str, matter_number: str, export_id: str, acknowledge_privilege: bool) -> dict:
    with get_conn() as conn:
        e = _decide_checks(p, _row(conn, matter_id, export_id))
        requester = _name(conn, e["requested_by"])
    if e["object_type"] == "document":
        from ..documents import repo as documents
        if not documents.visible(matter_id, e["object_id"], p["sub"]):
            raise Refused(404, "Not Found")           # an approver must be able to see what they release
    lab = _label(e)
    if lab in PROTECTED_LABELS and not acknowledge_privilege:
        raise Refused(409, f"This document is labelled {lab.replace('_', ' ')}. Confirm that releasing it is intended.")
    try:
        src = source.load(matter_id, e["object_type"], e["object_id"])
    except source.NotFound:
        raise Refused(409, "The source of this export no longer exists.")
    if e["source_sha256"] != _bound(src):
        with get_conn() as conn:
            conn.execute("UPDATE legal_exports SET status='failed', decided_by=?, decided_at=?, verification_json=? WHERE id=?",
                         (p["sub"], _now_iso(), json.dumps({"ok": False, "source_changed": True}), export_id))
            conn.commit()
        raise Refused(409, "This export was requested before exports were bound to their text. Ask for a new export."
                      if not e["source_sha256"] else
                      "The document changed after this export was requested, so its redactions were planned on other text. "
                      "Ask for a new export.")
    if json.loads(e["plan_json"]).get("approval"):
        _approval(e["object_type"], e["object_id"], src)             # still the approved copy
    strings = [i["text"] for i in json.loads(e["plan_json"])["items"] if i["redact"]]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    watermark = f"CONFIDENTIAL — {matter_number} — {requester} — {stamp}"
    data = render.render({"format": e["format"], "source": src, "items": strings, "watermark": watermark})
    result = verify.verify(e["format"], data, strings, watermark)
    fields = {"decided_by": p["sub"], "decided_at": _now_iso(), "privileged_ack": int(bool(acknowledge_privilege)),
              "watermark": watermark, "verification_json": json.dumps(result)}
    if result["ok"]:
        fields |= {"status": "ready", "wrapped_key": blobs.store(matter_id, export_id, data, FOLDER),
                   "output_sha256": hashlib.sha256(data).hexdigest(), "output_bytes": len(data),
                   "expires_at": (datetime.now(timezone.utc) + timedelta(hours=settings.LEGAL_EXPORT_TTL_HOURS))
                   .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"}
    else:
        fields["status"] = "failed"
    with get_conn() as conn:
        conn.execute(f"UPDATE legal_exports SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), export_id))
        conn.commit()
    return get(matter_id, export_id)


def reject(p: dict, matter_id: str, export_id: str, note: Optional[str]) -> dict:
    with get_conn() as conn:
        _decide_checks(p, _row(conn, matter_id, export_id))
        conn.execute("UPDATE legal_exports SET status='rejected', decided_by=?, decided_at=?, decision_note=? WHERE id=?",
                     (p["sub"], _now_iso(), note, export_id))
        conn.commit()
    return get(matter_id, export_id)


def file(p: dict, matter_id: str, export_id: str) -> tuple[dict, bytes]:
    with get_conn() as conn:
        e = _row(conn, matter_id, export_id)
        if not e or p["sub"] not in (e["requested_by"], e["decided_by"]):
            raise Refused(404, "Not Found")
        if e["object_type"] == "document":
            from ..documents import repo as documents
            if not documents.visible(matter_id, e["object_id"], p["sub"]):
                raise Refused(404, "Not Found")
        e = _expire_if_due(conn, e)
        if e["status"] == "expired":
            raise Refused(410, "This export has expired. Ask for a new one.")
        if e["status"] != "ready":
            raise Refused(409, "This export is not ready to download.")
        data = blobs.load(matter_id, export_id, e["wrapped_key"], FOLDER)
        conn.execute("UPDATE legal_exports SET downloads=downloads+1 WHERE id=?", (export_id,))
        conn.commit()
    return e, data


def expire_due() -> int:
    """Remove released files past their time (called by the job worker)."""
    with get_conn() as conn:
        rows = [dict(zip(_COLS, r)) for r in conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_exports WHERE status='ready' AND expires_at<=?",
                                                         (_now_iso(),)).fetchall()]
        for e in rows:
            _expire_if_due(conn, e)
    return len(rows)


# ── Used by the deletion service: exports are copies of what is being purged ─────
def count_for(conn, object_type: str, object_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM legal_exports WHERE object_type=? AND object_id=?", (object_type, object_id)).fetchone()[0]


def purge_for(conn, object_type: str, object_id: str) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT matter_id, id FROM legal_exports WHERE object_type=? AND object_id=?", (object_type, object_id)).fetchall()
    conn.execute("DELETE FROM legal_exports WHERE object_type=? AND object_id=?", (object_type, object_id))
    return [(r[0], r[1]) for r in rows]


def file_exists(matter_id: str, export_id: str) -> bool:
    import os
    return os.path.exists(blobs._path(matter_id, export_id, FOLDER))
