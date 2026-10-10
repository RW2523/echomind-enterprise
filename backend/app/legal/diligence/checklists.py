"""Due-diligence checklists (S3.7, DOC-06; decision D-051).

A checklist is a firm template: items grouped by section, each with the phrases that name the
documents expected ("articles of association", "board minutes") and the questions asked of the
documents that answer it. Versions are fixed (a trigger); a managing partner other than the author
approves exactly the version read (its hash), which retires the version before it. Runs use approved
versions only. Lawyers who do not manage checklists see approved versions only.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

from ...core.db import get_conn
from .. import policy

QUESTIONS = ("change_of_control", "assignment", "termination", "governing_law", "liability", "renewal")
_COLS = ("id", "org_id", "family_id", "version", "name", "content_json", "content_sha256", "status", "created_by", "created_at",
         "approved_by", "approved_at")


class Invalid(ValueError):
    pass


class Conflict(Exception):
    pass


def sha(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate(name: str, items: list[dict], expected_law: Optional[str]) -> dict:
    if not name.strip():
        raise Invalid("Give the checklist a name.")
    if not items:
        raise Invalid("A checklist needs at least one item.")
    seen, out = set(), []
    for it in items:
        iid = str(it.get("id") or "").strip()
        title = str(it.get("title") or "").strip()
        expect = [str(x).strip().casefold() for x in (it.get("expect") or []) if str(x).strip()]
        questions = [q for q in (it.get("questions") or [])]
        if not iid or not title or not expect:
            raise Invalid("Each item needs an id, a title and the words that name the documents expected.")
        if iid in seen:
            raise Invalid(f"The item id {iid} is used twice.")
        bad = [q for q in questions if q not in QUESTIONS]
        if bad:
            raise Invalid(f"Unknown question(s): {', '.join(bad)}. Use: {', '.join(QUESTIONS)}.")
        seen.add(iid)
        out.append({"id": iid, "section": str(it.get("section") or "").strip() or "General", "title": title, "expect": expect,
                    "questions": questions})
    return {"name": name.strip(), "expected_law": (expected_law or "").strip() or None, "items": out}


def _username(conn, uid: Optional[str]) -> Optional[str]:
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _out(conn, r) -> dict:
    d = dict(zip(_COLS, r))
    content = json.loads(d["content_json"])
    return {"id": d["id"], "family_id": d["family_id"], "version": d["version"], "name": d["name"], "expected_law": content.get("expected_law"),
            "items": content["items"], "content_sha256": d["content_sha256"], "status": d["status"],
            "created_by": _username(conn, d["created_by"]), "created_at": d["created_at"], "approved_by": _username(conn, d["approved_by"]),
            "approved_at": d["approved_at"]}


def create(org_id: str, name: str, items: list[dict], expected_law: Optional[str], by: str, family_id: Optional[str] = None) -> dict:
    content = validate(name, items, expected_law)
    cid = str(uuid.uuid4())
    with get_conn() as conn:
        if family_id:
            top = conn.execute("SELECT MAX(version) FROM legal_dd_checklists WHERE family_id=? AND org_id=?", (family_id, org_id)).fetchone()[0]
            if top is None:
                raise LookupError(family_id)
            version = top + 1
        else:
            family_id, version = str(uuid.uuid4()), 1
        conn.execute("INSERT INTO legal_dd_checklists (id, org_id, family_id, version, name, content_json, content_sha256, status, created_by, "
                     "created_at, approved_by, approved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (cid, org_id, family_id, version, content["name"], json.dumps(content, ensure_ascii=False), sha(content), "draft", by,
                      policy.now_iso(), None, None))
        conn.commit()
        return _out(conn, conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_dd_checklists WHERE id=?", (cid,)).fetchone())


def get(org_id: str, checklist_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_dd_checklists WHERE id=? AND org_id=?", (checklist_id, org_id)).fetchone()
        return _out(conn, r) if r else None


def approve(org_id: str, checklist_id: str, by: str, content_sha256: str) -> dict:
    with get_conn() as conn:
        r = conn.execute("SELECT family_id, status, created_by, content_sha256 FROM legal_dd_checklists WHERE id=? AND org_id=?",
                         (checklist_id, org_id)).fetchone()
        if not r:
            raise LookupError(checklist_id)
        family, status, author, digest = r
        if status != "draft":
            raise Conflict(f"This version is {status}.")
        if author == by:
            raise Conflict("A checklist is approved by a managing partner who did not write it.")
        if digest != content_sha256:
            raise Conflict("This version is not the one you read. Reload it and check it again.")
        now = policy.now_iso()
        conn.execute("UPDATE legal_dd_checklists SET status='retired' WHERE family_id=? AND status='approved'", (family,))
        conn.execute("UPDATE legal_dd_checklists SET status='approved', approved_by=?, approved_at=? WHERE id=?", (by, now, checklist_id))
        conn.commit()
    return get(org_id, checklist_id)


def listing(org_id: str, drafts: bool) -> list[dict]:
    where = "" if drafts else "AND status='approved'"
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_dd_checklists WHERE org_id=? {where} ORDER BY name, version DESC",
                            (org_id,)).fetchall()
        return [_out(conn, r) for r in rows]


PRESETS = [
    {"name": "Share acquisition — core documents", "expected_law": "England and Wales", "items": [
        {"id": "corp-articles", "section": "Corporate", "title": "Articles of association", "expect": ["articles of association", "constitution", "bylaws"]},
        {"id": "corp-sha", "section": "Corporate", "title": "Shareholders' agreement", "expect": ["shareholders agreement", "shareholders' agreement"]},
        {"id": "corp-minutes", "section": "Corporate", "title": "Board minutes", "expect": ["minutes", "written resolution"]},
        {"id": "corp-captable", "section": "Corporate", "title": "Register of members or cap table", "expect": ["register of members", "cap table"]},
        {"id": "contracts", "section": "Commercial", "title": "Material contracts",
         "expect": ["services agreement", "supply agreement", "distribution agreement", "licence agreement", "license agreement"],
         "questions": ["change_of_control", "assignment", "termination", "governing_law", "liability", "renewal"]},
        {"id": "property", "section": "Property", "title": "Leases", "expect": ["lease"], "questions": ["termination", "governing_law", "renewal"]},
        {"id": "people", "section": "Employment", "title": "Key employment contracts", "expect": ["employment contract", "service agreement"],
         "questions": ["change_of_control", "termination"]},
        {"id": "ip", "section": "Intellectual property", "title": "IP assignments and licences", "expect": ["ip assignment", "intellectual property"]},
        {"id": "insurance", "section": "Insurance", "title": "Insurance policies", "expect": ["insurance policy", "policy schedule"]},
        {"id": "disputes", "section": "Disputes", "title": "Litigation and claims", "expect": ["claim form", "letter before action", "litigation"]},
    ]},
]
