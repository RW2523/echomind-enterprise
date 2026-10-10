"""Firm templates (S3.1, DOC-01; decision D-045).

A template is the firm's precedent, written in a small markup:

    # Heading                    (## and ### for lower levels)
    A paragraph with {{client_name}} and {{fee}}.     (blank lines separate paragraphs; a single
                                                       line break stays a line break, as in an address)
    [[write: an instruction]]    (a passage the model writes from the matter's documents, citing them)

Each `{{key}}` is a declared field with a source:
- `documents`: found in the matter's documents, with the exact quote that states it (checked);
- `matter`: from the matter record (`matter_number`, `matter_title`, `client`);
- `person`: always entered by a lawyer;
- `today`: the date of drafting.

A version's wording and fields are fixed (a database trigger); a change is a new version. Firm
styles (a Word file) may be attached while a version is a draft; they are part of its hash. A
managing partner other than the author approves a version for exactly its hash; approving it
retires the version approved before it. Only approved versions are drafted from.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

KINDS = ("engagement_letter", "nda", "demand_letter", "memo", "contract_review", "board_minutes", "custom")
FIELD_KINDS = ("text", "party", "date", "money", "period", "percent", "address", "agreement")   # agreement: s3.1c
SOURCES = ("documents", "matter", "person", "today")
MATTER_FIELDS = {"matter_number": "number", "matter_title": "title", "client": "client", "matter_client": "client"}
_REF = re.compile(r"\{\{\s*([a-z][a-z0-9_]{0,40})\s*\}\}")
_WRITE = re.compile(r"^\[\[write:\s*(.+?)\s*\]\]$", re.S)
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


class Invalid(ValueError):
    pass


class Conflict(Exception):
    pass


def parse(body: str) -> list[dict]:
    """The template as blocks: headings, paragraphs (with {{fields}}) and passages to write. A heading
    line and a [[write: …]] line are blocks of their own; other lines run together into a paragraph
    (keeping their line breaks) until a blank line."""
    blocks, para = [], []

    def close():
        if para:
            blocks.append({"type": "paragraph", "raw": "\n".join(para)})
            para.clear()
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        h = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        w = _WRITE.match(stripped)
        if not stripped:
            close()
        elif h:
            close()
            blocks.append({"type": "heading", "level": len(h.group(1)), "raw": h.group(2).strip()})
        elif w:
            close()
            blocks.append({"type": "write", "instruction": " ".join(w.group(1).split())})
        else:
            para.append(line.rstrip())
    close()
    return blocks


def validate(body: str, fields: list[dict]) -> list[dict]:
    if not (body or "").strip():
        raise Invalid("The template has no text.")
    if len(body) > 60_000:
        raise Invalid("The template is too long.")
    keys, clean = set(), []
    for f in fields:
        key = str(f.get("key", ""))
        if not _KEY.match(key):
            raise Invalid(f"A field key must be lowercase letters, digits and underscores: {key!r}.")
        if key in keys:
            raise Invalid(f"The field {key} is declared twice.")
        keys.add(key)
        kind, source = f.get("kind", "text"), f.get("source", "documents")
        if kind not in FIELD_KINDS:
            raise Invalid(f"The field {key} has an unknown kind {kind!r}.")
        if source not in SOURCES:
            raise Invalid(f"The field {key} has an unknown source {source!r}.")
        if source == "documents" and not str(f.get("question", "")).strip():
            raise Invalid(f"The field {key} is taken from the documents, so it needs the question to ask them.")
        if source == "matter" and (f.get("from") or MATTER_FIELDS.get(key)) not in ("number", "title", "client"):
            raise Invalid(f"The field {key} is taken from the matter: name it matter_number, matter_title or client.")
        clean.append({"key": key, "label": str(f.get("label") or key.replace("_", " ").capitalize())[:80], "kind": kind,
                      "source": source, "question": str(f.get("question", "")).strip()[:300] or None,
                      "from": (f.get("from") or MATTER_FIELDS.get(key)) if source == "matter" else None})
    used = set(_REF.findall(body))
    if used - keys:
        raise Invalid("The template uses fields it does not declare: " + ", ".join(sorted(used - keys)) + ".")
    if keys - used:
        raise Invalid("The template declares fields it never uses: " + ", ".join(sorted(keys - used)) + ".")
    if not parse(body):
        raise Invalid("The template has no text.")
    return clean


def digest(body: str, fields: list[dict], styles_sha256: Optional[str]) -> str:
    return hashlib.sha256(json.dumps({"body": body, "fields": fields, "styles": styles_sha256}, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


_VCOLS = ("id", "template_id", "version", "status", "body", "fields_json", "styles_name", "styles_sha256", "styles_wrapped_key",
          "content_sha256", "created_by", "created_at", "approved_by", "approved_at")


def _name(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _version(conn, r) -> dict:
    v = dict(zip(_VCOLS, r))
    v["fields"] = json.loads(v.pop("fields_json"))
    v["styles"] = v.pop("styles_name")
    v.pop("styles_wrapped_key")
    v["created_by_id"], v["approved_by_id"] = v["created_by"], v["approved_by"]
    v["created_by"], v["approved_by"] = (_name(conn, v["created_by"]) or ("standard preset" if v["created_by"] == "system" else None),
                                          _name(conn, v["approved_by"]))
    return v


def get(org_id: str, template_id: str, drafts: bool = True) -> Optional[dict]:
    with get_conn() as conn:
        t = conn.execute("SELECT id, name, kind, preset_key, created_by, created_at FROM legal_templates WHERE id=? AND org_id=?",
                         (template_id, org_id)).fetchone()
        if not t:
            return None
        rows = conn.execute(f"SELECT {', '.join(_VCOLS)} FROM legal_template_versions WHERE template_id=? ORDER BY version DESC",
                            (template_id,)).fetchall()
        versions = [_version(conn, r) for r in rows if drafts or r[3] == "approved"]
        return {"id": t[0], "name": t[1], "kind": t[2], "preset": t[3], "created_by": _name(conn, t[4]), "created_at": t[5],
                "versions": versions}


def listing(org_id: str, drafts: bool = True) -> list[dict]:
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM legal_templates WHERE org_id=? ORDER BY name", (org_id,))]
    out = [get(org_id, i, drafts) for i in ids]
    return [t for t in out if t and t["versions"]]


def create(org_id: str, name: str, kind: str, body: str, fields: list[dict], by: str, preset_key: Optional[str] = None) -> dict:
    if kind not in KINDS:
        raise Invalid(f"Unknown kind {kind!r}.")
    clean = validate(body, fields)
    tid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_templates (id, org_id, name, kind, preset_key, created_by, created_at) VALUES (?,?,?,?,?,?,?)",
                     (tid, org_id, name.strip()[:200], kind, preset_key, by, now_iso()))
        _insert_version(conn, tid, 1, body, clean, by)
        conn.commit()
    return get(org_id, tid)


def _insert_version(conn, template_id: str, n: int, body: str, fields: list[dict], by: str) -> str:
    vid = str(uuid.uuid4())
    conn.execute("INSERT INTO legal_template_versions (id, template_id, version, status, body, fields_json, content_sha256, created_by, "
                 "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (vid, template_id, n, "draft", body, json.dumps(fields, ensure_ascii=False), digest(body, fields, None), by, now_iso()))
    return vid


def add_version(org_id: str, template_id: str, body: str, fields: list[dict], by: str) -> dict:
    clean = validate(body, fields)
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if not conn.execute("SELECT 1 FROM legal_templates WHERE id=? AND org_id=?", (template_id, org_id)).fetchone():
                raise LookupError(template_id)
            n = conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM legal_template_versions WHERE template_id=?", (template_id,)).fetchone()[0]
            _insert_version(conn, template_id, n, body, clean, by)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(org_id, template_id)


def _row(conn, template_id: str, version_id: str) -> Optional[dict]:
    r = conn.execute(f"SELECT {', '.join(_VCOLS)} FROM legal_template_versions WHERE id=? AND template_id=?", (version_id, template_id)).fetchone()
    return dict(zip(_VCOLS, r)) if r else None


def set_styles(org_id: str, template_id: str, version_id: str, filename: str, data: bytes, by: str) -> dict:
    """Attach the firm's Word file whose styles (fonts, headings, margins, header) drafts are rendered in."""
    import io

    import docx
    from ..documents import blobs
    try:
        docx.Document(io.BytesIO(data))
    except Exception:                                   # noqa: BLE001 — anything python-docx cannot open
        raise Invalid("That file is not a Word document (.docx).")
    with get_conn() as conn:
        if not conn.execute("SELECT 1 FROM legal_templates WHERE id=? AND org_id=?", (template_id, org_id)).fetchone():
            raise LookupError(template_id)
        v = _row(conn, template_id, version_id)
        if not v:
            raise LookupError(version_id)
        if v["status"] != "draft":
            raise Conflict("Firm styles can only be attached to a draft version.")
        sha = hashlib.sha256(data).hexdigest()
        wrapped = blobs.store(org_id, version_id, data, folder="templates")
        conn.execute("UPDATE legal_template_versions SET styles_name=?, styles_sha256=?, styles_wrapped_key=?, content_sha256=? WHERE id=?",
                     (filename[:200], sha, wrapped, digest(v["body"], json.loads(v["fields_json"]), sha), version_id))
        conn.commit()
    return get(org_id, template_id)


def styles(org_id: str, version: dict) -> Optional[bytes]:
    from ..documents import blobs
    with get_conn() as conn:
        r = conn.execute("SELECT styles_wrapped_key FROM legal_template_versions WHERE id=?", (version["id"],)).fetchone()
    return blobs.load(org_id, version["id"], r[0], folder="templates") if r and r[0] else None


def approve(org_id: str, template_id: str, version_id: str, by: str, content_sha256: str) -> dict:
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if not conn.execute("SELECT 1 FROM legal_templates WHERE id=? AND org_id=?", (template_id, org_id)).fetchone():
                raise LookupError(template_id)
            v = _row(conn, template_id, version_id)
            if not v:
                raise LookupError(version_id)
            if v["status"] != "draft":
                raise Conflict(f"This version is {v['status']}.")
            if v["created_by"] == by:
                raise Conflict("A template version is approved by someone other than its author.")
            if v["content_sha256"] != content_sha256:
                raise Conflict("This version changed since you opened it. Review the current one.")
            conn.execute("UPDATE legal_template_versions SET status='retired' WHERE template_id=? AND status='approved'", (template_id,))
            conn.execute("UPDATE legal_template_versions SET status='approved', approved_by=?, approved_at=? WHERE id=?",
                         (by, now_iso(), version_id))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(org_id, template_id)


def retire(org_id: str, template_id: str, version_id: str) -> dict:
    with get_conn() as conn:
        if not conn.execute("SELECT 1 FROM legal_templates WHERE id=? AND org_id=?", (template_id, org_id)).fetchone():
            raise LookupError(template_id)
        conn.execute("UPDATE legal_template_versions SET status='retired' WHERE id=? AND template_id=? AND status='approved'",
                     (version_id, template_id))
        conn.commit()
    return get(org_id, template_id)


def approved(org_id: str, template_id: str) -> Optional[tuple[dict, dict]]:
    """The template and its approved version, if it has one."""
    t = get(org_id, template_id, drafts=False)
    if not t or not t["versions"]:
        return None
    return t, t["versions"][0]


def install_presets(org_id: str, by: str) -> int:
    from . import presets
    n = 0
    for p in presets.PRESETS:
        with get_conn() as conn:
            if conn.execute("SELECT 1 FROM legal_templates WHERE org_id=? AND preset_key=?", (org_id, p["key"])).fetchone():
                continue
        create(org_id, p["name"], p["kind"], p["body"], p["fields"], by, preset_key=p["key"])
        n += 1
    return n
