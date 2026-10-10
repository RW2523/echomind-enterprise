"""Playbooks and their positions (S3.2, DOC-02a; decision D-046).

A playbook holds the firm's positions for one kind of contract and side ("Commercial services,
seller"). A position covers one topic: the standard wording, the fallbacks the firm accepts (each
with a note, such as who must agree), guidance, and an owner. Positions are versioned and fixed; a
managing partner other than the author approves a version for exactly its hash, which retires the
version approved before it. Lawyers without the right to manage playbooks see approved positions only.

A precedent (a firm document) can be read into proposed positions: its clauses, each with the topic
its words suggest, for a lawyer to accept or not.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

# topic → (label, words that suggest it)
TOPICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "liability_cap": ("Limitation of liability", ("liabilit", "shall not exceed", "aggregate", "cap ")),
    "exclusions": ("Excluded losses", ("indirect", "consequential", "loss of profit", "exclud")),
    "term": ("Term and renewal", ("term of", "commenc", "continues for", "renew")),
    "termination": ("Termination", ("terminat",)),
    "payment": ("Payment", ("invoice", "payable", "pay ", "fee", "payment")),
    "confidentiality": ("Confidentiality", ("confidential", "disclos")),
    "governing_law": ("Governing law and disputes", ("governed by", "governing law", "courts of", "arbitration")),
    "assignment": ("Assignment", ("assign", "transfer", "subcontract")),
    "indemnity": ("Indemnity", ("indemnif", "hold harmless")),
    "data_protection": ("Data protection", ("personal data", "data protection", "processor", "gdpr")),
    "force_majeure": ("Force majeure", ("force majeure", "beyond its reasonable control")),
    "warranties": ("Warranties", ("warrant", "represents that")),
}


class Invalid(ValueError):
    pass


class Conflict(Exception):
    pass


def topic_of(text: str) -> Optional[str]:
    """The topic whose words the text uses most (ties: the first listed); None when none."""
    low = " " + (text or "").casefold() + " "
    best, score = None, 0
    for key, (_, words) in TOPICS.items():
        n = sum(low.count(w) for w in words)
        if n > score:
            best, score = key, n
    return best


def digest(pos: dict) -> str:
    keep = {k: pos.get(k) for k in ("topic", "title", "standard", "fallbacks", "guidance", "owner")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _name(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


_PCOLS = ("id", "playbook_id", "topic", "version", "status", "title", "standard_text", "fallbacks_json", "guidance", "owner",
          "content_sha256", "created_by", "created_at", "approved_by", "approved_at")


def _position(conn, r) -> dict:
    p = dict(zip(_PCOLS, r))
    p["standard"], p["fallbacks"] = p.pop("standard_text"), json.loads(p.pop("fallbacks_json"))
    p["topic_label"] = TOPICS.get(p["topic"], (p["topic"],))[0]
    p["created_by_id"] = p["created_by"]
    p["created_by"], p["approved_by"], p["owner_name"] = _name(conn, p["created_by"]), _name(conn, p["approved_by"]), _name(conn, p["owner"])
    return p


def get(org_id: str, playbook_id: str, drafts: bool = True) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT id, name, description, jurisdiction, created_by, created_at FROM legal_playbooks WHERE id=? AND org_id=?",
                         (playbook_id, org_id)).fetchone()
        if not r:
            return None
        order = {k: i for i, k in enumerate(TOPICS)}
        rows = conn.execute(f"SELECT {', '.join(_PCOLS)} FROM legal_playbook_positions WHERE playbook_id=?", (playbook_id,)).fetchall()
        positions = [_position(conn, x) for x in rows if drafts or x[4] == "approved"]
        positions.sort(key=lambda p: (order.get(p["topic"], 99), -p["version"]))
        return {"id": r[0], "name": r[1], "description": r[2], "jurisdiction": r[3], "created_by": _name(conn, r[4]), "created_at": r[5],
                "positions": positions}


def listing(org_id: str, drafts: bool = True) -> list[dict]:
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM legal_playbooks WHERE org_id=? ORDER BY name", (org_id,))]
    return [p for p in (get(org_id, i, drafts) for i in ids) if p]


def create(org_id: str, name: str, description: Optional[str], jurisdiction: Optional[str], by: str) -> dict:
    pid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_playbooks (id, org_id, name, description, jurisdiction, created_by, created_at) VALUES (?,?,?,?,?,?,?)",
                     (pid, org_id, name.strip()[:200], (description or "").strip()[:2000] or None, (jurisdiction or "").strip()[:100] or None,
                      by, now_iso()))
        conn.commit()
    return get(org_id, pid)


def add_position(org_id: str, playbook_id: str, pos: dict, by: str) -> dict:
    if pos["topic"] not in TOPICS:
        raise Invalid(f"Unknown topic {pos['topic']!r}.")
    if not pos["standard"].strip():
        raise Invalid("A position needs its standard wording.")
    fallbacks = [{"text": f["text"].strip(), "note": (f.get("note") or "").strip() or None} for f in pos.get("fallbacks", []) if f["text"].strip()]
    clean = {"topic": pos["topic"], "title": (pos.get("title") or TOPICS[pos["topic"]][0]).strip()[:200], "standard": pos["standard"].strip(),
             "fallbacks": fallbacks, "guidance": (pos.get("guidance") or "").strip() or None, "owner": pos.get("owner") or by}
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if not conn.execute("SELECT 1 FROM legal_playbooks WHERE id=? AND org_id=?", (playbook_id, org_id)).fetchone():
                raise LookupError(playbook_id)
            n = conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM legal_playbook_positions WHERE playbook_id=? AND topic=?",
                             (playbook_id, clean["topic"])).fetchone()[0]
            conn.execute("INSERT INTO legal_playbook_positions (id, playbook_id, topic, version, status, title, standard_text, fallbacks_json, "
                         "guidance, owner, content_sha256, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (str(uuid.uuid4()), playbook_id, clean["topic"], n, "draft", clean["title"], clean["standard"],
                          json.dumps(fallbacks, ensure_ascii=False), clean["guidance"], clean["owner"], digest(clean), by, now_iso()))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(org_id, playbook_id)


def approve(org_id: str, playbook_id: str, position_id: str, by: str, content_sha256: str) -> dict:
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if not conn.execute("SELECT 1 FROM legal_playbooks WHERE id=? AND org_id=?", (playbook_id, org_id)).fetchone():
                raise LookupError(playbook_id)
            r = conn.execute("SELECT topic, status, created_by, content_sha256 FROM legal_playbook_positions WHERE id=? AND playbook_id=?",
                             (position_id, playbook_id)).fetchone()
            if not r:
                raise LookupError(position_id)
            topic, status, author, sha = r
            if status != "draft":
                raise Conflict(f"This position is {status}.")
            if author == by:
                raise Conflict("A position is approved by someone other than its author.")
            if sha != content_sha256:
                raise Conflict("This position changed since you opened it. Review the current one.")
            conn.execute("UPDATE legal_playbook_positions SET status='retired' WHERE playbook_id=? AND topic=? AND status='approved'",
                         (playbook_id, topic))
            conn.execute("UPDATE legal_playbook_positions SET status='approved', approved_by=?, approved_at=? WHERE id=?", (by, now_iso(), position_id))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(org_id, playbook_id)


def approved_positions(org_id: str, playbook_id: str) -> list[dict]:
    p = get(org_id, playbook_id, drafts=False)
    return p["positions"] if p else []


def proposals(filename: str, data: bytes) -> list[dict]:
    """A precedent's clauses as proposed positions: the leaf clauses (those with no sub-clauses), each
    with the heading of the clause above it and the topic its words suggest."""
    from ..documents import extract as extract_mod
    from ..documents import structure
    x = extract_mod.extract(filename, data)
    found = structure.clauses(x)
    parents = {c.parent for c in found if c.parent}
    by_number = {c.number: c for c in found}
    out = []
    for c in found:
        if c.number in parents:
            continue
        text = x.text[c.start:c.end].strip()
        if len(text) < 20:
            continue
        head = c.heading or (by_number[c.parent].heading if c.parent in by_number else None)
        out.append({"clause": c.number, "title": head or TOPICS.get(topic_of(text) or "", (None,))[0], "text": text,
                    "topic": topic_of(f"{head or ''} {text}")})
    return out
