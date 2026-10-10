"""Which version of a document answers come from, and where versions disagree (S2.2a; D-036; T05).

Searchable versions: each document's current version, and its latest signed (`executed`) version
when that is an older one. Every other version leaves the index (its text, clauses and marks stay
for the reader and for comparison).

Two policies choose among them for a search or an answer:
- `signed` (default): the latest signed version; a document never signed is used in its current
  version, marked unsigned;
- `current`: every document's current version, signed or not.

A conflict is a clause that the other searchable version of the same document words differently:
for a quote from the signed version, the current draft; for a quote from the current draft, the
signed version. Clause text is compared after folding whitespace and case.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from ...core.db import get_conn

POLICIES = ("signed", "current")
_LATEST_SIGNED = ("(SELECT e.id FROM legal_document_versions e WHERE e.document_id = {doc} AND e.status = 'executed' "
                  "ORDER BY e.number DESC LIMIT 1)")


def policy_sql(policy: str, chunk: str = "c", doc: str = "d") -> str:
    """SQL condition: this chunk's version is the one `policy` answers from."""
    if policy == "current":
        return f"{chunk}.version_id = {doc}.current_version_id"
    signed = _LATEST_SIGNED.format(doc=f"{chunk}.document_id")
    return f"{chunk}.version_id = COALESCE({signed}, {doc}.current_version_id)"


def searchable(conn, document_id: str) -> set[str]:
    current = conn.execute("SELECT current_version_id FROM legal_documents WHERE id=?", (document_id,)).fetchone()
    signed = conn.execute("SELECT " + _LATEST_SIGNED.format(doc="?"), (document_id,)).fetchone()
    return {v for v in ((current or [None])[0], (signed or [None])[0]) if v}


def sync_active(conn, document_id: str) -> tuple[list[tuple[str, str, bytes]], list[str]]:
    """Make exactly the searchable versions' passages active. Returns (activated rows (id, text, vec),
    deactivated ids) for the matter's index. Runs inside the caller's transaction."""
    keep = searchable(conn, document_id)
    rows = conn.execute("SELECT id, version_id, active, text, vec FROM legal_chunks WHERE document_id=?", (document_id,)).fetchall()
    on = [(r[0], r[3], r[4]) for r in rows if r[1] in keep and not r[2] and r[4] is not None]
    off = [r[0] for r in rows if r[1] not in keep and r[2]]
    if on:
        conn.executemany("UPDATE legal_chunks SET active=1 WHERE id=?", [(r[0],) for r in on])
    if off:
        conn.executemany("UPDATE legal_chunks SET active=0 WHERE id=?", [(i,) for i in off])
    qs = ",".join("?" * len(keep)) or "NULL"
    conn.execute(f"UPDATE legal_document_versions SET index_status='superseded' WHERE document_id=? AND index_status='indexed' "
                 f"AND id NOT IN ({qs})", (document_id, *keep))
    conn.execute(f"UPDATE legal_document_versions SET index_status='indexed' WHERE document_id=? AND index_status='superseded' "
                 f"AND id IN ({qs}) AND chunk_count > 0", (document_id, *keep))
    return on, off


def apply_to_index(matter_id: str, on: list[tuple[str, str, bytes]], off: list[str]) -> None:
    import numpy as np

    from ..retrieval import registry
    if not on and not off:
        return
    idx = registry.for_matter(matter_id)
    with idx.lock:
        if off:
            idx.remove(off)
        if on:
            idx.add([(i, text, np.frombuffer(vec, dtype=np.float32)) for i, text, vec in on])


def resync(document_id: str) -> None:
    """After a status change or a restore: bring the index in line with the searchable versions."""
    with get_conn() as conn:
        row = conn.execute("SELECT matter_id FROM legal_documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            on, off = sync_active(conn, document_id)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    apply_to_index(row[0], on, off)


# ── Conflicts ────────────────────────────────────────────────────────────────────
def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def _clause_text(conn, version_id: str, number: str) -> Optional[str]:
    row = conn.execute("SELECT c.char_start, c.char_end, t.text FROM legal_clauses c JOIN legal_version_text t ON t.version_id = c.version_id "
                       "WHERE c.version_id=? AND c.number=? ORDER BY c.char_start LIMIT 1", (version_id, number)).fetchone()
    return row[2][row[0]:row[1]] if row else None


def conflicts(sources: Iterable[dict], policy: str) -> list[dict]:
    """For each cited clause, the other searchable version's wording when it differs."""
    out, seen = [], set()
    with get_conn() as conn:
        for s in sources:
            clause, vid, did = s.get("clause"), s.get("version_id"), s.get("document_id")
            if not clause or not vid or (did, clause) in seen:
                continue
            seen.add((did, clause))
            others = [v for v in searchable(conn, did) if v != vid]
            mine = _clause_text(conn, vid, clause)
            if mine is None:
                continue
            for other in others:
                theirs = _clause_text(conn, other, clause)
                if theirs is None or _fold(theirs) == _fold(mine):
                    continue
                meta = {r[0]: {"number": r[1], "status": r[2]} for r in conn.execute(
                    "SELECT id, number, status FROM legal_document_versions WHERE id IN (?, ?)", (vid, other))}
                out.append({"document_id": did, "document_title": s.get("document_title"), "clause": clause,
                            "version": {**meta[vid], "id": vid, "text": mine[:600]},
                            "other": {**meta[other], "id": other, "text": theirs[:600]}})
    return out
