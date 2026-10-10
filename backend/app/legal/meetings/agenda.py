"""A meeting's agenda and board pack (S2.6, MEET-01; decision D-042).

Each item has a title and the matter documents it relies on (the board pack for that item). During
capture, the person recording marks which item the meeting has reached (`{"type": "agenda"}`), so
the minutes can file what was said under the right item. Documents are checked against the person's
visibility when attached and when listed; an item never reveals a document the reader cannot see.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from ...core.db import get_conn
from ..documents.repo import VISIBLE_TO
from ..policy import now_iso


class Invalid(Exception):
    pass


def _visible(conn, matter_id: str, document_ids: list[str], uid: str) -> dict[str, str]:
    if not document_ids:
        return {}
    qs = ",".join("?" * len(document_ids))
    return {r[0]: r[1] for r in conn.execute(f"SELECT d.id, d.title FROM legal_documents d WHERE d.matter_id=? AND d.id IN ({qs}) "
                                             f"AND {VISIBLE_TO}", (matter_id, *document_ids, uid))}


def items(matter_id: str, meeting_id: str, uid: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, position, title, document_ids_json FROM legal_agenda_items WHERE meeting_id=? AND matter_id=? "
                            "AND removed_at IS NULL ORDER BY position, created_at", (meeting_id, matter_id)).fetchall()
        out = []
        for iid, pos, title, dj in rows:
            seen = _visible(conn, matter_id, json.loads(dj), uid)
            out.append({"id": iid, "position": pos, "title": title,
                        "documents": [{"id": d, "title": t} for d, t in seen.items()]})
    return out


def add(matter_id: str, meeting_id: str, title: str, document_ids: list[str], uid: str) -> dict:
    with get_conn() as conn:
        seen = _visible(conn, matter_id, list(dict.fromkeys(document_ids)), uid)
        if len(seen) != len(set(document_ids)):
            raise Invalid("A document is not available in this matter.")
        pos = conn.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM legal_agenda_items WHERE meeting_id=? AND removed_at IS NULL",
                           (meeting_id,)).fetchone()[0]
        iid = str(uuid.uuid4())
        conn.execute("INSERT INTO legal_agenda_items (id, meeting_id, matter_id, position, title, document_ids_json, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (iid, meeting_id, matter_id, pos, title, json.dumps(list(seen)), uid, now_iso()))
        conn.commit()
    return next(i for i in items(matter_id, meeting_id, uid) if i["id"] == iid)


def remove(matter_id: str, meeting_id: str, item_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("UPDATE legal_agenda_items SET removed_at=? WHERE id=? AND meeting_id=? AND matter_id=? AND removed_at IS NULL",
                           (now_iso(), item_id, meeting_id, matter_id))
        conn.commit()
    return cur.rowcount > 0


def exists(meeting_id: str, item_id: Optional[str]) -> bool:
    if item_id is None:
        return True
    with get_conn() as conn:
        return bool(conn.execute("SELECT 1 FROM legal_agenda_items WHERE id=? AND meeting_id=? AND removed_at IS NULL",
                                 (item_id, meeting_id)).fetchone())
