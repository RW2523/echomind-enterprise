"""SQLite persistence for session notes."""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from ..core.db import get_conn
from ..schemas.assistant_suggestion import SuggestionOut
from ..schemas.session_note import SessionNoteOut, row_to_session_note_out
from ..schemas.silent_finding import CorrectionFindingOut
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)


def _select_cols() -> str:
    return (
        "id, session_id, source_type, source_id, title, body, citations_json, tags_json, "
        "pinned, created_at, updated_at"
    )


def get_note(note_id: str) -> Optional[SessionNoteOut]:
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {_select_cols()} FROM session_notes WHERE id = ?",
            (note_id,),
        ).fetchone()
    if not row:
        return None
    return row_to_session_note_out(tuple(row))


def find_by_source(session_id: str, source_type: str, source_id: str) -> Optional[SessionNoteOut]:
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {_select_cols()} FROM session_notes WHERE session_id = ? AND source_type = ? AND source_id = ?",
            (session_id, source_type, source_id),
        ).fetchone()
    if not row:
        return None
    return row_to_session_note_out(tuple(row))


def upsert_note(
    session_id: str,
    source_type: str,
    source_id: str,
    title: str,
    body: str,
    citations: List[dict],
    tags: List[str],
    pinned: bool,
) -> SessionNoteOut:
    """One row per (session_id, source_type, source_id); pinned OR-merge; refresh snapshot fields."""
    nid = new_id("note")
    ts = now_iso()
    cj = json.dumps(citations or [], ensure_ascii=False)
    tj = json.dumps(tags or [], ensure_ascii=False)
    p = 1 if pinned else 0
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_notes (
                id, session_id, source_type, source_id, title, body, citations_json, tags_json, pinned, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id, source_type, source_id) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                citations_json = excluded.citations_json,
                tags_json = excluded.tags_json,
                pinned = CASE WHEN session_notes.pinned = 1 OR excluded.pinned = 1 THEN 1 ELSE 0 END,
                updated_at = excluded.updated_at
            """,
            (nid, session_id, source_type, source_id, title, body, cj, tj, p, ts, ts),
        )
        conn.commit()
    row = find_by_source(session_id, source_type, source_id)
    assert row is not None
    return row


def note_from_suggestion(s: SuggestionOut) -> SessionNoteOut:
    cat = s.category.value if hasattr(s.category, "value") else str(s.category)
    body_parts = [s.short_text or ""]
    if s.speak_text and (s.speak_text.strip() != (s.short_text or "").strip()):
        body_parts.append("Speak text:\n" + (s.speak_text[:8000]))
    if s.reason:
        body_parts.append("Reason:\n" + (s.reason[:4000]))
    body = "\n\n".join(p for p in body_parts if p.strip())
    tags = ["assistant", "suggestion", cat]
    cites: List[dict] = []
    if s.citations:
        for c in s.citations:
            if isinstance(c, dict):
                cites.append(c)
            else:
                try:
                    cites.append(c.model_dump() if hasattr(c, "model_dump") else dict(c))  # type: ignore[arg-type]
                except Exception:
                    pass
    return upsert_note(
        s.session_id,
        "suggestion",
        s.id,
        s.title or "Suggestion",
        body,
        cites,
        tags,
        pinned=False,
    )


def note_from_correction(f: CorrectionFindingOut, *, pinned: bool) -> SessionNoteOut:
    cat = f.category.value if hasattr(f.category, "value") else str(f.category)
    parts: List[str] = []
    if f.original_text:
        parts.append(f"Original\n{f.original_text}")
    if f.reason:
        parts.append(f"Reason\n{f.reason}")
    if f.suggested_correction:
        parts.append(f"Suggested correction\n{f.suggested_correction}")
    body = "\n\n".join(parts)
    tags = ["silent", "correction", cat]
    if pinned:
        tags.append("pinned")
    cites: List[dict] = []
    if f.citations:
        for c in f.citations:
            if isinstance(c, dict):
                cites.append(c)
            else:
                try:
                    cites.append(c.model_dump() if hasattr(c, "model_dump") else dict(c))  # type: ignore[arg-type]
                except Exception:
                    pass
    title = f"Correction ({cat})"
    return upsert_note(
        f.session_id,
        "correction",
        f.id,
        title[:500],
        body,
        cites,
        tags,
        pinned=pinned,
    )


def list_notes(
    session_id: str,
    *,
    pinned_only: Optional[bool] = None,
    source_type: Optional[str] = None,
    q: Optional[str] = None,
) -> List[SessionNoteOut]:
    clauses = ["session_id = ?"]
    params: List[object] = [session_id]
    if pinned_only is True:
        clauses.append("pinned = 1")
    elif pinned_only is False:
        clauses.append("pinned = 0")
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if q and q.strip():
        needle = "%" + q.strip().replace("%", "\\%").replace("_", "\\_") + "%"
        clauses.append("(title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\')")
        params.extend([needle, needle])
    where = " AND ".join(clauses)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT {_select_cols()} FROM session_notes WHERE {where} ORDER BY pinned DESC, updated_at DESC",
            params,
        ).fetchall()
    return [row_to_session_note_out(tuple(r)) for r in rows]


def search_notes(session_id: str, q: str) -> List[SessionNoteOut]:
    return list_notes(session_id, q=q)


def set_pinned(note_id: str, pinned: bool) -> Optional[SessionNoteOut]:
    ts = now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE session_notes SET pinned = ?, updated_at = ? WHERE id = ?",
            (1 if pinned else 0, ts, note_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_note(note_id)


def delete_note(note_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM session_notes WHERE id = ?", (note_id,))
        conn.commit()
        return cur.rowcount > 0
