"""REST API for local session notes (saved suggestions, corrections, manual)."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ...assistant import suggestion_store
from ...schemas.session_note import SessionNoteCreate, SessionNoteOut
from ...session_notes import notes_store

router = APIRouter(prefix="/session-notes", tags=["session-notes"])


@router.post("", response_model=SessionNoteOut)
def create_note(body: SessionNoteCreate) -> SessionNoteOut:
    st = body.source_type.value if hasattr(body.source_type, "value") else str(body.source_type)
    return notes_store.upsert_note(
        body.session_id,
        st,
        body.source_id,
        body.title,
        body.body,
        list(body.citations or []),
        list(body.tags or []),
        body.pinned,
    )


@router.get("/sessions/{session_id}/notes", response_model=List[SessionNoteOut])
def list_session_notes(
    session_id: str,
    pinned_only: Optional[bool] = Query(None, description="If true, only pinned; if false, only unpinned; omit for all."),
    source_type: Optional[str] = Query(None, description="Filter by NoteSourceType value."),
    q: Optional[str] = Query(None, description="Search in title and body (simple LIKE)."),
) -> List[SessionNoteOut]:
    st = None
    if source_type and source_type not in ("any", "all", "*"):
        st = source_type
    return notes_store.list_notes(session_id, pinned_only=pinned_only, source_type=st, q=q)


@router.get("/sessions/{session_id}/notes/search", response_model=List[SessionNoteOut])
def search_session_notes(session_id: str, q: str = Query(..., min_length=1, max_length=500)) -> List[SessionNoteOut]:
    return notes_store.search_notes(session_id, q)


@router.post("/{note_id}/pin", response_model=SessionNoteOut)
def pin_note(note_id: str) -> SessionNoteOut:
    row = notes_store.set_pinned(note_id, True)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    return row


@router.post("/{note_id}/unpin", response_model=SessionNoteOut)
def unpin_note(note_id: str) -> SessionNoteOut:
    row = notes_store.set_pinned(note_id, False)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    return row


@router.delete("/{note_id}")
def delete_note(note_id: str) -> dict:
    ok = notes_store.delete_note(note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"ok": True, "deleted": note_id}


@router.post("/from-suggestion/{suggestion_id}", response_model=SessionNoteOut)
def capture_suggestion_note(suggestion_id: str) -> SessionNoteOut:
    s = suggestion_store.get_suggestion(suggestion_id)
    if not s:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return notes_store.note_from_suggestion(s)
