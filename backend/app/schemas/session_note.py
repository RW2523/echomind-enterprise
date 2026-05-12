"""Session-scoped notes (saved suggestions, corrections, manual entries)."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class NoteSourceType(str, Enum):
    suggestion = "suggestion"
    correction = "correction"
    grounded_answer = "grounded_answer"
    action_item = "action_item"
    manual_note = "manual_note"


class SessionNoteCreate(BaseModel):
    session_id: str = Field(..., min_length=4, max_length=128)
    source_type: NoteSourceType
    source_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="", max_length=100_000)
    citations: List[dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    pinned: bool = False


class SessionNoteOut(BaseModel):
    id: str
    session_id: str
    source_type: NoteSourceType
    source_id: str
    title: str
    body: str
    citations: List[dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    pinned: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def row_to_session_note_out(row: tuple) -> SessionNoteOut:
    (
        nid,
        session_id,
        source_type,
        source_id,
        title,
        body,
        citations_json,
        tags_json,
        pinned,
        created_at,
        updated_at,
    ) = row
    cites: List[Any] = []
    if citations_json:
        try:
            cites = json.loads(citations_json)
            if not isinstance(cites, list):
                cites = []
        except Exception:
            cites = []
    tags: List[Any] = []
    if tags_json:
        try:
            tags = json.loads(tags_json)
            if not isinstance(tags, list):
                tags = []
        except Exception:
            tags = []
    return SessionNoteOut(
        id=nid,
        session_id=session_id,
        source_type=NoteSourceType(source_type),
        source_id=source_id or "",
        title=title or "",
        body=body or "",
        citations=cites,
        tags=[str(t) for t in tags],
        pinned=bool(pinned),
        created_at=created_at,
        updated_at=updated_at,
    )
