"""
Shared logic: save transcript to DB (transcripts table + RAG index).
Used by POST /api/transcribe/store and by WebSocket combine→LLM→save flow.
"""
from __future__ import annotations
import json
import logging
import re
from ..utils.ids import new_id, now_iso
from ..core.db import get_conn
from ..rag.index import index
from ..rag.llm import OpenAICompatChat
from ..core.config import settings

logger = logging.getLogger(__name__)
_chat = None


def _get_chat():
    global _chat
    if _chat is None:
        _chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
    return _chat


def _title_for_transcript(tid: str, echodate: str) -> str:
    """Human-readable title: date and time + short id (e.g. 2025-02-10 14:30_abc12def)."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", echodate)
    if m:
        date_part = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
    else:
        date_part = echodate[:16].replace("T", " ") if len(echodate) >= 16 else echodate
    short_id = tid.replace("trn_", "")[:8] if tid.startswith("trn_") else tid[:8]
    return f"{date_part}_{short_id}"


def default_transcript_name_from_time() -> str:
    """Default name when user chooses Default: transcript_YYYY-MM-DD_HH-MM."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return f"transcript_{now.strftime('%Y-%m-%d_%H-%M')}"


async def store_transcript_to_db(
    raw_text: str,
    refined_text: str | None = None,
    echotag: str | None = None,
    name: str | None = None,
    location: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """
    Save a transcript to the transcripts table and RAG index.
    - raw_text: required.
    - refined_text: optional; if None, only raw is stored and indexed.
    - echotag: optional; if None, derived from LLM-generated tags or name/location.
    - name: optional display name (e.g. from Start popup or default transcript_YYYY-MM-DD_HH-MM).
    - location: optional; default "default" if not provided.
    - tags: optional list of manual tags; if provided, used instead of LLM-extracted tags.
    Returns: { transcript_id, title, name, location, tags, echotag, echodate, created_at }.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text is required")
    tid = new_id("trn")
    echodate = now_iso()
    name_val = (name or "").strip() or None
    location_val = (location or "").strip() or "default"
    title = name_val or _title_for_transcript(tid, echodate)
    tags_list = [t.strip() for t in tags] if tags else []
    tags_list = [t for t in tags_list if t][:16]
    if not tags_list:
        try:
            chat = _get_chat()
            tag_txt = await chat.chat(
                [
                    {"role": "system", "content": "Extract 3-6 short topic tags. Return comma-separated tags only."},
                    {"role": "user", "content": raw_text[:3500]},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            tags_list = [t.strip() for t in (tag_txt or "").split(",") if t.strip()][:8]
        except Exception:
            pass
    echotag = (echotag or "").strip() or (",".join(tags_list) if tags_list else (name_val or "transcript"))
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transcripts (id, title, raw_text, polished_text, tags_json, echotag, echodate, created_at, updated_at, name, location) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (tid, title, raw_text, refined_text, json.dumps(tags_list), echotag, echodate, echodate, echodate, name_val, location_val),
        )
        conn.commit()
    try:
        index_text = raw_text + ("\n\n" + refined_text if refined_text else "")
        await index.add_text(
            f"transcript_{tid}",
            index_text,
            {"type": "transcript", "tags": tags_list, "echotag": echotag, "echodate": echodate, "created_at": echodate},
        )
    except Exception as e:
        logger.warning("Failed to index transcript %s in RAG: %s", tid, e)
    return {
        "transcript_id": tid,
        "title": title,
        "name": name_val,
        "location": location_val,
        "tags": tags_list,
        "echotag": echotag,
        "echodate": echodate,
        "created_at": echodate,
    }


def create_transcript_for_session(
    name: str | None = None,
    location: str | None = None,
    started_at_iso: str | None = None,
    initial_text: str | None = None,
) -> str:
    """
    Create a single transcript row for a live session (grouped by session).
    Called on first auto-store in a session. Returns transcript_id.
    """
    tid = new_id("trn")
    echodate = started_at_iso or now_iso()
    name_val = (name or "").strip() or None
    location_val = (location or "").strip() or "default"
    title = name_val or _title_for_transcript(tid, echodate)
    raw = (initial_text or "").strip()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transcripts (id, title, raw_text, polished_text, tags_json, echotag, echodate, created_at, updated_at, name, location) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (tid, title, raw, None, json.dumps([]), name_val or "transcript", echodate, echodate, echodate, name_val, location_val),
        )
        conn.commit()
    return tid


def append_transcript_chunk(transcript_id: str, chunk_text: str) -> None:
    """Append text to an existing transcript row (every 1 min or on stop). Updates raw_text and updated_at."""
    if not transcript_id or not (chunk_text or "").strip():
        return
    updated = now_iso()
    with get_conn() as conn:
        row = conn.execute("SELECT raw_text FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
        if not row:
            return
        existing = (row[0] or "").strip()
        new_raw = (existing + "\n\n" + chunk_text.strip()).strip() if existing else chunk_text.strip()
        conn.execute(
            "UPDATE transcripts SET raw_text = ?, updated_at = ? WHERE id = ?",
            (new_raw, updated, transcript_id),
        )
        conn.commit()


def update_transcript_tags_and_echotag(transcript_id: str, tags: list[str], echotag: str | None = None) -> None:
    """Update tags_json (and optionally echotag) for a transcript. Used after auto-store so tags are visible in list."""
    if not transcript_id:
        return
    tags_list = [t.strip() for t in tags if (t or "").strip()][:16]
    with get_conn() as conn:
        if echotag is not None:
            conn.execute(
                "UPDATE transcripts SET tags_json = ?, echotag = ? WHERE id = ?",
                (json.dumps(tags_list), (echotag or "").strip() or (",".join(tags_list) if tags_list else ""), transcript_id),
            )
        else:
            conn.execute(
                "UPDATE transcripts SET tags_json = ? WHERE id = ?",
                (json.dumps(tags_list), transcript_id),
            )
        conn.commit()
