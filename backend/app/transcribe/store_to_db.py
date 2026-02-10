"""
Shared logic: save transcript to DB (transcripts table + RAG index).
Used by POST /api/transcribe/store and by WebSocket combine→LLM→save flow.
"""
from __future__ import annotations
import json
import re
from ..utils.ids import new_id, now_iso
from ..core.db import get_conn
from ..rag.index import index
from ..rag.llm import OpenAICompatChat
from ..core.config import settings

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


async def store_transcript_to_db(
    raw_text: str,
    refined_text: str | None = None,
    echotag: str | None = None,
) -> dict:
    """
    Save a transcript to the transcripts table and RAG index.
    - raw_text: required.
    - refined_text: optional; if None, only raw is stored and indexed.
    - echotag: optional; if None, derived from LLM-generated tags.
    Returns: { transcript_id, title, tags, echotag, echodate, created_at }.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text is required")
    tid = new_id("trn")
    echodate = now_iso()
    title = _title_for_transcript(tid, echodate)
    tags = []
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
        tags = [t.strip() for t in (tag_txt or "").split(",") if t.strip()][:8]
    except Exception:
        tags = []
    echotag = (echotag or "").strip() or (",".join(tags) if tags else "transcript")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transcripts (id, title, raw_text, polished_text, tags_json, echotag, echodate, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (tid, title, raw_text, refined_text, json.dumps(tags), echotag, echodate, echodate),
        )
        conn.commit()
    try:
        index_text = raw_text + ("\n\n" + refined_text if refined_text else "")
        await index.add_text(
            f"transcript_{tid}",
            index_text,
            {"type": "transcript", "tags": tags, "echotag": echotag, "echodate": echodate, "created_at": echodate},
        )
    except Exception:
        pass
    return {"transcript_id": tid, "title": title, "tags": tags, "echotag": echotag, "echodate": echodate, "created_at": echodate}
