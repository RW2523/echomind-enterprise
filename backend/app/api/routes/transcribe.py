import json
from fastapi import APIRouter, WebSocket
from pydantic import BaseModel
from ...core.db import get_conn
from ...refine import refine_text
from ...transcribe.ws import handler as ws_handler
from ...transcribe.store_to_db import store_transcript_to_db
from ...tagging import get_metadata

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


def _list_transcripts_impl(conn, since_iso: str | None = None, last_hours: float | None = None):
    """Shared logic: list transcripts, optionally filtered by time (since_iso or last_hours)."""
    extra = ""
    params = []
    if since_iso or last_hours is not None:
        if last_hours is not None:
            try:
                from datetime import datetime, timezone, timedelta
                since_dt = datetime.now(timezone.utc) - timedelta(hours=float(last_hours))
                since_iso = since_dt.isoformat()
            except Exception:
                pass
        if since_iso:
            extra = " AND created_at >= ?"
            params.append(since_iso)
    try:
        rows = conn.execute(
            "SELECT id, title, tags_json, echotag, created_at, raw_text, polished_text FROM transcripts WHERE 1=1" + extra + " ORDER BY created_at DESC",
            params,
        ).fetchall()
        has_title = True
        has_content = True
    except Exception:
        try:
            rows = conn.execute(
                "SELECT id, title, tags_json, echotag, created_at FROM transcripts WHERE 1=1" + extra + " ORDER BY created_at DESC",
                params,
            ).fetchall()
            has_title = True
            has_content = False
        except Exception:
            rows = conn.execute(
                "SELECT id, raw_text, tags_json, created_at FROM transcripts WHERE 1=1" + extra + " ORDER BY created_at DESC",
                params,
            ).fetchall()
            has_title = False
            has_content = True
    return rows, has_title, has_content


@router.get("/list")
def list_transcripts(since: str | None = None, last_hours: float | None = None):
    """List transcripts with optional time filter. Query params: since (ISO datetime), last_hours (e.g. 2)."""
    with get_conn() as conn:
        rows, has_title, has_content = _list_transcripts_impl(conn, since_iso=since, last_hours=last_hours)
    out = []
    for r in rows:
        if has_title and has_content and len(r) >= 7:
            tid, title, tags_json, echotag, created_at, raw_text, polished_text = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        elif has_title and len(r) >= 5:
            tid, title, tags_json, echotag, created_at = r[0], r[1], r[2], r[3], r[4]
            raw_text = polished_text = None
        else:
            tid, _raw, tags_json, created_at = r[0], r[1], r[2], r[3]
            title = None
            echotag = ""
            raw_text = _raw
            polished_text = None
        tags = []
        if tags_json:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
            except Exception:
                pass
        created_at = created_at or ""
        if not title and created_at:
            date_part = created_at[:16].replace("T", " ") if len(created_at) >= 16 else created_at
            short_id = (tid or "").replace("trn_", "")[:8]
            title = f"{date_part}_{short_id}" if short_id else date_part
        if not title:
            title = tid or ""
        item = {
            "id": tid,
            "title": title,
            "tags": tags,
            "echotag": (echotag or ""),
            "created_at": created_at,
        }
        if raw_text is not None:
            item["raw_text"] = raw_text
        if polished_text is not None:
            item["polished_text"] = polished_text
        out.append(item)
    return {"transcripts": out}

@router.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_handler(ws)

class TagsIn(BaseModel):
    raw_text: str

@router.post("/tags")
def preview_tags(inp: TagsIn):
    """Preview possible tags and conversation type for the given transcript text."""
    text = (inp.raw_text or "").strip()
    if not text:
        return {"tags": [], "conversation_type": "casual"}
    conv_type, tags = get_metadata(text)
    return {"tags": tags, "conversation_type": conv_type}

class RefineIn(BaseModel):
    raw_text: str

@router.post("/refine")
async def refine(inp: RefineIn):
    refined = await refine_text(inp.raw_text)
    return {"refined": refined}

class StoreIn(BaseModel):
    raw_text: str
    refined_text: str | None = None  # Refined/structured notes (API name); stored in polished_text column
    polished_text: str | None = None  # Legacy alias for refined_text
    echotag: str | None = None  # Optional tag/category; if not set, derived from tags

@router.post("/store")
async def store(inp: StoreIn):
    refined_value = inp.refined_text if inp.refined_text is not None else inp.polished_text
    result = await store_transcript_to_db(
        raw_text=inp.raw_text,
        refined_text=refined_value,
        echotag=inp.echotag,
    )
    return result
