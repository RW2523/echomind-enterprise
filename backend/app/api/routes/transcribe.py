import json
import httpx
import logging
from fastapi import APIRouter, WebSocket, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from ...core.db import get_conn
from ...refine import refine_text
from ...transcribe.ws import handler as ws_handler
from ...transcribe.store_to_db import store_transcript_to_db
from ...tagging import get_metadata
from ...rag.llm import OpenAICompatChat
from ...core.config import settings

logger = logging.getLogger(__name__)

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
    has_name_location = False
    try:
        rows = conn.execute(
            "SELECT id, title, tags_json, echotag, created_at, raw_text, polished_text, name, location FROM transcripts WHERE 1=1" + extra + " ORDER BY created_at DESC",
            params,
        ).fetchall()
        has_title = True
        has_content = True
        has_name_location = True
    except Exception:
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
    return rows, has_title, has_content, has_name_location


@router.get("/list")
def list_transcripts(since: str | None = None, last_hours: float | None = None):
    """List transcripts with optional time filter. Query params: since (ISO datetime), last_hours (e.g. 2)."""
    with get_conn() as conn:
        rows, has_title, has_content, has_name_location = _list_transcripts_impl(conn, since_iso=since, last_hours=last_hours)
    out = []
    for r in rows:
        name_val = None
        location_val = None
        if has_title and has_content and has_name_location and len(r) >= 9:
            tid, title, tags_json, echotag, created_at, raw_text, polished_text, name_val, location_val = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
        elif has_title and has_content and len(r) >= 7:
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
        if name_val is not None:
            item["name"] = name_val
        if location_val is not None:
            item["location"] = location_val
        if raw_text is not None:
            item["raw_text"] = raw_text
        if polished_text is not None:
            item["polished_text"] = polished_text
        out.append(item)
    return {"transcripts": out}

@router.websocket("/ws")
async def ws(ws: WebSocket):
    # Auth gate (Phase 0b): when enabled, require a valid session cookie on the WS handshake.
    from ...core.config import settings as _settings
    if _settings.AUTH_ENABLED:
        from ...core.auth import decode_token
        if not decode_token(ws.cookies.get("echomind_token", "")):
            await ws.close(code=1008)
            return
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
    name: str | None = None  # Display name (from Start popup or default)
    location: str | None = None  # Location; default "default" if not set
    tags: list[str] | None = None  # Manual tags (overrides LLM extraction when provided)

@router.post("/store")
async def store(inp: StoreIn):
    refined_value = inp.refined_text if inp.refined_text is not None else inp.polished_text
    result = await store_transcript_to_db(
        raw_text=inp.raw_text,
        refined_text=refined_value,
        echotag=inp.echotag,
        name=inp.name,
        location=inp.location,
        tags=inp.tags,
    )
    return result


class UpdateTranscriptIn(BaseModel):
    name: str | None = None
    location: str | None = None
    tags: list[str] | None = None


@router.get("/transcripts/{transcript_id}")
def get_transcript(transcript_id: str):
    """Get a single transcript by ID with full raw_text and polished_text (for preview popup)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, raw_text, polished_text, tags_json, echotag, created_at, name, location FROM transcripts WHERE id = ?",
            (transcript_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    tid, title, raw_text, polished_text, tags_json, echotag, created_at, name_val, location_val = row
    tags = []
    if tags_json:
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
        except Exception:
            pass
    return {
        "id": tid,
        "title": title or tid,
        "raw_text": raw_text or "",
        "polished_text": polished_text or "",
        "tags": tags,
        "echotag": echotag or "",
        "created_at": created_at or "",
        "name": name_val,
        "location": location_val,
    }


@router.delete("/transcripts/{transcript_id}")
async def delete_transcript(transcript_id: str):
    """Delete a transcript from the transcripts table and from the RAG index (embeddings, chunks, documents)."""
    from ...rag.index import index

    with get_conn() as conn:
        row = conn.execute("SELECT id FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcript not found")
        # Collect every KB document tied to this transcript. Live/auto-stored transcripts are
        # indexed under filename transcript_<kb_id> (NOT transcript_<transcript_id>) with the
        # transcript_id only in meta_json — so match on both, else embeddings are orphaned. (audit H1)
        doc_ids: set = set()
        legacy = conn.execute(
            "SELECT id FROM documents WHERE filename = ?", (f"transcript_{transcript_id}",),
        ).fetchone()
        if legacy:
            doc_ids.add(legacy[0])
        for did, meta_json in conn.execute(
            "SELECT id, meta_json FROM documents WHERE meta_json LIKE ?", (f"%{transcript_id}%",),
        ).fetchall():
            try:
                if json.loads(meta_json or "{}").get("transcript_id") == transcript_id:
                    doc_ids.add(did)
            except Exception:
                pass
    for doc_id in doc_ids:
        try:
            await index.delete_document(doc_id)
        except Exception as e:
            logger.warning("delete_transcript: failed to delete KB doc %s: %s", doc_id, e)
    with get_conn() as conn:
        conn.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
        conn.commit()
    return {"ok": True, "deleted": transcript_id, "kb_docs_removed": len(doc_ids)}


@router.get("/transcripts/{transcript_id}/analysis")
def get_transcript_analysis(transcript_id: str):
    """Return all analysis cards for a given transcript (stored during live session).

    Also searches by session_id for the transcript row, so cards saved before the
    first auto-store (which had transcript_id=NULL but session_id set) are included.
    """
    with get_conn() as conn:
        trow = conn.execute("SELECT session_id FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
        session_id = trow[0] if trow else None
        # Include cards saved before the first auto-store (transcript_id NULL, session_id set) and
        # any that finished after the last backfill — they'd otherwise be invisible. (audit M)
        if session_id:
            rows = conn.execute(
                """SELECT id, segment_id, segment_text, label, confidence, explanation, source_refs, created_at
                   FROM transcript_analysis
                   WHERE transcript_id = ? OR (transcript_id IS NULL AND session_id = ?)
                   ORDER BY created_at ASC""",
                (transcript_id, session_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, segment_id, segment_text, label, confidence, explanation, source_refs, created_at
                   FROM transcript_analysis WHERE transcript_id = ? ORDER BY created_at ASC""",
                (transcript_id,),
            ).fetchall()
    cards = []
    for row in rows:
        aid, seg_id, seg_text, label, conf, expl, src_refs, created_at = row
        try:
            sources = json.loads(src_refs or "[]")
        except Exception:
            sources = []
        cards.append({
            "id": aid,
            "segment_id": seg_id,
            "segment_text": seg_text,
            "label": label,
            "confidence": conf,
            "explanation": expl,
            "source_chunks": sources,
            "created_at": created_at,
        })
    return {"cards": cards}


@router.get("/chunks/{chunk_id}/preview")
def get_chunk_preview(chunk_id: str):
    """Return source chunk text and document metadata for analysis card source preview."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT text, source_json, doc_id FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        text, src_json, doc_id = row
        source = {}
        try:
            source = json.loads(src_json or "{}")
        except Exception:
            pass
        doc_title = ""
        if doc_id:
            doc_row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if doc_row:
                doc_title = doc_row[0]
    return {
        "chunk_id": chunk_id,
        "text": text or "",
        "doc_title": doc_title or source.get("filename") or "",
        "doc_id": doc_id or source.get("doc_id") or "",
        "source": source,
    }


class SpeakIn(BaseModel):
    mode: str = "card"  # "card" | "summary"
    text: str | None = None  # for mode=card: the text to speak
    cards: list | None = None  # for mode=summary: list of {label, segment_text, explanation}
    transcript_id: str | None = None


@router.post("/speak")
async def speak(inp: SpeakIn):
    """
    Generate spoken audio for an analysis card or a summary of all cards.
    Uses the voice service Piper TTS if available; returns text fallback otherwise.
    Frontend should use browser SpeechSynthesis if audio_b64 is null.
    """
    if inp.mode == "card" and inp.text:
        speak_text = inp.text
    elif inp.mode == "summary" and inp.cards:
        # Build a summary text from all cards
        lines = ["Here is the analysis summary from the live transcript:"]
        for card in inp.cards[:20]:
            label = card.get("label", "")
            seg = (card.get("segment_text") or "")[:120]
            expl = (card.get("explanation") or "")[:200]
            lines.append(f"{label}: {seg}. {expl}")
        speak_text = " ".join(lines)
    else:
        speak_text = inp.text or ""

    if not speak_text.strip():
        raise HTTPException(status_code=400, detail="No text to speak")

    # Try voice service TTS
    voice_url = getattr(settings, "VOICE_TTS_URL", "http://voice:8002/speak")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                voice_url,
                json={"text": speak_text},
            )
            if r.is_success:
                import base64
                audio_b64 = base64.b64encode(r.content).decode()
                return {"audio_b64": audio_b64, "format": "wav", "text": speak_text}
    except Exception as e:
        logger.debug("Voice TTS unavailable, returning text fallback: %s", e)

    # Fallback: return text for browser SpeechSynthesis
    return {"audio_b64": None, "format": None, "text": speak_text}


@router.patch("/transcripts/{transcript_id}")
async def update_transcript(transcript_id: str, inp: UpdateTranscriptIn):
    """Update transcript metadata (name, location, tags). Editable bar uses this."""
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcript not found")
        updates = []
        params = []
        if inp.name is not None:
            updates.append("name = ?")
            params.append((inp.name or "").strip() or None)
        if inp.location is not None:
            updates.append("location = ?")
            params.append((inp.location or "").strip() or "default")
        if inp.tags is not None:
            updates.append("tags_json = ?")
            tags_list = [t.strip() for t in inp.tags if (t or "").strip()][:16]
            params.append(json.dumps(tags_list))
        if not updates:
            row = conn.execute(
                "SELECT id, title, name, location, tags_json, echotag, created_at FROM transcripts WHERE id = ?",
                (transcript_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Transcript not found")
            _, title, name, location, tags_json, echotag, created_at = row
            tags = json.loads(tags_json) if tags_json else []
            return {"transcript_id": transcript_id, "title": title, "name": name, "location": location, "tags": tags, "echotag": echotag, "created_at": created_at}
        params.append(transcript_id)
        conn.execute(
            "UPDATE transcripts SET " + ", ".join(updates) + " WHERE id = ?",
            params,
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, title, name, location, tags_json, echotag, created_at FROM transcripts WHERE id = ?",
            (transcript_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    _, title, name, location, tags_json, echotag, created_at = row
    tags = json.loads(tags_json) if tags_json else []
    return {"transcript_id": transcript_id, "title": title, "name": name, "location": location, "tags": tags, "echotag": echotag, "created_at": created_at}
