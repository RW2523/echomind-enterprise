from fastapi import APIRouter, WebSocket
from pydantic import BaseModel
from ...rag.llm import OpenAICompatChat
from ...core.config import settings
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.index import index
from ...transcribe.ws import handler as ws_handler
from ...tagging import get_metadata
import json

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

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
    sys = "Refine the transcript into clear, well-structured notes with headings and bullet points. Keep meaning."
    refined = await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": inp.raw_text}], temperature=0.2, max_tokens=700)
    return {"refined": refined}

class StoreIn(BaseModel):
    raw_text: str
    refined_text: str | None = None  # Refined/structured notes (API name); stored in polished_text column
    polished_text: str | None = None  # Legacy alias for refined_text
    echotag: str | None = None  # Optional tag/category; if not set, derived from tags

@router.post("/store")
async def store(inp: StoreIn):
    tid = new_id("trn")
    echodate = now_iso()
    tags = []
    try:
        tag_txt = await chat.chat(
            [{"role":"system","content":"Extract 3-6 short topic tags. Return comma-separated tags only."},
             {"role":"user","content":inp.raw_text[:3500]}],
            temperature=0.0, max_tokens=60
        )
        tags = [t.strip() for t in tag_txt.split(",") if t.strip()][:8]
    except Exception:
        tags = []
    echotag = (inp.echotag or "").strip() or (",".join(tags) if tags else "transcript")
    refined_value = inp.refined_text if inp.refined_text is not None else inp.polished_text
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transcripts (id, raw_text, polished_text, tags_json, echotag, echodate, created_at) VALUES (?,?,?,?,?,?,?)",
            (tid, inp.raw_text, refined_value, json.dumps(tags), echotag, echodate, echodate),
        )
        conn.commit()
    try:
        await index.add_text(
            f"transcript_{tid}",
            inp.raw_text + ("\n\n" + refined_value if refined_value else ""),
            {"type": "transcript", "tags": tags, "echotag": echotag, "echodate": echodate, "created_at": echodate},
        )
    except Exception:
        pass
    return {"transcript_id": tid, "tags": tags, "echotag": echotag, "echodate": echodate, "created_at": echodate}
