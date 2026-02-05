from fastapi import APIRouter, WebSocket
from pydantic import BaseModel
from ...rag.llm import OpenAICompatChat
from ...core.config import settings
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.index import index
from ...transcribe.ws import handler as ws_handler
import json

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

@router.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_handler(ws)

class PolishIn(BaseModel):
    raw_text: str

@router.post("/polish")
async def polish(inp: PolishIn):
    sys="Polish the transcript into clear, well-structured notes with headings and bullet points. Keep meaning."
    polished = await chat.chat([{"role":"system","content":sys},{"role":"user","content":inp.raw_text}], temperature=0.2, max_tokens=700)
    return {"polished": polished}

class StoreIn(BaseModel):
    raw_text: str
    polished_text: str | None = None

@router.post("/store")
async def store(inp: StoreIn):
    tid = new_id("trn")
    tags=[]
    try:
        tag_txt = await chat.chat(
            [{"role":"system","content":"Extract 3-6 short topic tags. Return comma-separated tags only."},
             {"role":"user","content":inp.raw_text[:3500]}],
            temperature=0.0, max_tokens=60
        )
        tags=[t.strip() for t in tag_txt.split(",") if t.strip()][:8]
    except Exception:
        tags=[]
    with get_conn() as conn:
        conn.execute("INSERT INTO transcripts (id, raw_text, polished_text, tags_json, created_at) VALUES (?,?,?,?,?)",
                     (tid, inp.raw_text, inp.polished_text, json.dumps(tags), now_iso()))
        conn.commit()
    try:
        await index.add_text(f"transcript_{tid}", inp.raw_text + ("\n\n"+inp.polished_text if inp.polished_text else ""), {"type":"transcript","tags":tags,"created_at":now_iso()})
    except Exception:
        pass
    return {"transcript_id": tid, "tags": tags, "created_at": now_iso()}
