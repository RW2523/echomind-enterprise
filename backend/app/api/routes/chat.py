from fastapi import APIRouter
from pydantic import BaseModel
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.advanced import answer as answer_with_citations

router = APIRouter(prefix="/chat", tags=["chat"])

class CreateChatIn(BaseModel):
    title: str = "EchoMind Chat"

@router.post("/create")
async def create_chat(inp: CreateChatIn):
    cid = new_id("chat")
    with get_conn() as conn:
        conn.execute("INSERT INTO chats (id, title, created_at) VALUES (?,?,?)", (cid, inp.title, now_iso()))
        conn.commit()
    return {"chat_id": cid}

class AskIn(BaseModel):
    chat_id: str
    message: str

@router.post("/ask")
async def ask(inp: AskIn):
    with get_conn() as conn:
        rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC", (inp.chat_id,)).fetchall()
    history=[{"role":r[0], "content":r[1]} for r in rows]
    out = await answer_with_citations(inp.message, history)

    with get_conn() as conn:
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "user", inp.message, now_iso()))
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "assistant", out["answer"], now_iso()))
        conn.commit()

    return out
