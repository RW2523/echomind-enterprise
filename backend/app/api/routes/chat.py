import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.advanced import answer as answer_with_citations, answer_stream

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


@router.post("/ask-stream")
async def ask_stream(inp: AskIn):
    async def gen():
        with get_conn() as conn:
            rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC", (inp.chat_id,)).fetchall()
        history = [{"role": r[0], "content": r[1]} for r in rows]

        with get_conn() as conn:
            conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                         (new_id("msg"), inp.chat_id, "user", inp.message, now_iso()))
            conn.commit()

        try:
            async for kind, text, citations in answer_stream(inp.message, history):
                if kind == "chunk":
                    yield json.dumps({"type": "chunk", "text": text or ""}) + "\n"
                elif kind == "done":
                    with get_conn() as conn:
                        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                                     (new_id("msg"), inp.chat_id, "assistant", text or "", now_iso()))
                        conn.commit()
                    yield json.dumps({"type": "done", "answer": text or "", "citations": citations or []}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
