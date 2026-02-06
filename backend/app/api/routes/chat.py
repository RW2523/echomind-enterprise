import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.advanced import answer as answer_with_citations, answer_stream, update_conversation_summary

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_conversation_summary(chat_id: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT conversation_summary FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return row[0] if row and row[0] else None


def _set_conversation_summary(chat_id: str, summary: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE chats SET conversation_summary = ? WHERE id = ?", (summary, chat_id))
        conn.commit()


class CreateChatIn(BaseModel):
    title: str = "EchoMind Chat"


@router.post("/create")
async def create_chat(inp: CreateChatIn):
    cid = new_id("chat")
    with get_conn() as conn:
        conn.execute("INSERT INTO chats (id, title, created_at, conversation_summary) VALUES (?,?,?,?)",
                     (cid, inp.title, now_iso(), None))
        conn.commit()
    return {"chat_id": cid}


class AskIn(BaseModel):
    chat_id: str
    message: str
    persona: str | None = None
    context_window: str | None = None


@router.post("/ask")
async def ask(inp: AskIn):
    with get_conn() as conn:
        rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC", (inp.chat_id,)).fetchall()
    history = [{"role": r[0], "content": r[1]} for r in rows]
    conversation_summary = _get_conversation_summary(inp.chat_id)

    out = await answer_with_citations(
        inp.message,
        history,
        persona=inp.persona,
        context_window=inp.context_window or "all",
        conversation_summary=conversation_summary,
    )

    with get_conn() as conn:
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "user", inp.message, now_iso()))
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "assistant", out["answer"], now_iso()))
        conn.commit()

    new_summary = await update_conversation_summary(conversation_summary, inp.message, out["answer"])
    _set_conversation_summary(inp.chat_id, new_summary)

    return out


@router.post("/ask-stream")
async def ask_stream(inp: AskIn):
    async def gen():
        with get_conn() as conn:
            rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC", (inp.chat_id,)).fetchall()
        history = [{"role": r[0], "content": r[1]} for r in rows]
        conversation_summary = _get_conversation_summary(inp.chat_id)

        with get_conn() as conn:
            conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                         (new_id("msg"), inp.chat_id, "user", inp.message, now_iso()))
            conn.commit()

        full_answer: str | None = None
        try:
            async for kind, text, citations in answer_stream(
                inp.message,
                history,
                persona=inp.persona,
                context_window=inp.context_window or "all",
                conversation_summary=conversation_summary,
            ):
                if kind == "chunk":
                    yield json.dumps({"type": "chunk", "text": text or ""}) + "\n"
                elif kind == "done":
                    full_answer = text or ""
                    with get_conn() as conn:
                        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                                     (new_id("msg"), inp.chat_id, "assistant", full_answer, now_iso()))
                        conn.commit()
                    yield json.dumps({"type": "done", "answer": full_answer, "citations": citations or []}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        else:
            if full_answer is not None:
                new_summary = await update_conversation_summary(conversation_summary, inp.message, full_answer)
                _set_conversation_summary(inp.chat_id, new_summary)

    return StreamingResponse(gen(), media_type="application/x-ndjson")
