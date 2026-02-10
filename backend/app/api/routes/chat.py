import json
import logging
import re
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ...utils.ids import new_id, now_iso
from ...core.db import get_conn
from ...rag.advanced import answer as answer_with_citations, answer_stream, update_conversation_summary, _answer_general

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def _update_summary_background(prev_summary: str | None, user_msg: str, assistant_msg: str, chat_id: str) -> None:
    """Run conversation summary update in background; do not block response. Logs errors."""
    try:
        new_summary = await update_conversation_summary(prev_summary, user_msg, assistant_msg)
        _set_conversation_summary(chat_id, new_summary)
    except Exception as e:
        logger.warning("Conversation summary update failed (background): %s", e)


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
    use_knowledge_base: bool = True
    advanced_rag: bool = False


class AskVoiceIn(BaseModel):
    message: str
    persona: str | None = None
    context_window: str | None = None
    use_knowledge_base: bool = True
    advanced_rag: bool = True


def _parse_transcript_time_query(message: str) -> float | None:
    """If the message asks for transcripts in a time range (e.g. 'last 2 hours'), return hours as float; else None."""
    m = (message or "").strip().lower()
    if not m or "transcript" not in m and "speak" not in m and "say" not in m and "talk" not in m:
        return None
    # e.g. "last 2 hours", "last 1 hour", "in the last 3 hours", "past 2 hours"
    match = re.search(r"(?:last|past|in the last)\s+(\d+(?:\.\d+)?)\s*(hour|hours?)", m, re.I)
    if match:
        n = float(match.group(1))
        return n if n > 0 else None
    return None


def _fetch_transcripts_since_hours(last_hours: float) -> list[dict]:
    """Return list of transcript items with raw_text or polished_text, created_at >= (now - last_hours)."""
    since = (datetime.now(timezone.utc) - timedelta(hours=last_hours)).isoformat()
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT id, raw_text, polished_text, created_at FROM transcripts WHERE created_at >= ? ORDER BY created_at ASC",
                (since,),
            ).fetchall()
        except Exception:
            try:
                rows = conn.execute(
                    "SELECT id, raw_text, created_at FROM transcripts WHERE created_at >= ? ORDER BY created_at ASC",
                    (since,),
                ).fetchall()
            except Exception:
                return []
    out = []
    for r in rows:
        raw = r[1] if len(r) > 1 else None
        polished = r[2] if len(r) >= 4 else None  # 4 cols: id, raw_text, polished_text, created_at
        out.append({"raw_text": raw, "polished_text": polished, "created_at": r[-1] if r else None})
    return out


@router.post("/ask-voice")
async def ask_voice(inp: AskVoiceIn):
    msg = (inp.message or "").strip()
    last_hours = _parse_transcript_time_query(msg)
    if last_hours is not None:
        transcripts = _fetch_transcripts_since_hours(last_hours)
        parts = []
        for t in transcripts:
            text = (t.get("polished_text") or t.get("raw_text") or "").strip()
            if text:
                parts.append(text)
        if not parts:
            return {"answer": "I couldn't find any transcripts in that time range."}
        transcript_block = "\n\n---\n\n".join(parts)
        user_content = f"Question: {msg}\n\nTranscripts from the last {int(last_hours)} hour(s):\n\n{transcript_block}\n\nAnswer based on these transcripts only."
        out = await _answer_general(user_content, history=[], persona=inp.persona, conversation_summary=None)
        return {"answer": out["answer"]}

    out = await answer_with_citations(
        inp.message,
        history=[],
        persona=inp.persona,
        context_window=inp.context_window or "all",
        conversation_summary=None,
        use_knowledge_base=inp.use_knowledge_base,
        advanced_rag=inp.advanced_rag,
    )
    return {"answer": out["answer"]}


@router.post("/ask")
async def ask(inp: AskIn, background_tasks: BackgroundTasks):
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
        use_knowledge_base=inp.use_knowledge_base,
        advanced_rag=inp.advanced_rag,
    )

    with get_conn() as conn:
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "user", inp.message, now_iso()))
        conn.execute("INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                     (new_id("msg"), inp.chat_id, "assistant", out["answer"], now_iso()))
        conn.commit()

    background_tasks.add_task(_update_summary_background, conversation_summary, inp.message, out["answer"], inp.chat_id)
    return out


@router.post("/ask-stream")
async def ask_stream(inp: AskIn, background_tasks: BackgroundTasks):
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
                use_knowledge_base=inp.use_knowledge_base,
                advanced_rag=inp.advanced_rag,
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
                    if full_answer:
                        background_tasks.add_task(_update_summary_background, conversation_summary, inp.message, full_answer, inp.chat_id)
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
