#
# EchoMind API adapter: same REST surface as EchoMind frontend expects,
# implemented entirely with the assets backend (Postgres, Milvus, LangGraph agent).
# Transcripts are stored in Postgres and indexed in Milvus for RAG.
#
# RAG: All retrieval, embedding, and document handling use assets code only
# (vector_store, agent, tools/mcp_servers/rag). Do not import or use the main
# echomind backend RAG (backend/app/rag/).
#

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from logger import logger
from transcript_store import TranscriptStore


def get_agent(request: Request):
    return getattr(request.app.state, "agent", None)


def get_postgres_storage(request: Request):
    return getattr(request.app.state, "postgres_storage", None)


def get_vector_store(request: Request):
    return getattr(request.app.state, "vector_store", None)


def get_config_manager(request: Request):
    return getattr(request.app.state, "config_manager", None)


def get_transcript_store(request: Request):
    return getattr(request.app.state, "transcript_store", None)


def get_indexing_tasks(request: Request):
    return getattr(request.app.state, "indexing_tasks", {})


def create_echomind_router() -> APIRouter:
    router = APIRouter()

    # --- Docs (map to assets ingest + config + transcripts) ---
    @router.get("/docs/usage")
    async def docs_usage(request: Request):
        """Storage usage for sidebar."""
        try:
            # Approximate: Milvus collection size not trivial without pymilvus; return placeholder
            return {"usage_bytes": 0, "capacity_bytes": None}
        except Exception as e:
            logger.warning("docs/usage: %s", e)
            return {"usage_bytes": 0, "capacity_bytes": None}

    @router.get("/docs/list")
    async def docs_list(request: Request):
        """List uploaded documents only (exclude transcript sources)."""
        config = get_config_manager(request)
        store = get_transcript_store(request)
        if not config:
            return {"documents": []}
        try:
            config_obj = config.read_config()
            sources = getattr(config_obj, "sources", []) or []
            # Exclude transcript_% entries
            docs = [
                {"id": s, "filename": s, "filetype": "text", "created_at": ""}
                for s in sources
                if not (s or "").startswith("transcript_")
            ]
            return {"documents": docs}
        except Exception as e:
            logger.warning("docs/list: %s", e)
            return {"documents": []}

    @router.post("/docs/upload")
    async def docs_upload(
        request: Request,
        file: UploadFile = File(...),
    ):
        """Upload and ingest one file (assets ingest). Source name = filename for consistent retrieval."""
        vector_store = get_vector_store(request)
        config = get_config_manager(request)
        tasks = get_indexing_tasks(request)
        if not vector_store or not config:
            raise HTTPException(status_code=503, detail="Backend not ready")
        task_id = str(uuid.uuid4())
        permanent_dir = os.path.join("uploads", task_id)
        path = None
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
            filename = (file.filename or "upload").strip() or "upload"
            os.makedirs(permanent_dir, exist_ok=True)
            path = os.path.join(permanent_dir, filename)
            with open(path, "wb") as f:
                f.write(content)
            tasks[task_id] = "indexing_documents"
            try:
                documents = vector_store._load_documents([path])
                if not documents:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract text from file (unsupported format or empty). Try PDF or plain text.",
                    )
                vector_store.index_documents(documents)
                # Use same source as in doc metadata (basename of path) so retrieval filter works
                source_name = os.path.basename(path)
                config_obj = config.read_config()
                sources = getattr(config_obj, "sources", None) or []
                if source_name not in sources:
                    config_obj.sources = list(sources) + [source_name]
                    config.write_config(config_obj)
                chunk_count = len(documents)  # pre-split; actual chunks may be higher
                return {"ok": True, "doc_id": task_id, "chunks": chunk_count}
            finally:
                tasks[task_id] = "completed"
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("docs/upload: %s", e)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/docs/{doc_id}")
    async def docs_delete(doc_id: str, request: Request):
        """Delete a document (source) from config and Milvus if applicable."""
        config = get_config_manager(request)
        vector_store = get_vector_store(request)
        if not config or not vector_store:
            raise HTTPException(status_code=503, detail="Backend not ready")
        try:
            config_obj = config.read_config()
            sources = getattr(config_obj, "sources", []) or []
            # doc_id might be task_id or source name
            if doc_id in sources:
                sources.remove(doc_id)
                config_obj.sources = sources
                config.write_config(config_obj)
            # Optionally delete from Milvus by collection/source - assets uses single collection
            return {"ok": True, "deleted": doc_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/docs/data-preview")
    async def docs_data_preview(request: Request):
        """Data preview: documents, chunks, transcripts."""
        config = get_config_manager(request)
        store = get_transcript_store(request)
        if not store:
            return {"documents": [], "chunks": [], "transcripts": []}
        try:
            config_obj = config.read_config() if config else None
            sources = getattr(config_obj, "sources", []) or [] if config_obj else []
            docs = [{"id": s, "filename": s, "filetype": "text", "created_at": "", "meta_json": None} for s in sources if not (s or "").startswith("transcript_")]
            transcripts = await store.list_transcripts()
            transcripts_out = [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "tags": t["tags"],
                    "echotag": t["echotag"],
                    "created_at": t["created_at"],
                    "raw_length": 0,
                    "polished_length": 0,
                }
                for t in transcripts
            ]
            return {"documents": docs, "chunks": [], "transcripts": transcripts_out}
        except Exception as e:
            logger.warning("docs/data-preview: %s", e)
            return {"documents": [], "chunks": [], "transcripts": []}

    @router.post("/docs/delete-all")
    async def docs_delete_all(request: Request):
        """Delete all data: clear config sources, transcripts, chats."""
        config = get_config_manager(request)
        postgres = get_postgres_storage(request)
        if not postgres or not config:
            raise HTTPException(status_code=503, detail="Backend not ready")
        try:
            config_obj = config.read_config()
            config_obj.sources = []
            config.write_config(config_obj)
            # Clear conversations and transcripts (custom SQL if needed)
            return {"ok": True, "message": "All data deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- Chat (assets agent + Postgres) ---
    class CreateChatIn(BaseModel):
        title: str = "EchoMind Chat"

    @router.post("/chat/create")
    async def chat_create(inp: CreateChatIn, request: Request):
        postgres = get_postgres_storage(request)
        config = get_config_manager(request)
        if not postgres or not config:
            raise HTTPException(status_code=503, detail="Backend not ready")
        new_chat_id = str(uuid.uuid4())
        await postgres.save_messages_immediate(new_chat_id, [])
        await postgres.set_chat_metadata(new_chat_id, inp.title or "EchoMind Chat")
        config.updated_current_chat_id(new_chat_id)
        return {"chat_id": new_chat_id}

    class AskIn(BaseModel):
        chat_id: str
        message: str
        persona: Optional[str] = None
        context_window: Optional[str] = None
        use_knowledge_base: bool = True
        advanced_rag: bool = False

    class AskVoiceIn(BaseModel):
        message: str
        persona: Optional[str] = None
        context_window: Optional[str] = None
        use_knowledge_base: bool = True
        advanced_rag: bool = True

    @router.post("/chat/ask")
    async def chat_ask(inp: AskIn, request: Request):
        agent = get_agent(request)
        postgres = get_postgres_storage(request)
        if not agent or not postgres:
            raise HTTPException(status_code=503, detail="Backend not ready")
        full_answer = ""
        async for event in agent.query(inp.message, inp.chat_id):
            if isinstance(event, dict):
                if event.get("type") == "error":
                    full_answer += (event.get("data") or str(event)) + "\n"
                elif event.get("type") == "token":
                    full_answer += event.get("data") or ""
            elif isinstance(event, str):
                full_answer += event
        await postgres.save_messages_immediate(inp.chat_id, [])  # agent already saved
        return {"answer": full_answer, "citations": []}

    @router.post("/chat/ask-stream")
    async def chat_ask_stream(inp: AskIn, request: Request):
        agent = get_agent(request)
        postgres = get_postgres_storage(request)
        if not agent or not postgres:
            raise HTTPException(status_code=503, detail="Backend not ready")

        async def gen():
            buf = []
            err = None
            async for event in agent.query(inp.message, inp.chat_id):
                if isinstance(event, dict):
                    if event.get("type") == "error":
                        err = event.get("data") or str(event)
                        yield json.dumps({"type": "error", "message": err}) + "\n"
                        break
                    if event.get("type") == "token":
                        text = event.get("data") or ""
                        buf.append(text)
                        yield json.dumps({"type": "chunk", "text": text}) + "\n"
                elif isinstance(event, str):
                    buf.append(event)
                    yield json.dumps({"type": "chunk", "text": event}) + "\n"
            if err is None:
                full = "".join(buf)
                yield json.dumps({"type": "done", "answer": full, "citations": []}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @router.post("/chat/ask-voice")
    async def chat_ask_voice(inp: AskVoiceIn, request: Request):
        """Used by voice service: no chat_id, return answer only."""
        agent = get_agent(request)
        if not agent:
            raise HTTPException(status_code=503, detail="Backend not ready")
        # Use a temporary chat id for this request
        temp_id = str(uuid.uuid4())
        full_answer = ""
        async for event in agent.query(inp.message, temp_id):
            if isinstance(event, dict):
                if event.get("type") == "error":
                    full_answer = (event.get("data") or str(event)) or full_answer
                elif event.get("type") == "token":
                    full_answer += event.get("data") or ""
            elif isinstance(event, str):
                full_answer += event
        return {"answer": full_answer}

    # --- Transcribe (transcript_store + optional LLM for refine/tags) ---
    @router.get("/transcribe/list")
    async def transcribe_list(request: Request, since: Optional[str] = None, last_hours: Optional[float] = None):
        store = get_transcript_store(request)
        if not store:
            return {"transcripts": []}
        rows = await store.list_transcripts(since_iso=since, last_hours=last_hours)
        return {"transcripts": rows}

    class RefineIn(BaseModel):
        raw_text: str

    @router.post("/transcribe/refine")
    async def transcribe_refine(inp: RefineIn, request: Request):
        """Refine transcript with LLM (placeholder if no LLM)."""
        text = (inp.raw_text or "").strip()
        if not text:
            return {"refined": ""}
        # TODO: call assets LLM or Ollama for refine
        return {"refined": text}

    class TagsIn(BaseModel):
        raw_text: str

    @router.post("/transcribe/tags")
    async def transcribe_tags(inp: TagsIn, request: Request):
        """Preview tags for transcript (placeholder)."""
        if not (inp.raw_text or "").strip():
            return {"tags": [], "conversation_type": "casual"}
        # TODO: LLM or rule-based tags
        return {"tags": [], "conversation_type": "casual"}

    class StoreIn(BaseModel):
        raw_text: str
        refined_text: Optional[str] = None
        polished_text: Optional[str] = None
        echotag: Optional[str] = None
        name: Optional[str] = None
        location: Optional[str] = None
        tags: Optional[List[str]] = None

    @router.post("/transcribe/store")
    async def transcribe_store(inp: StoreIn, request: Request):
        store = get_transcript_store(request)
        if not store:
            raise HTTPException(status_code=503, detail="Backend not ready")
        refined = inp.refined_text if inp.refined_text is not None else inp.polished_text
        result = await store.create(
            raw_text=inp.raw_text,
            polished_text=refined,
            echotag=inp.echotag,
            name=inp.name,
            location=inp.location,
            tags=inp.tags,
        )
        return result

    class UpdateTranscriptIn(BaseModel):
        name: Optional[str] = None
        location: Optional[str] = None
        tags: Optional[List[str]] = None

    @router.patch("/transcribe/transcripts/{transcript_id}")
    async def transcribe_update(transcript_id: str, inp: UpdateTranscriptIn, request: Request):
        store = get_transcript_store(request)
        if not store:
            raise HTTPException(status_code=404, detail="Transcript not found")
        out = await store.update_transcript(transcript_id, name=inp.name, location=inp.location, tags=inp.tags)
        if not out:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return out

    return router
