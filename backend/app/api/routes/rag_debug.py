"""RAG debug endpoint: run retrieval + answer and return intent, chunks, and answer for the test page."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...rag.advanced import rag_debug_run

router = APIRouter(prefix="/rag", tags=["rag"])


class RagDebugRequest(BaseModel):
    question: str = ""
    advanced_rag: bool = False


@router.post("/debug")
async def rag_debug(body: RagDebugRequest):
    """
    Run the full RAG flow for a question and return debug info:
    - intent: document | transcript | general | advanced_rag
    - chunks: list of { score, text_preview, filename, doc_id, chunk_index, filetype }
    - answer: the model answer (same as chat would return)
    - message: optional info (e.g. why no retrieval was done)
    """
    q = (body.question or "").strip()
    result = await rag_debug_run(q, advanced_rag=body.advanced_rag)
    return result

