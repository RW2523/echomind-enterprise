"""Query API."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Body

from app.router.orchestrator import answer

router = APIRouter(tags=["query"])


@router.post("/query")
async def query(
    user_query: str = Body(..., embed=True),
    mode: Optional[str] = Body(None, embed=True),
    doc_id: Optional[str] = Body(None, embed=True),
) -> dict:
    """
    Run RAG query: classify intent -> retrieve -> generate.
    Returns { answer, evidence[], source_used, from_sources }.
    """
    result = answer(user_query=user_query, mode=mode, doc_id=doc_id, include_evidence_block=True)
    return {
        "answer": result["answer"],
        "evidence": result["evidence"],
        "source_used": result["source_used"],
        "from_sources": result["from_sources"],
    }
