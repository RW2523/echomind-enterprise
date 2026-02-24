"""
Query API: POST /query returns { answer, evidence[], source_used }.
"""
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from ..router.orchestrator import answer_with_evidence

router = APIRouter(tags=["query"])

class QueryIn(BaseModel):
    query: str
    context_window: str | None = None
    max_context_chunks: int = 10

@router.post("/query")
async def query(inp: QueryIn):
    result = await answer_with_evidence(inp.query, filters=None, max_context_chunks=inp.max_context_chunks)
    return result
