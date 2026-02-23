"""Debug endpoints for RAG retrieval sanity and index management. Not for production auth; gate behind env if needed."""
from __future__ import annotations
from fastapi import APIRouter, Query
from ...core.db import get_conn
from ...rag.index import index
from ...rag.advanced import _weighted_rrf, choose_rrf_weights

router = APIRouter(prefix="/debug", tags=["debug"])


def _hit_to_debug(h: dict) -> dict:
    """Minimal hit for debug output: chunk_id, score, filename."""
    src = h.get("source") or {}
    return {
        "chunk_id": h.get("chunk_id"),
        "score": round(float(h.get("score") or 0), 4),
        "filename": src.get("filename"),
    }


@router.get("/retrieve")
async def debug_retrieve(q: str = Query(..., description="Query string"), k: int = Query(10, ge=1, le=50)):
    """Run retrieval for query and return top dense, sparse, and fused hits (chunk_id, score, filename) for debugging."""
    query = (q or "").strip() or " "
    dense_hits = await index.search(query, max(k, 4))
    sparse_hits = index.sparse.search(query, max(k, 4))
    dense_w, sparse_w = choose_rrf_weights(query)
    fused = _weighted_rrf([dense_hits], [sparse_hits], k, dense_weight=dense_w, sparse_weight=sparse_w)
    return {
        "query": query,
        "dense": [_hit_to_debug(h) for h in dense_hits[:k]],
        "sparse": [_hit_to_debug(h) for h in sparse_hits[:k]],
        "fused": [_hit_to_debug(h) for h in fused[:k]],
    }


@router.post("/clear-index")
async def clear_index(full_reset: bool = Query(False, description="Also clear documents and chunks tables")):
    """Clear RAG index files and in-memory state. Use full_reset=1 to also truncate documents/chunks so next ingest rebuilds from empty."""
    index.clear_all()
    if full_reset:
        with get_conn() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.commit()
        return {"ok": True, "message": "Index and documents/chunks cleared. Re-upload or re-add content to rebuild."}
    return {"ok": True, "message": "Index files cleared. DB documents/chunks unchanged; re-ingest will rebuild index."}
