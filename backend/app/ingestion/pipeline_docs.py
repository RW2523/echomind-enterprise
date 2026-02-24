"""
Documents ingestion pipeline: extract -> chunk -> tag -> embed -> upsert to Qdrant (or existing FAISS index).
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from ..core.config import settings
from ..utils.ids import new_id, now_iso
from ..core.timeutils import now_utc_ts
from .extractors.pdf import extract_pdf
from .extractors.docx import extract_docx
from .extractors.pptx import extract_pptx
from .extractors.txt import extract_txt
from .extractors.csv_xlsx import extract_csv, extract_xlsx
from . import chunking as ing_chunking
from . import tagging as ing_tagging
from ..models.embedder import embed_texts
from ..qdrant.client import is_qdrant_enabled, get_qdrant_client
from ..qdrant.upsert import upsert_document_points
from ..qdrant.collections import ensure_collections, DOCUMENTS_COLLECTION

logger = logging.getLogger(__name__)


def _extract_blocks(filename: str, data: bytes) -> tuple[str, List[Dict[str, Any]]]:
    """Dispatch by extension; return (file_type, list of blocks)."""
    f = (filename or "").strip().lower() or "file.txt"
    if f.endswith(".pdf"):
        return "pdf", extract_pdf(data)
    if f.endswith(".docx"):
        return "docx", extract_docx(data)
    if f.endswith(".pptx"):
        return "pptx", extract_pptx(data)
    if f.endswith(".csv"):
        return "csv", extract_csv(data)
    if f.endswith(".xlsx") or f.endswith(".xls"):
        return "xlsx", extract_xlsx(data)
    return "txt", extract_txt(data)


async def run_pipeline_docs(
    filename: str,
    data: bytes,
    doc_title: Optional[str] = None,
    version: str = "1",
) -> Dict[str, Any]:
    """
    Run full document pipeline: extract, chunk, tag, embed, upsert.
    Returns {"doc_id", "chunks_count", "file_type"}.
    When Qdrant is disabled, delegates to existing FAISS index (see main/routes).
    """
    file_type, blocks = _extract_blocks(filename, data)
    full_text = "\n\n".join(b.get("text", "") for b in blocks).strip()
    if not full_text:
        return {"doc_id": None, "chunks_count": 0, "file_type": file_type, "error": "No text extracted"}

    doc_id = new_id("doc")
    ingested_at = now_utc_ts()
    title = doc_title or filename

    # Use existing RAG chunk_document for compatibility (same chunk size/overlap)
    from ..rag.chunking import chunk_document
    rag_chunks = chunk_document(full_text, doc_id)
    embed_chunks = [c for c in rag_chunks if not c.is_parent]
    if not embed_chunks:
        return {"doc_id": doc_id, "chunks_count": 0, "file_type": file_type}

    texts = [c.text for c in embed_chunks]
    tags_per_chunk = [ing_tagging.tag_chunk(t) for t in texts]

    if is_qdrant_enabled():
        vecs = await embed_texts(texts)
        vector_size = vecs.shape[1]
        ensure_collections(vector_size)
        points = []
        for i, c in enumerate(embed_chunks):
            payload = {
                "source_type": "document",
                "doc_id": doc_id,
                "chunk_id": c.chunk_id,
                "doc_title": title,
                "doc_type": getattr(c.doc_type, "value", "user"),
                "file_type": file_type,
                "section_path": c.section or "",
                "page_start": None,
                "page_end": None,
                "row_start": None,
                "row_end": None,
                "tags": tags_per_chunk[i] if i < len(tags_per_chunk) else [],
                "ingested_at": ingested_at,
                "version": version,
                "text_preview": (c.text or "")[:2000],
            }
            # Qdrant point id: use chunk_id hash or uuid
            try:
                point_id = hash(c.chunk_id) & 0x7FFFFFFFFFFFFFFF
            except Exception:
                point_id = i
            points.append({"id": point_id, "payload": payload})
        await upsert_document_points(points, vectors=[v.tolist() for v in vecs])
        logger.info("Upserted %s document chunks to Qdrant", len(points))
        return {"doc_id": doc_id, "chunks_count": len(points), "file_type": file_type}

    # Fallback: use existing FAISS index and SQLite documents/chunks
    from ..rag.index import index
    await index.add_document(filename, file_type, full_text, {"filename": filename, "filetype": file_type})
    try:
        from ..catalog.dao import insert_document
        insert_document(doc_id, title=filename, file_type=file_type, num_chunks=len(embed_chunks))
    except Exception:
        pass
    return {"doc_id": doc_id, "chunks_count": len(embed_chunks), "file_type": file_type}
