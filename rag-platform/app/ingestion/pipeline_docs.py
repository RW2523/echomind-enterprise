"""
Document ingestion: extract text+structure -> chunk (token-aware) -> tag -> embed -> Qdrant + catalog.
"""
from __future__ import annotations
import re
import time
import uuid
import logging
from typing import Optional, Tuple

from app.core.config import settings
from app.ingestion.extractors import extract_pdf, extract_docx, extract_pptx, extract_txt, extract_csv, extract_xlsx
from app.ingestion.chunking import chunk_text_with_structure
from app.ingestion.tagging import get_tags
from app.qdrant.upsert import upsert_document_points
from app.catalog.dao import insert_catalog

logger = logging.getLogger(__name__)

DOC_TYPES = ("gov_rules", "tax", "faq", "book", "records")
FILE_TYPES = ("pdf", "docx", "pptx", "txt", "md", "csv", "xlsx")


def _detect_doc_type(text: str, file_type: str) -> str:
    """Simple heuristic: faq if Q&A pattern, book if long and has chapters, else records."""
    t = (text or "").lower()[:5000]
    if re.search(r"\b(?:q(?:uestion)?\s*[.:]|a(?:nswer)?\s*[.:])\s*", t):
        return "faq"
    if "chapter" in t and len(text) > 10000:
        return "book"
    if "income tax" in t or "gst" in t or "deduction" in t:
        return "tax"
    return "records"


def _extract(filename: str, data: bytes) -> Tuple[str, list, str]:
    f = filename.lower()
    if f.endswith(".pdf"):
        full, structure = extract_pdf(data)
        return full, structure, "pdf"
    if f.endswith(".docx"):
        full, structure = extract_docx(data)
        return full, structure, "docx"
    if f.endswith(".pptx"):
        full, structure = extract_pptx(data)
        return full, structure, "pptx"
    if f.endswith(".csv"):
        full, structure = extract_csv(data)
        return full, structure, "csv"
    if f.endswith(".xlsx"):
        full, structure = extract_xlsx(data)
        return full, structure, "xlsx"
    full, structure = extract_txt(data)
    return full, structure, "txt" if not f.endswith(".md") else "md"


def run_document_pipeline(
    filename: str,
    data: bytes,
    doc_id: Optional[str] = None,
    doc_type_override: Optional[str] = None,
    source_path: Optional[str] = None,
) -> str:
    """
    Full pipeline: extract -> chunk -> tag -> embed -> upsert to Qdrant + catalog.
    Returns doc_id.
    """
    doc_id = doc_id or str(uuid.uuid4())
    full_text, structure, file_type = _extract(filename, data)
    if not full_text.strip():
        logger.warning("Empty extraction for %s", filename)
        insert_catalog(
            doc_id=doc_id,
            title=filename,
            file_type=file_type,
            num_pages=0,
            num_chunks=0,
            source_path=source_path,
        )
        return doc_id

    doc_type = doc_type_override or _detect_doc_type(full_text, file_type)
    chunks = chunk_text_with_structure(full_text, structure, file_type, filename)
    if not chunks:
        chunks = [{"text": full_text[:8000], "section_path": "", "page_start": None, "page_end": None, "row_start": None, "row_end": None, "chunk_id": str(uuid.uuid4())}]

    ingested_at = int(time.time())
    num_pages = max(p.get("page") or p.get("slide") or 0 for p in structure) if structure else 0
    all_tags = []
    texts = []
    payloads = []
    for c in chunks:
        tags = get_tags(c["text"])
        all_tags.extend(tags)
        texts.append(c["text"])
        payloads.append({
            "doc_id": doc_id,
            "chunk_id": c["chunk_id"],
            "doc_title": filename,
            "doc_type": doc_type,
            "file_type": file_type,
            "section_path": c.get("section_path", ""),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "row_start": c.get("row_start"),
            "row_end": c.get("row_end"),
            "tags": list(set(tags))[:20],
            "ingested_at": ingested_at,
            "version": "v1",
            "text_preview": (c["text"])[:300],
        })
    upsert_document_points(texts, payloads)
    unique_tags = list(dict.fromkeys(all_tags))[:30]
    insert_catalog(
        doc_id=doc_id,
        title=filename,
        doc_type=doc_type,
        file_type=file_type,
        uploaded_at=ingested_at,
        tags=unique_tags,
        num_pages=num_pages or None,
        num_chunks=len(chunks),
        source_path=source_path,
    )
    logger.info("Ingested doc %s: %s chunks", doc_id, len(chunks))
    return doc_id
