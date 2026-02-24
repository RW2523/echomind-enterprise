"""
PDF extractor: extract text per page for downstream chunking with page boundaries.
"""
from __future__ import annotations
from typing import List, Dict, Any
from io import BytesIO
from pypdf import PdfReader


def extract_pdf(data: bytes) -> List[Dict[str, Any]]:
    """
    Extract text per page. Returns list of {"text", "page_start", "page_end", "section_path"}.
    """
    blocks = []
    reader = PdfReader(BytesIO(data))
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append({
                "text": text,
                "page_start": i + 1,
                "page_end": i + 1,
                "section_path": f"page_{i + 1}",
            })
    return blocks
