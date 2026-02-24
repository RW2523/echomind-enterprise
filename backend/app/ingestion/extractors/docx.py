"""
DOCX extractor: extract text with heading structure for chunking.
"""
from __future__ import annotations
from typing import List, Dict, Any
from io import BytesIO
from docx import Document


def extract_docx(data: bytes) -> List[Dict[str, Any]]:
    """
    Extract paragraphs with optional heading hierarchy. Returns list of {"text", "section_path", "page_start", "page_end"}.
    DOCX has no page numbers by default; we use section/heading path.
    """
    blocks = []
    doc = Document(BytesIO(data))
    current_heading = ""
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        style = (p.style and p.style.name or "").lower()
        if "heading" in style:
            current_heading = text
        section = current_heading or "body"
        blocks.append({
            "text": text,
            "page_start": None,
            "page_end": None,
            "section_path": section,
        })
    if not blocks:
        return []
    return blocks
