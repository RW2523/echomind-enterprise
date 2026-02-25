"""PDF extraction: text + structure (pages)."""
from __future__ import annotations
from io import BytesIO
from typing import List, Tuple

from pypdf import PdfReader


def extract_pdf(data: bytes) -> Tuple[str, List[dict]]:
    """
    Extract full text and per-page structure.
    Returns (full_text, [{"page": 1, "text": "..."}, ...]).
    """
    r = PdfReader(BytesIO(data))
    pages = []
    full_parts = []
    for i, p in enumerate(r.pages):
        text = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": text})
        full_parts.append(text)
    full_text = "\n\n".join(full_parts)
    return full_text, pages
