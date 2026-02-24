"""
Plain text extractor: full text as a single block for chunking.
"""
from __future__ import annotations
from typing import List, Dict, Any


def extract_txt(data: bytes) -> List[Dict[str, Any]]:
    """
    Extract full text. Returns single block {"text", "page_start", "page_end", "section_path"}.
    """
    text = (data or b"").decode("utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"text": text, "page_start": None, "page_end": None, "section_path": "body"}]
