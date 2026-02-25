"""TXT/MD extraction: raw text, optional section split by headers."""
from __future__ import annotations
import re
from typing import List, Tuple


def extract_txt(data: bytes) -> Tuple[str, List[dict]]:
    full_text = data.decode("utf-8", errors="replace")
    sections = _split_md_sections(full_text)
    return full_text, sections


def _split_md_sections(text: str) -> List[dict]:
    # Split by ## or ### etc.
    parts = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
    sections = []
    current = ""
    current_title = ""
    for i, p in enumerate(parts):
        if p.strip().startswith("#"):
            if current.strip():
                sections.append({"section": current_title, "text": current.strip()})
            current_title = p.strip().lstrip("#").strip()
            current = ""
        else:
            current += p
    if current.strip():
        sections.append({"section": current_title, "text": current.strip()})
    if not sections:
        return [{"section": "", "text": text}]
    return sections
