from __future__ import annotations
from io import BytesIO
from typing import List, Optional, Tuple

from pypdf import PdfReader
from docx import Document
from pptx import Presentation


def parse_pdf(data: bytes) -> Tuple[str, int, List[Tuple[int, int]]]:
    """Return (text, page_count, page_char_offsets).

    page_char_offsets: list of (start_char_offset, page_number_1indexed) sorted by
    ascending offset. Allows O(log n) lookup of page number for any character offset.
    """
    r = PdfReader(BytesIO(data))
    pages_text: List[str] = []
    page_offsets: List[Tuple[int, int]] = []
    offset = 0
    for i, page in enumerate(r.pages):
        page_text = page.extract_text() or ""
        page_offsets.append((offset, i + 1))
        pages_text.append(page_text)
        offset += len(page_text) + 1  # +1 for the \n separator
    text = "\n".join(pages_text)
    return text, len(r.pages), page_offsets


def page_for_offset(offset: int, page_offsets: List[Tuple[int, int]]) -> Optional[int]:
    """Binary-search page_offsets to find the 1-indexed page number for a character offset.

    page_offsets must be sorted ascending by start_char_offset (guaranteed by parse_pdf).
    Returns None when page_offsets is empty.
    """
    if not page_offsets:
        return None
    lo, hi = 0, len(page_offsets) - 1
    result = page_offsets[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if page_offsets[mid][0] <= offset:
            result = page_offsets[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def parse_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs])


def parse_pptx(data: bytes) -> str:
    prs = Presentation(BytesIO(data))
    parts = []
    for s in prs.slides:
        for sh in s.shapes:
            if hasattr(sh, "text") and sh.text:
                parts.append(sh.text)
    return "\n".join(parts)


def parse_any(filename: str, data: bytes) -> Tuple[str, str, int, List[Tuple[int, int]]]:
    """Return (filetype, text, estimated_pages, page_char_offsets).

    estimated_pages=0 and page_char_offsets=[] when page info is unavailable.
    """
    f = filename.lower()
    if f.endswith(".pdf"):
        text, pages, page_offsets = parse_pdf(data)
        return "pdf", text, pages, page_offsets
    if f.endswith(".docx"):
        return "docx", parse_docx(data), 0, []
    if f.endswith(".pptx"):
        return "pptx", parse_pptx(data), 0, []
    return "txt", data.decode("utf-8", errors="ignore"), 0, []
