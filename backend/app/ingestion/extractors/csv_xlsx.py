"""
CSV/XLSX extractor: extract rows with headers for chunking (row_start, row_end).
"""
from __future__ import annotations
from typing import List, Dict, Any
from io import BytesIO


def extract_csv(data: bytes) -> List[Dict[str, Any]]:
    """Extract CSV rows; first row as header. Returns blocks with row_start, row_end."""
    import csv
    text = (data or b"").decode("utf-8", errors="ignore")
    blocks = []
    reader = csv.reader(BytesIO(text.encode("utf-8")))
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    header_text = " | ".join(header)
    for i, row in enumerate(rows[1:], start=2):
        row_text = " | ".join(str(c) for c in row)
        blocks.append({
            "text": header_text + "\n" + row_text,
            "page_start": None,
            "page_end": None,
            "section_path": "table",
            "row_start": i,
            "row_end": i,
        })
    return blocks


def extract_xlsx(data: bytes) -> List[Dict[str, Any]]:
    """Extract first sheet rows; first row as header. Returns blocks with row_start, row_end."""
    try:
        import openpyxl
    except ImportError:
        return extract_txt_fallback(data)
    wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    blocks = []
    sheet = wb.active
    if not sheet:
        return []
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(c or "") for c in rows[0]]
    header_text = " | ".join(header)
    for i, row in enumerate(rows[1:], start=2):
        row_text = " | ".join(str(c or "") for c in row)
        blocks.append({
            "text": header_text + "\n" + row_text,
            "page_start": None,
            "page_end": None,
            "section_path": "sheet",
            "row_start": i,
            "row_end": i,
        })
    return blocks


def extract_txt_fallback(data: bytes) -> List[Dict[str, Any]]:
    """Fallback when openpyxl not available: treat as plain text."""
    from .txt import extract_txt
    return extract_txt(data)
