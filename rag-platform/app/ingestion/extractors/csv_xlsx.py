"""CSV/XLSX extraction: rows + headers, chunk by row blocks."""
from __future__ import annotations
import csv
from io import BytesIO, StringIO
from typing import List, Tuple

try:
    import openpyxl
    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False


def extract_csv(data: bytes) -> Tuple[str, List[dict]]:
    """Extract CSV: header row + rows. Structure: list of row dicts with keys = header."""
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    if not rows:
        return "", [{"section": "csv", "text": "", "rows": []}]
    header = rows[0]
    row_dicts = [dict(zip(header, r)) for r in rows[1:] if len(r) >= len(header)]
    full_text = "\n".join([", ".join(str(v) for v in r.values()) for r in row_dicts])
    return full_text, [{"section": "csv", "text": full_text, "header": header, "rows": row_dicts}]


def extract_xlsx(data: bytes) -> Tuple[str, List[dict]]:
    """Extract first sheet: header + rows. Structure per sheet/section."""
    if not HAS_XLSX:
        raise RuntimeError("openpyxl required for XLSX")
    wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return "", [{"section": "sheet", "text": "", "rows": []}]
    header = [str(c or "") for c in rows[0]]
    row_dicts = []
    for r in rows[1:]:
        row_dicts.append(dict(zip(header, [str(c) if c is not None else "" for c in r])))
    full_text = "\n".join([", ".join(str(v) for v in r.values()) for r in row_dicts])
    return full_text, [{"section": "sheet", "text": full_text, "header": header, "rows": row_dicts}]
