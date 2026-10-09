"""Upload checks (S1.6): the name, the declared type and the content must agree, and limits hold.

Accepted: PDF, Word (.docx), plain text and Markdown. A .docx must be a real Word package whose
uncompressed size stays under LEGAL_DOCX_MAX_UNCOMPRESSED_MB (no zip bombs); text must be UTF-8
without NUL bytes; a PDF must open and stay under LEGAL_UPLOAD_MAX_PAGES pages.
"""
from __future__ import annotations

import io
import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass

from ...core.config import settings

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TYPES = {
    ".pdf": ("application/pdf", {"application/pdf"}),
    ".docx": (DOCX, {DOCX}),
    ".txt": ("text/plain", {"text/plain"}),
    ".md": ("text/markdown", {"text/markdown", "text/x-markdown", "text/plain"}),
}
_GENERIC = {"", "application/octet-stream"}


class Rejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Checked:
    filename: str
    mime: str
    pages: int | None


def safe_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = "".join(ch for ch in unicodedata.normalize("NFC", base) if unicodedata.category(ch)[0] != "C")
    base = re.sub(r"\s+", " ", base).strip(" .")
    return base[:200] or "document"


def max_bytes() -> int:
    return settings.LEGAL_UPLOAD_MAX_MB * 1024 * 1024


def check(filename: str, declared_mime: str | None, data: bytes) -> Checked:
    name = safe_filename(filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in TYPES:
        raise Rejected(415, "Only PDF, Word (.docx), text and Markdown files can be imported.")
    if not data:
        raise Rejected(422, "The file is empty.")
    mime, accepted = TYPES[ext]
    declared = (declared_mime or "").split(";")[0].strip().lower()
    if declared not in _GENERIC and declared not in accepted:
        raise Rejected(415, "The file's type does not match its name.")
    pages = None
    if ext == ".pdf":
        pages = _pdf_pages(data)
    elif ext == ".docx":
        _check_docx(data)
    else:
        _check_text(data)
    return Checked(name, mime, pages)


def _pdf_pages(data: bytes) -> int:
    if b"%PDF-" not in data[:1024]:
        raise Rejected(415, "The file is not a valid PDF.")
    try:
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise Rejected(415, "Password-protected PDFs cannot be imported. Remove the password first.")
            pages = doc.page_count
    except Rejected:
        raise
    except Exception:
        raise Rejected(415, "The PDF could not be read.")
    if pages > settings.LEGAL_UPLOAD_MAX_PAGES:
        raise Rejected(413, f"The PDF has {pages} pages; the limit is {settings.LEGAL_UPLOAD_MAX_PAGES}.")
    return pages


def _check_docx(data: bytes) -> None:
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise Rejected(415, "The file is not a valid Word document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            names = {i.filename for i in infos}
            total = sum(i.file_size for i in infos)
    except zipfile.BadZipFile:
        raise Rejected(415, "The file is not a valid Word document.")
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise Rejected(415, "The file is not a valid Word document.")
    if len(infos) > 10_000 or total > settings.LEGAL_DOCX_MAX_UNCOMPRESSED_MB * 1024 * 1024:
        raise Rejected(415, "The Word document is too large once unpacked.")


def _check_text(data: bytes) -> None:
    if b"\x00" in data:
        raise Rejected(415, "The file contains binary data, not text.")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise Rejected(415, "Text files must be UTF-8.")
