from __future__ import annotations
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from pptx import Presentation

def parse_pdf(data: bytes) -> str:
    r = PdfReader(BytesIO(data))
    return "\n".join([(p.extract_text() or "") for p in r.pages])

def parse_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs])

def parse_pptx(data: bytes) -> str:
    prs = Presentation(BytesIO(data))
    parts=[]
    for s in prs.slides:
        for sh in s.shapes:
            if hasattr(sh, "text") and sh.text:
                parts.append(sh.text)
    return "\n".join(parts)

def parse_any(filename: str, data: bytes) -> tuple[str,str]:
    f = filename.lower()
    if f.endswith(".pdf"): return "pdf", parse_pdf(data)
    if f.endswith(".docx"): return "docx", parse_docx(data)
    if f.endswith(".pptx"): return "pptx", parse_pptx(data)
    return "txt", data.decode("utf-8", errors="ignore")
