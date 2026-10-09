"""What an export is made from (S1.10).

The content of a document or transcript as blocks — paragraphs (style + runs) and tables —
read the way a reader sees it: in Word files, tracked insertions are accepted, deletions,
hidden (vanished) runs and comments are left out; everything outside the main body (comments,
custom XML, metadata) is never read. PDFs keep their original bytes so redaction can remove
text in place; their text is read too, for detection.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

from ...core.db import get_conn
from ..documents import blobs

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class Source:
    kind: str                       # "pdf" | "docx" | "text"
    title: str
    text: str                       # what a reader sees, for detection
    blocks: list = field(default_factory=list)
    pdf: Optional[bytes] = None
    version_id: Optional[str] = None
    image_pages: list = field(default_factory=list)


# ── Word ─────────────────────────────────────────────────────────────────────────
def _runs(p) -> list[dict]:
    runs = []
    for r in p.iter(f"{W}r"):
        if any(a.tag == f"{W}del" for a in r.iterancestors()):
            continue                                  # deleted by a tracked change
        rpr = r.find(f"{W}rPr")
        if rpr is not None and rpr.find(f"{W}vanish") is not None:
            continue                                  # hidden text
        text = []
        for el in r:
            if el.tag == f"{W}t":
                text.append(el.text or "")
            elif el.tag == f"{W}tab":
                text.append("\t")
            elif el.tag in (f"{W}br", f"{W}cr"):
                text.append("\n")
        if text:
            flag = (lambda tag: rpr is not None and rpr.find(f"{W}{tag}") is not None
                    and rpr.find(f"{W}{tag}").get(f"{W}val") not in ("0", "false"))
            runs.append({"text": "".join(text), "b": flag("b"), "i": flag("i"), "u": flag("u")})
    return runs


def _para(p) -> dict:
    ps = p.find(f"{W}pPr/{W}pStyle")
    return {"type": "p", "style": ps.get(f"{W}val") if ps is not None else None, "runs": _runs(p)}


def _blocks(parent) -> list[dict]:
    out = []
    for el in parent:
        if el.tag == f"{W}p":
            out.append(_para(el))
        elif el.tag == f"{W}tbl":
            rows = []
            for tr in el.findall(f"{W}tr"):
                rows.append([[_para(p) for p in tc.iter(f"{W}p")] for tc in tr.findall(f"{W}tc")])
            out.append({"type": "table", "rows": rows})
        elif el.tag == f"{W}sdt":
            content = el.find(f"{W}sdtContent")
            if content is not None:
                out += _blocks(content)
    return out


def docx_blocks(data: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = etree.fromstring(z.read("word/document.xml"), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    body = root.find(f"{W}body")
    return _blocks(body) if body is not None else []


def blocks_text(blocks: list[dict]) -> str:
    lines = []
    for b in blocks:
        if b["type"] == "p":
            lines.append("".join(r["text"] for r in b["runs"]))
        else:
            for row in b["rows"]:
                lines.append(" | ".join(" ".join("".join(r["text"] for r in p["runs"]) for p in cell) for cell in row))
    return "\n".join(lines)


# ── Plain text ───────────────────────────────────────────────────────────────────
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def text_blocks(text: str) -> list[dict]:
    out = []
    for line in (text or "").splitlines():
        m = _MD_HEADING.match(line)
        if m:
            out.append({"type": "p", "style": f"Heading{min(len(m.group(1)), 3)}", "runs": [{"text": m.group(2), "b": False, "i": False, "u": False}]})
        else:
            out.append({"type": "p", "style": None, "runs": [{"text": line, "b": False, "i": False, "u": False}]})
    return out


# ── Loading ──────────────────────────────────────────────────────────────────────
class NotFound(Exception):
    pass


def load(matter_id: str, object_type: str, object_id: str) -> Source:
    with get_conn() as conn:
        if object_type == "document":
            r = conn.execute("SELECT d.title, v.id, v.filename, v.wrapped_key FROM legal_documents d JOIN legal_document_versions v "
                             "ON v.id = d.current_version_id WHERE d.id=? AND d.matter_id=? AND d.deleted_at IS NULL",
                             (object_id, matter_id)).fetchone()
            if not r:
                raise NotFound(object_id)
            title, vid, filename, wrapped = r
        else:
            r = conn.execute("SELECT title FROM legal_meetings WHERE id=? AND matter_id=? AND transcript_deleted_at IS NULL",
                             (object_id, matter_id)).fetchone()
            if not r:
                raise NotFound(object_id)
            segs = [s[0] for s in conn.execute("SELECT text FROM legal_transcript_segments s JOIN legal_captures c ON c.id = s.capture_id "
                                               "WHERE s.meeting_id=? ORDER BY c.started_at, s.seq", (object_id,))]
            text = f"# Transcript — {r[0]}\n" + "\n".join(segs)
            return Source("text", f"Transcript of {r[0]}", text, text_blocks(text))
    data = blobs.load(matter_id, vid, wrapped)
    low = filename.lower()
    if low.endswith(".pdf"):
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = [p.get_text() for p in doc]
            images = [i + 1 for i, p in enumerate(doc) if p.get_images() and len(pages[i].strip()) < 200]
        text = "\n".join(pages)
        blocks = [{"type": "p", "style": None, "runs": [{"text": line, "b": False, "i": False, "u": False}]}
                  for line in text.splitlines()]
        return Source("pdf", title, text, blocks, pdf=data, version_id=vid, image_pages=images)
    if low.endswith(".docx"):
        blocks = docx_blocks(data)
        return Source("docx", title, blocks_text(blocks), blocks, version_id=vid)
    text = data.decode("utf-8-sig")
    return Source("text", title, text, text_blocks(text), version_id=vid)
