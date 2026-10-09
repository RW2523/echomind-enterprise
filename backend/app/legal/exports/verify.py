"""Checking an export before it is released (S1.10, spec test T09).

The rendered file is opened again as a recipient would and inspected:
- redactions_absent: no planned string in any text, metadata or package part (the watermark
  line itself excepted);
- metadata_clean: no author, title, subject or keywords of the original; no XMP;
- structure_clean: PDF — no annotations, form fields or attachments; Word — no comments, custom
  XML, tracked changes or hidden runs;
- watermark: on every PDF page, in the Word header.
Anything short of all four, and the export is not released.
"""
from __future__ import annotations

import html
import io
import re
import zipfile


def _absent(haystack: str, strings: list[str], watermark: str) -> bool:
    body = haystack.replace(watermark, "").replace(html.escape(watermark), "").lower()
    return all(s.lower() not in body and html.escape(s).lower() not in body for s in strings if s)


def verify_pdf(data: bytes, strings: list[str], watermark: str) -> dict:
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    texts = [p.get_text() for p in doc]
    meta = {k: v for k, v in (doc.metadata or {}).items() if k not in ("format", "encryption")}
    xmp = doc.get_xml_metadata() or ""
    raw = doc.tobytes(garbage=4, expand=255).decode("latin-1")
    checks = {
        "redactions_absent": _absent("\n".join(texts) + str(meta) + xmp, strings, watermark)
        and all(s.lower() not in raw.lower() for s in strings if s and s not in watermark),
        "metadata_clean": not any(meta.get(k) for k in ("author", "title", "subject", "keywords")) and not xmp.strip(),
        "structure_clean": all(not list(p.annots() or []) and not list(p.widgets() or []) for p in doc) and doc.embfile_count() == 0,
        "watermark": bool(texts) and all(watermark in t for t in texts),
    }
    return {"ok": all(checks.values()), "checks": checks, "pages": len(texts)}


_FORBIDDEN_PARTS = re.compile(r"^(customXml/|word/comments|word/people\.xml|word/commentsExtended|word/commentsIds|word/vbaProject)")


def verify_docx(data: bytes, strings: list[str], watermark: str) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        parts = {n: z.read(n).decode("utf-8", "replace") for n in names if n.endswith((".xml", ".rels"))}
    body = parts.get("word/document.xml", "")
    core = parts.get("docProps/core.xml", "")
    headers = "".join(v for k, v in parts.items() if re.match(r"word/header\d*\.xml$", k))
    creator = re.search(r"<dc:creator>(.*?)</dc:creator>", core)
    title = re.search(r"<dc:title>(.*?)</dc:title>", core)
    checks = {
        "redactions_absent": _absent("".join(parts.values()), strings, watermark),
        "metadata_clean": (not creator or creator.group(1) in ("", "EchoMind Law")) and (not title or not title.group(1)),
        "structure_clean": not any(_FORBIDDEN_PARTS.match(n) for n in names)
        and not any(f"<{t}" in body for t in ("w:ins ", "w:ins>", "w:del ", "w:del>", "w:vanish", "w:commentReference", "w:delText")),
        "watermark": html.escape(watermark) in headers or watermark in headers,
    }
    return {"ok": all(checks.values()), "checks": checks}


def verify(fmt: str, data: bytes, strings: list[str], watermark: str) -> dict:
    return verify_pdf(data, strings, watermark) if fmt == "pdf" else verify_docx(data, strings, watermark)
