#!/usr/bin/env python3
"""Chunking content-loss regression test.

Guards the bug class that repeatedly cost this corpus real content: chunking that produces
healthy-looking output while silently dropping part of the document. Two separate incidents
came from exactly this — a section splitter whose body started after a truncated heading
(41% lost), and an indexing validator that `continue`d past chunks it didn't recognize
(90% lost on one chapter).

Run inside the backend container (it needs the app + the corpus):

    docker cp eval/test_chunk_coverage.py echomind-backend:/tmp/t.py \
      && docker exec echomind-backend python3 /tmp/t.py

Exit code 0 when every document meets the coverage floor, 1 otherwise (CI-friendly).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")

UPLOADS = os.getenv("ECHOMIND_UPLOAD_DIR", "/data/uploads")
MIN_COVERAGE = float(os.getenv("CHUNK_MIN_COVERAGE", "0.98"))

# Facts that must survive chunking. Each was lost by a real regression at some point.
CANARIES = [
    ("doc_ac04549f3765.pdf", "nine months"),   # 14_03 — dropped by the truncated-title bug
    ("doc_ac04549f3765.pdf", "four months"),
]


def main() -> int:
    from app.rag.parse import parse_any, normalize_extracted_text
    from app.rag.chunking import chunk_document
    from app.rag.chunking.pipeline import _coverage_ratio
    from app.rag.chunking.sanitize import sanitize_text

    pdfs = sorted(f for f in os.listdir(UPLOADS) if f.endswith(".pdf"))
    if not pdfs:
        print(f"no PDFs in {UPLOADS} — nothing to check")
        return 0

    failures: list[str] = []
    print(f"{'document':<26}{'chunks':>7}{'parents':>8}{'coverage':>10}")
    chunks_by_file = {}
    for f in pdfs:
        data = open(os.path.join(UPLOADS, f), "rb").read()
        ftype, text, pages, offsets = parse_any(f, data)
        base = text if offsets else normalize_extracted_text(text)
        clean, _, _ = sanitize_text(base)
        chunks = chunk_document(
            text, "covtest", estimated_pages=pages,
            page_offsets=offsets, already_normalized=bool(offsets),
        )
        chunks_by_file[f] = chunks
        cov = _coverage_ratio(clean, chunks)
        parents = sum(1 for c in chunks if getattr(c, "is_parent", False))
        flag = "" if cov >= MIN_COVERAGE else "  <-- BELOW FLOOR"
        print(f"{f[:26]:<26}{len(chunks):>7}{parents:>8}{cov*100:>9.1f}%{flag}")
        if cov < MIN_COVERAGE:
            failures.append(f"{f}: coverage {cov*100:.1f}% < {MIN_COVERAGE*100:.0f}%")

    for fname, needle in CANARIES:
        chunks = chunks_by_file.get(fname)
        if not chunks:
            continue
        if not any(needle.lower() in (c.text or "").lower() for c in chunks):
            failures.append(f"{fname}: canary phrase {needle!r} missing from all chunks")

    print()
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS — all {len(pdfs)} documents >= {MIN_COVERAGE*100:.0f}% coverage, canaries intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
