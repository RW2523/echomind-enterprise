#!/usr/bin/env python3
"""Remove stale PDF document rows left behind by a failed re-ingest.

Identifies superseded documents: filename ends in .pdf, and the raw upload file
/data/uploads/<doc_id>.pdf no longer exists (the re-ingest writes the file under the
NEW doc_id and removes the old one). Those rows are the pre-re-ingest duplicates.

Safety: dry-run by default; never touches transcripts, markdown or non-PDF docs;
refuses to delete a document whose filename has no surviving replacement.

  python3 /app/scripts/dedupe_stale_pdfs.py --dry-run
  python3 /app/scripts/dedupe_stale_pdfs.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys

sys.path.insert(0, "/app")

DB = os.getenv("ECHOMIND_DB", "/data/echomind.sqlite")
UPLOAD_DIR = os.getenv("ECHOMIND_UPLOAD_DIR", "/data/uploads")


def find_stale() -> list[tuple[str, str, int]]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, filename FROM documents "
        "WHERE LOWER(filename) LIKE '%.pdf' AND filename NOT LIKE 'transcript_%'"
    ).fetchall()
    by_name: dict[str, list[tuple[str, bool]]] = {}
    for doc_id, filename in rows:
        has_file = os.path.exists(os.path.join(UPLOAD_DIR, f"{doc_id}.pdf"))
        by_name.setdefault(filename, []).append((doc_id, has_file))

    stale: list[tuple[str, str, int]] = []
    for filename, entries in by_name.items():
        survivors = [d for d, ok in entries if ok]
        missing = [d for d, ok in entries if not ok]
        if not missing:
            continue
        if not survivors:
            print(f"  SKIP {filename}: no surviving copy with an upload file — keeping all")
            continue
        for doc_id in missing:
            n = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]
            stale.append((doc_id, filename, n))
    conn.close()
    return stale


async def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)

    stale = find_stale()
    print(f"{'APPLY' if apply else 'DRY-RUN'}: {len(stale)} stale PDF document(s)\n")
    total = 0
    for doc_id, filename, n in stale:
        print(f"  {'DELETING' if apply else 'would delete'} {filename[:34]:<36} {doc_id[:14]} ({n} chunks)")
        total += n
        if apply:
            from app.rag.index import index
            await index.delete_document(doc_id)
    print(f"\n{'Removed' if apply else 'Would remove'} {total} chunks")
    if not apply:
        print("(no changes written — re-run with --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
