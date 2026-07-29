#!/usr/bin/env python3
"""Re-ingest PDF documents in place, preserving doc_id, filename and namespace.

Why: chunking/parse fixes (BookRAG detection, section salvage, FAQ split) only affect
NEW ingests — existing chunks keep whatever the old code produced. This re-chunks and
re-embeds the PDFs already in the KB so they benefit, WITHOUT touching transcripts,
chats, users or the vertical markdown KBs.

Safety:
  - Only documents whose filename ends in .pdf AND whose source file still exists.
  - Transcript documents (transcript_*) are never selected.
  - --dry-run (default) prints the plan and changes nothing.
  - Each document is deleted+re-added individually; a failure on one leaves the rest intact.

Usage (inside the backend container):
  python3 /app/scripts/reingest_pdfs.py --dry-run
  python3 /app/scripts/reingest_pdfs.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys

sys.path.insert(0, "/app")

DB = os.getenv("ECHOMIND_DB", "/data/echomind.sqlite")
UPLOAD_DIR = os.getenv("ECHOMIND_UPLOAD_DIR", "/data/uploads")


def select_targets() -> list[dict]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, filename, filetype, meta_json FROM documents "
        "WHERE LOWER(filename) LIKE '%.pdf' AND filename NOT LIKE 'transcript_%'"
    ).fetchall()
    out = []
    for doc_id, filename, filetype, meta_json in rows:
        path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        if not os.path.exists(path):
            print(f"  SKIP {filename}: source file missing ({path})")
            continue
        meta = json.loads(meta_json or "{}")
        n = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]
        out.append({
            "doc_id": doc_id, "filename": filename, "filetype": filetype,
            "namespace": meta.get("namespace") or "default", "meta": meta,
            "path": path, "old_chunks": n,
        })
    conn.close()
    return out


async def reingest(targets: list[dict], apply: bool) -> None:
    from app.rag.index import index, set_active_namespace
    from app.rag.parse import parse_any

    total_old = total_new = 0
    for t in targets:
        data = open(t["path"], "rb").read()
        ft, text, pages, offsets = parse_any(t["filename"], data)
        if not (text or "").strip():
            print(f"  SKIP {t['filename']}: parsed empty")
            continue

        if not apply:
            from app.rag.chunking import chunk_document
            ch = chunk_document(text, "dry", estimated_pages=pages,
                                page_offsets=offsets, already_normalized=bool(offsets))
            kids = [c for c in ch if not getattr(c, "is_parent", False)]
            pars = [c for c in ch if getattr(c, "is_parent", False)]
            sp = sum(1 for c in ch if getattr(c, "section_path", None))
            cov = 100 * sum(len(c.text) for c in kids) / max(len(text), 1)
            print(f"  {t['filename'][:30]:<32} ns={t['namespace']:<9} "
                  f"{t['old_chunks']:>3} -> {len(kids)} chunks (+{len(pars)} parents, "
                  f"{sp} with section_path, {cov:.0f}% coverage)")
            total_old += t["old_chunks"]; total_new += len(kids)
            continue

        # add_document mints a fresh doc_id, so mirror what /docs/upload does:
        # add first, re-persist the raw file under the NEW id (for in-browser preview),
        # then delete the old row + its file. Add-before-delete means a crash mid-way
        # leaves a duplicate (recoverable) rather than a missing document.
        set_active_namespace(t["namespace"] if t["namespace"] != "default" else None)
        meta = dict(t["meta"]) or {}
        meta.setdefault("filename", t["filename"])
        meta.setdefault("filetype", ft)
        res = await index.add_document(
            t["filename"], ft, text, meta,
            estimated_pages=pages, page_offsets=offsets, namespace=t["namespace"],
        )
        new_doc_id = res.get("doc_id")
        if not new_doc_id:
            print(f"  FAILED {t['filename']}: add_document returned {res}")
            continue
        ext = os.path.splitext(t["path"])[1] or ".pdf"
        try:
            with open(os.path.join(UPLOAD_DIR, f"{new_doc_id}{ext}"), "wb") as fh:
                fh.write(data)
        except Exception as exc:
            print(f"  WARN {t['filename']}: could not persist preview file: {exc}")
        index.delete_document(t["doc_id"])
        try:
            os.remove(t["path"])
        except OSError:
            pass

        conn = sqlite3.connect(DB)
        n = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (new_doc_id,)).fetchone()[0]
        conn.close()
        print(f"  RE-INGESTED {t['filename'][:30]:<32} {t['old_chunks']:>3} -> {n} chunks "
              f"(doc_id {t['doc_id'][:12]} -> {new_doc_id[:12]})")
        total_old += t["old_chunks"]; total_new += n

    print(f"\nTOTAL chunks: {total_old} -> {total_new}")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)

    targets = select_targets()
    print(f"{'APPLY' if apply else 'DRY-RUN'}: {len(targets)} PDF document(s) selected "
          f"(transcripts/markdown/chats untouched)\n")
    if not targets:
        return 0
    asyncio.run(reingest(targets, apply))
    if not apply:
        print("\n(no changes written — re-run with --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
