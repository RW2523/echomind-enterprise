#!/usr/bin/env python3
"""Rebuild the FAISS + BM25 indexes from scratch by re-embedding every chunk in the DB.

Why this exists: delete_document rebuilds survivors by RECONSTRUCTING vectors from the
existing index (an O(corpus) optimization). If index/meta ever desync, that shortcut
propagates the misalignment instead of correcting it — chunk_ids and vectors drift apart
and retrieval starts returning the wrong chunk for a query. Symptom: a chunk no longer
retrieves ITSELF when you search its own text.

This script is the ground-truth repair: it ignores the existing vectors entirely and
re-embeds from the chunks table, so chunk_id <-> vector alignment is correct by
construction. Also rebuilds the transcript index and BM25.

  python3 /app/scripts/rebuild_index.py --check    # verify alignment only
  python3 /app/scripts/rebuild_index.py --apply    # full rebuild
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

sys.path.insert(0, "/app")

import faiss  # noqa: E402
import numpy as np  # noqa: E402


async def check() -> int:
    """Self-retrieval test: does each sampled chunk retrieve itself?"""
    from app.core.db import get_conn
    from app.rag.index import index, set_active_namespace

    set_active_namespace(None)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ch.id, ch.text FROM chunks ch JOIN documents d ON ch.doc_id = d.id "
            "WHERE d.filename NOT LIKE 'transcript_%' AND LENGTH(ch.text) > 400 LIMIT 12"
        ).fetchall()
    ok = bad = 0
    for cid, text in rows:
        hits = await index.search_document_only(text[:1500], 3)
        top = hits[0]["chunk_id"] if hits else None
        if top == cid:
            ok += 1
        else:
            bad += 1
    print(f"self-retrieval: {ok} aligned / {bad} MISALIGNED (of {len(rows)} sampled)")
    return 0 if bad == 0 else 1


async def rebuild() -> int:
    from app.core.db import get_conn
    from app.rag.index import index, IVF_THRESHOLD, _build_ivf_index

    t0 = time.time()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, COALESCE(contextualized_text, text), source_json FROM chunks "
            "ORDER BY doc_id, chunk_index"
        ).fetchall()

    ids, texts, source_by_chunk = [], [], {}
    for cid, text, src_json in rows:
        src = json.loads(src_json) if isinstance(src_json, str) else (src_json or {})
        if src.get("is_parent"):
            continue  # parents are context-only, never embedded
        ids.append(cid)
        texts.append(text or "")
        source_by_chunk[cid] = src

    print(f"re-embedding {len(ids)} chunks (parents excluded) ...")
    vecs_list = []
    BATCH = 256
    for i in range(0, len(texts), BATCH):
        part = await index.emb.embed(texts[i : i + BATCH])
        vecs_list.append(np.asarray(part, dtype=np.float32))
        done = min(i + BATCH, len(texts))
        print(f"  {done}/{len(texts)}  ({time.time()-t0:.0f}s)", flush=True)
    vecs = np.vstack(vecs_list).astype(np.float32)
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index.index = (
        _build_ivf_index(vecs, dim) if vecs.shape[0] >= IVF_THRESHOLD else faiss.IndexFlatIP(dim)
    )
    if vecs.shape[0] < IVF_THRESHOLD:
        index.index.add(vecs)
    index.meta["chunk_ids"] = ids
    index.meta["source_by_chunk"] = source_by_chunk
    index._save()
    index.sparse.rebuild_from_chunk_ids(ids)
    await index._rebuild_transcript_index()

    print(f"rebuilt: {index.index.ntotal} vectors, {len(ids)} ids, BM25 {len(index.sparse.chunk_ids)} "
          f"in {time.time()-t0:.0f}s")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.apply:
        rc = await rebuild()
        print("\nverifying ...")
        return await check() or rc
    return await check()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
