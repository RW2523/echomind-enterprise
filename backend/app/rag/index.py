from __future__ import annotations
import os, json
import numpy as np
import faiss
from typing import Dict, List
from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .embeddings import OllamaEmbeddings
from .sparse import Bm25Index
from .chunking import chunk_document
from .structure import extract_headings_from_chunks, store_doc_headings

class FaissIndex:
    def __init__(self):
        self.emb = OllamaEmbeddings()
        self.index = None
        self.meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.sparse = Bm25Index()
        self._load()

    def _load(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(settings.FAISS_PATH) and os.path.exists(settings.META_PATH):
            self.index = faiss.read_index(settings.FAISS_PATH)
            with open(settings.META_PATH,"r",encoding="utf-8") as f:
                self.meta = json.load(f)
            if self.meta.get("chunk_ids") and not self.sparse.chunk_ids:
                self.sparse.rebuild_from_chunk_ids(self.meta["chunk_ids"])

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, settings.FAISS_PATH)
        with open(settings.META_PATH,"w",encoding="utf-8") as f:
            json.dump(self.meta,f)

    async def _ensure_index(self, dim:int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    async def add_document(self, filename: str, filetype: str, text: str, meta: dict) -> dict:
        doc_id = new_id("doc")
        all_chunks = chunk_document(text or "", doc_id)
        if not all_chunks:
            raise ValueError("No text extracted")
        embed_chunks = [c for c in all_chunks if not c.is_parent]
        texts_to_embed = [c.text for c in embed_chunks]
        vecs = await self.emb.embed(texts_to_embed)
        faiss.normalize_L2(vecs)
        await self._ensure_index(int(vecs.shape[1]))

        with get_conn() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, filetype, created_at, meta_json) VALUES (?,?,?,?,?)",
                (doc_id, filename, filetype, now_iso(), json.dumps(meta)),
            )
            for c in all_chunks:
                src = c.to_source_dict(filename, filetype)
                conn.execute(
                    "INSERT INTO chunks (id, doc_id, chunk_index, text, source_json) VALUES (?,?,?,?,?)",
                    (c.chunk_id, doc_id, c.chunk_index, c.text, json.dumps(src)),
                )
            conn.commit()
            # Structure index: extract headings from all chunks for TOC/structure fallback
            chunks_for_structure = [(c.chunk_id, c.chunk_index, c.text) for c in all_chunks]
            heading_rows = extract_headings_from_chunks(doc_id, chunks_for_structure)
            if heading_rows:
                store_doc_headings(doc_id, heading_rows)

        for c in embed_chunks:
            self.meta["chunk_ids"].append(c.chunk_id)
            self.meta["source_by_chunk"][c.chunk_id] = c.to_source_dict(filename, filetype)
        self.index.add(vecs.astype(np.float32))
        self._save()
        self.sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)
        return {"doc_id": doc_id, "chunks": len(embed_chunks)}

    async def add_text(self, title:str, text:str, meta:dict) -> dict:
        return await self.add_document(title, "text", text, meta)

    async def delete_document(self, doc_id: str) -> None:
        """Remove document and its chunks from DB, FAISS, and sparse index. Rebuilds both indexes from remaining chunks."""
        with get_conn() as conn:
            conn.execute("DELETE FROM doc_headings WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.commit()
            rows = conn.execute("SELECT id, text, source_json FROM chunks ORDER BY doc_id, chunk_index").fetchall()
        remaining_ids = []
        remaining_texts = []
        source_by_chunk = {}
        for r in rows:
            src = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            if src.get("is_parent"):
                continue
            remaining_ids.append(r[0])
            remaining_texts.append(r[1])
            source_by_chunk[r[0]] = src
        if not remaining_ids:
            self.meta = {"chunk_ids": [], "source_by_chunk": {}}
            self.index = None
            self._save()
            if os.path.exists(settings.FAISS_PATH):
                os.remove(settings.FAISS_PATH)
            self.sparse.chunk_ids = []
            self.sparse.corpus_tokens = []
            self.sparse._bm25 = None
            self.sparse._save()
            return
        vecs = await self.emb.embed(remaining_texts)
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs.astype(np.float32))
        self.meta["chunk_ids"] = remaining_ids
        self.meta["source_by_chunk"] = source_by_chunk
        self._save()
        self.sparse.rebuild_from_chunk_ids(remaining_ids)

    async def search(self, query:str, k:int) -> List[Dict]:
        if self.index is None or self.index.ntotal==0:
            return []
        qv = await self.emb.embed([query])
        faiss.normalize_L2(qv)
        D,I = self.index.search(qv.astype(np.float32), k)
        out=[]
        chunk_ids=self.meta["chunk_ids"]
        with get_conn() as conn:
            for rank, idx in enumerate(I[0].tolist()):
                if idx<0 or idx>=len(chunk_ids): 
                    continue
                cid=chunk_ids[idx]
                row=conn.execute("SELECT text, source_json FROM chunks WHERE id=?", (cid,)).fetchone()
                if not row: continue
                text, src_json = row
                out.append({"chunk_id":cid,"score":float(D[0][rank]),"text":text,"source":json.loads(src_json)})
        return out

index = FaissIndex()
