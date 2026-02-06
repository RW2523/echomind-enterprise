from __future__ import annotations
import os, json
import numpy as np
import faiss
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .embeddings import OllamaEmbeddings
from .sparse import Bm25Index

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

    async def add_document(self, filename:str, filetype:str, text:str, meta:dict) -> dict:
        splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks=[c.strip() for c in splitter.split_text(text or "") if c.strip()]
        if not chunks: raise ValueError("No text extracted")
        vecs = await self.emb.embed(chunks)
        faiss.normalize_L2(vecs)
        await self._ensure_index(int(vecs.shape[1]))

        doc_id = new_id("doc")
        with get_conn() as conn:
            conn.execute("INSERT INTO documents (id, filename, filetype, created_at, meta_json) VALUES (?,?,?,?,?)",
                         (doc_id, filename, filetype, now_iso(), json.dumps(meta)))
            for i, chunk_text in enumerate(chunks):
                chunk_id = new_id("chk")
                src={"doc_id":doc_id,"filename":filename,"chunk_index":i,"filetype":filetype}
                conn.execute("INSERT INTO chunks (id, doc_id, chunk_index, text, source_json) VALUES (?,?,?,?,?)",
                             (chunk_id, doc_id, i, chunk_text, json.dumps(src)))
                self.meta["chunk_ids"].append(chunk_id)
                self.meta["source_by_chunk"][chunk_id]=src
            conn.commit()

        self.index.add(vecs.astype(np.float32))
        self._save()
        new_ids = self.meta["chunk_ids"][-len(chunks):]
        self.sparse.add_chunks(new_ids, chunks)
        return {"doc_id": doc_id, "chunks": len(chunks)}

    async def add_text(self, title:str, text:str, meta:dict) -> dict:
        return await self.add_document(title, "text", text, meta)

    async def delete_document(self, doc_id: str) -> None:
        """Remove document and its chunks from DB, FAISS, and sparse index. Rebuilds both indexes from remaining chunks."""
        with get_conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.commit()
            rows = conn.execute("SELECT id, text, source_json FROM chunks ORDER BY doc_id, chunk_index").fetchall()
        remaining_ids = [r[0] for r in rows]
        remaining_texts = [r[1] for r in rows]
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
        self.meta["source_by_chunk"] = {r[0]: json.loads(r[2]) for r in rows}
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
