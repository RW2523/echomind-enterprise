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

def _is_transcript_doc(filename: str, meta: dict) -> bool:
    """True if this document is a transcript (stored via add_text with transcript_ prefix or type)."""
    if (filename or "").startswith("transcript_"):
        return True
    return (meta or {}).get("type") == "transcript"


class FaissIndex:
    def __init__(self):
        self.emb = OllamaEmbeddings()
        self.index = None
        self.meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.sparse = Bm25Index()
        # Transcript-only index: used when intent=transcript so retrieval runs only over transcripts.
        self.transcript_index = None
        self.transcript_meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.transcript_sparse = Bm25Index(settings.SPARSE_TRANSCRIPT_META_PATH)
        self._load()

    def _load(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(settings.FAISS_PATH) and os.path.exists(settings.META_PATH):
            self.index = faiss.read_index(settings.FAISS_PATH)
            with open(settings.META_PATH,"r",encoding="utf-8") as f:
                self.meta = json.load(f)
            if self.meta.get("chunk_ids") and not self.sparse.chunk_ids:
                self.sparse.rebuild_from_chunk_ids(self.meta["chunk_ids"])
        self._load_transcript()

    def _load_transcript(self):
        if os.path.exists(settings.FAISS_TRANSCRIPT_PATH) and os.path.exists(settings.META_TRANSCRIPT_PATH):
            self.transcript_index = faiss.read_index(settings.FAISS_TRANSCRIPT_PATH)
            with open(settings.META_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
                self.transcript_meta = json.load(f)
            if self.transcript_meta.get("chunk_ids") and not self.transcript_sparse.chunk_ids:
                self.transcript_sparse.rebuild_from_chunk_ids(self.transcript_meta["chunk_ids"])

    def _save_transcript(self):
        if self.transcript_index is not None:
            faiss.write_index(self.transcript_index, settings.FAISS_TRANSCRIPT_PATH)
        with open(settings.META_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            json.dump(self.transcript_meta, f)

    async def _rebuild_transcript_index(self) -> None:
        """Rebuild transcript-only index from DB (chunks whose document has filename LIKE 'transcript_%')."""
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT c.id, c.text, c.source_json FROM chunks c
                   INNER JOIN documents d ON c.doc_id = d.id
                   WHERE d.filename LIKE 'transcript_%' ORDER BY c.doc_id, c.chunk_index"""
            ).fetchall()
        transcript_ids = []
        transcript_texts = []
        source_by_chunk = {}
        for r in rows:
            src = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            if src.get("is_parent"):
                continue
            transcript_ids.append(r[0])
            transcript_texts.append(r[1])
            source_by_chunk[r[0]] = src
        self.transcript_meta = {"chunk_ids": transcript_ids, "source_by_chunk": source_by_chunk}
        if not transcript_ids:
            self.transcript_index = None
            if os.path.exists(settings.FAISS_TRANSCRIPT_PATH):
                os.remove(settings.FAISS_TRANSCRIPT_PATH)
            self.transcript_sparse.chunk_ids = []
            self.transcript_sparse.corpus_tokens = []
            self.transcript_sparse._bm25 = None
            self.transcript_sparse._save()
            self._save_transcript()
            return
        vecs = await self.emb.embed(transcript_texts)
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.transcript_index = faiss.IndexFlatIP(dim)
        self.transcript_index.add(vecs.astype(np.float32))
        self.transcript_sparse.rebuild_from_chunk_ids(transcript_ids)
        self._save_transcript()

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, settings.FAISS_PATH)
        with open(settings.META_PATH,"w",encoding="utf-8") as f:
            json.dump(self.meta,f)
        self._save_transcript()

    async def _ensure_index(self, dim:int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    async def _ensure_transcript_index(self, dim: int):
        if self.transcript_index is None:
            self.transcript_index = faiss.IndexFlatIP(dim)

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

        for c in embed_chunks:
            self.meta["chunk_ids"].append(c.chunk_id)
            self.meta["source_by_chunk"][c.chunk_id] = c.to_source_dict(filename, filetype)
        self.index.add(vecs.astype(np.float32))
        if _is_transcript_doc(filename, meta):
            await self._ensure_transcript_index(int(vecs.shape[1]))
            for c in embed_chunks:
                self.transcript_meta["chunk_ids"].append(c.chunk_id)
                self.transcript_meta["source_by_chunk"][c.chunk_id] = c.to_source_dict(filename, filetype)
            self.transcript_index.add(vecs.astype(np.float32))
            self.transcript_sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)
        self._save()
        self.sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)
        return {"doc_id": doc_id, "chunks": len(embed_chunks)}

    async def add_text(self, title:str, text:str, meta:dict) -> dict:
        return await self.add_document(title, "text", text, meta)

    async def delete_document(self, doc_id: str) -> None:
        """Remove document and its chunks from DB, FAISS, and sparse index. Rebuilds both indexes from remaining chunks."""
        with get_conn() as conn:
            row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
            was_transcript = row and (row[0] or "").startswith("transcript_")
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
            await self._rebuild_transcript_index()
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
        if was_transcript:
            await self._rebuild_transcript_index()

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

    async def search_transcript_only(self, query: str, k: int) -> List[Dict]:
        """Search only over the transcript-only index. Returns same shape as search(). Empty if no transcripts."""
        if self.transcript_index is None:
            await self._rebuild_transcript_index()
        if self.transcript_index is None or self.transcript_index.ntotal == 0:
            return []
        qv = await self.emb.embed([query])
        faiss.normalize_L2(qv)
        D, I = self.transcript_index.search(qv.astype(np.float32), k)
        out = []
        chunk_ids = self.transcript_meta["chunk_ids"]
        with get_conn() as conn:
            for rank, idx in enumerate(I[0].tolist()):
                if idx < 0 or idx >= len(chunk_ids):
                    continue
                cid = chunk_ids[idx]
                row = conn.execute("SELECT text, source_json FROM chunks WHERE id=?", (cid,)).fetchone()
                if not row:
                    continue
                text, src_json = row
                out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": json.loads(src_json)})
        return out

index = FaissIndex()
