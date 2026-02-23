from __future__ import annotations
import logging
import os
import json
import numpy as np
import faiss
from typing import Dict, List

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .embeddings import OllamaEmbeddings
from .sparse import Bm25Index
from .chunking import chunk_document

logger = logging.getLogger(__name__)

def _is_transcript_doc(filename: str, meta: dict) -> bool:
    """True if this document is a transcript (stored via add_text with transcript_ prefix or type)."""
    if (filename or "").startswith("transcript_"):
        return True
    return (meta or {}).get("type") == "transcript"


def _faiss_index_type_name(index) -> str:
    """Return 'flat' or 'hnsw' from a loaded or created FAISS index."""
    if index is None:
        return "flat"
    name = type(index).__name__
    return "hnsw" if "HNSW" in name else "flat"


def _current_embed_stamp(dim: int | None = None) -> dict:
    """Return current embedding config stamp for index compatibility checks. Pass dim from index.d when available."""
    return {
        "embed_model": getattr(settings, "OLLAMA_EMBED_MODEL", ""),
        "embed_endpoint": getattr(settings, "OLLAMA_EMBED_URL", ""),
        "embed_dim": dim,
        "normalized": getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True),
    }


# GPU: faiss-cpu has no index_cpu_to_gpu / StandardGpuResources; faiss-gpu does. Detect once and reuse.
_faiss_gpu_init_done = False
_faiss_gpu_capable = False
_faiss_gpu_resources = None
_faiss_index_cpu_to_gpu = None


def _init_faiss_gpu() -> bool:
    """Detect if FAISS GPU is available (faiss-gpu + CUDA). Safe to call repeatedly; only initializes once."""
    global _faiss_gpu_init_done, _faiss_gpu_capable, _faiss_gpu_resources, _faiss_index_cpu_to_gpu
    if _faiss_gpu_init_done:
        return _faiss_gpu_capable
    _faiss_gpu_init_done = True
    try:
        _faiss_gpu_resources = faiss.StandardGpuResources()
        _faiss_index_cpu_to_gpu = faiss.index_cpu_to_gpu
        ngpus = faiss.get_num_gpus()
        _faiss_gpu_capable = ngpus > 0
        if not _faiss_gpu_capable:
            logger.info("FAISS GPU: not used (get_num_gpus()=0). Using CPU index for search.")
        return _faiss_gpu_capable
    except (AttributeError, ImportError, Exception) as e:
        logger.info("FAISS GPU: GPU FAISS unavailable (%s), using CPU", e)
        _faiss_gpu_resources = None
        _faiss_index_cpu_to_gpu = None
        _faiss_gpu_capable = False
        return False


def _create_faiss_index(dim: int):
    """Create a new FAISS index for cosine similarity (normalized L2 + inner product). Uses FAISS_INDEX_TYPE and HNSW settings."""
    index_type = (getattr(settings, "FAISS_INDEX_TYPE", "flat") or "flat").lower().strip()
    if index_type == "hnsw":
        M = max(4, min(64, getattr(settings, "FAISS_HNSW_M", 32)))
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = max(4, getattr(settings, "FAISS_HNSW_EF_CONSTRUCTION", 200))
        index.hnsw.efSearch = max(4, getattr(settings, "FAISS_HNSW_EF_SEARCH", 64))
        logger.info(
            "FAISS index created: type=hnsw dim=%d M=%d efConstruction=%d efSearch=%d",
            dim,
            M,
            index.hnsw.efConstruction,
            index.hnsw.efSearch,
        )
        return index
    index = faiss.IndexFlatIP(dim)
    logger.info("FAISS index created: type=flat dim=%d", dim)
    return index


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
        # GPU clones for search when FAISS_USE_GPU and faiss-gpu available. CPU index remains source of truth for persistence.
        self.gpu_index = None
        self.gpu_transcript_index = None
        self._load()

    def _load(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(settings.FAISS_PATH) and os.path.exists(settings.META_PATH):
            self.index = faiss.read_index(settings.FAISS_PATH)
            loaded_type = _faiss_index_type_name(self.index)
            expected_type = (getattr(settings, "FAISS_INDEX_TYPE", "flat") or "flat").lower().strip()
            if expected_type != loaded_type:
                logger.warning(
                    "FAISS main index type mismatch: loaded=%s configured=%s; using loaded index unchanged. Change FAISS_INDEX_TYPE or clear index to recreate.",
                    loaded_type,
                    expected_type,
                )
            else:
                logger.info("FAISS main index loaded: type=%s ntotal=%d", loaded_type, self.index.ntotal)
            with open(settings.META_PATH,"r",encoding="utf-8") as f:
                self.meta = json.load(f)
            # Index compatibility: warn if index was built with different embeddings.
            stored = self.meta.get("_embed_stamp")
            if stored and self.index is not None:
                current = _current_embed_stamp(dim=int(self.index.d))
                diff = [k for k in current if current.get(k) != stored.get(k)]
                if diff:
                    logger.warning(
                        "Index built with different embeddings. You must rebuild. Stored=%s current=%s diff=%s",
                        stored,
                        current,
                        diff,
                    )
            if self.meta.get("chunk_ids") and not self.sparse.chunk_ids:
                self.sparse.rebuild_from_chunk_ids(self.meta["chunk_ids"])
        self._load_transcript()

    def _load_transcript(self):
        if os.path.exists(settings.FAISS_TRANSCRIPT_PATH) and os.path.exists(settings.META_TRANSCRIPT_PATH):
            self.transcript_index = faiss.read_index(settings.FAISS_TRANSCRIPT_PATH)
            loaded_type = _faiss_index_type_name(self.transcript_index)
            expected_type = (getattr(settings, "FAISS_INDEX_TYPE", "flat") or "flat").lower().strip()
            if expected_type != loaded_type:
                logger.warning(
                    "FAISS transcript index type mismatch: loaded=%s configured=%s; using loaded index unchanged.",
                    loaded_type,
                    expected_type,
                )
            else:
                logger.info("FAISS transcript index loaded: type=%s ntotal=%d", loaded_type, self.transcript_index.ntotal)
            with open(settings.META_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
                self.transcript_meta = json.load(f)
            stored = self.transcript_meta.get("_embed_stamp")
            if stored and self.transcript_index is not None:
                current = _current_embed_stamp(dim=int(self.transcript_index.d))
                diff = [k for k in current if current.get(k) != stored.get(k)]
                if diff:
                    logger.warning(
                        "Transcript index built with different embeddings. You must rebuild. Stored=%s current=%s diff=%s",
                        stored,
                        current,
                        diff,
                    )
            if self.transcript_meta.get("chunk_ids") and not self.transcript_sparse.chunk_ids:
                self.transcript_sparse.rebuild_from_chunk_ids(self.transcript_meta["chunk_ids"])
        self._sync_gpu_indexes()

    def _sync_gpu_indexes(self) -> None:
        """Build or refresh GPU clones from CPU indexes for search. No-op if FAISS_USE_GPU is False or faiss-gpu unavailable."""
        use_gpu = getattr(settings, "FAISS_USE_GPU", True)
        if not use_gpu:
            self.gpu_index = None
            self.gpu_transcript_index = None
            logger.info("FAISS running on CPU (FAISS_USE_GPU disabled)")
            return
        if not _init_faiss_gpu() or _faiss_gpu_resources is None or _faiss_index_cpu_to_gpu is None:
            self.gpu_index = None
            self.gpu_transcript_index = None
            logger.info("FAISS running on CPU (GPU FAISS unavailable)")
            return
        device_id = getattr(settings, "FAISS_GPU_DEVICE", 0)
        ngpus = faiss.get_num_gpus()
        if ngpus <= 0:
            self.gpu_index = None
            self.gpu_transcript_index = None
            logger.info("FAISS running on CPU (no GPUs)")
            return
        device_id = max(0, min(device_id, ngpus - 1))
        try:
            if self.index is not None and self.index.ntotal > 0:
                self.gpu_index = _faiss_index_cpu_to_gpu(_faiss_gpu_resources, device_id, self.index)
                logger.info("FAISS GPU: main index cloned to device %d (ntotal=%d). Search will use GPU.", device_id, self.index.ntotal)
            else:
                self.gpu_index = None
        except (RuntimeError, Exception) as e:
            logger.warning("FAISS GPU: failed to clone main index to device %d (%s). Search will use CPU.", device_id, e)
            self.gpu_index = None
        try:
            if self.transcript_index is not None and self.transcript_index.ntotal > 0:
                self.gpu_transcript_index = _faiss_index_cpu_to_gpu(_faiss_gpu_resources, device_id, self.transcript_index)
                logger.info("FAISS GPU: transcript index cloned to device %d (ntotal=%d).", device_id, self.transcript_index.ntotal)
            else:
                self.gpu_transcript_index = None
        except (RuntimeError, Exception) as e:
            logger.warning("FAISS GPU: failed to clone transcript index to device %d (%s). Search will use CPU.", device_id, e)
            self.gpu_transcript_index = None
        if self.gpu_index is not None:
            logger.info("FAISS running on GPU")
        else:
            logger.info("FAISS running on CPU")

    def _save_transcript(self):
        if self.transcript_index is not None:
            faiss.write_index(self.transcript_index, settings.FAISS_TRANSCRIPT_PATH)
            self.transcript_meta["_embed_stamp"] = _current_embed_stamp(dim=int(self.transcript_index.d))
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
        if not getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True):
            faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.transcript_index = _create_faiss_index(dim)
        self.transcript_index.add(vecs.astype(np.float32))
        self.transcript_sparse.rebuild_from_chunk_ids(transcript_ids)
        self._save_transcript()

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, settings.FAISS_PATH)
            self.meta["_embed_stamp"] = _current_embed_stamp(dim=int(self.index.d))
        with open(settings.META_PATH,"w",encoding="utf-8") as f:
            json.dump(self.meta,f)
        self._save_transcript()

    async def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = _create_faiss_index(dim)

    async def _ensure_transcript_index(self, dim: int):
        if self.transcript_index is None:
            self.transcript_index = _create_faiss_index(dim)

    async def add_document(self, filename: str, filetype: str, text: str, meta: dict) -> dict:
        doc_id = new_id("doc")
        all_chunks = chunk_document(text or "", doc_id)
        if not all_chunks:
            raise ValueError("No text extracted")
        embed_chunks = [c for c in all_chunks if not c.is_parent]
        texts_to_embed = [c.text for c in embed_chunks]
        vecs = await self.emb.embed(texts_to_embed)
        if not getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True):
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
        self._sync_gpu_indexes()
        self._save()
        self.sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)
        return {"doc_id": doc_id, "chunks": len(embed_chunks)}

    async def add_text(self, title:str, text:str, meta:dict) -> dict:
        return await self.add_document(title, "text", text, meta)

    def clear_all(self) -> None:
        """Clear all indexes and persisted files in one shot (no re-embedding). Call after DB tables are cleared."""
        self.gpu_index = None
        self.gpu_transcript_index = None
        self.index = None
        self.meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.sparse.chunk_ids = []
        self.sparse.corpus_tokens = []
        self.sparse._bm25 = None
        self.sparse._save()
        for path in (settings.FAISS_PATH, settings.META_PATH, settings.SPARSE_META_PATH):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        self.transcript_index = None
        self.transcript_meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.transcript_sparse.chunk_ids = []
        self.transcript_sparse.corpus_tokens = []
        self.transcript_sparse._bm25 = None
        self.transcript_sparse._save()
        for path in (settings.FAISS_TRANSCRIPT_PATH, settings.META_TRANSCRIPT_PATH, settings.SPARSE_TRANSCRIPT_META_PATH):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

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
        if not getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True):
            faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.index = _create_faiss_index(dim)
        self.index.add(vecs.astype(np.float32))
        self.meta["chunk_ids"] = remaining_ids
        self.meta["source_by_chunk"] = source_by_chunk
        self._save()
        self.sparse.rebuild_from_chunk_ids(remaining_ids)
        if was_transcript:
            await self._rebuild_transcript_index()
        self._sync_gpu_indexes()

    async def search(self, query:str, k:int) -> List[Dict]:
        if self.index is None or self.index.ntotal==0:
            return []
        # Single-query embed: uses OllamaEmbeddings query cache (LRU) when EMBED_QUERY_CACHE_SIZE > 0.
        qv = await self.emb.embed([query])
        if not getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True):
            faiss.normalize_L2(qv)
        search_index = self.gpu_index if self.gpu_index is not None else self.index
        D, I = search_index.search(qv.astype(np.float32), k)
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
        # Single-query embed: uses OllamaEmbeddings query cache (LRU) when EMBED_QUERY_CACHE_SIZE > 0.
        qv = await self.emb.embed([query])
        if not getattr(settings, "EMBEDDINGS_ALREADY_NORMALIZED", True):
            faiss.normalize_L2(qv)
        search_index = self.gpu_transcript_index if self.gpu_transcript_index is not None else self.transcript_index
        D, I = search_index.search(qv.astype(np.float32), k)
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
