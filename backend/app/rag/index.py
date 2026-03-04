from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .embeddings import OllamaEmbeddings
from .sparse import Bm25Index
from .chunking import chunk_document
from .section_index import SectionIndex
from .glossary_index import GlossaryIndex, is_glossary_section
from .cross_ref_graph import CrossRefGraph, extract_references

logger = logging.getLogger(__name__)


def _is_transcript_doc(filename: str, meta: dict) -> bool:
    """True if this document is a transcript (stored via add_text with transcript_ prefix or type)."""
    if (filename or "").startswith("transcript_"):
        return True
    return (meta or {}).get("type") == "transcript"


def _build_book_sections_from_chunks(chunks, doc_id: str) -> List[Dict]:
    """Reconstruct section-level entries from parent chunks (grouped by section_path).

    Returns list of {section_id, doc_id, section_title, section_path, full_section_text}.
    """
    sections_by_path: Dict[str, Dict] = {}
    for c in chunks:
        if not c.is_parent:
            continue
        sp = (c.section_path or "").strip() or "__root__"
        if sp not in sections_by_path:
            sections_by_path[sp] = {
                "section_id": new_id("sec"),
                "doc_id": doc_id,
                "section_title": c.section or sp,
                "section_path": sp,
                "texts": [],
            }
        sections_by_path[sp]["texts"].append(c.text)

    result = []
    for sp, v in sections_by_path.items():
        combined = " ".join(v["texts"])
        result.append({
            "section_id": v["section_id"],
            "doc_id": doc_id,
            "section_title": v["section_title"],
            "section_path": sp,
            "full_section_text": combined[:6000],
        })
    return result


def _store_book_sections_in_db(sections: List[Dict]) -> None:
    """Persist section metadata to book_sections table."""
    if not sections:
        return
    with get_conn() as conn:
        for s in sections:
            conn.execute(
                "INSERT OR IGNORE INTO book_sections "
                "(section_id, doc_id, section_title, section_path, full_section_text, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    s["section_id"],
                    s["doc_id"],
                    s["section_title"],
                    s["section_path"],
                    s["full_section_text"],
                    now_iso(),
                ),
            )
        conn.commit()


class FaissIndex:
    def __init__(self):
        self.emb = OllamaEmbeddings()
        self.index = None
        self.meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.sparse = Bm25Index()
        # Transcript-only index
        self.transcript_index = None
        self.transcript_meta = {"chunk_ids": [], "source_by_chunk": {}}
        self.transcript_sparse = Bm25Index(settings.SPARSE_TRANSCRIPT_META_PATH)
        # Hierarchical section index
        self.section_index = SectionIndex(
            self.emb, settings.FAISS_SECTION_PATH, settings.SECTION_META_PATH
        )
        # Glossary priority index
        self.glossary_index = GlossaryIndex(
            self.emb, settings.FAISS_GLOSSARY_PATH, settings.GLOSSARY_META_PATH
        )
        # Cross-reference graph
        self.cross_ref_graph = CrossRefGraph(settings.CROSS_REF_GRAPH_PATH)
        self._load()

    def _load(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(settings.FAISS_PATH) and os.path.exists(settings.META_PATH):
            self.index = faiss.read_index(settings.FAISS_PATH)
            with open(settings.META_PATH, "r", encoding="utf-8") as f:
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
        """Rebuild transcript-only index from DB."""
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
        with open(settings.META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)
        self._save_transcript()

    async def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    async def _ensure_transcript_index(self, dim: int):
        if self.transcript_index is None:
            self.transcript_index = faiss.IndexFlatIP(dim)

    async def add_document(
        self,
        filename: str,
        filetype: str,
        text: str,
        meta: dict,
        estimated_pages: int = 0,
        page_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> dict:
        doc_id = new_id("doc")
        all_chunks = chunk_document(
            text or "", doc_id,
            estimated_pages=estimated_pages,
            page_offsets=page_offsets,
        )
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

        # ── Hierarchical section + glossary indexing (BOOK documents only) ────
        has_parents = any(c.is_parent for c in all_chunks)
        if has_parents and not _is_transcript_doc(filename, meta):
            await self._index_book_sections(all_chunks, doc_id)

        return {"doc_id": doc_id, "chunks": len(embed_chunks)}

    async def _index_book_sections(self, all_chunks, doc_id: str) -> None:
        """Build and store section-level embeddings, glossary entries, and cross-references."""
        book_sections = _build_book_sections_from_chunks(all_chunks, doc_id)
        if not book_sections:
            return

        # Store metadata in DB
        _store_book_sections_in_db(book_sections)

        # Add to section FAISS index
        await self.section_index.add_sections(book_sections)
        logger.info("index: indexed %d sections for doc_id=%s", len(book_sections), doc_id)

        # Extract cross-references from section texts
        all_refs = []
        for s in book_sections:
            refs = extract_references(
                s.get("full_section_text") or "",
                s.get("section_path") or "",
                doc_id,
            )
            all_refs.extend(refs)
        if all_refs:
            self.cross_ref_graph.add_section_refs(all_refs)
            self.cross_ref_graph.store_refs_in_db(all_refs)
            logger.info("index: stored %d cross-references for doc_id=%s", len(all_refs), doc_id)

        # Glossary index: collect glossary sections
        glossary_entries = []
        for s in book_sections:
            if is_glossary_section(s.get("section_title")):
                glossary_entries.append({
                    "doc_id": doc_id,
                    "section_path": s.get("section_path"),
                    "text": s.get("full_section_text") or "",
                })
        if glossary_entries:
            await self.glossary_index.add_entries(glossary_entries)
            logger.info("index: added %d glossary entries for doc_id=%s", len(glossary_entries), doc_id)

    async def add_text(self, title: str, text: str, meta: dict) -> dict:
        return await self.add_document(title, "text", text, meta)

    def clear_all(self) -> None:
        """Clear all indexes and persisted files (no re-embedding)."""
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
        # Clear new indexes
        self.section_index.clear_all()
        self.glossary_index.clear_all()
        self.cross_ref_graph.clear_all()
        # Clear DB tables
        with get_conn() as conn:
            conn.execute("DELETE FROM book_sections")
            conn.execute("DELETE FROM section_references")
            conn.commit()

    async def delete_document(self, doc_id: str) -> None:
        """Remove document and its chunks from DB, FAISS, and sparse index. Rebuilds indexes from remaining chunks."""
        with get_conn() as conn:
            row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
            was_transcript = row and (row[0] or "").startswith("transcript_")
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.execute("DELETE FROM book_sections WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM section_references WHERE doc_id=?", (doc_id,))
            conn.commit()
            rows = conn.execute("SELECT id, text, source_json FROM chunks ORDER BY doc_id, chunk_index").fetchall()

        # Clean up hierarchical indexes
        self.section_index.clear_doc(doc_id)
        self.glossary_index.clear_doc(doc_id)
        self.cross_ref_graph.clear_doc(doc_id)

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

    async def search(self, query: str, k: int) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        qv = await self.emb.embed([query])
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv.astype(np.float32), k)
        out = []
        chunk_ids = self.meta["chunk_ids"]
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

    async def search_transcript_only(self, query: str, k: int, query_vector: Optional[np.ndarray] = None) -> List[Dict]:
        """Search only over the transcript-only index. Returns same shape as search()."""
        if self.transcript_index is None:
            await self._rebuild_transcript_index()
        if self.transcript_index is None or self.transcript_index.ntotal == 0:
            return []
        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32) if not isinstance(qv, np.ndarray) else qv.astype(np.float32)
            faiss.normalize_L2(qv)
        D, I = self.transcript_index.search(qv, k)
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

    def _is_transcript_chunk(self, source: dict) -> bool:
        fn = (source or {}).get("filename") or ""
        return (fn or "").startswith("transcript_")

    async def search_document_only(self, query: str, k: int, query_vector: Optional[np.ndarray] = None) -> List[Dict]:
        """Search only over uploaded documents (exclude transcripts)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32) if not isinstance(qv, np.ndarray) else qv.astype(np.float32)
            faiss.normalize_L2(qv)
        fetch_k = min(k * 4, self.index.ntotal)
        D, I = self.index.search(qv, fetch_k)
        out = []
        chunk_ids = self.meta["chunk_ids"]
        with get_conn() as conn:
            for rank, idx in enumerate(I[0].tolist()):
                if idx < 0 or idx >= len(chunk_ids) or len(out) >= k:
                    continue
                cid = chunk_ids[idx]
                row = conn.execute("SELECT text, source_json FROM chunks WHERE id=?", (cid,)).fetchone()
                if not row:
                    continue
                text, src_json = row
                src = json.loads(src_json)
                if self._is_transcript_chunk(src):
                    continue
                out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": src})
        return out

    def search_document_only_sparse(self, query: str, k: int) -> List[Dict]:
        """Sparse (BM25) search over documents only (exclude transcripts)."""
        if not self.sparse._bm25 or not self.sparse.chunk_ids:
            return []
        raw = self.sparse.search(query, min(k * 4, len(self.sparse.chunk_ids)))
        out = [h for h in raw if not self._is_transcript_chunk(h.get("source") or {})]
        return out[:k]

    async def search_document_restricted(
        self,
        query: str,
        k: int,
        allowed_section_paths: List[str],
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Dense search restricted to chunks whose section_path starts with one of allowed_section_paths.

        Fetches up to k*8 candidates from FAISS then filters by section_path prefix.
        Falls back to search_document_only if no hits pass the filter.
        """
        if not allowed_section_paths:
            return await self.search_document_only(query, k, query_vector=query_vector)

        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32) if not isinstance(qv, np.ndarray) else qv.astype(np.float32)
            faiss.normalize_L2(qv)

        fetch_k = min(k * 8, self.index.ntotal)
        D, I = self.index.search(qv, fetch_k)
        out = []
        chunk_ids = self.meta["chunk_ids"]
        with get_conn() as conn:
            for rank, idx in enumerate(I[0].tolist()):
                if idx < 0 or idx >= len(chunk_ids) or len(out) >= k:
                    continue
                cid = chunk_ids[idx]
                row = conn.execute("SELECT text, source_json FROM chunks WHERE id=?", (cid,)).fetchone()
                if not row:
                    continue
                text, src_json = row
                src = json.loads(src_json)
                if self._is_transcript_chunk(src):
                    continue
                sp = src.get("section_path") or ""
                if any(sp.startswith(allowed) for allowed in allowed_section_paths):
                    out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": src})

        if not out:
            logger.info("search_document_restricted: no hits for section filter, falling back to global search")
            return await self.search_document_only(query, k, query_vector=qv)
        return out

    def search_document_sparse_restricted(self, query: str, k: int, allowed_section_paths: List[str]) -> List[Dict]:
        """BM25 search restricted by section_path prefix. Falls back to global sparse if no hits."""
        if not self.sparse._bm25 or not self.sparse.chunk_ids:
            return []
        if not allowed_section_paths:
            return self.search_document_only_sparse(query, k)

        raw = self.sparse.search(query, min(k * 8, len(self.sparse.chunk_ids)))
        out = []
        for h in raw:
            if self._is_transcript_chunk(h.get("source") or {}):
                continue
            sp = (h.get("source") or {}).get("section_path") or ""
            if any(sp.startswith(allowed) for allowed in allowed_section_paths):
                out.append(h)
            if len(out) >= k:
                break

        if not out:
            logger.info("search_document_sparse_restricted: no hits for section filter, falling back to global sparse")
            return self.search_document_only_sparse(query, k)
        return out


index = FaissIndex()
