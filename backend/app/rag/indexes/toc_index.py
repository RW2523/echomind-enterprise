"""
TOC (Table of Contents) FAISS index for BookRAG routing.

Indexes TOC nodes by text_for_embedding. Used for chapter/section locator when
no explicit refs in query — TOC search returns allowed section_paths for child retrieval.

Optional hybrid search: vector + BM25 with weighted reciprocal rank fusion (RRF).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

import faiss
import numpy as np

from ...core.config import settings
from ..book.toc_builder import build_toc_from_sections, save_toc_nodes

logger = logging.getLogger(__name__)

TOC_FAISS_PATH = os.path.join(settings.DATA_DIR, "faiss_toc.index")
TOC_META_PATH = os.path.join(settings.DATA_DIR, "toc_meta.json")
TOC_SPARSE_PATH = os.path.join(settings.DATA_DIR, "toc_sparse.json")

RRF_K = 60  # RRF constant


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


class TocIndex:
    """FAISS index over TOC node embeddings for routing. Optional BM25 for hybrid search."""

    def __init__(self, emb, faiss_path: str = TOC_FAISS_PATH, meta_path: str = TOC_META_PATH, sparse_path: str = TOC_SPARSE_PATH):
        self.emb = emb
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self.sparse_path = sparse_path
        self._index: Optional[faiss.Index] = None
        self._meta: Dict = {"node_ids": [], "node_by_id": {}}
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.faiss_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info("TocIndex: loaded %d nodes", len(self._meta.get("node_ids", [])))
            except Exception as e:
                logger.warning("TocIndex: load failed: %s", e)
        if os.path.exists(self.sparse_path):
            try:
                with open(self.sparse_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._corpus_tokens = data.get("corpus_tokens", [])
                if self._corpus_tokens and len(self._corpus_tokens) == len(self._meta.get("node_ids", [])):
                    from rank_bm25 import BM25Okapi
                    self._bm25 = BM25Okapi(self._corpus_tokens)
            except Exception as e:
                logger.warning("TocIndex: sparse load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.faiss_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
            if self._corpus_tokens:
                with open(self.sparse_path, "w", encoding="utf-8") as f:
                    json.dump({"corpus_tokens": self._corpus_tokens}, f)
        except Exception as e:
            logger.warning("TocIndex: save failed: %s", e)

    def clear_all(self) -> None:
        """Clear TOC index in-memory and remove persisted files."""
        self._index = None
        self._bm25 = None
        self._corpus_tokens = []
        self._meta = {"node_ids": [], "node_by_id": {}}
        for path in (self.faiss_path, self.meta_path, self.sparse_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("TocIndex: could not remove %s: %s", path, e)

    async def rebuild(self) -> None:
        """Rebuild TOC index from book_sections (vector + optional BM25)."""
        nodes = build_toc_from_sections()
        if not nodes:
            self._index = None
            self._bm25 = None
            self._corpus_tokens = []
            self._meta = {"node_ids": [], "node_by_id": {}}
            self._save()
            return
        texts = [n.get("text_for_embedding") or n.get("title") or "" for n in nodes]
        vecs = await self.emb.embed(texts, kind="document")
        vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self._meta = {"node_ids": [], "node_by_id": {}}
        self._corpus_tokens = [_tokenize(t) for t in texts]
        for n in nodes:
            nid = n.get("toc_node_id") or n.get("section_path", "")
            self._meta["node_ids"].append(nid)
            self._meta["node_by_id"][nid] = n
        if self._corpus_tokens:
            try:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(self._corpus_tokens)
            except ImportError:
                self._bm25 = None
        self._save()
        save_toc_nodes(nodes)
        logger.info("TocIndex: rebuilt %d nodes (vector + BM25=%s)", len(nodes), self._bm25 is not None)

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.2,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Return top-k TOC nodes with score >= threshold.

        When RAG_TOC_HYBRID_SEARCH is True and BM25 is available, merges vector + BM25
        with weighted reciprocal rank fusion (RRF).
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        use_hybrid = getattr(settings, "RAG_TOC_HYBRID_SEARCH", False) and self._bm25 is not None
        node_ids = self._meta["node_ids"]
        node_by_id = self._meta["node_by_id"]

        if use_hybrid:
            # Vector search
            if query_vector is not None:
                qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
            else:
                qv = await self.emb.embed([query])
                qv = np.array(qv, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(qv)
            D, I = self._index.search(qv, min(k * 3, self._index.ntotal))
            vec_rank_by_nid: Dict[str, int] = {}
            for rank, idx in enumerate(I[0].tolist()):
                if idx < 0 or idx >= len(node_ids):
                    continue
                nid = node_ids[idx]
                vec_rank_by_nid[nid] = rank

            # BM25 search
            q_tokens = _tokenize(query)
            bm25_ranks: Dict[str, int] = {}
            if q_tokens:
                scores = self._bm25.get_scores(q_tokens)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k * 3]
                for rank, idx in enumerate(top_indices):
                    if scores[idx] <= 0 or idx >= len(node_ids):
                        continue
                    nid = node_ids[idx]
                    bm25_ranks[nid] = rank

            # Weighted RRF: w_vec * 1/(k+rank_vec) + w_bm25 * 1/(k+rank_bm25)
            w_vec, w_bm25 = 0.6, 0.4
            rrf_scores: Dict[str, float] = {}
            all_nids = set(vec_rank_by_nid) | set(bm25_ranks)
            for nid in all_nids:
                rrf = 0.0
                if nid in vec_rank_by_nid:
                    rrf += w_vec / (RRF_K + vec_rank_by_nid[nid])
                if nid in bm25_ranks:
                    rrf += w_bm25 / (RRF_K + bm25_ranks[nid])
                rrf_scores[nid] = rrf
            sorted_nids = sorted(rrf_scores, key=lambda n: -rrf_scores[n])[:k]
            out = []
            for nid in sorted_nids:
                info = node_by_id.get(nid, {})
                out.append({
                    "toc_node_id": nid,
                    "section_path": info.get("section_path"),
                    "section_ids": info.get("section_ids", []),
                    "title": info.get("title"),
                    "score": rrf_scores[nid],
                })
            return out

        # Vector-only
        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)
        D, I = self._index.search(qv, min(k * 2, self._index.ntotal))
        out: List[Dict] = []
        for rank, idx in enumerate(I[0].tolist()):
            if idx < 0 or idx >= len(node_ids) or len(out) >= k:
                continue
            score = float(D[0][rank])
            if score < threshold:
                continue
            nid = node_ids[idx]
            info = node_by_id.get(nid, {})
            out.append({
                "toc_node_id": nid,
                "section_path": info.get("section_path"),
                "section_ids": info.get("section_ids", []),
                "title": info.get("title"),
                "score": score,
            })
        return out

    def get_section_paths_from_nodes(self, nodes: List[Dict]) -> List[str]:
        """Extract section_paths from TOC search results."""
        paths: List[str] = []
        seen: set = set()
        for n in nodes:
            p = n.get("section_path")
            if p and p not in seen:
                seen.add(p)
                paths.append(p)
        return paths
