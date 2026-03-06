"""
TOC (Table of Contents) FAISS index for BookRAG routing.

Indexes TOC nodes by text_for_embedding. Used for chapter/section locator when
no explicit refs in query — TOC search returns allowed section_paths for child retrieval.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import faiss
import numpy as np

from ...core.config import settings
from ..book.toc_builder import build_toc_from_sections, save_toc_nodes

logger = logging.getLogger(__name__)

TOC_FAISS_PATH = os.path.join(settings.DATA_DIR, "faiss_toc.index")
TOC_META_PATH = os.path.join(settings.DATA_DIR, "toc_meta.json")


class TocIndex:
    """FAISS index over TOC node embeddings for routing."""

    def __init__(self, emb, faiss_path: str = TOC_FAISS_PATH, meta_path: str = TOC_META_PATH):
        self.emb = emb
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self._index: Optional[faiss.Index] = None
        self._meta: Dict = {"node_ids": [], "node_by_id": {}}
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

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.faiss_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning("TocIndex: save failed: %s", e)

    def clear_all(self) -> None:
        """Clear TOC index in-memory and remove persisted files."""
        self._index = None
        self._meta = {"node_ids": [], "node_by_id": {}}
        for path in (self.faiss_path, self.meta_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("TocIndex: could not remove %s: %s", path, e)

    async def rebuild(self) -> None:
        """Rebuild TOC index from book_sections."""
        nodes = build_toc_from_sections()
        if not nodes:
            self._index = None
            self._meta = {"node_ids": [], "node_by_id": {}}
            self._save()
            return
        texts = [n.get("text_for_embedding") or n.get("title") or "" for n in nodes]
        vecs = await self.emb.embed(texts)
        vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self._meta = {"node_ids": [], "node_by_id": {}}
        for n in nodes:
            nid = n.get("toc_node_id") or n.get("section_path", "")
            self._meta["node_ids"].append(nid)
            self._meta["node_by_id"][nid] = n
        self._save()
        save_toc_nodes(nodes)
        logger.info("TocIndex: rebuilt %d nodes", len(nodes))

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.2,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Return top-k TOC nodes with score >= threshold."""
        if self._index is None or self._index.ntotal == 0:
            return []
        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)
        D, I = self._index.search(qv, min(k * 2, self._index.ntotal))
        out: List[Dict] = []
        node_ids = self._meta["node_ids"]
        node_by_id = self._meta["node_by_id"]
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
