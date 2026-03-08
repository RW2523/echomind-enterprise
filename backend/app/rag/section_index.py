"""
Hierarchical section-level FAISS index for BOOK documents (Step 1).

During ingestion:
  - Each unique section_path from parent chunks becomes one section embedding.
  - The full section text (joined parent chunks) is embedded and stored.

During retrieval:
  - Embed query → search section_index → top-N matching section_paths.
  - Child FAISS/BM25 searches are restricted to chunks whose section_path
    starts with one of those top paths (hierarchical prefix match).
  - If no section scores above threshold → fall back to global child search.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import faiss
import numpy as np

from ..core.config import settings
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

# Max chars to embed per section. Must fit within embedding model context window.
# nomic-embed-text default context varies; 2000 chars is safe for all Ollama builds.
_SECTION_EMBED_MAX_CHARS = int(os.getenv("ECHOMIND_SECTION_EMBED_MAX_CHARS", "2000"))


class SectionIndex:
    """FAISS flat-IP index over section embeddings for hierarchical retrieval."""

    def __init__(self, emb, faiss_path: str, meta_path: str):
        self.emb = emb
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self._index: Optional[faiss.Index] = None
        # section_ids: ordered list matching FAISS row positions
        # section_by_id: {section_id: {doc_id, section_title, section_path}}
        self._meta: Dict = {"section_ids": [], "section_by_id": {}}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.faiss_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info(
                    "SectionIndex: loaded %d sections",
                    len(self._meta.get("section_ids", [])),
                )
            except Exception as e:
                logger.warning("SectionIndex: load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.faiss_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning("SectionIndex: save failed: %s", e)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def add_sections(self, sections: List[Dict]) -> None:
        """Embed and store section-level entries.

        Each entry: {section_id, doc_id, section_title, section_path, full_section_text}.
        """
        if not sections:
            return

        texts = [
            (s.get("full_section_text") or s.get("section_title") or "")[:_SECTION_EMBED_MAX_CHARS]
            for s in sections
        ]
        vecs = await self.emb.embed(texts)
        vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)

        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

        for s in sections:
            sid = s["section_id"]
            self._meta["section_ids"].append(sid)
            self._meta["section_by_id"][sid] = {
                "doc_id": s.get("doc_id"),
                "section_title": s.get("section_title"),
                "section_path": s.get("section_path"),
                "canonical_section_id": s.get("canonical_section_id"),
            }
        self._save()
        logger.info("SectionIndex: added %d sections (total=%d)", len(sections), len(self._meta["section_ids"]))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 3,
        query_vector: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Return top-k matching sections as list of {section_id, doc_id, section_path, section_title, score}.

        Optionally filter by doc_ids (list of allowed doc_id values).
        If query_vector provided (shape [1, dim], L2-normalised) embedding is skipped.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_vector is not None:
            qv = (
                query_vector.astype(np.float32)
                if query_vector.ndim == 2
                else query_vector.reshape(1, -1).astype(np.float32)
            )
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)

        fetch_k = min(max(k * 5, 15), self._index.ntotal)
        D, I = self._index.search(qv, fetch_k)

        out: List[Dict] = []
        section_ids = self._meta["section_ids"]
        sec_by_id = self._meta["section_by_id"]

        for rank, idx in enumerate(I[0].tolist()):
            if idx < 0 or idx >= len(section_ids) or len(out) >= k:
                continue
            sid = section_ids[idx]
            info = sec_by_id.get(sid, {})
            if doc_ids is not None and info.get("doc_id") not in doc_ids:
                continue
            out.append({
                "section_id": sid,
                "doc_id": info.get("doc_id"),
                "section_path": info.get("section_path"),
                "section_title": info.get("section_title"),
                "score": float(D[0][rank]),
            })

        return out

    def get_paths_containing_codes(self, codes: List[str]) -> List[str]:
        """Return section_paths that contain any of the given codes (e.g. 0301, 0402).

        Used for comparison queries to ensure both sections are in retrieval scope.
        Checks section_path and canonical_section_id (DoD code from section text).
        """
        if not codes:
            return []
        paths: List[str] = []
        seen: set = set()
        for info in self._meta.get("section_by_id", {}).values():
            sp = (info.get("section_path") or "").strip()
            if not sp or sp in seen:
                continue
            canon = (info.get("canonical_section_id") or "").strip()
            for code in codes:
                if not code:
                    continue
                if code == canon:
                    paths.append(sp)
                    seen.add(sp)
                    break
                if code in sp or sp.endswith(code) or (">" + code) in sp:
                    paths.append(sp)
                    seen.add(sp)
                    break
        return paths

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear_doc(self, doc_id: str) -> None:
        """Remove all section entries for a doc_id from meta (index vectors stay but are ignored).

        Note: FAISS IndexFlatIP does not support in-place deletion.  The orphaned
        vectors will not be returned because their section_id is removed from meta.
        A full index rebuild is triggered by delete_document in FaissIndex.
        """
        surviving = [
            sid for sid in self._meta["section_ids"]
            if self._meta["section_by_id"].get(sid, {}).get("doc_id") != doc_id
        ]
        for sid in list(self._meta["section_by_id"].keys()):
            if self._meta["section_by_id"][sid].get("doc_id") == doc_id:
                del self._meta["section_by_id"][sid]
        self._meta["section_ids"] = surviving
        self._save()

    def clear_all(self) -> None:
        self._index = None
        self._meta = {"section_ids": [], "section_by_id": {}}
        for path in (self.faiss_path, self.meta_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
