"""
Glossary priority index (Step 7).

During ingestion:
  - Detect sections whose title matches "Glossary", "Definitions", "Terms", etc.
  - Embed those sections into a separate FAISS index (glossary_index).
  - Also store the chunk texts for direct lookup.

During retrieval (when RAG_USE_GLOSSARY_PRIORITY=true and query_type == "definition"):
  - Search glossary_index first.
  - If best hit score > RAG_GLOSSARY_SCORE_THRESHOLD → return that hit exclusively.
  - Else fall back to normal retrieval pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

import faiss
import numpy as np

from ..core.config import settings
from ..utils.ids import new_id

logger = logging.getLogger(__name__)

_GLOSSARY_TITLE_RE = re.compile(
    r"^(glossary|definitions?|terms?\s+and\s+definitions?|abbreviations?\s+and\s+terms?|"
    r"key\s+terms?|terminology|defined\s+terms?)",
    re.I,
)

_GLOSSARY_EMBED_MAX_CHARS = int(os.getenv("ECHOMIND_GLOSSARY_EMBED_MAX_CHARS", "4000"))


def is_glossary_section(section_title: Optional[str]) -> bool:
    """True when the section title looks like a glossary/definitions section."""
    if not section_title:
        return False
    return bool(_GLOSSARY_TITLE_RE.match(section_title.strip()))


class GlossaryIndex:
    """FAISS index restricted to glossary/definitions sections."""

    def __init__(self, emb, faiss_path: str, meta_path: str):
        self.emb = emb
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self._index: Optional[faiss.Index] = None
        # ids: list of entry IDs in FAISS row order
        # by_id: {entry_id: {doc_id, section_path, text}}
        self._meta: Dict = {"ids": [], "by_id": {}}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.faiss_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info(
                    "GlossaryIndex: loaded %d entries",
                    len(self._meta.get("ids", [])),
                )
            except Exception as e:
                logger.warning("GlossaryIndex: load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.faiss_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning("GlossaryIndex: save failed: %s", e)

    async def add_entries(self, entries: List[Dict]) -> None:
        """Add glossary entries. Each entry: {doc_id, section_path, text}."""
        if not entries:
            return
        texts = [(e.get("text") or "")[:_GLOSSARY_EMBED_MAX_CHARS] for e in entries]
        vecs = await self.emb.embed(texts)
        vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)

        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

        for e in entries:
            eid = new_id("gls")
            self._meta["ids"].append(eid)
            self._meta["by_id"][eid] = {
                "doc_id": e.get("doc_id"),
                "section_path": e.get("section_path"),
                "text": (e.get("text") or "")[:800],
            }

        self._save()
        logger.info("GlossaryIndex: added %d entries (total=%d)", len(entries), len(self._meta["ids"]))

    async def search(
        self,
        query: str,
        k: int = 3,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Search glossary index. Returns list of {score, text, section_path, doc_id}."""
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

        fetch_k = min(max(k * 3, 5), self._index.ntotal)
        D, I = self._index.search(qv, fetch_k)

        out: List[Dict] = []
        ids = self._meta["ids"]
        by_id = self._meta["by_id"]

        for rank, idx in enumerate(I[0].tolist()):
            if idx < 0 or idx >= len(ids) or len(out) >= k:
                continue
            eid = ids[idx]
            info = by_id.get(eid, {})
            out.append({
                "score": float(D[0][rank]),
                "text": info.get("text", ""),
                "section_path": info.get("section_path"),
                "doc_id": info.get("doc_id"),
            })

        return out

    def clear_doc(self, doc_id: str) -> None:
        surviving = [
            eid for eid in self._meta["ids"]
            if self._meta["by_id"].get(eid, {}).get("doc_id") != doc_id
        ]
        for eid in list(self._meta["by_id"].keys()):
            if self._meta["by_id"][eid].get("doc_id") == doc_id:
                del self._meta["by_id"][eid]
        self._meta["ids"] = surviving
        self._save()

    def clear_all(self) -> None:
        self._index = None
        self._meta = {"ids": [], "by_id": {}}
        for path in (self.faiss_path, self.meta_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
