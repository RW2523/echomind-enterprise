"""
Section Summary Index: first-class retrieval layer for concept queries.

Embeds section_summary (LLM-generated) from book_sections. Used BEFORE child retrieval:
query → section summary search → rerank sections → top 1–3 sections → child retrieval.

Improves routing for: administrative control of funds, certifying officers,
unmatched disbursements, commitments and obligations.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import faiss
import numpy as np

from ...core.config import settings
from ...core.db import get_conn

logger = logging.getLogger(__name__)

SECTION_SUMMARY_FAISS_PATH = os.path.join(settings.DATA_DIR, "faiss_section_summary.index")
SECTION_SUMMARY_META_PATH = os.path.join(settings.DATA_DIR, "section_summary_meta.json")


def _section_summary_text_for_embedding(
    section_path: str,
    section_title: str,
    section_summary: Optional[str],
    full_section_text: Optional[str],
) -> str:
    """Build text for embedding: prefer section_summary, else title + excerpt."""
    if section_summary and section_summary.strip():
        return f"{section_path}. {section_summary.strip()}"[:2000]
    parts = [section_path]
    if section_title:
        parts.append(section_title)
    if full_section_text:
        parts.append(full_section_text[:800].replace("\n", " "))
    return " ".join(parts).strip()[:2000]


class SectionSummaryIndex:
    """FAISS index over section summary embeddings for hierarchical retrieval."""

    def __init__(self, emb, faiss_path: str = SECTION_SUMMARY_FAISS_PATH, meta_path: str = SECTION_SUMMARY_META_PATH):
        self.emb = emb
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self._index: Optional[faiss.Index] = None
        self._meta: Dict = {"section_ids": [], "section_by_id": {}}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.faiss_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info(
                    "SectionSummaryIndex: loaded %d sections",
                    len(self._meta.get("section_ids", [])),
                )
            except Exception as e:
                logger.warning("SectionSummaryIndex: load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, self.faiss_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning("SectionSummaryIndex: save failed: %s", e)

    async def rebuild(self) -> None:
        """Rebuild from book_sections (section_summary, section_path, etc.)."""
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT section_id, section_path, section_title, section_summary, full_section_text
                   FROM book_sections WHERE section_path IS NOT NULL"""
            ).fetchall()

        if not rows:
            self._index = None
            self._meta = {"section_ids": [], "section_by_id": {}}
            self._save()
            return

        texts = []
        sections = []
        for row in rows:
            sec_id, sp, title, summary, full_text = row[0], row[1] or "", row[2] or "", row[3] or "", row[4] or ""
            if not sp:
                continue
            text = _section_summary_text_for_embedding(sp, title, summary, full_text)
            texts.append(text)
            sections.append({
                "section_id": sec_id,
                "section_path": sp,
                "section_title": title,
                "section_summary": summary,
            })

        vecs = await self.emb.embed(texts, kind="document")
        vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)

        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self._meta = {"section_ids": [], "section_by_id": {}}
        for s in sections:
            sid = s["section_id"]
            self._meta["section_ids"].append(sid)
            self._meta["section_by_id"][sid] = s
        self._save()
        logger.info("SectionSummaryIndex: rebuilt %d sections", len(sections))

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.25,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Return top-k sections by summary relevance."""
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_vector is not None:
            qv = query_vector.astype(np.float32) if query_vector.ndim == 2 else query_vector.reshape(1, -1).astype(np.float32)
        else:
            qv = await self.emb.embed([query])
            qv = np.array(qv, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)

        fetch_k = min(max(k * 3, 10), self._index.ntotal)
        D, I = self._index.search(qv, fetch_k)

        out: List[Dict] = []
        section_ids = self._meta["section_ids"]
        sec_by_id = self._meta["section_by_id"]
        for rank, idx in enumerate(I[0].tolist()):
            if idx < 0 or idx >= len(section_ids) or len(out) >= k:
                continue
            score = float(D[0][rank])
            if score < threshold:
                continue
            sid = section_ids[idx]
            info = sec_by_id.get(sid, {})
            out.append({
                "section_id": sid,
                "section_path": info.get("section_path"),
                "section_title": info.get("section_title"),
                "score": score,
            })
        return out

    def clear_all(self) -> None:
        self._index = None
        self._meta = {"section_ids": [], "section_by_id": {}}
        for path in (self.faiss_path, self.meta_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
