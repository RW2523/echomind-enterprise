"""
Sparse (BM25) index for hybrid RAG. Tokenizes chunks and uses BM25Okapi for keyword retrieval.
Persists chunk_ids + tokenized corpus so the index can be rebuilt on load.
"""
from __future__ import annotations
import os
import re
import json
from typing import Dict, List

from ..core.config import settings
from ..core.db import get_conn


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, min length 2."""
    tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    return tokens


class Bm25Index:
    def __init__(self, meta_path: str | None = None):
        self._meta_path = meta_path if meta_path is not None else settings.SPARSE_META_PATH
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25 = None
        self._load()

    def _load(self) -> None:
        path = self._meta_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunk_ids = data.get("chunk_ids", [])
            self.corpus_tokens = data.get("corpus_tokens", [])
            if len(self.chunk_ids) != len(self.corpus_tokens):
                self.chunk_ids = []
                self.corpus_tokens = []
                return
            if self.corpus_tokens:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.chunk_ids = []
            self.corpus_tokens = []
            self._bm25 = None

    def _save(self) -> None:
        path = self._meta_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chunk_ids": self.chunk_ids, "corpus_tokens": self.corpus_tokens}, f, ensure_ascii=False)

    def rebuild_from_chunk_ids(self, chunk_ids: List[str]) -> None:
        """Rebuild full BM25 from DB (e.g. when sparse_meta was missing but FAISS has data).

        Uses batch queries (500 at a time) for performance with large corpora.
        """
        self.chunk_ids = []
        self.corpus_tokens = []
        batch_size = 500
        with get_conn() as conn:
            # Detect if contextualized_text column exists
            use_ctx = True
            try:
                conn.execute("SELECT contextualized_text FROM chunks LIMIT 1")
            except Exception:
                use_ctx = False
            for start in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[start:start + batch_size]
                placeholders = ",".join("?" for _ in batch)
                col = "COALESCE(contextualized_text, text)" if use_ctx else "text"
                rows = conn.execute(
                    f"SELECT id, {col} FROM chunks WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                row_map = {r[0]: r[1] for r in rows}
                for cid in batch:
                    txt = row_map.get(cid)
                    if txt is None:
                        continue
                    self.chunk_ids.append(cid)
                    self.corpus_tokens.append(_tokenize(txt))
        if self.corpus_tokens:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None
        self._save()

    def add_chunks(self, chunk_ids: List[str], texts: List[str]) -> None:
        """Append chunks and rebuild BM25. ids and texts must be same length and order as in FAISS."""
        from rank_bm25 import BM25Okapi
        for cid, text in zip(chunk_ids, texts):
            self.chunk_ids.append(cid)
            self.corpus_tokens.append(_tokenize(text))
        self._bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        self._save()

    def search(self, query: str, k: int) -> List[Dict]:
        """Return top-k chunks by BM25 score. Same dict shape as FaissIndex.search (chunk_id, score, text, source)."""
        if not self._bm25 or not self.chunk_ids:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # argsort descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k]
        out = []
        with get_conn() as conn:
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                cid = self.chunk_ids[idx]
                row = conn.execute("SELECT text, source_json FROM chunks WHERE id=?", (cid,)).fetchone()
                if not row:
                    continue
                text, src_json = row
                out.append({
                    "chunk_id": cid,
                    "score": float(scores[idx]),
                    "text": text,
                    "source": json.loads(src_json),
                })
        return out
