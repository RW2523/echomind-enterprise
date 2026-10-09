"""One partition's searchable passages (decision D-005, option a).

An exact inner-product FAISS index over L2-normalised embeddings plus a BM25 corpus, persisted
together in one directory. The directory records which partition it belongs to; files that name
another partition (or disagree with each other) are never served: the partition is rebuilt from
the database instead, where every passage keeps its matter id and its embedding.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterable, Optional

import faiss
import numpy as np

from ...core.db import get_conn
from ...rag.sparse import _tokenize

logger = logging.getLogger(__name__)

_META = "meta.json"
_DENSE = "dense.faiss"


class PartitionIndex:
    def __init__(self, partition: str, matter_id: str, directory: str):
        self.partition = partition
        self.matter_id = matter_id
        self.dir = directory
        self.lock = threading.RLock()
        self.chunk_ids: list[str] = []
        self.tokens: list[list[str]] = []
        self.dense: Optional[faiss.Index] = None
        self._bm25 = None
        self.files_mtime = 0       # meta.json as last read or written by this process

    # ── Loading and saving ───────────────────────────────────────────────────────
    @classmethod
    def open(cls, partition: str, matter_id: str, directory: str) -> "PartitionIndex":
        p = cls(partition, matter_id, directory)
        if not p._load_files():
            p.rebuild_from_db()
        return p

    def _load_files(self) -> bool:
        meta_path, dense_path = os.path.join(self.dir, _META), os.path.join(self.dir, _DENSE)
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("partition") != self.partition:
                logger.warning("Retrieval: index files in %s belong to another partition; rebuilding", self.dir)
                return False
            ids, tokens = list(meta.get("chunk_ids") or []), list(meta.get("tokens") or [])
            dense = faiss.read_index(dense_path) if ids else None
            if len(ids) != len(tokens) or (dense is not None and dense.ntotal != len(ids)):
                logger.warning("Retrieval: index files in %s are inconsistent; rebuilding", self.dir)
                return False
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Retrieval: index files in %s unreadable (%s); rebuilding", self.dir, type(e).__name__)
            return False
        self.chunk_ids, self.tokens, self.dense = ids, tokens, dense
        self._bm25 = None
        self.files_mtime = os.stat(meta_path).st_mtime_ns
        return True

    def files_changed(self) -> bool:
        """True when another process rewrote this partition since this copy was loaded."""
        try:
            return os.stat(os.path.join(self.dir, _META)).st_mtime_ns != self.files_mtime
        except OSError:
            return True

    def save(self) -> None:
        with self.lock:
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            dense_path = os.path.join(self.dir, _DENSE)
            if self.dense is not None and self.dense.ntotal:
                faiss.write_index(self.dense, dense_path + ".tmp")
                os.replace(dense_path + ".tmp", dense_path)
            elif os.path.exists(dense_path):
                os.remove(dense_path)
            tmp = os.path.join(self.dir, _META + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"partition": self.partition, "chunk_ids": self.chunk_ids, "tokens": self.tokens}, f)
            os.replace(tmp, os.path.join(self.dir, _META))
            self.files_mtime = os.stat(os.path.join(self.dir, _META)).st_mtime_ns

    def rebuild_from_db(self) -> None:
        """Recreate the partition from the database (its passages, their stored embeddings)."""
        with get_conn() as conn:
            rows = conn.execute("SELECT id, text, vec FROM legal_chunks WHERE matter_id=? AND active=1 "
                                "ORDER BY version_id, chunk_index", (self.matter_id,)).fetchall()
        with self.lock:
            self.chunk_ids, self.tokens, self.dense, self._bm25 = [], [], None, None
            self._append([(r[0], r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows if r[2]])
            self.save()

    # ── Changing ─────────────────────────────────────────────────────────────────
    def _append(self, items: list[tuple[str, str, np.ndarray]]) -> None:
        if not items:
            return
        vecs = np.vstack([v for _, _, v in items]).astype(np.float32)
        if self.dense is None:
            self.dense = faiss.IndexFlatIP(vecs.shape[1])
        self.dense.add(vecs)
        self.chunk_ids += [cid for cid, _, _ in items]
        self.tokens += [_tokenize(text) for _, text, _ in items]
        self._bm25 = None

    def add(self, items: list[tuple[str, str, np.ndarray]]) -> None:
        """items: (chunk_id, text, L2-normalised float32 vector)."""
        with self.lock:
            self._append(items)
            self.save()

    def remove(self, chunk_ids: Iterable[str]) -> None:
        drop = set(chunk_ids)
        with self.lock:
            keep = [i for i, cid in enumerate(self.chunk_ids) if cid not in drop]
            if len(keep) == len(self.chunk_ids):
                return
            vecs = [self.dense.reconstruct(i) for i in keep] if self.dense is not None else []
            ids, toks = [self.chunk_ids[i] for i in keep], [self.tokens[i] for i in keep]
            self.dense = None
            if vecs:
                self.dense = faiss.IndexFlatIP(len(vecs[0]))
                self.dense.add(np.vstack(vecs).astype(np.float32))
            self.chunk_ids, self.tokens, self._bm25 = ids, toks, None
            self.save()

    # ── Searching ────────────────────────────────────────────────────────────────
    def search_dense(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self.lock:
            if self.dense is None or not self.dense.ntotal:
                return []
            if qvec.shape[-1] != self.dense.d:
                logger.warning("Retrieval: query embedding size %d does not match the index (%d)", qvec.shape[-1], self.dense.d)
                return []
            scores, idx = self.dense.search(qvec.reshape(1, -1).astype(np.float32), min(k, self.dense.ntotal))
            return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def search_sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = _tokenize(query)
        with self.lock:
            if not self.chunk_ids or not terms:
                return []
            if self._bm25 is None:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(self.tokens)           # this partition's own statistics only
            scores = self._bm25.get_scores(terms)
            # A passage must share at least one term with the query: BM25 can give tiny or negative
            # scores to every passage of a very small corpus, which would make "found" meaningless.
            qset = set(terms)
            hits = [(self.chunk_ids[i], float(s)) for i, s in enumerate(scores) if qset & set(self.tokens[i])]
            return sorted(hits, key=lambda h: h[1], reverse=True)[:k]
