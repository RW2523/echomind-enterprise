from __future__ import annotations
import json
import logging
import math
import os
import re
import contextvars
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .embeddings import OllamaEmbeddings
from .sparse import Bm25Index
from .chunking import chunk_document
from .chunking.models import DocType
from .section_index import SectionIndex
from .glossary_index import GlossaryIndex, is_glossary_section
from .cross_ref_graph import CrossRefGraph, extract_references
from .book.section_id import section_id_from_path, extract_section_id, extract_all_codes
from .metadata_validation import validate_and_fix_source, validate_book_chunk_for_indexing, validate_chunk_metadata

# ── KB namespacing: per-vertical / per-tenant isolation, set per request (contextvar = async-safe).
# namespace None  -> no filter (existing single-KB behavior, fully backward compatible).
# namespace "X"   -> search returns only chunks tagged namespace == "X".
_active_namespace: contextvars.ContextVar = contextvars.ContextVar("active_namespace", default=None)


def set_active_namespace(ns: Optional[str]) -> None:
    _active_namespace.set((ns or "").strip() or None)


def _ns_ok(src: dict) -> bool:
    ns = _active_namespace.get()
    if ns is None:
        return True
    return ((src or {}).get("namespace") or "default") == ns


def _ns_fetch_k(k: int, base_mult: int, total: int) -> int:
    """Candidate-pool size for FAISS/BM25 fetches, namespace-aware.

    Searches rank GLOBALLY then filter by namespace. With a large default corpus
    (e.g. 10k FMR chunks) a small vertical namespace (~60 chunks) can be entirely
    absent from a k*4 global top — retrieval returns 0 hits for questions the
    namespace answers easily. When a namespace is active, widen the pool a lot;
    FlatIP over ~10k vectors costs milliseconds, so the recall is cheap.
    """
    if _active_namespace.get() is not None:
        return min(max(k * 50, 512), total)
    # Unfiltered (whole-KB) path: keep the pool NARROW. Widening it to 192 was tried and
    # measured WORSE — the golden eval dropped 47->43 with doc-precision 0.98->0.86, and
    # all six regressions were whole-KB questions. This corpus is ~97% transcript chunks,
    # so a bigger candidate pool mostly admits near-miss transcript noise that crowds the
    # correct document out of the reranker's top-N. Override per-deployment if your corpus
    # is document-dominated rather than transcript-dominated.
    floor = int(getattr(settings, "RAG_GLOBAL_FETCH_FLOOR", 0))
    return min(max(k * base_mult, floor), total)
from .contextualizer import (
    build_context_header_from_chunk,
    build_contextualized_text,
    generate_section_summaries_batch,
    generate_chunk_roles_batch,
)

logger = logging.getLogger(__name__)

# Stopwords excluded from keyword/grep fallback (keep acronyms and meaningful terms)
_KEYWORD_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "what", "which", "when", "where",
    "who", "how", "why", "that", "this", "these", "those", "it", "its",
})


def _extract_keyword_search_terms(query: str) -> List[str]:
    """Extract terms for literal substring search: acronyms (2+ caps), words >= 2 chars, numbers. Excludes stopwords."""
    if not (query or "").strip():
        return []
    q = query.strip()
    terms: List[str] = []
    seen: set = set()
    # Acronyms: 2+ consecutive uppercase letters (e.g. DDRS, DoD, FMR)
    for m in re.finditer(r"[A-Z]{2,}", q):
        t = m.group(0)
        if t.lower() not in _KEYWORD_STOP and t not in seen:
            terms.append(t)
            seen.add(t)
    # Words: alphanumeric, min 2 chars
    for m in re.findall(r"[a-zA-Z0-9]{2,}", q):
        t = m
        if t.lower() in _KEYWORD_STOP or t in seen:
            continue
        terms.append(t)
        seen.add(t)
    return terms[:10]  # Cap to avoid huge LIKE chains

IVF_THRESHOLD = 10_000
IVF_NPROBE = 32


def _build_ivf_index(vecs: np.ndarray, dim: int) -> faiss.Index:
    """Build an IVF index trained on the given vectors.

    nlist = sqrt(n) (clamped 16–1024). The index uses IndexFlatIP as the
    quantizer so inner-product scores remain comparable.
    """
    n = vecs.shape[0]
    nlist = max(16, min(1024, int(math.sqrt(n))))
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vecs)
    ivf.add(vecs)
    ivf.nprobe = IVF_NPROBE
    logger.info("index: built IVF index — %d vectors, nlist=%d, nprobe=%d", n, nlist, IVF_NPROBE)
    return ivf


def _set_nprobe(idx: faiss.Index) -> None:
    """Ensure nprobe is set when loading an existing IVF index from disk."""
    try:
        if hasattr(idx, "nprobe"):
            idx.nprobe = IVF_NPROBE
    except Exception:
        pass


def _is_transcript_doc(filename: str, meta: dict) -> bool:
    """True if this document is a transcript (stored via add_text with transcript_ prefix or type)."""
    if (filename or "").startswith("transcript_"):
        return True
    return (meta or {}).get("type") == "transcript"


def _build_book_sections_from_chunks(chunks, doc_id: str) -> List[Dict]:
    """Reconstruct section-level entries from parent chunks (grouped by section_path).

    Returns list of {section_id, doc_id, section_title, section_path, full_section_text, canonical_section_id}.
    canonical_section_id: DoD-style code (e.g. 030201) for deterministic matching.
    """
    sections_by_path: Dict[str, Dict] = {}
    for c in chunks:
        if not c.is_parent:
            continue
        sp = (c.section_path or "").strip() or "__root__"
        if sp not in sections_by_path:
            canonical_sid = section_id_from_path(sp) or extract_section_id(c.section or sp) or extract_section_id(sp)
            # Fallback: extract DoD codes from section text (for "Segment N" docs that mention codes in body)
            if not canonical_sid and c.text:
                codes = extract_all_codes(c.text)
                if codes:
                    canonical_sid = max(codes, key=len)
            sections_by_path[sp] = {
                "section_id": new_id("sec"),
                "doc_id": doc_id,
                "section_title": c.section or sp,
                "section_path": sp,
                "texts": [],
                "canonical_section_id": canonical_sid,
            }
        sections_by_path[sp]["texts"].append(c.text)

    result = []
    for sp, v in sections_by_path.items():
        combined = " ".join(v["texts"])
        canonical_sid = v.get("canonical_section_id")
        # Fallback: extract DoD codes from combined section text (for "Segment N" docs)
        if not canonical_sid and combined:
            codes = extract_all_codes(combined)
            if codes:
                canonical_sid = max(codes, key=len)
        result.append({
            "section_id": v["section_id"],
            "doc_id": doc_id,
            "section_title": v["section_title"],
            "section_path": sp,
            "full_section_text": combined[:6000],
            "canonical_section_id": canonical_sid,
        })
    return result


def _store_book_sections_in_db(sections: List[Dict]) -> None:
    """Persist section metadata to book_sections table."""
    if not sections:
        return
    with get_conn() as conn:
        for s in sections:
            summary = s.get("section_summary")
            conn.execute(
                "INSERT OR IGNORE INTO book_sections "
                "(section_id, doc_id, section_title, section_path, full_section_text, created_at, section_summary) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    s["section_id"],
                    s["doc_id"],
                    s["section_title"],
                    s["section_path"],
                    s["full_section_text"],
                    now_iso(),
                    summary,
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
        # TOC index for BookRAG routing
        try:
            from .indexes.toc_index import TocIndex
            self.toc_index = TocIndex(self.emb)
        except Exception:
            self.toc_index = None
        # Section summary index for concept query routing (embeds section_summary)
        try:
            from .indexes.section_summary_index import SectionSummaryIndex
            self.section_summary_index = SectionSummaryIndex(self.emb)
        except Exception:
            self.section_summary_index = None
        self._load()

    def _load(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(settings.FAISS_PATH) and os.path.exists(settings.META_PATH):
            self.index = faiss.read_index(settings.FAISS_PATH)
            _set_nprobe(self.index)
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
        # Capture the current vectors' positions so survivors can be reconstructed instead of
        # re-embedded (re-embedding every transcript chunk on each delete is the slow path). (audit perf)
        _old_t_ids = list(self.transcript_meta.get("chunk_ids", []))
        _old_t_pos = {cid: i for i, cid in enumerate(_old_t_ids)}
        _old_t_index = self.transcript_index
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
        vecs = None
        if (
            _old_t_index is not None
            and _old_t_index.ntotal == len(_old_t_ids)
            and all(cid in _old_t_pos for cid in transcript_ids)
        ):
            try:
                vecs = np.vstack(
                    [_old_t_index.reconstruct(int(_old_t_pos[cid])) for cid in transcript_ids]
                ).astype(np.float32)
            except Exception as e:
                logger.warning("_rebuild_transcript_index: reconstruct failed (%s); re-embedding", e)
                vecs = None
        if vecs is None:
            vecs = await self.emb.embed(transcript_texts)
            faiss.normalize_L2(vecs)
            vecs = vecs.astype(np.float32)
        dim = vecs.shape[1]
        self.transcript_index = faiss.IndexFlatIP(dim)
        self.transcript_index.add(vecs)
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

    def _maybe_upgrade_to_ivf(self) -> None:
        """Upgrade the main index from FlatIP to IVF when it exceeds IVF_THRESHOLD.

        This is a no-op if the index is already IVF or below threshold.
        Reads all child-chunk vectors from the existing FlatIP, builds a trained
        IVF index, and swaps+saves in place.
        """
        if self.index is None:
            return
        n = self.index.ntotal
        if n < IVF_THRESHOLD:
            return
        if isinstance(self.index, faiss.IndexIVFFlat):
            return
        try:
            # faiss.IndexIVFFlat is a subclass check but read_index may return
            # the C++ type; also guard against swigfaiss types.
            idx_type = type(self.index).__name__
            if "IVF" in idx_type:
                return
        except Exception:
            pass

        logger.info("index: upgrading FlatIP (%d vectors) to IVF", n)
        dim = self.index.d
        vecs = self.index.reconstruct_n(0, n).astype(np.float32)
        self.index = _build_ivf_index(vecs, dim)
        self._save()

    async def add_document(
        self,
        filename: str,
        filetype: str,
        text: str,
        meta: dict,
        estimated_pages: int = 0,
        page_offsets: Optional[List[Tuple[int, int]]] = None,
        namespace: str = "default",
    ) -> dict:
        doc_id = new_id("doc")
        ns = (namespace or "default").strip() or "default"
        meta = {**(meta or {}), "namespace": ns}
        all_chunks = chunk_document(
            text or "", doc_id,
            estimated_pages=estimated_pages,
            page_offsets=page_offsets,
            # PDFs are normalized per-page in parse_pdf (keeps page_offsets aligned); other
            # types still get normalized inside chunk_document.
            already_normalized=bool(page_offsets),
        )
        if not all_chunks:
            raise ValueError("No text extracted")

        # Metadata validation for BOOK chunks. DEGRADE, NEVER DROP: a chunk whose section
        # metadata is unrecognized is still indexed as an ordinary chunk (it just loses the
        # section breadcrumb/citation precision). Previously such chunks were `continue`d at
        # logger.debug — measured content loss of 90% on one FMR chapter (47 chunks -> 4)
        # the first time BookRAG detection started firing, with nothing above DEBUG to show it.
        valid_chunks: List = []
        degraded = 0
        for c in all_chunks:
            if getattr(c, "doc_type", None) == DocType.BOOK:
                src = c.to_source_dict(filename, filetype)
                _, ok = validate_book_chunk_for_indexing(src)
                if not ok:
                    c.doc_type = DocType.USER
                    c.section_path = None
                    c.section = None
                    c.section_id = None
                    degraded += 1
            valid_chunks.append(c)
        if degraded:
            logger.warning(
                "index: %d/%d chunk(s) in %s had unrecognized BOOK section metadata — "
                "indexed WITHOUT section info (content preserved, citations lose section/page precision)",
                degraded, len(all_chunks), filename,
            )
        all_chunks = valid_chunks

        embed_chunks = [c for c in all_chunks if not c.is_parent]
        texts_to_embed: List[str] = []
        contextualized_by_chunk: Dict[str, str] = {}
        has_parents = any(c.is_parent for c in all_chunks)
        use_contextual = (
            getattr(settings, "RAG_USE_CONTEXTUAL_RETRIEVAL", True)
            and has_parents
            and not _is_transcript_doc(filename, meta)
        )

        section_summaries: Dict[str, str] = {}
        if use_contextual:
            book_sections = _build_book_sections_from_chunks(all_chunks, doc_id)
            section_summaries = await generate_section_summaries_batch(
                book_sections,
                max_concurrent=getattr(settings, "RAG_CONTEXTUAL_SUMMARY_CONCURRENCY", 3),
            )
            for s in book_sections:
                sp = s.get("section_path") or ""
                if sp and sp not in section_summaries and s.get("full_section_text"):
                    pass
            chunk_roles_input = [(c.chunk_id, c.text) for c in embed_chunks]
            chunk_roles = await generate_chunk_roles_batch(
                chunk_roles_input,
                max_concurrent=getattr(settings, "RAG_CONTEXTUAL_SUMMARY_CONCURRENCY", 3),
            )
            doc_title = (filename or "").replace(".pdf", "").replace(".docx", "").replace("_", " ")
            for c in embed_chunks:
                parent_summary = section_summaries.get(c.section_path or "") if c.section_path else None
                chunk_role = chunk_roles.get(c.chunk_id)
                header = build_context_header_from_chunk(c, doc_title, parent_summary, chunk_role)
                ctx_text = build_contextualized_text(header, c.text)
                contextualized_by_chunk[c.chunk_id] = ctx_text
                texts_to_embed.append(ctx_text)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "contextual_retrieval: chunk=%s raw_preview=%s ctx_preview=%s",
                        c.chunk_id[:12],
                        (c.text or "")[:80],
                        (ctx_text or "")[:120],
                    )
        else:
            texts_to_embed = [c.text for c in embed_chunks]

        vecs = await self.emb.embed(texts_to_embed)
        faiss.normalize_L2(vecs)
        await self._ensure_index(int(vecs.shape[1]))

        validated_sources: dict = {}
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, filetype, created_at, meta_json) VALUES (?,?,?,?,?)",
                (doc_id, filename, filetype, now_iso(), json.dumps(meta)),
            )
            for c in all_chunks:
                src = c.to_source_dict(filename, filetype)
                if (src.get("doc_type") or "").lower() == "book":
                    src, _ = validate_chunk_metadata(src)
                else:
                    src, _ = validate_and_fix_source(src, doc_type=src.get("doc_type", ""))
                    src["metadata_valid"] = True
                src["namespace"] = ns
                validated_sources[c.chunk_id] = src
                ctx_text = contextualized_by_chunk.get(c.chunk_id) if c.chunk_id in contextualized_by_chunk else None
                conn.execute(
                    "INSERT INTO chunks (id, doc_id, chunk_index, text, source_json, contextualized_text) VALUES (?,?,?,?,?,?)",
                    (c.chunk_id, doc_id, c.chunk_index, c.text, json.dumps(src), ctx_text),
                )
            conn.commit()

        for c in embed_chunks:
            self.meta["chunk_ids"].append(c.chunk_id)
            self.meta["source_by_chunk"][c.chunk_id] = validated_sources.get(c.chunk_id, c.to_source_dict(filename, filetype))
        self.index.add(vecs.astype(np.float32))

        if _is_transcript_doc(filename, meta):
            await self._ensure_transcript_index(int(vecs.shape[1]))
            for c in embed_chunks:
                self.transcript_meta["chunk_ids"].append(c.chunk_id)
                self.transcript_meta["source_by_chunk"][c.chunk_id] = validated_sources.get(c.chunk_id, c.to_source_dict(filename, filetype))
            self.transcript_index.add(vecs.astype(np.float32))
            self.transcript_sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)

        self._save()
        self.sparse.add_chunks([c.chunk_id for c in embed_chunks], texts_to_embed)

        self._maybe_upgrade_to_ivf()

        # ── Hierarchical section + glossary indexing (BOOK documents only) ────
        if has_parents and not _is_transcript_doc(filename, meta):
            await self._index_book_sections(
                all_chunks, doc_id,
                section_summaries=section_summaries if use_contextual else {},
            )

        return {"doc_id": doc_id, "chunks": len(embed_chunks)}

    async def _index_book_sections(self, all_chunks, doc_id: str, section_summaries: Optional[Dict[str, str]] = None) -> None:
        """Build and store section-level embeddings, glossary entries, and cross-references."""
        book_sections = _build_book_sections_from_chunks(all_chunks, doc_id)
        if not book_sections:
            return

        summaries = section_summaries or {}
        for s in book_sections:
            sp = s.get("section_path") or ""
            if sp in summaries:
                s["section_summary"] = summaries[sp]

        # Store metadata in DB
        _store_book_sections_in_db(book_sections)

        # Add to section FAISS index
        await self.section_index.add_sections(book_sections)
        logger.info("index: indexed %d sections for doc_id=%s", len(book_sections), doc_id)

        # Rebuild section summary index (for concept query routing)
        if self.section_summary_index:
            try:
                await self.section_summary_index.rebuild()
            except Exception as e:
                logger.warning("index: section summary rebuild failed: %s", e)

        # Rebuild TOC index for BookRAG routing
        if self.toc_index:
            try:
                await self.toc_index.rebuild()
            except Exception as e:
                logger.warning("index: TOC rebuild failed: %s", e)

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
        if self.toc_index:
            try:
                self.toc_index.clear_all()
            except Exception:
                pass
        if getattr(self, "section_summary_index", None):
            try:
                self.section_summary_index.clear_all()
            except Exception:
                pass
        # Remove TOC nodes cache (toc_builder persists this)
        toc_nodes_path = os.path.join(settings.DATA_DIR, "toc_nodes.json")
        if os.path.exists(toc_nodes_path):
            try:
                os.remove(toc_nodes_path)
            except OSError:
                pass
        # Clear DB tables (book_sections, section_references)
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
            try:
                rows = conn.execute(
                    "SELECT id, COALESCE(contextualized_text, text), source_json FROM chunks ORDER BY doc_id, chunk_index"
                ).fetchall()
            except Exception:
                rows = conn.execute("SELECT id, text, source_json FROM chunks ORDER BY doc_id, chunk_index").fetchall()

        # Clean up hierarchical indexes
        self.section_index.clear_doc(doc_id)
        self.glossary_index.clear_doc(doc_id)
        self.cross_ref_graph.clear_doc(doc_id)
        if self.toc_index:
            try:
                await self.toc_index.rebuild()
            except Exception as e:
                logger.warning("TOC rebuild after delete failed: %s", e)
        if getattr(self, "section_summary_index", None):
            try:
                await self.section_summary_index.rebuild()
            except Exception as e:
                logger.warning("section summary rebuild after delete failed: %s", e)

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

        # Rebuild vectors for the survivors. Prefer reconstructing the existing (already-normalized)
        # vectors from the current index — re-embedding the whole corpus on every delete is O(corpus)
        # and made deleting a single transcript take many seconds/minutes on a large KB. Fall back to
        # re-embedding only when the index can't be reconstructed (e.g. IVF without a direct map, or a
        # prior index/meta desync). (audit perf)
        old_ids = self.meta.get("chunk_ids", [])
        old_pos = {cid: i for i, cid in enumerate(old_ids)}
        vecs = None
        if (
            self.index is not None
            and self.index.ntotal == len(old_ids)
            and remaining_ids
            and all(cid in old_pos for cid in remaining_ids)
        ):
            try:
                vecs = np.vstack(
                    [self.index.reconstruct(int(old_pos[cid])) for cid in remaining_ids]
                ).astype(np.float32)
            except Exception as e:
                logger.warning("delete_document: vector reconstruct failed (%s); re-embedding", e)
                vecs = None
        if vecs is None:
            vecs = await self.emb.embed(remaining_texts)
            faiss.normalize_L2(vecs)
            vecs = vecs.astype(np.float32)
        dim = vecs.shape[1]
        if vecs.shape[0] >= IVF_THRESHOLD:
            self.index = _build_ivf_index(vecs, dim)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vecs)
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
                src = json.loads(src_json)
                if not _ns_ok(src):
                    continue
                out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": src})
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
                src = json.loads(src_json)
                if not _ns_ok(src):
                    continue
                out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": src})
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
        fetch_k = _ns_fetch_k(k, 4, self.index.ntotal)
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
                if not _ns_ok(src):
                    continue
                out.append({"chunk_id": cid, "score": float(D[0][rank]), "text": text, "source": src})
        return out

    def search_document_only_sparse(self, query: str, k: int) -> List[Dict]:
        """Sparse (BM25) search over documents only (exclude transcripts)."""
        if not self.sparse._bm25 or not self.sparse.chunk_ids:
            return []
        raw = self.sparse.search(query, _ns_fetch_k(k, 4, len(self.sparse.chunk_ids)))
        out = [h for h in raw if not self._is_transcript_chunk(h.get("source") or {}) and _ns_ok(h.get("source") or {})]
        return out[:k]

    def search_document_keyword_grep(self, query: str, k: int) -> List[Dict]:
        """Literal substring search (grep-like) over document chunks. Fallback when semantic/BM25 miss exact terms (e.g. acronyms like DDRS).

        Extracts search terms from query (acronyms, words >= 2 chars), runs SQL LIKE over chunk text,
        excludes transcripts. Returns same shape as other search methods: {chunk_id, score, text, source}.
        Score: 0.5 base + 0.1 per extra term matched (capped 0.8).
        """
        terms = _extract_keyword_search_terms(query)
        if not terms:
            return []
        with get_conn() as conn:
            # Match ANY term (OR) for recall; score by match count in Python
            conditions = []
            params: List[str] = []
            for t in terms:
                conditions.append("LOWER(COALESCE(c.contextualized_text, c.text)) LIKE ?")
                params.append(f"%{t.lower()}%")
            where_clause = " OR ".join(conditions)
            sql = f"""
                SELECT c.id, c.text, c.source_json, c.contextualized_text
                FROM chunks c
                INNER JOIN documents d ON c.doc_id = d.id
                WHERE (d.filename IS NULL OR d.filename NOT LIKE 'transcript_%')
                AND ({where_clause})
            """
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []
        # Score by number of terms matched (more = higher)
        scored: Dict[str, Dict] = {}
        for row in rows:
            cid, text, src_json, ctx_text = row
            content = (ctx_text or text or "").lower()
            matches = sum(1 for t in terms if t.lower() in content)
            if matches == 0:
                continue
            src = json.loads(src_json) if src_json else {}
            if self._is_transcript_chunk(src):
                continue
            # Namespace isolation: without this the grep fallback leaks out-of-namespace
            # chunks (score 0.8) that outrank real in-namespace hits AND get killed by the
            # downstream _ns_ok safety net → net effect was 0 hits for vertical queries.
            if not _ns_ok(src):
                continue
            # Base 0.5 + 0.1 per term matched, cap 0.8
            score = min(0.8, 0.5 + 0.1 * matches)
            if cid not in scored or scored[cid]["score"] < score:
                scored[cid] = {"chunk_id": cid, "score": score, "text": text or "", "source": src}
        out = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:k]
        return out

    async def search_document_restricted(
        self,
        query: str,
        k: int,
        allowed_section_paths: List[str],
        query_vector: Optional[np.ndarray] = None,
        no_global_fallback: bool = False,
    ) -> List[Dict]:
        """Dense search restricted to chunks whose section_path starts with one of allowed_section_paths.

        Fetches up to k*8 candidates from FAISS then filters by section_path prefix.
        When no_global_fallback=False, falls back to search_document_only if no hits pass the filter.
        When no_global_fallback=True (explicit refs), returns empty list instead of falling back.
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

        fetch_k = _ns_fetch_k(k, 8, self.index.ntotal)
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

        if not out and not no_global_fallback:
            logger.info("search_document_restricted: no hits for section filter, falling back to global search")
            return await self.search_document_only(query, k, query_vector=qv)
        if not out and no_global_fallback:
            logger.info("search_document_restricted: no hits for section filter (explicit refs), no global fallback")
        return out

    def search_document_sparse_restricted(
        self,
        query: str,
        k: int,
        allowed_section_paths: List[str],
        no_global_fallback: bool = False,
    ) -> List[Dict]:
        """BM25 search restricted by section_path prefix. When no_global_fallback=False, falls back to global sparse if no hits."""
        if not self.sparse._bm25 or not self.sparse.chunk_ids:
            return []
        if not allowed_section_paths:
            return self.search_document_only_sparse(query, k)

        raw = self.sparse.search(query, _ns_fetch_k(k, 8, len(self.sparse.chunk_ids)))
        out = []
        for h in raw:
            if self._is_transcript_chunk(h.get("source") or {}):
                continue
            sp = (h.get("source") or {}).get("section_path") or ""
            if any(sp.startswith(allowed) for allowed in allowed_section_paths):
                out.append(h)
            if len(out) >= k:
                break

        if not out and not no_global_fallback:
            logger.info("search_document_sparse_restricted: no hits for section filter, falling back to global sparse")
            return self.search_document_only_sparse(query, k)
        if not out and no_global_fallback:
            logger.info("search_document_sparse_restricted: no hits for section filter (explicit refs), no global fallback")
        return out


index = FaissIndex()
