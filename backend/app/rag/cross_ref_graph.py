"""
Cross-reference graph for BOOK documents (Step 4).

During ingestion:
  - Scan each section text for DoD-style cross-references:
      "See paragraph 030201", "Refer to Volume 5, Chapter 10", "See section 080302"
  - Build adjacency: graph[source_section_path] = [referenced_path, ...]
  - Persist to JSON.

During retrieval:
  - For each retrieved chunk, follow outgoing references (BFS, max 2 levels deep).
  - Append referenced chunks as supplemental context (labeled "(ref: path)").
  - Hard limit: at most RAG_GRAPH_MAX_ADDITIONS (default 3) extra sections total.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set

from ..core.db import get_conn
from ..utils.ids import new_id

logger = logging.getLogger(__name__)

# ── Reference extraction patterns ────────────────────────────────────────────

_XREF_PATTERNS = [
    # "See paragraph 030201" / "See paragraph 030201.2"
    re.compile(r"\b[Ss]ee\s+[Pp]aragraph\s+([\d]{4,6}(?:\.\d{1,4}){0,3})", re.I),
    # "Refer to Volume 5, Chapter 10"
    re.compile(r"\b[Rr]efer\s+to\s+[Vv]olume\s+(\d+),?\s+[Cc]hapter\s+(\d+)", re.I),
    # "See section 080302"
    re.compile(r"\b[Ss]ee\s+[Ss]ection\s+([\d]{3,6}(?:\.\d{1,4}){0,3})", re.I),
    # "See Chapter 5"
    re.compile(r"\b[Ss]ee\s+[Cc]hapter\s+(\d+)", re.I),
    # "See Volume 7"
    re.compile(r"\b[Ss]ee\s+[Vv]olume\s+(\d+)", re.I),
    # "See Part III"
    re.compile(r"\b[Ss]ee\s+[Pp]art\s+([IVXivx\d]+)", re.I),
    # Bare 6-digit code after "see" / "reference"
    re.compile(r"\b(?:[Ss]ee|[Rr]eference(?:s|d)?)\s+(\d{5,6}(?:\.\d{1,4}){0,3})\b", re.I),
]


def extract_references(text: str, source_section_path: str, doc_id: str) -> List[Dict]:
    """Extract cross-references from section text.

    Returns list of dicts:
      {source_section_path, referenced_section_path, doc_id, reference_text}
    """
    refs: List[Dict] = []
    seen: Set[tuple] = set()

    for pattern in _XREF_PATTERNS:
        for m in pattern.finditer(text):
            groups = [g.strip() for g in m.groups() if g and g.strip()]
            if not groups:
                continue
            ref_path = " ".join(groups)
            key = (source_section_path, ref_path)
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "source_section_path": source_section_path,
                "referenced_section_path": ref_path,
                "doc_id": doc_id,
                "reference_text": m.group(0)[:200],
            })

    return refs


# ── Graph class ───────────────────────────────────────────────────────────────

class CrossRefGraph:
    """Adjacency map with BFS expansion; persisted as JSON."""

    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        # graph[source_path] = [target1, target2, ...]
        self._graph: Dict[str, List[str]] = {}
        # refs list (for doc-level deletion)
        self._refs: List[Dict] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._graph = data.get("graph", {})
                self._refs = data.get("refs", [])
                logger.info(
                    "CrossRefGraph: loaded %d source nodes, %d refs",
                    len(self._graph),
                    len(self._refs),
                )
            except Exception as e:
                logger.warning("CrossRefGraph: load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.graph_path) or ".", exist_ok=True)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump({"graph": self._graph, "refs": self._refs}, f)
        except Exception as e:
            logger.warning("CrossRefGraph: save failed: %s", e)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_section_refs(self, refs: List[Dict]) -> None:
        """Merge extracted reference list into the graph and persist."""
        for r in refs:
            src = r.get("source_section_path") or ""
            tgt = r.get("referenced_section_path") or ""
            if not src or not tgt:
                continue
            targets = self._graph.setdefault(src, [])
            if tgt not in targets:
                targets.append(tgt)
            self._refs.append(r)
        self._save()

    def store_refs_in_db(self, refs: List[Dict]) -> None:
        """Persist extracted references to the section_references DB table."""
        if not refs:
            return
        try:
            with get_conn() as conn:
                for r in refs:
                    conn.execute(
                        "INSERT OR IGNORE INTO section_references (id, source_section_path, referenced_section_path, doc_id, reference_text) VALUES (?,?,?,?,?)",
                        (
                            new_id("xref"),
                            r.get("source_section_path") or "",
                            r.get("referenced_section_path") or "",
                            r.get("doc_id") or "",
                            r.get("reference_text") or "",
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.warning("CrossRefGraph: DB store failed: %s", e)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_referenced_paths(
        self,
        section_path: str,
        max_depth: int = 2,
        max_total: int = 3,
    ) -> List[str]:
        """BFS from section_path following outgoing edges up to max_depth hops.

        Returns a list of referenced section paths (deduplicated, limited to max_total).
        """
        if not section_path or section_path not in self._graph:
            return []

        visited: Set[str] = {section_path}
        results: List[str] = []
        queue = [section_path]
        depth = 0

        while queue and depth < max_depth and len(results) < max_total:
            next_queue: List[str] = []
            for path in queue:
                for target in self._graph.get(path, []):
                    if target not in visited:
                        visited.add(target)
                        results.append(target)
                        next_queue.append(target)
                        if len(results) >= max_total:
                            return results
            queue = next_queue
            depth += 1

        return results

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear_doc(self, doc_id: str) -> None:
        """Remove all references for a doc_id and rebuild the graph."""
        self._refs = [r for r in self._refs if r.get("doc_id") != doc_id]
        # Rebuild adjacency from remaining refs
        self._graph = {}
        for r in self._refs:
            src = r.get("source_section_path") or ""
            tgt = r.get("referenced_section_path") or ""
            if src and tgt:
                targets = self._graph.setdefault(src, [])
                if tgt not in targets:
                    targets.append(tgt)
        self._save()

    def clear_all(self) -> None:
        self._graph = {}
        self._refs = []
        if os.path.exists(self.graph_path):
            try:
                os.remove(self.graph_path)
            except OSError:
                pass
