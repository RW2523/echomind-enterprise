"""
TOC (Table of Contents) builder for BookRAG.

If a separate TOC document exists: parse it into a structured tree.
Else: derive TOC from section headings detected during chunking (from book_sections).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from ...core.config import settings
from ...core.db import get_conn
from .section_id import extract_section_id, section_id_from_path

logger = logging.getLogger(__name__)

TOC_PATH = os.path.join(settings.DATA_DIR, "toc_nodes.json")


def build_toc_from_sections() -> List[Dict]:
    """Build TOC nodes from book_sections (section_path, section_title).

    Each section becomes a TOC node with:
      toc_node_id, title, level, parent_id, section_ids, path_prefix, text_for_embedding
    """
    nodes: List[Dict] = []
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT section_id, section_path, section_title, full_section_text
                   FROM book_sections WHERE section_path IS NOT NULL ORDER BY section_path"""
            ).fetchall()
    except Exception as e:
        logger.warning("TOC builder: failed to read book_sections: %s", e)
        return []

    path_to_id: Dict[str, str] = {}
    for row in rows:
        sec_id, section_path, section_title, full_text = row[0], row[1] or "", row[2] or "", row[3] or ""
        if not section_path:
            continue
        sid = section_id_from_path(section_path) or extract_section_id(section_title) or sec_id
        # Infer level from path depth
        level = section_path.count(">") + 1
        parent_id = ""
        parts = [p.strip() for p in section_path.split(">")]
        if len(parts) > 1:
            parent_path = " > ".join(parts[:-1])
            parent_id = path_to_id.get(parent_path, "")
        node_id = f"toc_{sec_id}"
        path_to_id[section_path] = node_id
        # text_for_embedding: title + first 200 chars of content + codes
        text_parts = [section_title or section_path]
        if full_text:
            text_parts.append(full_text[:200].replace("\n", " "))
        if sid:
            text_parts.append(sid)
        text_for_embedding = " ".join(text_parts).strip()
        nodes.append({
            "toc_node_id": node_id,
            "title": section_title or section_path,
            "level": level,
            "parent_id": parent_id,
            "section_ids": [sid] if sid else [],
            "path_prefix": section_path,
            "section_path": section_path,
            "text_for_embedding": text_for_embedding[:2000],
        })
    logger.info("TOC builder: built %d nodes from book_sections", len(nodes))
    return nodes


def load_toc_nodes() -> List[Dict]:
    """Load TOC nodes from disk or build from sections."""
    if os.path.exists(TOC_PATH):
        try:
            with open(TOC_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                nodes = data.get("nodes", [])
                if nodes:
                    return nodes
        except Exception as e:
            logger.warning("TOC: failed to load %s: %s", TOC_PATH, e)
    return build_toc_from_sections()


def save_toc_nodes(nodes: List[Dict]) -> None:
    """Persist TOC nodes to disk."""
    try:
        os.makedirs(os.path.dirname(TOC_PATH) or ".", exist_ok=True)
        with open(TOC_PATH, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes}, f, indent=2)
    except Exception as e:
        logger.warning("TOC: failed to save: %s", e)
