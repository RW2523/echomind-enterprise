#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""EchoMind transcript storage: Postgres + Milvus indexing for RAG.

Transcripts are stored in Postgres and their text is indexed in the same
Milvus collection as documents, with source=transcript_<id> and type=transcript,
so RAG retrieval returns both document and transcript chunks.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from logger import logger


def _title_for_transcript(tid: str, echodate: str) -> str:
    """Human-readable title: date and time + short id."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", echodate)
    if m:
        date_part = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
    else:
        date_part = echodate[:16].replace("T", " ") if len(echodate) >= 16 else echodate
    short_id = (tid or "").replace("trn_", "")[:8] if tid else ""
    return f"{date_part}_{short_id}" if short_id else date_part


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranscriptStore:
    """EchoMind transcript CRUD and Milvus indexing."""

    def __init__(self, pool, vector_store, config_manager):
        self.pool = pool
        self.vector_store = vector_store
        self.config_manager = config_manager

    async def create(
        self,
        raw_text: str,
        polished_text: Optional[str] = None,
        echotag: Optional[str] = None,
        name: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a transcript in Postgres and index its text in Milvus. Returns transcript record."""
        tid = f"trn_{uuid.uuid4().hex[:20]}"
        echodate = _now_iso()
        name_val = (name or "").strip() or None
        location_val = (location or "").strip() or "default"
        title = name_val or _title_for_transcript(tid, echodate)
        tags_list = [t.strip() for t in tags] if tags else []
        tags_list = [t for t in tags_list if t][:16]
        echotag_val = (echotag or "").strip() or (",".join(tags_list) if tags_list else (name_val or "transcript"))

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO transcripts (id, title, raw_text, polished_text, tags_json, echotag, echodate, name, location)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                tid,
                title,
                raw_text,
                polished_text,
                json.dumps(tags_list),
                echotag_val,
                echodate,
                name_val,
                location_val,
            )

        # Index in Milvus so RAG can retrieve transcript chunks
        source_name = f"transcript_{tid}"
        index_text = raw_text + ("\n\n" + (polished_text or "")) if polished_text else raw_text
        docs = [
            Document(
                page_content=index_text,
                metadata={
                    "source": source_name,
                    "filename": title,
                    "type": "transcript",
                    "transcript_id": tid,
                    "tags": tags_list,
                    "echotag": echotag_val,
                    "echodate": echodate,
                },
            )
        ]
        try:
            self.vector_store.index_documents(docs)
            # Register as a source so RAG can include it when selected
            config = self.config_manager.read_config()
            if hasattr(config, "sources") and source_name not in config.sources:
                config.sources.append(source_name)
                self.config_manager.write_config(config)
        except Exception as e:
            logger.warning("Failed to index transcript %s in Milvus: %s", tid, e)

        return {
            "transcript_id": tid,
            "title": title,
            "name": name_val,
            "location": location_val,
            "tags": tags_list,
            "echotag": echotag_val,
            "echodate": echodate,
            "created_at": echodate,
        }

    async def list_transcripts(
        self,
        since_iso: Optional[str] = None,
        last_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """List transcripts, optionally filtered by time."""
        extra = ""
        params: List[Any] = []
        if last_hours is not None:
            from datetime import timedelta
            since_dt = datetime.now(timezone.utc) - timedelta(hours=float(last_hours))
            since_iso = since_dt.isoformat()
        if since_iso:
            extra = " AND created_at >= $1"
            params.append(since_iso)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, title, tags_json, echotag, created_at, raw_text, polished_text, name, location
                FROM transcripts WHERE 1=1 {extra}
                ORDER BY created_at DESC
                """,
                *params,
            )

        out = []
        for r in rows:
            tags = []
            if r["tags_json"]:
                try:
                    tags = json.loads(r["tags_json"]) if isinstance(r["tags_json"], str) else (r["tags_json"] or [])
                except Exception:
                    pass
            created_at = (r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"])) if r["created_at"] else ""
            title = r["title"] or (created_at[:16].replace("T", " ") + "_" + (r["id"] or "").replace("trn_", "")[:8]) if created_at else (r["id"] or "")
            out.append({
                "id": r["id"],
                "title": title,
                "tags": tags,
                "echotag": r["echotag"] or "",
                "created_at": created_at,
                "name": r["name"],
                "location": r["location"],
            })
        return out

    async def get_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Get one transcript by id."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, title, raw_text, polished_text, tags_json, echotag, echodate, created_at, name, location FROM transcripts WHERE id = $1",
                transcript_id,
            )
        if not row:
            return None
        tags = []
        if row["tags_json"]:
            try:
                tags = json.loads(row["tags_json"]) if isinstance(row["tags_json"], str) else (row["tags_json"] or [])
            except Exception:
                pass
        created_at = (row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"])) if row["created_at"] else ""
        return {
            "id": row["id"],
            "title": row["title"],
            "raw_text": row["raw_text"],
            "polished_text": row["polished_text"],
            "tags": tags,
            "echotag": row["echotag"] or "",
            "echodate": row["echodate"] or "",
            "created_at": created_at,
            "name": row["name"],
            "location": row["location"],
        }

    async def update_transcript(
        self,
        transcript_id: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update transcript metadata."""
        updates = []
        params: List[Any] = []
        if name is not None:
            updates.append("name = $%d" % (len(params) + 1))
            params.append((name or "").strip() or None)
        if location is not None:
            updates.append("location = $%d" % (len(params) + 1))
            params.append((location or "").strip() or "default")
        if tags is not None:
            updates.append("tags_json = $%d" % (len(params) + 1))
            params.append(json.dumps([t.strip() for t in tags if (t or "").strip()][:16]))
        if not updates:
            return await self.get_transcript(transcript_id)
        params.append(transcript_id)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE transcripts SET " + ", ".join(updates) + " WHERE id = $%d" % len(params),
                *params,
            )
        return await self.get_transcript(transcript_id)

    async def append_chunk(self, transcript_id: str, chunk_text: str) -> None:
        """Append text to an existing transcript (e.g. auto-store). Updates raw_text and updated_at."""
        if not (chunk_text or "").strip():
            return
        updated = _now_iso()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT raw_text FROM transcripts WHERE id = $1", transcript_id)
            if not row:
                return
            existing = (row["raw_text"] or "").strip()
            new_raw = (existing + "\n\n" + chunk_text.strip()).strip() if existing else chunk_text.strip()
            await conn.execute(
                "UPDATE transcripts SET raw_text = $1, updated_at = $2 WHERE id = $3",
                new_raw,
                updated,
                transcript_id,
            )
