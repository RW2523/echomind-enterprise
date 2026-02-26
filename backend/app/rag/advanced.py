from __future__ import annotations
import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, AsyncIterator, Tuple, Optional

import faiss
import numpy as np

from ..core.config import settings
from ..core.db import get_conn
from .index import index
from .llm import OpenAICompatChat

logger = logging.getLogger(__name__)
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

CONTEXT_WINDOW_VALUES = ("24h", "48h", "1w", "all")

# Deterministic message when document/transcript intent but retrieval is insufficient (no hallucination fallback to general chat).
INSUFFICIENT_CONTEXT_MSG = "The provided documents do not contain this information."

def _parse_iso_date(created_at: Optional[str]) -> Optional[datetime]:
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return None

def _filter_hits_by_context_window(hits: List[Dict], context_window: str) -> List[Dict]:
    """
    Keep only hits whose document created_at falls within context_window (24h, 48h, 1w). 'all' = no filter.
    Policy: when context_window != 'all', batch-fetch created_at via _get_doc_info_for_hits; hits without
    doc_id or without created_at are excluded (treated as out-of-window) to avoid incorrect inclusion.
    """
    if not context_window or context_window == "all" or not hits:
        return hits
    now = datetime.now(timezone.utc)
    if context_window == "24h":
        cutoff = now - timedelta(hours=24)
    elif context_window == "48h":
        cutoff = now - timedelta(hours=48)
    elif context_window == "1w":
        cutoff = now - timedelta(days=7)
    else:
        return hits
    doc_created, _ = _get_doc_info_for_hits(hits)
    filtered = []
    for h in hits:
        doc_id = (h.get("source") or {}).get("doc_id")
        if not doc_id:
            continue
        created_at = doc_created.get(doc_id)
        if created_at is None:
            continue
        dt = _parse_iso_date(created_at)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt >= cutoff:
            filtered.append(h)
    return filtered

# Max chars for parent chunk expansion; config overrides to limit context domination (RAG_PARENT_CONTEXT_MAX_CHARS).
def _parent_context_max_chars() -> int:
    return getattr(settings, "RAG_PARENT_CONTEXT_MAX_CHARS", 2400) or 2400

# General conversation: greetings, thanks, small talk. Skip RAG entirely—answer directly with LLM.
_GENERAL_PHRASES = frozenset({
    "hi", "hello", "hey", "hey there", "hi there", "howdy",
    "thanks", "thank you", "thanks!", "thank you!", "thx", "ty",
    "how are you", "how are you?", "what's up", "whats up", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "see ya", "take care",
    "ok", "okay", "yes", "no", "sure", "alright",
    "good", "great", "cool", "nice", "perfect", "awesome",
    "please", "excuse me", "sorry", "my bad",
    "what's the weather", "tell me a joke", "how's it going",
})

# Short 1–2 word inputs that are likely real queries, not small talk (do not skip RAG).
_SHORT_QUERY_WORDS = frozenset({
    "pricing", "price", "cost", "errors", "setup", "status", "help", "login", "docs", "guide",
    "tutorial", "api", "support", "summary", "overview", "list", "find", "search", "export",
})


def _is_general_conversation(question: str) -> bool:
    """
    True for obvious greetings, thanks, or small talk. Skip RAG—answer directly with LLM.
    Uses phrase matching for speed; semantic search handles gray-area questions.
    """
    t = (question or "").strip().lower()
    if not t:
        return True
    if t in _GENERAL_PHRASES:
        return True
    words = t.split()
    if len(words) <= 2:
        if t in _SHORT_QUERY_WORDS or any(w in _SHORT_QUERY_WORDS for w in words):
            return False
        if "?" not in t and not any(w in t for w in ("what", "which", "when", "where", "who", "how", "why", "can", "does", "is", "are", "do")):
            return True
    return False


def _dedupe_best(items: List[Dict]) -> List[Dict]:
    best={}
    for it in items:
        cid=it["chunk_id"]
        if cid not in best or it["score"]>best[cid]["score"]:
            best[cid]=it
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)

RRF_K = 60  # reciprocal rank fusion constant


def _weighted_rrf(
    dense_hits_per_query: List[List[Dict]],
    sparse_hits_per_query: List[List[Dict]],
    k: int,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict]:
    """Merge dense + sparse with weighted RRF. Higher dense_weight favors semantic similarity; sparse_weight favors keyword match. Improves precision/recall balance."""
    fused: Dict[str, Dict] = {}
    for hit_list in dense_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = dense_weight * (1.0 / (RRF_K + rank))
            if cid not in fused:
                fused[cid] = {"chunk_id": cid, "rrf": 0.0, "dense_score": 0.0, "text": h["text"], "source": h["source"]}
            fused[cid]["rrf"] += rrf
            fused[cid]["dense_score"] = max(fused[cid]["dense_score"], h["score"])
    for hit_list in sparse_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = sparse_weight * (1.0 / (RRF_K + rank))
            if cid not in fused:
                fused[cid] = {"chunk_id": cid, "rrf": 0.0, "dense_score": 0.0, "text": h["text"], "source": h["source"]}
            fused[cid]["rrf"] += rrf
    sorted_ids = sorted(fused.keys(), key=lambda cid: fused[cid]["rrf"], reverse=True)[:k]
    out = []
    max_rrf = fused[sorted_ids[0]]["rrf"] if sorted_ids else 1.0
    for cid in sorted_ids:
        f = fused[cid]
        score = f["dense_score"] if f["dense_score"] > 0 else min(1.0, 0.3 + 0.7 * (f["rrf"] / max_rrf))
        out.append({"chunk_id": cid, "score": score, "text": f["text"], "source": f["source"]})
    return out


def _reciprocal_rank_fusion(
    dense_hits_per_query: List[List[Dict]],
    sparse_hits_per_query: List[List[Dict]],
    k: int,
) -> List[Dict]:
    """Legacy equal-weight RRF (used when weighted RRF weights not configured)."""
    return _weighted_rrf(dense_hits_per_query, sparse_hits_per_query, k, dense_weight=0.5, sparse_weight=0.5)


def _doc_epoch_from_meta(meta: dict, doc_created: Optional[str]) -> Optional[float]:
    """Get Unix epoch for a document from meta (epoch) or fallback to echodate/created_at. Returns None if unavailable."""
    epoch = meta.get("epoch")
    if epoch is not None:
        try:
            return float(epoch)
        except (TypeError, ValueError):
            pass
    ts = meta.get("echodate") or doc_created
    if ts:
        dt = _parse_iso_date(ts)
        if dt:
            return dt.timestamp()
    return None


def _get_doc_info_for_hits(hits: List[Dict]) -> Tuple[Dict[str, Optional[str]], Dict[str, dict]]:
    """Fetch created_at and meta_json for each unique doc_id in hits. Returns (doc_id -> created_at, doc_id -> meta)."""
    doc_ids = list({(h.get("source") or {}).get("doc_id") for h in hits if (h.get("source") or {}).get("doc_id")})
    created_map: Dict[str, Optional[str]] = {}
    meta_map: Dict[str, dict] = {}
    if not doc_ids:
        return created_map, meta_map
    with get_conn() as conn:
        for doc_id in doc_ids:
            row = conn.execute("SELECT created_at, meta_json FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row:
                created_map[doc_id] = row[0]
                try:
                    meta_map[doc_id] = json.loads(row[1]) if isinstance(row[1], str) else (row[1] or {})
                except Exception:
                    meta_map[doc_id] = {}
            else:
                created_map[doc_id] = None
                meta_map[doc_id] = {}
    return created_map, meta_map


def _apply_time_decay(hits: List[Dict], doc_created: Dict[str, Optional[str]], halflife_days: float) -> List[Dict]:
    """True half-life decay: decay = exp(-ln(2) * age_days / halflife_days). At age_days = halflife_days, score is halved. halflife_days=0 means no decay."""
    if halflife_days <= 0 or not hits:
        return hits
    now = datetime.now(timezone.utc)
    ln2 = math.log(2)
    out = []
    for h in hits:
        doc_id = (h.get("source") or {}).get("doc_id")
        created = doc_created.get(doc_id) if doc_id else None
        dt = _parse_iso_date(created)
        if dt is None:
            out.append(h)
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (now - dt).total_seconds() / 86400
        decay = 1.0 if age_days <= 0 else max(0.1, math.exp(-ln2 * age_days / halflife_days))
        out.append({**h, "score": h["score"] * decay})
    return out


def _apply_tag_boost(hits: List[Dict], query: str, doc_meta: Dict[str, dict], factor: float) -> List[Dict]:
    """Boost score for transcript chunks whose tags overlap query terms. Improves recall when tags describe content."""
    if factor <= 0 or not query:
        return hits
    q_lower = query.lower()
    q_words = set(re.findall(r"[a-z0-9]{2,}", q_lower))
    out = []
    for h in hits:
        doc_id = (h.get("source") or {}).get("doc_id")
        meta = doc_meta.get(doc_id, {}) if doc_id else {}
        if meta.get("type") != "transcript":
            out.append(h)
            continue
        tags = meta.get("tags") or []
        if not tags:
            out.append(h)
            continue
        tag_set = set()
        for t in tags:
            tag_set.update(re.findall(r"[a-z0-9]{2,}", (t or "").lower()))
        overlap = len(q_words & tag_set) / max(1, len(q_words))
        boost = 1.0 + factor * overlap
        out.append({**h, "score": min(1.0, h["score"] * boost)})
    return out


def _is_authoritative(source: dict) -> bool:
    """True if chunk is from an uploaded document (PDF/DOCX/PPTX), not a transcript. Used for tie-break preference."""
    ft = (source or {}).get("filetype") or ""
    fn = (source or {}).get("filename") or ""
    if (fn.startswith("transcript_") and ft == "text") or ft == "text":
        return False
    return ft in ("pdf", "docx", "pptx", "txt")


def _prefer_authoritative_sort(hits: List[Dict]) -> List[Dict]:
    """Sort by (score desc, authoritative first). When scores are close, authoritative docs win (reduces transcript noise)."""
    return sorted(hits, key=lambda h: (-h["score"], 0 if _is_authoritative(h.get("source") or {}) else 1))


# --- Deterministic query expansion (pre-LLM): typos, quoted phrases, TOC variants ---
_COMMON_TYPOS = {"mathew": "matthew", "matthew": "matthew", "meriton": "merton", "outler": "outlier", "outlers": "outliers"}


def get_deterministic_query_variants(question: str) -> List[str]:
    """Pre-LLM query variants: fix typos, add quoted phrase for named concepts, add TOC/structure terms when relevant. Return list to prepend to LLM-generated queries."""
    if not (question or "").strip():
        return []
    q = question.strip()
    q_lower = q.lower()
    out: List[str] = []
    # Typo fixes: produce corrected query
    for wrong, right in _COMMON_TYPOS.items():
        if wrong in q_lower:
            out.append(q_lower.replace(wrong, right))
    # Quoted phrase for named concepts (e.g. "matthew effect" -> exact phrase search helps BM25)
    if "matthew" in q_lower and "effect" in q_lower:
        out.append('"matthew effect"')
    if "table of contents" in q_lower or "toc" in q_lower or "contents" in q_lower or "chapters" in q_lower or "chapter list" in q_lower:
        out.extend(["table of contents", "contents", "chapter"])
    return out


def is_toc_chapters_query(question: str) -> bool:
    """True if the user is asking for chapters, table of contents, or section list. Used for TOC guardrail and book retrieval tuning."""
    if not (question or "").strip():
        return False
    t = question.strip().lower()
    toc_phrases = (
        "table of contents", "toc", "list of chapters", "chapter list", "list the chapters",
        "what are the chapters", "chapters in", "contents of the book", "book structure",
        "section list", "list sections", "list of sections", "chapter titles", "section titles",
    )
    if any(p in t for p in toc_phrases):
        return True
    words = set(re.findall(r"[a-z0-9]+", t))
    if "chapters" in words and ("list" in words or "what" in words or "name" in words or "which" in words):
        return True
    if "contents" in words and ("list" in words or "table" in words):
        return True
    return False


def _get_original_blocks_for_toc(hits: List[Dict]) -> List[str]:
    """Build list of original (uncompressed) chunk texts in display order for TOC guardrail. Avoids false negatives when compression strips TOC signals."""
    out: List[str] = []
    seen_parent: set = set()
    for h in hits:
        src = h.get("source") or {}
        parent_chunk_id = src.get("parent_chunk_id") if isinstance(src.get("parent_chunk_id"), str) else None
        if parent_chunk_id and parent_chunk_id not in seen_parent:
            seen_parent.add(parent_chunk_id)
            parent_text = _get_chunk_text(parent_chunk_id)
            if parent_text:
                out.append(parent_text)
        out.append((h.get("text") or "").strip())
    return out


def has_toc_signals_in_context(blocks: List[str]) -> bool:
    """True if the combined context blocks contain TOC-like signals (Contents, CHAPTER N, roman numerals list, etc.). Use original (uncompressed) blocks for guardrail to avoid false negatives."""
    combined = " ".join((b or "" for b in blocks)).lower()
    if "table of contents" in combined or "contents" in combined and ("chapter" in combined or "part " in combined):
        return True
    if re.search(r"chapter\s+[0-9]+", combined, re.I) or re.search(r"chapter\s+one|two|three|four|five|six|seven|eight|nine|ten", combined, re.I):
        return True
    if re.search(r"\b(i{1,3}|iv|v|vi{0,3}|ix|x|xi|xiv|xv)\s+[a-z]", combined):
        return True
    if "part i" in combined or "part ii" in combined or "part 1" in combined or "part 2" in combined:
        return True
    return False


def _parse_last_n_transcripts(question: str) -> Optional[int]:
    """If question asks for 'last N transcripts' or 'summary of last N', return N. Otherwise None."""
    if not (question or "").strip():
        return None
    t = question.lower()
    # "last 15 transcripts", "last 15 transcript", "summary of last 15", "past 10 transcripts"
    m = re.search(r"(?:last|past|recent)\s+(\d+)\s*(?:transcript|recording|meeting)s?", t)
    if m:
        return min(100, max(1, int(m.group(1))))
    m = re.search(r"(?:summary|recap)\s+(?:of\s+)?(?:the\s+)?last\s+(\d+)", t)
    if m:
        return min(100, max(1, int(m.group(1))))
    return None


def _parse_last_time_window(question: str) -> Optional[timedelta]:
    """If question asks for 'last N hours', 'last N days', or 'last N minutes', return timedelta. Used to filter transcript hits by time."""
    if not (question or "").strip():
        return None
    t = question.lower()
    # "last 5 minutes", "last 5 mins", "past 10 minutes"
    m = re.search(r"(?:last|past|for\s+last)\s+(\d+)\s*(minute|minutes?|mins?)", t)
    if m:
        return timedelta(minutes=min(10080, max(1, int(m.group(1)))))  # cap 7 days
    # "last 15 hours", "for last 15 hours", "past 2 hours", "last 3 days"
    m = re.search(r"(?:last|past|for\s+last)\s+(\d+)\s*(hour|hours?)", t)
    if m:
        return timedelta(hours=min(720, max(1, int(m.group(1)))))
    m = re.search(r"(?:last|past|for\s+last)\s+(\d+)\s*(day|days?)", t)
    if m:
        return timedelta(days=min(365, max(1, int(m.group(1)))))
    return None


# Weekday names for "last Monday" etc. (Monday=0, Sunday=6)
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
_WEEKDAYS_SHORT = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _has_explicit_time_mention(question: str) -> bool:
    """True if the question explicitly mentions a time period, date, or temporal reference. Used to decide whether to apply time-range filtering."""
    if not (question or "").strip():
        return False
    t = question.lower()
    # Relative: last N hours/days/minutes, past N, for N hours
    if re.search(r"(?:last|past|for)\s+(\d+)\s*(?:hour|minute|day|min|hr)s?", t):
        return True
    if re.search(r"\b\d+\s*(?:hour|hours?|minute|minutes?|day|days?)\b", t):
        return True
    # Named: yesterday, today, tomorrow
    if re.search(r"\b(yesterday|today|tomorrow)\b", t):
        return True
    # "today at 2 pm", "at 2pm"
    if re.search(r"(?:today|yesterday)\s+at\s+\d+", t) or re.search(r"\bat\s+\d+\s*(?:am|pm|o'clock)?", t):
        return True
    # Weekday: last Monday, on Monday
    if re.search(r"\b(?:last|on|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t, re.I):
        return True
    # Absolute date: 2 Feb 2026, Feb 2 2026, 2026-02-02
    if re.search(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b", t, re.I):
        return True
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\s*,?\s*\d{4}\b", t, re.I):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", t):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", t):
        return True
    return False


def _parse_time_range_from_question(question: str) -> Optional[Tuple[float, float]]:
    """
    Parse time range from question using semantic + context rules. Returns (start_epoch, end_epoch) or None.
    Only returns a range when time is explicitly mentioned. Used for transcript retrieval filtering.
    Supports: last N hours/days/minutes, 48 hours, yesterday, today, today at 2 pm, last Monday, 2 Feb 2026.
    """
    if not (question or "").strip() or not _has_explicit_time_mention(question):
        return None
    t = question.lower().strip()
    now = datetime.now(timezone.utc)
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    # 1. Relative: "last 2 hours", "past 48 hours", "for last 3 days", "summary for 2 hours"
    m = re.search(r"(?:last|past|for\s+last|for)\s+(\d+)\s*(minute|minutes?|mins?)", t)
    if m:
        n = min(10080, max(1, int(m.group(1))))
        start_dt = now - timedelta(minutes=n)
        end_dt = now
    if start_dt is None:
        m = re.search(r"(?:last|past|for\s+last|for)\s+(\d+)\s*(hour|hours?)", t)
        if m:
            n = min(720, max(1, int(m.group(1))))
            start_dt = now - timedelta(hours=n)
            end_dt = now
    if start_dt is None:
        m = re.search(r"(?:last|past|for\s+last|for)\s+(\d+)\s*(day|days?)", t)
        if m:
            n = min(365, max(1, int(m.group(1))))
            start_dt = now - timedelta(days=n)
            end_dt = now
    # "48 hours" without "last" (e.g. "summary for 48 hours")
    if start_dt is None and re.search(r"\b(\d+)\s*(?:hour|hours?)\b", t):
        m = re.search(r"\b(\d+)\s*(?:hour|hours?)\b", t)
        if m:
            n = min(720, max(1, int(m.group(1))))
            start_dt = now - timedelta(hours=n)
            end_dt = now
    if start_dt is None and re.search(r"\b(\d+)\s*(?:day|days?)\b", t):
        m = re.search(r"\b(\d+)\s*(?:day|days?)\b", t)
        if m:
            n = min(365, max(1, int(m.group(1))))
            start_dt = now - timedelta(days=n)
            end_dt = now

    # 2. "today at 2 pm" - from 2pm today until now
    if start_dt is None and re.search(r"today\s+at\s+(\d{1,2})\s*(?:(am|pm))?", t):
        m = re.search(r"today\s+at\s+(\d{1,2})\s*(?:(am|pm))?", t)
        if m:
            hour = int(m.group(1)) % 24
            pm = (m.group(2) or "").lower() == "pm"
            am = (m.group(2) or "").lower() == "am"
            if pm and hour < 12:
                hour += 12
            elif am and hour == 12:
                hour = 0
            start_dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            end_dt = now
            if start_dt > end_dt:
                start_dt = start_dt - timedelta(days=1)

    # 3. "yesterday at 2 pm" - from 2pm to end of yesterday
    if start_dt is None and re.search(r"yesterday\s+at\s+(\d{1,2})\s*(?:(am|pm))?", t):
        m = re.search(r"yesterday\s+at\s+(\d{1,2})\s*(?:(am|pm))?", t)
        if m:
            hour = int(m.group(1)) % 24
            pm = (m.group(2) or "").lower() == "pm"
            am = (m.group(2) or "").lower() == "am"
            if pm and hour < 12:
                hour += 12
            elif am and hour == 12:
                hour = 0
            start_dt = (now - timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            end_dt = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

    # 4. "today" - from midnight today until now
    if start_dt is None and re.search(r"\btoday\b", t):
        start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = now

    # 5. "yesterday" - full day
    if start_dt is None and re.search(r"\byesterday\b", t):
        start_dt = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    # 6. "last Monday" (or any weekday)
    if start_dt is None:
        for name, wd in {**_WEEKDAYS, **_WEEKDAYS_SHORT}.items():
            if re.search(rf"last\s+{name}\b", t):
                today_wd = now.weekday()  # Monday=0
                days_back = (today_wd - wd) % 7
                if days_back == 0:
                    days_back = 7  # "last Monday" when today is Monday = previous Monday
                start_dt = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                end_dt = start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                break

    # 7. Absolute date: "2 Feb 2026", "Feb 2 2026", "2026-02-02"
    if start_dt is None:
        # "2 Feb 2026" or "2 February 2026"
        m = re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b", t, re.I)
        if m:
            try:
                from datetime import date
                months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                month = months.get(m.group(2).lower()[:3], 1)
                d = date(int(m.group(3)), month, min(28, int(m.group(1))))
                start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                end_dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
            except (ValueError, KeyError):
                pass
        if start_dt is None:
            # "Feb 2 2026" or "February 2, 2026"
            m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\s*,?\s*(\d{4})\b", t, re.I)
            if m:
                try:
                    from datetime import date
                    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                    month = months.get(m.group(1).lower()[:3], 1)
                    d = date(int(m.group(3)), month, min(28, int(m.group(2))))
                    start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                    end_dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
                except (ValueError, KeyError):
                    pass
        if start_dt is None:
            # "2026-02-02"
            m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
            if m:
                try:
                    from datetime import date
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                    end_dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    pass
        if start_dt is None:
            # "12/1/2025", "10/10/2025" (MM/DD/YYYY)
            m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t)
            if m:
                try:
                    from datetime import date
                    d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                    start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                    end_dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    pass

    if start_dt is None or end_dt is None:
        return None
    return (start_dt.timestamp(), end_dt.timestamp())


def _parse_location_from_question(question: str) -> Optional[str]:
    """If question mentions a location (e.g. 'in office', 'at home', 'in the meeting room'), return it for transcript filtering."""
    if not (question or "").strip():
        return None
    t = question.lower()
    # "in office", "at office", "from office", "in the office", "in office"
    m = re.search(r"(?:in|at|from)\s+(?:the\s+)?([a-z0-9][a-z0-9\s\-]{1,40}?)(?:\s+transcript|\s+recording|\s+meeting|$|\.|\?)", t)
    if m:
        loc = m.group(1).strip()
        if len(loc) >= 2 and loc not in ("a", "an", "the", "my", "last"):
            return loc
    # "office" alone when nearby "transcript" or "last"
    if re.search(r"\b(office|home|meeting\s*room|conference)\b", t):
        for word in ("office", "home", "meeting room", "meeting room", "conference"):
            if word in t:
                return word
    return None


def _parse_latest_or_recent_transcript(question: str) -> bool:
    """True when user asks for 'latest transcript', 'most recent transcript', 'recent transcript' without a number (no time/location required)."""
    if not (question or "").strip():
        return False
    t = question.lower()
    if not re.search(r"\b(transcript|recording|meeting|conversation)\b", t):
        return False
    return bool(
        re.search(r"\b(latest|most\s+recent|recent)\s+(?:transcript|recording|meeting|conversation)", t)
        or re.search(r"(?:summary|recap|give)\s+(?:me\s+)?(?:the\s+)?(?:latest|most\s+recent|recent)\b", t)
        or re.search(r"(?:the\s+)?(latest|most\s+recent)\b", t)
    )


def _get_most_recent_transcript_id() -> Optional[str]:
    """Return transcript id of the most recently updated transcript (one row per session)."""
    with get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT id FROM transcripts ORDER BY updated_at DESC LIMIT 1",
                ()
            ).fetchone()
        except Exception:
            row = conn.execute(
                "SELECT id FROM transcripts ORDER BY created_at DESC LIMIT 1",
                ()
            ).fetchone()
    return row[0] if row else None


def _get_recent_transcript_doc_ids(n: int) -> set:
    """Return set of document ids for the N most recent transcript documents (filename like 'transcript_%')."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM documents WHERE filename LIKE 'transcript_%' ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()
    return {r[0] for r in rows if r}


async def _rerank_hits(question: str, hits: List[Dict], top_n: int) -> List[Dict]:
    """Optional LLM rerank: score each chunk 0-10 for relevance to the question, then reorder. Keeps top_n. Adds latency."""
    if not hits or top_n <= 0:
        return hits
    # Batch prompt: list chunks and ask for scores (one line per score, same order). Reduces round-trips.
    prompt_lines = [f"Question: {question}\n"]
    for i, h in enumerate(hits):
        snippet = (h.get("text") or "")[:400].replace("\n", " ")
        prompt_lines.append(f"[{i}] {snippet}")
    prompt_lines.append("\nScore each [i] 0-10 for relevance. Reply with one integer per line, in order 0..n-1.")
    try:
        reply = await chat.chat(
            [{"role": "system", "content": "You score relevance of each excerpt to the question. Reply with only the scores, one per line, same order as the excerpts."}, {"role": "user", "content": "\n".join(prompt_lines)}],
            temperature=0.0,
            max_tokens=min(200, len(hits) * 4),
        )
        scores_list = []
        for line in (reply or "").splitlines():
            line = line.strip()
            m = re.search(r"\b(\d+)\b", line)
            if m:
                scores_list.append(min(10, max(0, int(m.group(1)))))
        if len(scores_list) >= len(hits):
            scored = [(hits[i], scores_list[i] / 10.0) for i in range(len(hits))]
            scored.sort(key=lambda x: -x[1])
            return [{"chunk_id": h["chunk_id"], "score": s, "text": h["text"], "source": h["source"]} for h, s in scored[:top_n]]
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
    return hits[:top_n]


async def retrieve_single_query(question: str, k: int, context_window: str = "all") -> List[Dict]:
    if not (question or "").strip():
        return []
    hits = await index.search(question.strip(), max(k, 4))
    hits = _filter_hits_by_context_window(hits, context_window or "all")
    doc_created, doc_meta = _get_doc_info_for_hits(hits)
    halflife = getattr(settings, "RAG_TIME_DECAY_HALFLIFE_DAYS", 0) or 0
    if halflife > 0:
        hits = _apply_time_decay(hits, doc_created, halflife)
    return hits[:k]


async def retrieve_semantic_first(
    question: str,
    k: int,
    context_window: str = "all",
) -> Tuple[str, List[Dict]]:
    """
    Embedding-first routing: search transcript first, then document. No LLM intent classification.
    Returns (source_type, hits) where source_type is "transcript" | "document" | "general".
    If transcript has relevant hits (best_score >= threshold) → use transcript.
    Else if document has relevant hits → use document.
    Else → general (empty hits).
    """
    if not (question or "").strip():
        return ("general", [])
    q = question.strip()
    k_per = max(k, 4)
    dense_w = getattr(settings, "RAG_DENSE_RRF_WEIGHT", 0.6)
    sparse_w = getattr(settings, "RAG_SPARSE_RRF_WEIGHT", 0.4)

    # 1. Embed query once (reused for both transcript and document dense search)
    qv = await index.emb.embed([q])
    qv = np.array(qv, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(qv)

    # 2. Run transcript and document search in parallel (dense + sparse each)
    transcript_dense_task = index.search_transcript_only(q, k_per, query_vector=qv)
    doc_dense_task = index.search_document_only(q, k_per, query_vector=qv)
    transcript_sparse_task = asyncio.to_thread(index.transcript_sparse.search, q, k_per)
    doc_sparse_task = asyncio.to_thread(index.search_document_only_sparse, q, k_per)

    transcript_dense, doc_dense, transcript_sparse, doc_sparse = await asyncio.gather(
        transcript_dense_task, doc_dense_task, transcript_sparse_task, doc_sparse_task
    )

    # 3. Merge dense + sparse with RRF for each source
    transcript_hits = _weighted_rrf(
        [transcript_dense], [transcript_sparse], k_per, dense_weight=dense_w, sparse_weight=sparse_w
    )
    transcript_hits = _filter_hits_by_context_window(transcript_hits, context_window or "all")
    doc_created_t, doc_meta_t = _get_doc_info_for_hits(transcript_hits)
    halflife = getattr(settings, "RAG_TIME_DECAY_HALFLIFE_DAYS", 0) or 0
    if halflife > 0:
        transcript_hits = _apply_time_decay(transcript_hits, doc_created_t, halflife)
    if getattr(settings, "RAG_TAG_BOOST_ENABLED", False):
        tag_factor = getattr(settings, "RAG_TAG_BOOST_FACTOR", 0.08)
        transcript_hits = _apply_tag_boost(transcript_hits, q, doc_meta_t, tag_factor)
    # Transcript-specific filters (last N, time range, location, latest)
    # Time filter only when user explicitly mentions time (e.g. "last 2 hours", "yesterday", "2 Feb 2026")
    n = _parse_last_n_transcripts(q)
    if n is not None and n > 0:
        recent_ids = _get_recent_transcript_doc_ids(n)
        if recent_ids:
            transcript_hits = [h for h in transcript_hits if (h.get("source") or {}).get("doc_id") in recent_ids]
    time_range = _parse_time_range_from_question(q)
    location_filter = _parse_location_from_question(q)
    latest_only = _parse_latest_or_recent_transcript(q)
    if time_range is not None or location_filter:
        start_epoch, end_epoch = time_range if time_range else (None, None)
        filtered = []
        for h in transcript_hits:
            doc_id = (h.get("source") or {}).get("doc_id")
            if not doc_id:
                continue
            meta = doc_meta_t.get(doc_id) or {}
            if time_range is not None and start_epoch is not None and end_epoch is not None:
                doc_epoch = _doc_epoch_from_meta(meta, doc_created_t.get(doc_id))
                if doc_epoch is None:
                    continue
                if not (start_epoch <= doc_epoch <= end_epoch):
                    continue
            if location_filter:
                doc_location = (meta.get("location") or "").strip().lower()
                loc_lower = location_filter.strip().lower()
                if not doc_location or (loc_lower not in doc_location and doc_location not in loc_lower):
                    continue
            filtered.append(h)
        transcript_hits = filtered
    elif latest_only:
        recent_tid = _get_most_recent_transcript_id()
        if recent_tid:
            filtered = [
                h for h in transcript_hits
                if (doc_meta_t.get((h.get("source") or {}).get("doc_id")) or {}).get("transcript_id") == recent_tid
            ]
            if filtered:
                transcript_hits = filtered
    transcript_hits = transcript_hits[:k]

    # 4. Document hits (already fetched in parallel above)
    document_hits = _weighted_rrf(
        [doc_dense], [doc_sparse], k_per, dense_weight=dense_w, sparse_weight=sparse_w
    )
    document_hits = _filter_hits_by_context_window(document_hits, context_window or "all")
    doc_created_d, doc_meta_d = _get_doc_info_for_hits(document_hits)
    if halflife > 0:
        document_hits = _apply_time_decay(document_hits, doc_created_d, halflife)
    if getattr(settings, "RAG_PREFER_AUTHORITATIVE", False):
        document_hits = _prefer_authoritative_sort(document_hits)
    document_hits = document_hits[:k]

    # 5. Pick source by semantic relevance (best score)
    transcript_best = transcript_hits[0]["score"] if transcript_hits else 0.0
    document_best = document_hits[0]["score"] if document_hits else 0.0
    threshold = settings.RAG_RELEVANCE_THRESHOLD

    if transcript_hits and transcript_best >= threshold:
        logger.info("RAG semantic-first: transcript wins (score=%.3f) question=%s", transcript_best, (q[:80] + "…") if len(q) > 80 else q)
        return ("transcript", transcript_hits)
    if document_hits and document_best >= threshold:
        logger.info("RAG semantic-first: document wins (score=%.3f) question=%s", document_best, (q[:80] + "…") if len(q) > 80 else q)
        return ("document", document_hits)
    logger.info("RAG semantic-first: no relevant hits (transcript=%.3f, document=%.3f) → general", transcript_best, document_best)
    return ("general", [])


def _get_chunk_text(chunk_id: str) -> str | None:
    """Fetch chunk text by id from DB (for parent expansion). Returns None if not found."""
    if not chunk_id:
        return None
    with get_conn() as conn:
        row = conn.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else None


def _rag_sentences(text: str) -> List[str]:
    """Split text into sentences for RAG dedupe only (simple: by ., !, ?). Distinct from chunking._sentences to avoid collision."""
    if not (text or "").strip():
        return []
    s = re.sub(r"[.!?]+\s*", "\n", text)
    return [x.strip() for x in s.splitlines() if x.strip()]


def _word_set(s: str) -> set:
    """Normalized word set for overlap check."""
    return set(re.findall(r"[a-z0-9]{2,}", (s or "").lower()))


# Stopwords to exclude when deciding "chunk contains key query terms" for verbatim inclusion.
_QUERY_TERM_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "about", "into", "through",
    "what", "which", "when", "where", "who", "how", "why", "does", "say", "said", "anything", "something",
})


def _key_query_terms(question: str, min_len: int = 3) -> set:
    """Extract significant query terms (e.g. 'matthew', 'effect') for verbatim-chunk bypass and evidence grounding."""
    words = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    return {w for w in words if len(w) >= min_len and w not in _QUERY_TERM_STOP}


def _dedupe_overlapping_sentences(blocks: List[str], overlap_ratio: float) -> List[str]:
    """Remove sentences from later blocks that overlap too much with already-seen content. Reduces redundancy in context."""
    if overlap_ratio <= 0 or not blocks:
        return blocks
    seen_word_sets: List[set] = []
    out = []
    for block in blocks:
        sentences = _rag_sentences(block)
        kept = []
        for sent in sentences:
            ws = _word_set(sent)
            if len(ws) < 3:
                kept.append(sent)
                seen_word_sets.append(ws)
                continue
            overlap_any = False
            for prev in seen_word_sets:
                if len(prev) < 3:
                    continue
                inter = len(ws & prev)
                jaccard = inter / max(len(ws | prev), 1)
                if jaccard >= overlap_ratio:
                    overlap_any = True
                    break
            if not overlap_any:
                kept.append(sent)
                seen_word_sets.append(ws)
        out.append(" ".join(kept) if kept else block)
    return out


# Compression prompt: extract only answer-critical sentences; label partial/conflicting to improve faithfulness.
COMPRESS_SYSTEM = """Extract only sentences that are directly needed to answer the question. Copy verbatim; do not paraphrase or add interpretation.
- If the excerpt is only partially relevant, prefix with "[Partial]: " and keep only the relevant part.
- If the excerpt contradicts or differs from something you already extracted, include it and note "[Conflicting]: ".
- Omit sentences that do not help answer the question. Keep the result short (at most a few sentences)."""


async def compress(question: str, chunk_text: str, src: dict) -> str:
    """Extract only answer-critical sentences; label partial/conflicting. Reduces token use and improves grounding."""
    usr = f"Question: {question}\n\nRelevant excerpt:\n{chunk_text[:2000]}"
    try:
        return await chat.chat([{"role": "system", "content": COMPRESS_SYSTEM}, {"role": "user", "content": usr}], temperature=0.0, max_tokens=180)
    except Exception:
        return chunk_text


# Document-type-aware response rules: adapt style and certainty to doc_type (BOOK, FAQ, GOVERNMENT, RECORDS).


# RAG generation rules: faithfulness, grounding, and explicit "insufficient context" when needed.
def _rag_system_prompt(persona: Optional[str] = None) -> str:
    rag_doc_type_rules = """
    DOCUMENT TYPES & STYLES:

    1. 📘 Books (doc_type: book)
    - Treat as conceptual and explanatory.
    - Paraphrase when appropriate; combine chunks if they align.
    - Style: natural language, structured paragraphs or bullets.
    - Cite chapter/section when available.
    - Do not quote large blocks unless requested.

    2. ❓ FAQ Documents (doc_type: faq)
    - Treat each entry as authoritative Q&A.
    - Prefer exact answers over paraphrasing.
    - Style: direct, concise, one clear answer per question.
    - Do not merge unrelated FAQs. Note conflicts if they exist.

    3. 🏛️ Government Documents / Forms (doc_type: government/forms)
    - Treat as procedural/legal content.
    - Style: precise, step-by-step, neutral.
    - Reference form and section when possible.
    - Do not infer or extrapolate.

    4. 📊 Records / Structured Data (CSV, logs, spreadsheets) (doc_type: records)
    - Treat as factual data.
    - Prefer filtering, aggregation, or listing.
    - Include date ranges or filters when applicable.
    - Do not invent trends or values.

    5. 🎤 Live Transcripts (lectures, discussions, meetings) (doc_type: transcript)
    - Treat as ongoing speech content.
    - Summarize, extract key points, or answer questions using only the transcript.
    - Maintain speaker distinctions if provided.
    - Do not infer content beyond what was spoken.

    GENERAL GUIDELINES:
    - Only answer from the provided context. Do not invent information.

    """

    base_prompt = f"You are EchoMind, a retrieval-augmented assistant. Adapt your reasoning style and tone to the document type(s) of the provided context.\n\n{rag_doc_type_rules.strip()}"

    if persona:
        base_prompt = f"You are EchoMind in the role of: {persona}. Adapt your reasoning style and tone to this role.\n\n" + base_prompt

    return base_prompt

_GENERAL_SYSTEM = "You are EchoMind, a friendly enterprise assistant. The user is greeting you or making small talk. Reply briefly and warmly in one or two sentences. Do not mention documents or sources."

def _general_system_prompt(persona: Optional[str] = None) -> str:
    if persona:
        return f"You are EchoMind in the role of: {persona}. The user is greeting you or making small talk. Reply briefly and warmly in one or two sentences, in character. Do not mention documents or sources."
    return _GENERAL_SYSTEM

# Conversation summary: structured, compressed context for RAG (goals, constraints, decisions, key facts)
CONVERSATION_SUMMARY_SYSTEM = """You maintain a compressed, structured summary of an ongoing conversation.
Given the previous summary (or "None" if this is the first exchange) and the new exchange below, output an updated summary.
Capture: goals (what the user is trying to achieve), constraints (limits, preferences, requirements), decisions (conclusions or choices made), and key facts (important information stated or agreed).
Keep it concise: a few short paragraphs or bullet points. Preserve all signal that would help answer follow-up questions. Output only the updated summary, no preamble."""


async def update_conversation_summary(
    previous_summary: Optional[str],
    user_message: str,
    assistant_message: str,
) -> str:
    """Produce an updated conversation summary from the previous summary and the latest exchange. Does not write to DB."""
    prev = (previous_summary or "").strip() or "None"
    user = (user_message or "").strip() or "(no user message)"
    assistant = (assistant_message or "").strip() or "(no assistant message)"
    prompt = f"Previous summary:\n{prev}\n\nNew exchange:\nUser: {user}\nAssistant: {assistant}\n\nUpdated summary:"
    try:
        out = await chat.chat(
            [{"role": "system", "content": CONVERSATION_SUMMARY_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return (out or "").strip() or prev
    except Exception as e:
        logger.warning("Conversation summary update failed: %s", e)
        return prev

async def _build_rag_context(question: str, hits: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
    """Build numbered context blocks [1], [2], ... with parent expansion. No SOURCE lines (internal grounding only).
    Returns (block_texts, enriched_ctx_list for citations/audit, chunk_ids_used for audit).
    When RAG_COMPRESS_CONTEXT is False, chunks are not LLM-compressed (truncated only). When True, RAG_VERBATIM_QUERY_TERMS can still bypass compression for chunks with key query terms."""
    seen_parent_ids: set = set()
    blocks: List[str] = []
    enriched: List[Dict] = []
    chunk_ids_used: List[str] = []
    use_compression = False
    key_terms = _key_query_terms(question)
    use_verbatim = getattr(settings, "RAG_VERBATIM_QUERY_TERMS", True)
    verbatim_max = getattr(settings, "RAG_VERBATIM_MAX_CHARS", 1200)

    for h in hits:
        src = h.get("source") or {}
        parent_chunk_id = src.get("parent_chunk_id") if isinstance(src.get("parent_chunk_id"), str) else None

        if parent_chunk_id and parent_chunk_id not in seen_parent_ids:
            seen_parent_ids.add(parent_chunk_id)
            parent_text = _get_chunk_text(parent_chunk_id)
            if parent_text:
                max_chars = _parent_context_max_chars()
                truncated = parent_text if len(parent_text) <= max_chars else parent_text[:max_chars].rsplit(" ", 1)[0] + "…"
                blocks.append(_format_block_with_metadata(truncated, src))
                chunk_ids_used.append(parent_chunk_id)

        chunk_text = h.get("text") or ""
        chunk_lower = chunk_text.lower()
        use_verbatim_this = use_compression and use_verbatim and key_terms and any(term in chunk_lower for term in key_terms)
        if not use_compression or use_verbatim_this:
            # No compression: use chunk as-is, truncated to verbatim_max
            compressed = chunk_text if len(chunk_text) <= verbatim_max else chunk_text[:verbatim_max].rsplit(" ", 1)[0] + "…"
            compressed = compressed.strip()
        else:
            compressed = (await compress(question, chunk_text, src)).strip()
        blocks.append(_format_block_with_metadata(compressed, src))
        enriched.append({**h, "compressed": compressed})
        chunk_ids_used.append(h["chunk_id"])

    # Optional: deduplicate overlapping sentences across blocks to reduce token use and repetition.
    if getattr(settings, "RAG_DEDUPE_SENTENCES", False):
        ratio = getattr(settings, "RAG_DEDUPE_OVERLAP_RATIO", 0.6)
        blocks = _dedupe_overlapping_sentences(blocks, ratio)
    return blocks, enriched, chunk_ids_used


def _format_block_with_metadata(block_text: str, source: dict) -> str:
    """Prepend (doc_type, section) to the block when present so the model can adapt by document type."""
    src = source or {}
    doc_type = src.get("doc_type")
    section = src.get("section")
    if not doc_type and not section:
        return block_text
    parts = []
    if doc_type:
        parts.append(f"doc_type: {doc_type}")
    if section:
        parts.append(f"section: {section}")
    label = "(" + ", ".join(parts) + ")"
    return f"{label}\n{block_text}"


def _rag_context_block(blocks: List[str]) -> str:
    """Format blocks as [1] ... [2] ... (no SOURCE lines)."""
    return "\n\n".join([f"[{i+1}] {b}" for i, b in enumerate(blocks)])


async def _build_rag_context_fast(question: str, hits: List[Dict], max_chars_per_chunk: int = 1200) -> Tuple[List[str], List[Dict], List[str]]:
    """Fast context builder: no LLM compress, just truncated chunk text. Used for advanced_rag (single-query retrieval) path."""
    blocks: List[str] = []
    enriched: List[Dict] = []
    chunk_ids_used: List[str] = []
    for h in hits:
        src = h.get("source") or {}
        chunk_text = (h.get("text") or "").strip()
        if not chunk_text:
            continue
        truncated = chunk_text if len(chunk_text) <= max_chars_per_chunk else chunk_text[:max_chars_per_chunk].rsplit(" ", 1)[0] + "…"
        blocks.append(_format_block_with_metadata(truncated, src))
        enriched.append({**h, "compressed": truncated})
        chunk_ids_used.append(h["chunk_id"])
    return blocks, enriched, chunk_ids_used

def _is_follow_up_question(question: str) -> bool:
    """True when question seems like a follow-up (vague, short, or references prior context). Use summary only then."""
    t = (question or "").strip()
    if not t or len(t) > 200:
        return False
    words = t.split()
    if len(words) <= 4:
        return True
    follow_up_markers = ("that", "it", "this", "these", "those", "what about", "and", "also", "more", "explain", "elaborate", "tell me more", "go on")
    t_lower = t.lower()
    if any(m in t_lower for m in follow_up_markers):
        return True
    return False


def _build_user_content_with_summary(conversation_summary: Optional[str], question: str, context_block: Optional[str] = None) -> str:
    """User message: question first, RAG context second, summary last (only when follow-up). Do not prioritize summary."""
    parts = []
    parts.append(f"Current question: {question}")
    if context_block and context_block.strip():
        parts.append(f"Context (use only for factual claims):\n{context_block.strip()}")
    if conversation_summary and conversation_summary.strip() and _is_follow_up_question(question):
        parts.append(f"Optional context from earlier in conversation (use only if directly relevant to the current question):\n{conversation_summary.strip()}")
    return "\n\n".join(parts)


async def _answer_general(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> Dict:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip() and _is_follow_up_question(question):
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=None)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    return {"answer": ans, "citations": []}


# --- End-to-end RAG flow (embedding-first, semantic routing) ---
# 1. Fast path: obvious greetings/thanks/small talk (_is_general_conversation) → answer directly, no RAG.
# 2. Else: search transcript index first (dense + sparse, RRF).
# 3. Search document index (exclude transcripts).
# 4. If transcript best_score >= threshold → use transcript hits.
# 5. Else if document best_score >= threshold → use document hits.
# 6. Else → general (no RAG). Semantic search decides; no relevant hits = general.


async def answer(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    context_window: str = "all",
    conversation_summary: Optional[str] = None,
    use_knowledge_base: bool = True,
    advanced_rag: bool = False,
) -> Dict:
    if not use_knowledge_base:
        return await _answer_general(question, history, persona, conversation_summary)
    if _is_general_conversation(question):
        logger.info("RAG: general conversation (greeting/thanks) → answer directly, no RAG")
        return await _answer_general(question, history, persona, conversation_summary)

    source_type, hits = await retrieve_semantic_first(
        question, settings.TOP_K, context_window=context_window or "all"
    )

    if source_type == "general":
        return await _answer_general(question, history, persona, conversation_summary)

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    original_blocks = _get_original_blocks_for_toc(hits)
    if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(original_blocks):
        return {"answer": "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", "citations": []}

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer source=%s hits=%d", source_type, len(doc_ids))

    if conversation_summary and conversation_summary.strip() and _is_follow_up_question(question):
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=ctx_block)
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    return {"answer": ans, "citations": citations}


async def _answer_general_stream(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip() and _is_follow_up_question(question):
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=None)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    yield ("done", "".join(full).strip(), [])


async def answer_stream(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    context_window: str = "all",
    conversation_summary: Optional[str] = None,
    use_knowledge_base: bool = True,
    advanced_rag: bool = False,
) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    """
    Same RAG as answer(), but stream the final LLM response. Yields ("chunk", delta, None) then ("done", full_answer, citations).
    Uses embedding-first semantic routing (no LLM intent classification).
    """
    if not use_knowledge_base:
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return
    if _is_general_conversation(question):
        logger.info("RAG (stream): general conversation (greeting/thanks) → answer directly, no RAG")
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return

    source_type, hits = await retrieve_semantic_first(
        question, settings.TOP_K, context_window=context_window or "all"
    )

    if source_type == "general":
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    original_blocks = _get_original_blocks_for_toc(hits)
    if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(original_blocks):
        msg = "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text."
        yield ("chunk", msg, None)
        yield ("done", msg, [])
        return

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer (stream) source=%s hits=%d", source_type, len(doc_ids))

    if conversation_summary and conversation_summary.strip() and _is_follow_up_question(question):
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=ctx_block)
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    answer = "".join(full).strip()
    yield ("done", answer, citations)
