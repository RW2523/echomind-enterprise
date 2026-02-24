from __future__ import annotations
from typing import List, Dict, AsyncIterator, Tuple, Optional
import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta, date
from ..core.config import settings
from ..core.db import get_conn
from .index import index
from .llm import OpenAICompatChat

logger = logging.getLogger(__name__)
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

# Message when retrieval is insufficient and fallback-to-general is disabled.
INSUFFICIENT_CONTEXT_MSG = "I couldn't find relevant information to answer that."

def _parse_iso_date(created_at: Optional[str]) -> Optional[datetime]:
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return None

def _filter_hits_by_context_window(hits: List[Dict], context_window: str, now_utc: Optional[datetime] = None) -> List[Dict]:
    """
    Keep only hits whose document created_at falls within context_window (24h, 48h, 1w). 'all' = no filter.
    Uses now_utc as single reference time so all time calculations are consistent within a request.
    """
    if not context_window or context_window == "all" or not hits:
        return hits
    ref = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if context_window == "24h":
        cutoff = ref - timedelta(hours=24)
    elif context_window == "48h":
        cutoff = ref - timedelta(hours=48)
    elif context_window == "1w":
        cutoff = ref - timedelta(days=7)
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

# Greetings / small talk: answer generally, no RAG
_GENERAL_PHRASES = frozenset({
    "hi", "hello", "hey", "hey there", "hi there", "howdy",
    "thanks", "thank you", "thanks!", "thank you!",
    "how are you", "how are you?", "what's up", "whats up", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "ok", "okay", "yes", "no",
    "good", "great", "cool", "nice", "perfect",
})

# Short 1–2 word inputs that are likely real queries, not small talk (do not skip RAG).
_SHORT_QUERY_WORDS = frozenset({
    "pricing", "price", "cost", "errors", "setup", "status", "help", "login", "docs", "guide",
    "tutorial", "api", "support", "summary", "overview", "list", "find", "search", "export",
})


def _is_general_conversation(question: str) -> bool:
    """True only for empty, explicit greetings/thanks/bye, or clear small talk. Short real queries (e.g. 'pricing', 'setup') are not general."""
    t = question.strip().lower()
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


# Phrases that indicate the user is asking about knowledge-base content (documents, transcripts, etc.). Only then use RAG.
_RAG_INDICATOR_PHRASES = (
    "document", "documents", "resource", "resources",
    "live transcript", "live transcription", "transcript", "transcripts",
    "discussion", "discussions", "book", "books", "pdf", "pdfs",
    "file", "files", "uploaded", "saved transcript",
)


def _requires_rag_context(question: str) -> bool:
    """True when the user's message indicates they are asking about documents, resources, transcripts, discussion, book, pdf, or file. Only then use RAG backend flow."""
    t = (question or "").strip().lower()
    if not t:
        return False
    return any(phrase in t for phrase in _RAG_INDICATOR_PHRASES)


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


def _apply_time_decay(hits: List[Dict], doc_created: Dict[str, Optional[str]], halflife_days: float, now_utc: Optional[datetime] = None) -> List[Dict]:
    """True half-life decay: decay = exp(-ln(2) * age_days / halflife_days). Uses now_utc as reference when provided."""
    if halflife_days <= 0 or not hits:
        return hits
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)
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


# --- Source-based intent: where the user is asking from (general / document / transcript) ---

def _get_document_titles() -> Tuple[List[str], bool, List[str]]:
    """Return (uploaded_doc_titles, has_transcripts, transcript_echotags). Used for intent classification and query rewrite."""
    doc_titles: List[str] = []
    has_transcripts = False
    transcript_echotags: List[str] = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT filename, filetype FROM documents ORDER BY created_at DESC"
        ).fetchall()
        for fn, ft in rows or []:
            if (fn or "").startswith("transcript_"):
                has_transcripts = True
            else:
                doc_titles.append((fn or "").strip())
        if has_transcripts:
            tags = conn.execute(
                "SELECT DISTINCT echotag FROM transcripts WHERE echotag IS NOT NULL AND echotag != '' LIMIT 30"
            ).fetchall()
            transcript_echotags = [r[0].strip() for r in tags if r and r[0]]
    return (doc_titles[:50], has_transcripts, transcript_echotags)


def _build_intent_system_prompt(document_titles: List[str], has_transcripts: bool, transcript_echotags: List[str]) -> str:
    """Build intent classifier prompt. General comes from this LLM step. Document = embedding-first content search. Transcript = only when user clearly refers to recordings/conversations."""
    base = """You classify the user's intent so we can route correctly. Reply with only one word: general, document, or transcript.

PRIORITY: We use semantic search (embeddings) as the main way to find content. Document titles below are only to help you decide intent—not for exact matching.

- general: Only when the user is clearly greeting, doing small talk, or off-topic (e.g. "hi", "thanks", "what's the weather?", "tell me a joke"). Any factual or content-related question is NOT general.
- document: Default for content questions. The user is asking about something that could be in the knowledge base (uploaded files, books, PDFs). Use document if they ask about content by any wording: partial document name, "the book", "the report", "that file", or just a topic (e.g. "what is the Matthew effect?"). We will use embeddings to find the right chunks—no need for exact title match.
- transcript: When the user clearly refers to recordings, transcripts, or their own conversation. Examples: "summary of the recent transcript", "transcript for last 15 hours", "summarize my transcript", "give me a summary of the conversation/recording", "key points from my last meeting", "what was the latest conversation", "last 15 minutes", "last N transcripts", "last N hours/days", "summary of my recordings", "what did I say in the last meeting", "recent conversation". Any mention of transcript/recording/conversation together with a time (last X hours/days/minutes) or "recent" or "summary of" → use transcript. If they only ask a factual topic question with no mention of conversation/recording/transcript, use document."""
    if document_titles:
        base += f"\n\nDocument titles in the knowledge base (for context only; user may refer to part of a title or say 'the book'):\n" + "\n".join(f"- {t}" for t in document_titles[:40])
    if has_transcripts:
        base += "\n\nTranscripts (recordings) exist. Use intent 'transcript' only when the user explicitly asks about conversation/recording/transcript or time-bound speech (e.g. last 15 minutes, latest conversation)."
        if transcript_echotags:
            base += " Tags: " + ", ".join(transcript_echotags[:15]) + "."
    return base


# Intent-based query rewrite: tailor expansion to source (document vs transcript). Retrieval uses embeddings, so queries need not match titles exactly.
QUERY_REWRITE_BY_INTENT = {
    "general": "Return only the user's question as a single search query, no alternatives. (One line only.)",
    "document": "Rewrite into 2 concise search queries that capture the main concepts, key terms, and what the user is looking for. These will be used for semantic search (embeddings); include topic words, names, or phrases from the question so the right passages are retrieved. Partial or generic references (e.g. 'the book', 'the report') are fine—focus on the underlying content ask. Return only the queries, one per line.",
    "transcript": "Rewrite into 2 search queries for finding content in recorded transcripts. Preserve time-related intent: 'last N transcripts', 'recent', 'summary of last 15', 'what was said in the last meeting', 'last 10 minutes'. Use terms like transcript, meeting, recording, recent, summary where relevant. Return only the queries, one per line.",
}


async def _classify_intent(
    question: str,
    document_titles: Optional[List[str]] = None,
    has_transcripts: bool = False,
    transcript_echotags: Optional[List[str]] = None,
) -> str:
    """
    Classify where the user is asking from by sending the question to the LLM with the intent system prompt.
    Flow: build system prompt (with doc titles / transcript hints) → send question as user message → LLM replies with one word → we use that as intent.
    """
    if document_titles is None and not has_transcripts:
        doc_titles, has_transcripts, transcript_echotags = _get_document_titles()
    else:
        doc_titles = document_titles or []
        transcript_echotags = transcript_echotags or []
    sys = _build_intent_system_prompt(doc_titles, has_transcripts, transcript_echotags)
    user_msg = (question or "").strip()[:600]
    try:
        out = await chat.chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Question to classify:\n{user_msg}" if user_msg else "Question to classify: (empty)"},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        intent = (out or "").strip().lower()
        if intent in ("general", "document", "transcript"):
            return intent
    except Exception as e:
        logger.warning("Intent classification failed, using fallback: %s", e)
    return "document" if doc_titles or has_transcripts else "general"


def _question_clearly_asks_for_transcript(question: str, has_transcripts: bool) -> bool:
    """True when the user clearly asks for transcript/recording/conversation content (e.g. summary of recent transcript, last N hours).
    Used to override LLM intent so we always search the transcript-only index in these cases."""
    if not has_transcripts or not (question or "").strip():
        return False
    q = question.strip().lower()
    time_or_recent = (
        "recent" in q or "last" in q or "past" in q or "summary" in q
        or "hour" in q or "hours" in q or "day" in q or "days" in q or "minute" in q or "minutes" in q
        or "for last" in q
    )
    transcript_related = (
        "transcript" in q or "recording" in q or "recordings" in q
        or "conversation" in q or "conversations" in q or "meeting" in q or "meetings" in q
    )
    return bool(transcript_related and time_or_recent)


def _parse_last_n_transcripts(question: str) -> Optional[int]:
    """If question asks for 'last N transcripts' or 'summary of last N' or 'pick last N', return N. Otherwise None."""
    if not (question or "").strip():
        return None
    t = question.lower()
    # "last 15 transcripts", "last 15 transcript", "summary of last 15", "past 10 transcripts"
    m = re.search(r"(?:last|past|recent)\s+(\d+)\s*(?:transcript|recording|meeting)s?", t)
    if m:
        return min(100, max(1, int(m.group(1))))
    # "pick last 2", "last 2" (when transcript context), "recent 2 transcripts"
    m = re.search(r"(?:pick|get|give|summarize?|recap)\s+(?:the\s+)?(?:last|recent)\s+(\d+)\b", t)
    if m:
        return min(100, max(1, int(m.group(1))))
    m = re.search(r"\b(?:last|recent)\s+(\d+)\b", t)
    if m and re.search(r"\b(transcript|recording|meeting|conversation)\b", t):
        return min(100, max(1, int(m.group(1))))
    m = re.search(r"(?:summary|recap)\s+(?:of\s+)?(?:the\s+)?last\s+(\d+)", t)
    if m:
        return min(100, max(1, int(m.group(1))))
    return None


def _parse_last_time_window(question: str) -> Optional[timedelta]:
    """If question asks for 'last N hours/mins' or 'Nhrs' / 'N mins', return timedelta. Single reference time used by caller."""
    if not (question or "").strip():
        return None
    t = question.lower()
    # "last 5 minutes", "last 5 mins", "past 10 minutes", "summarise last 5 mins"
    m = re.search(r"(?:last|past|for\s+last|summarise?|summarize)\s+(?:the\s+)?(\d+)\s*(minute|minutes?|mins?)", t)
    if m:
        return timedelta(minutes=min(10080, max(1, int(m.group(1)))))  # cap 7 days
    # "last 5 mins" without "last" before number (e.g. "summary last 5 mins" already above)
    m = re.search(r"\b(\d+)\s*(?:minute|minutes?|mins?)\b", t)
    if m and re.search(r"\b(transcript|recording|meeting|last|recent)\b", t):
        return timedelta(minutes=min(10080, max(1, int(m.group(1)))))
    # "last 15 hours", "for last 15 hours", "past 2 hours"
    m = re.search(r"(?:last|past|for\s+last)\s+(\d+)\s*(hour|hours?)", t)
    if m:
        return timedelta(hours=min(720, max(1, int(m.group(1)))))
    # "24hrs", "48hrs", "36hrs", "24 hrs", "last 24hrs", "past 48hrs"
    m = re.search(r"(?:last|past|in\s+)?(\d+)\s*hrs?\b", t)
    if m:
        return timedelta(hours=min(720, max(1, int(m.group(1)))))
    m = re.search(r"\b(\d+)\s*(?:hour|hours?)\b", t)
    if m:
        return timedelta(hours=min(720, max(1, int(m.group(1)))))
    m = re.search(r"(?:last|past|for\s+last)\s+(\d+)\s*(day|days?)", t)
    if m:
        return timedelta(days=min(365, max(1, int(m.group(1)))))
    return None


def _parse_specific_date(question: str, now_utc: datetime) -> Optional[date]:
    """If question mentions a specific date (today, yesterday, or calendar date), return that date for transcript filtering."""
    if not (question or "").strip():
        return None
    t = question.lower().strip()
    # Must look like transcript/date context (avoid matching dates inside document content)
    if not re.search(r"\b(transcript|recording|meeting|conversation|discuss|said|on|from|for)\b", t):
        return None
    # today
    if re.search(r"\btoday\b", t):
        return now_utc.date()
    # yesterday
    if re.search(r"\byesterday\b", t):
        return (now_utc - timedelta(days=1)).date()
    # ISO: 2025-02-20 or 2025/02/20
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return date(y, mo, d)
        except (ValueError, TypeError):
            pass
    # Month name: Feb 20, February 20, 20 Feb, 20th February (current year if not given)
    year = now_utc.year
    months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    months_full = "january|february|march|april|may|june|july|august|september|october|november|december"
    m = re.search(rf"\b({months}|{months_full})\s*\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?\b", t)
    if m:
        try:
            from calendar import month_abbr, month_name
            month_str = m.group(1).lower()[:3]
            d = int(m.group(2))
            if m.lastindex >= 3 and m.group(3):
                year = int(m.group(3))
            for i in range(1, 13):
                if month_abbr[i].lower() == month_str or month_name[i].lower().startswith(month_str):
                    if 1 <= d <= 31:
                        return date(year, i, d)
                    break
        except (ValueError, TypeError, ImportError):
            pass
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({months}|{months_full})(?:\s*,?\s*(20\d{{2}}))?\b", t)
    if m:
        try:
            from calendar import month_abbr, month_name
            d = int(m.group(1))
            month_str = m.group(2).lower()[:3]
            if m.lastindex >= 3 and m.group(3):
                year = int(m.group(3))
            for i in range(1, 13):
                if month_abbr[i].lower() == month_str or month_name[i].lower().startswith(month_str):
                    if 1 <= d <= 31:
                        return date(year, i, d)
                    break
        except (ValueError, TypeError, ImportError):
            pass
    return None


# Default window for "at 2pm" style queries (1 hour)
_SPECIFIC_TIME_WINDOW_MINUTES = 60


def _parse_time_range_from_question(question: str, reference_ts: datetime, tz_str: str = "UTC") -> Optional[Tuple[datetime, datetime]]:
    """
    Parse a specific time or time range from the question (e.g. "at 2pm", "between 14:00 and 15:00 on Feb 20").
    Returns (start_utc, end_utc) for filtering transcript chunks by echodate. Times in the question are
    interpreted in the given timezone (TRANSCRIPT_QUERY_TZ) then converted to UTC.
    """
    if not (question or "").strip():
        return None
    t = question.lower()
    if not re.search(r"\b(transcript|recording|meeting|conversation|discuss|said|at|between|from)\b", t):
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_str)
    except Exception:
        tz = timezone.utc

    ref_local = reference_ts.astimezone(tz)
    year, month, day = ref_local.year, ref_local.month, ref_local.day

    def parse_time_only(s: str) -> Optional[Tuple[int, int]]:
        s = s.strip()
        m = re.match(r"(\d{1,2}):(\d{2})\s*(?:am|pm)?", s, re.I)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            if "pm" in s.lower() and h < 12:
                h += 12
            if "am" in s.lower() and h == 12:
                h = 0
            return (h, mi) if 0 <= h <= 23 and 0 <= mi <= 59 else None
        m = re.match(r"(\d{1,2})\s*(?:am|pm)\b", s, re.I)
        if m:
            h = int(m.group(1))
            if "pm" in s.lower() and h < 12:
                h += 12
            if "am" in s.lower() and h == 12:
                h = 0
            return (h, 0) if 0 <= h <= 23 else None
        return None

    def parse_date_part(s: str) -> Optional[Tuple[int, int, int]]:
        s = s.strip().lower()
        m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (y, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None
        months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
        m = re.search(rf"\b({months})\s*\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?\b", s)
        if m:
            from calendar import month_abbr
            mon_abbr = m.group(1).lower()[:3]
            d = int(m.group(2))
            y = int(m.group(3)) if m.lastindex >= 3 and m.group(3) else ref_local.year
            for i in range(1, 13):
                if month_abbr[i].lower() == mon_abbr:
                    return (y, i, d) if 1 <= d <= 31 else None
            return None
        if "yesterday" in s:
            rd = ref_local.date() - timedelta(days=1)
            return (rd.year, rd.month, rd.day)
        if "today" in s:
            return (ref_local.year, ref_local.month, ref_local.day)
        return None

    # "between 14:00 and 15:00" / "from 2pm to 3pm" / "from 14:00 to 15:00 on Feb 20"
    m = re.search(r"(?:between|from)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(?:and|to)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)(?:\s+on\s+([^.?]+))?", t, re.I)
    if m:
        t1 = parse_time_only(m.group(1))
        t2 = parse_time_only(m.group(2))
        date_part = parse_date_part(m.group(3) or "") if m.group(3) else (year, month, day)
        if t1 and t2 and date_part:
            y, mo, d = date_part
            start_local = datetime(y, mo, d, t1[0], t1[1], 0, tzinfo=tz)
            end_local = datetime(y, mo, d, t2[0], t2[1], 0, tzinfo=tz)
            if end_local <= start_local:
                end_local += timedelta(days=1)
            return (start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc))

    # "at 14:00" / "at 2pm" / "at 2:30pm on Feb 20" / "Feb 20 at 2pm"
    m = re.search(r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)(?:\s+on\s+([^.?]+)|\s+(?:on\s+)?([^.?]+))?", t, re.I)
    if not m:
        m = re.search(r"(?:Feb|Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", t, re.I)
    if m:
        time_part = parse_time_only(m.group(1))
        date_str = (m.group(2) or m.group(3) or "").strip() if m.lastindex >= 3 else ""
        date_part = parse_date_part(date_str) if date_str else (year, month, day)
        if time_part and date_part:
            y, mo, d = date_part
            h, mi = time_part
            start_local = datetime(y, mo, d, h, mi, 0, tzinfo=tz)
            end_local = start_local + timedelta(minutes=_SPECIFIC_TIME_WINDOW_MINUTES)
            return (start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc))

    return None


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


async def generate_queries(
    q: str,
    intent: Optional[str] = None,
    document_titles: Optional[List[str]] = None,
    has_transcripts: bool = False,
    transcript_echotags: Optional[List[str]] = None,
) -> List[str]:
    """Produce 1–3 alternative search queries. When RAG_INTENT_REWRITE is on, use source-based intent (general/document/transcript) to tailor rewrite."""
    use_intent = getattr(settings, "RAG_INTENT_REWRITE", True)
    if use_intent:
        if intent is None:
            intent = await _classify_intent(q, document_titles, has_transcripts, transcript_echotags)
        sys = QUERY_REWRITE_BY_INTENT.get(intent, QUERY_REWRITE_BY_INTENT["document"])
        try:
            txt = await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": (q or "")[:600]}], temperature=0.2, max_tokens=120)
            logger.info("RAG query rewrite (intent=%s) raw LLM response: %s", intent, (txt or "").strip()[:500])
            variants = []
            for line in txt.splitlines():
                line = re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
                if line and line.lower() != (q or "").strip().lower():
                    variants.append(line)
            out = [q.strip() or " "] + variants[:3]
            out = out[:4]
            logger.info("RAG query rewrite (intent=%s) final queries: %s", intent, out)
            return out
        except Exception:
            pass
    sys = "Rewrite the question into 2 alternative search queries (synonyms or key terms only). Return only the list, one per line."
    txt = await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": q}], temperature=0.2, max_tokens=120)
    logger.info("RAG query rewrite (fallback) raw LLM response: %s", (txt or "").strip()[:500])
    variants = []
    for line in txt.splitlines():
        line = re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
        if line:
            variants.append(line)
    out = [q.strip() or " "] + [v for v in variants if v.lower() != (q or "").strip().lower()]
    out = out[:4]
    logger.info("RAG query rewrite (fallback) final queries: %s", out)
    return out


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


async def retrieve(
    question: str,
    k: int,
    context_window: str = "all",
    intent: Optional[str] = None,
    document_titles: Optional[List[str]] = None,
    has_transcripts: bool = False,
    transcript_echotags: Optional[List[str]] = None,
) -> List[Dict]:
    """Hybrid retrieve: deterministic + optional LLM query expansion, dense + sparse, weighted RRF. Uses a single reference timestamp (now_utc) for all time/date filtering so context is consistent. When intent is transcript, filters by date (today/yesterday/specific), last N transcripts, or time window (24hrs, 48hrs, last 5 mins)."""
    reference_ts = datetime.now(timezone.utc)
    if getattr(settings, "RAG_QUERY_REWRITE_ENABLED", False):
        llm_qs = await generate_queries(question, intent=intent, document_titles=document_titles, has_transcripts=has_transcripts, transcript_echotags=transcript_echotags)
    else:
        llm_qs = [question.strip() or " "]
    det_qs = get_deterministic_query_variants(question)
    seen_lower: set = set()
    qs: List[str] = []
    for q in det_qs + llm_qs:
        key = (q or "").strip().lower()
        if key and key not in seen_lower:
            seen_lower.add(key)
            qs.append(q.strip())
    if not qs:
        qs = [question.strip() or " "]
    logger.info("RAG retrieve: intent=%s combined search queries (det + llm deduped): %s", intent, qs)

    is_toc = is_toc_chapters_query(question)
    if is_toc:
        k_per_query = getattr(settings, "RAG_BOOK_K_PER_QUERY", 20)
        sparse_w = getattr(settings, "RAG_BOOK_SPARSE_WEIGHT", 0.5)
        dense_w = 1.0 - sparse_w
    else:
        k_per_query = max(k, 4)
        dense_w = getattr(settings, "RAG_DENSE_RRF_WEIGHT", 0.5)
        sparse_w = getattr(settings, "RAG_SPARSE_RRF_WEIGHT", 0.5)

    dense_hits_per_query: List[List[Dict]] = []
    sparse_hits_per_query: List[List[Dict]] = []
    use_transcript_index = intent == "transcript"
    if use_transcript_index:
        logger.info("RAG retrieve: intent=transcript, using transcript-only index")
    for q in qs:
        if use_transcript_index:
            dense_hits_per_query.append(await index.search_transcript_only(q, k_per_query))
            sparse_hits_per_query.append(index.transcript_sparse.search(q, k_per_query))
        else:
            dense_hits_per_query.append(await index.search(q, k_per_query))
            sparse_hits_per_query.append(index.sparse.search(q, k_per_query))
    candidates_k = max(k, getattr(settings, "RAG_RERANK_CANDIDATES", k)) if getattr(settings, "RAG_RERANK_ENABLED", False) else k
    hits = _weighted_rrf(dense_hits_per_query, sparse_hits_per_query, candidates_k, dense_weight=dense_w, sparse_weight=sparse_w)
    # Hard filter by context_window when not 'all' (single reference time)
    hits = _filter_hits_by_context_window(hits, context_window or "all", reference_ts)
    # Time-decay scoring (soft; keeps older docs but down-weights)
    doc_created, doc_meta = _get_doc_info_for_hits(hits)
    halflife = getattr(settings, "RAG_TIME_DECAY_HALFLIFE_DAYS", 0) or 0
    if halflife > 0:
        hits = _apply_time_decay(hits, doc_created, halflife, reference_ts)
    # Tag boost for transcripts
    if getattr(settings, "RAG_TAG_BOOST_ENABLED", False):
        tag_factor = getattr(settings, "RAG_TAG_BOOST_FACTOR", 0.08)
        hits = _apply_tag_boost(hits, question, doc_meta, tag_factor)
    # Prefer authoritative docs when scores are close (skip when intent=transcript; all hits are from transcript index)
    if intent != "transcript" and getattr(settings, "RAG_PREFER_AUTHORITATIVE", False):
        hits = _prefer_authoritative_sort(hits)
    # Optional rerank
    if getattr(settings, "RAG_RERANK_ENABLED", False):
        rerank_n = getattr(settings, "RAG_RERANK_TOP_N", k)
        hits = await _rerank_hits(question, hits[: getattr(settings, "RAG_RERANK_CANDIDATES", 12)], rerank_n)
    else:
        hits = hits[:k]

    # When intent is transcript: filter by specific time range (UTC), then date, last N, time window, location, or latest
    if intent == "transcript":
        tz_str = getattr(settings, "TRANSCRIPT_QUERY_TZ", "UTC") or "UTC"
        time_range = _parse_time_range_from_question(question, reference_ts, tz_str)
        if time_range is not None:
            start_utc, end_utc = time_range
            filtered = []
            for h in hits:
                doc_id = (h.get("source") or {}).get("doc_id")
                if not doc_id:
                    continue
                meta = doc_meta.get(doc_id) or {}
                ts_at = meta.get("echodate") or doc_created.get(doc_id)
                if ts_at is None:
                    continue
                dt = _parse_iso_date(ts_at)
                if dt is None:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if start_utc <= dt < end_utc:
                    filtered.append(h)
            if filtered:
                hits = filtered
            # else: no chunks in that exact window; keep hits for answer to say "nothing in that time"
        else:
            # 1) Specific date (today, yesterday, or calendar date): keep only hits on that day
            specific_date = _parse_specific_date(question, reference_ts)
            if specific_date is not None:
                filtered = []
                for h in hits:
                    doc_id = (h.get("source") or {}).get("doc_id")
                    if not doc_id:
                        continue
                    meta = doc_meta.get(doc_id) or {}
                    ts_at = meta.get("echodate") or doc_created.get(doc_id)
                    if ts_at is None:
                        continue
                    dt = _parse_iso_date(ts_at)
                    if dt is None:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt.date() == specific_date:
                        filtered.append(h)
                if filtered:
                    hits = filtered
            else:
                # 2) Last N transcripts
                n = _parse_last_n_transcripts(question)
                if n is not None and n > 0:
                    recent_ids = _get_recent_transcript_doc_ids(n)
                    if recent_ids:
                        hits = [h for h in hits if ((h.get("source") or {}).get("doc_id")) in recent_ids]

                time_window = _parse_last_time_window(question)
                location_filter = _parse_location_from_question(question)
                latest_only = _parse_latest_or_recent_transcript(question)

                if time_window is not None or location_filter:
                    cutoff = (reference_ts - time_window) if time_window else None
                    filtered = []
                    for h in hits:
                        doc_id = (h.get("source") or {}).get("doc_id")
                        if not doc_id:
                            continue
                        meta = doc_meta.get(doc_id) or {}
                        if cutoff is not None:
                            ts_at = meta.get("echodate") or doc_created.get(doc_id)
                            if ts_at is None:
                                continue
                            dt = _parse_iso_date(ts_at)
                            if dt is None:
                                continue
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            if dt < cutoff:
                                continue
                        if location_filter:
                            doc_location = (meta.get("location") or "").strip().lower()
                            loc_lower = location_filter.strip().lower()
                            if not doc_location or (loc_lower not in doc_location and doc_location not in loc_lower):
                                continue
                        filtered.append(h)
                    hits = filtered
                elif latest_only:
                    recent_tid = _get_most_recent_transcript_id()
                    if recent_tid:
                        filtered = [
                            h for h in hits
                            if (doc_meta.get((h.get("source") or {}).get("doc_id")) or {}).get("transcript_id") == recent_tid
                        ]
                        if filtered:
                            hits = filtered
    return hits


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
    - Only answer from the provided Context block. Do not invent information or use other sources.
    - Be precise and concise. Base every factual claim only on the Context; if the Context does not support an answer, say so briefly.
    - Do not let prior conversation or summaries steer you to a different topic—answer only from the Context and the current question (use recent messages only to resolve pronouns like "it" or "that").
    - Answer in natural language. Do not mention "documents", "resources", or "sources" unless the user explicitly asks where the information came from.

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

def _build_user_content_with_summary(conversation_summary: Optional[str], question: str, context_block: Optional[str] = None) -> str:
    """Single user message: conversation summary (if any) + current question + optional RAG context. When context_block is present, it is the primary source; summary is for continuity only."""
    parts = []
    if context_block and context_block.strip():
        parts.append(
            "Context (primary source—base your factual answer only on this):\n" + context_block.strip()
        )
    parts.append(f"Current question: {question}")
    if conversation_summary and conversation_summary.strip():
        parts.append(
            f"Conversation summary (for continuity only; do not let it override the Context above):\n{conversation_summary.strip()}"
        )
    return "\n\n".join(parts)


async def _answer_general(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> Dict:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip():
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=None)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    return {"answer": ans, "citations": []}


# --- End-to-end RAG flow (embedding-first, intent from LLM) ---
# 1. Fast path: obvious greeting/small talk → _answer_general (no retrieval).
# 2. Load doc titles + transcript availability (for intent prompt only; no exact title matching).
# 3. Intent classification (LLM): question + system prompt with titles/hints → general | document | transcript.
#    General comes from this step when LLM says so. Document = content question (embedding-first). Transcript = only when user clearly says conversation/recording/transcript/last N minutes/latest conversation.
# 4. If general → _answer_general.
# 5. If document or transcript → retrieve(): query expansion (intent-aware), then embedding-first search (dense + sparse), time filter, optional "last N transcripts" for transcript intent.
# 6. Build context from hits, answer with LLM. Embedding is the priority for finding the right chunks.


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
        return await _answer_general(question, history, persona, conversation_summary)
    # When RAG_ALWAYS_TRY_FOR_CONTENT_QUESTIONS is False, use RAG only if user mentions document/transcript/etc.
    if not getattr(settings, "RAG_ALWAYS_TRY_FOR_CONTENT_QUESTIONS", True) and not _requires_rag_context(question):
        return await _answer_general(question, history, persona, conversation_summary)

    if advanced_rag:
        logger.info("RAG intent: (advanced_rag, no intent classification) question=%s", (question[:80] + "…") if len(question) > 80 else question)
        hits = await retrieve_single_query(question, settings.TOP_K, context_window=context_window or "all")
        best_score = hits[0]["score"] if hits else 0.0
        if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
            if getattr(settings, "RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", True):
                return await _answer_general(question, history, persona, conversation_summary)
            return {"answer": INSUFFICIENT_CONTEXT_MSG, "citations": []}
        blocks, enriched, chunk_ids_used = await _build_rag_context_fast(question, hits)
    else:
        doc_titles, has_transcripts, transcript_echotags = _get_document_titles()
        # Skip intent when KB is empty: go straight to general (faster).
        if not doc_titles and not has_transcripts:
            return await _answer_general(question, history, persona, conversation_summary)
        intent = await _classify_intent(question, doc_titles, has_transcripts, transcript_echotags)
        if has_transcripts and _question_clearly_asks_for_transcript(question, has_transcripts):
            intent = "transcript"
        logger.info("RAG intent: classified=%s question=%s", intent, (question[:80] + "…") if len(question) > 80 else question)
        if intent == "general":
            return await _answer_general(question, history, persona, conversation_summary)

        hits = await retrieve(
            question,
            settings.TOP_K,
            context_window=context_window or "all",
            intent=intent,
            document_titles=doc_titles,
            has_transcripts=has_transcripts,
            transcript_echotags=transcript_echotags,
        )
        best_score = hits[0]["score"] if hits else 0.0
        if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
            if getattr(settings, "RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", True):
                return await _answer_general(question, history, persona, conversation_summary)
            return {"answer": INSUFFICIENT_CONTEXT_MSG, "citations": []}

        blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
        original_blocks = _get_original_blocks_for_toc(hits)
        if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(original_blocks):
            return {"answer": "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", "citations": []}

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer intent=%s hits=%d", intent if not advanced_rag else "advanced_rag", len(doc_ids))

    # When answering from RAG context, do not inject conversation summary—it can steer the model to the wrong context and mislead. Use only history + Context + question.
    msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    rag_temp = getattr(settings, "RAG_LLM_TEMPERATURE", None)
    if rag_temp is None or rag_temp < 0:
        rag_temp = settings.LLM_TEMPERATURE
    ans = await chat.chat(msgs, temperature=rag_temp, max_tokens=settings.LLM_MAX_TOKENS)
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        src = lambda c: c.get("source") or {}
        citations = [{"filename": src(c).get("filename"), "chunk_index": src(c).get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    return {"answer": ans, "citations": citations}


# --- RAG debug: run full flow and return intent, retrieved chunks, and answer (for test/debug UI) ---
CHUNK_PREVIEW_CHARS = 400


async def rag_debug_run(question: str, advanced_rag: bool = False) -> Dict:
    """
    Run the full RAG flow and return debug info: intent, retrieved chunks (with score and text preview), and answer.
    Used by the RAG Test page to show how context is retrieved from embeddings and how the answer is produced.
    """
    q = (question or "").strip()
    if not q:
        return {"intent": "general", "chunks": [], "answer": "", "message": "No question provided."}

    # Same fast paths as answer()
    if _is_general_conversation(q):
        out = await _answer_general(q, [], None, None)
        return {"intent": "general", "chunks": [], "answer": out["answer"], "message": "Treated as general conversation (no retrieval)."}

    intent: Optional[str] = None
    hits: List[Dict] = []

    if advanced_rag:
        intent = "advanced_rag"
        hits = await retrieve_single_query(q, settings.TOP_K, context_window="all")
    else:
        doc_titles, has_transcripts, transcript_echotags = _get_document_titles()
        if not doc_titles and not has_transcripts:
            out = await _answer_general(q, [], None, None)
            return {"intent": "general", "chunks": [], "answer": out["answer"], "message": "No documents or transcripts in the knowledge base."}
        intent = await _classify_intent(q, doc_titles, has_transcripts, transcript_echotags)
        if has_transcripts and _question_clearly_asks_for_transcript(q, has_transcripts):
            intent = "transcript"
        if intent == "general":
            out = await _answer_general(q, [], None, None)
            return {"intent": "general", "chunks": [], "answer": out["answer"], "message": "Intent classified as general (no retrieval)."}
        hits = await retrieve(q, settings.TOP_K, context_window="all", intent=intent,
            document_titles=doc_titles, has_transcripts=has_transcripts, transcript_echotags=transcript_echotags)

    best_score = hits[0]["score"] if hits else 0.0
    if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
        if getattr(settings, "RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", True):
            out = await _answer_general(q, [], None, None)
            return {"intent": intent or "document", "chunks": [], "answer": out["answer"], "message": "No relevant chunks found; answered from general knowledge."}
        return {"intent": intent or "document", "chunks": [], "answer": INSUFFICIENT_CONTEXT_MSG, "message": "No relevant chunks found."}

    # Build chunks for debug (score + text preview + source)
    chunks_out = []
    for h in hits:
        src = h.get("source") or {}
        text = (h.get("text") or "")[:CHUNK_PREVIEW_CHARS]
        if len(h.get("text") or "") > CHUNK_PREVIEW_CHARS:
            text += "…"
        chunks_out.append({
            "score": round(h["score"], 4),
            "text_preview": text,
            "filename": src.get("filename"),
            "doc_id": src.get("doc_id"),
            "chunk_index": src.get("chunk_index"),
            "filetype": src.get("filetype"),
        })

    # Run full answer so we return the same answer the user would get
    if advanced_rag:
        blocks, enriched, _ = await _build_rag_context_fast(q, hits)
    else:
        blocks, enriched, _ = await _build_rag_context(q, hits)
        original_blocks = _get_original_blocks_for_toc(hits)
        if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(q) and not has_toc_signals_in_context(original_blocks):
            return {"intent": intent, "chunks": chunks_out, "answer": "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", "message": "TOC guardrail: no chapter list in context."}

    ctx_block = _rag_context_block(blocks)
    msgs = [{"role": "system", "content": _rag_system_prompt(None)}, {"role": "user", "content": f"Question: {q}\n\nContext:\n{ctx_block}"}]
    rag_temp = getattr(settings, "RAG_LLM_TEMPERATURE", None) or settings.LLM_TEMPERATURE
    ans = await chat.chat(msgs, temperature=rag_temp, max_tokens=settings.LLM_MAX_TOKENS)

    return {"intent": intent or "document", "chunks": chunks_out, "answer": ans, "message": None}


async def _answer_general_stream(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip():
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
    use_knowledge_base=False forces general answer. advanced_rag=True uses single-query retrieval (no intent/rewrite).
    """
    if not use_knowledge_base:
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return
    if _is_general_conversation(question):
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return
    if not getattr(settings, "RAG_ALWAYS_TRY_FOR_CONTENT_QUESTIONS", True) and not _requires_rag_context(question):
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return

    if advanced_rag:
        logger.info("RAG intent (stream): (advanced_rag, no intent classification) question=%s", (question[:80] + "…") if len(question) > 80 else question)
        hits = await retrieve_single_query(question, settings.TOP_K, context_window=context_window or "all")
        best_score = hits[0]["score"] if hits else 0.0
        if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
            if getattr(settings, "RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", True):
                async for ev in _answer_general_stream(question, history, persona, conversation_summary):
                    yield ev
                return
            yield ("chunk", INSUFFICIENT_CONTEXT_MSG, None)
            yield ("done", INSUFFICIENT_CONTEXT_MSG, [])
            return
        blocks, enriched, chunk_ids_used = await _build_rag_context_fast(question, hits)
    else:
        doc_titles, has_transcripts, transcript_echotags = _get_document_titles()
        if not doc_titles and not has_transcripts:
            async for ev in _answer_general_stream(question, history, persona, conversation_summary):
                yield ev
            return
        intent = await _classify_intent(question, doc_titles, has_transcripts, transcript_echotags)
        if has_transcripts and _question_clearly_asks_for_transcript(question, has_transcripts):
            intent = "transcript"
        logger.info("RAG intent (stream): classified=%s question=%s", intent, (question[:80] + "…") if len(question) > 80 else question)
        if intent == "general":
            async for ev in _answer_general_stream(question, history, persona, conversation_summary):
                yield ev
            return

        hits = await retrieve(
            question,
            settings.TOP_K,
            context_window=context_window or "all",
            intent=intent,
            document_titles=doc_titles,
            has_transcripts=has_transcripts,
            transcript_echotags=transcript_echotags,
        )
        best_score = hits[0]["score"] if hits else 0.0
        if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
            if getattr(settings, "RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", True):
                async for ev in _answer_general_stream(question, history, persona, conversation_summary):
                    yield ev
                return
            yield ("chunk", INSUFFICIENT_CONTEXT_MSG, None)
            yield ("done", INSUFFICIENT_CONTEXT_MSG, [])
            return

        blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
        original_blocks = _get_original_blocks_for_toc(hits)
        if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(original_blocks):
            yield ("chunk", "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", None)
            yield ("done", "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", [])
            return

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer (stream) intent=%s hits=%d", intent if not advanced_rag else "advanced_rag", len(doc_ids))

    # When answering from RAG context, do not inject conversation summary—it can steer the model to the wrong context and mislead. Use only history + Context + question.
    msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        src = lambda c: c.get("source") or {}
        citations = [{"filename": src(c).get("filename"), "chunk_index": src(c).get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    rag_temp = getattr(settings, "RAG_LLM_TEMPERATURE", None)
    if rag_temp is None or rag_temp < 0:
        rag_temp = settings.LLM_TEMPERATURE
    full = []
    async for delta in chat.chat_stream(msgs, temperature=rag_temp, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    answer = "".join(full).strip()
    yield ("done", answer, citations)
