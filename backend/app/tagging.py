"""
Tagging + metadata: conversation_type and topic tags for stored transcripts.
Lightweight heuristics now; interface allows LLM upgrade later.
"""
from __future__ import annotations
import re
from typing import List, Tuple
from collections import Counter

# Stopwords for keyphrase extraction (minimal set)
STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
}

CONVERSATION_KEYWORDS = {
    "meeting": ["meeting", "agenda", "minutes", "action items", "follow up", "schedule", "quarterly", "standup", "sync"],
    "lecture": ["lecture", "chapter", "slide", "today we", "today I", "students", "course", "exam", "homework"],
    "interview": ["interview", "candidate", "experience", "role", "team", "salary", "hiring", "position"],
    "brainstorming": ["idea", "brainstorm", "think about", "what if", "option", "concept", "design"],
    "casual": [],
}


def get_conversation_type(text: str) -> str:
    """
    Infer conversation type from text. Returns one of: meeting, lecture, interview, brainstorming, casual.
    """
    if not text or not text.strip():
        return "casual"
    lower = text.lower()
    scores = {}
    for ctype, keywords in CONVERSATION_KEYWORDS.items():
        scores[ctype] = sum(1 for k in keywords if k in lower)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "casual"


def get_tags(text: str, max_tags: int = 12) -> List[str]:
    """
    Lightweight keyphrase extraction: tokenize, remove stopwords, count, return top terms.
    RAKE-like: prefer longer phrases (bigrams then unigrams).
    """
    if not text or not text.strip():
        return []
    lower = text.lower()
    words = re.findall(r"[a-z0-9]+", lower)
    words = [w for w in words if len(w) > 1 and w not in STOP]
    if not words:
        return []
    unigrams = Counter(words)
    bigrams = Counter(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))
    scored = []
    for bigram, c in bigrams.most_common(max_tags * 2):
        scored.append((bigram, c * 1.5))
    added_phrases = set(scored[i][0] for i in range(len(scored)))
    for w, c in unigrams.most_common(max_tags * 2):
        if any(w in p for p in added_phrases if " " in p):
            continue
        scored.append((w, c))
    scored.sort(key=lambda x: -x[1])
    seen = set()
    tags = []
    for phrase, _ in scored:
        if phrase in seen:
            continue
        seen.add(phrase)
        tags.append(phrase)
        if len(tags) >= max_tags:
            break
    return tags[:max_tags]


def get_metadata(text: str) -> Tuple[str, List[str]]:
    """Returns (conversation_type, tags)."""
    return get_conversation_type(text), get_tags(text)
