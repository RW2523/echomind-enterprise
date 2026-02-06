"""
Tagging and metadata: conversation_type, topic tags (5–12). Simple heuristics; interface ready for LLM upgrade.
"""
from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple

# Conversation type keywords (simple heuristic)
CONVERSATION_PATTERNS = {
    "meeting": ["meeting", "agenda", "action items", "follow up", "next steps", "quarterly", "standup", "sync", "review"],
    "lecture": ["lecture", "today we will", "chapter", "section", "slide", "professor", "class", "students", "today's topic"],
    "interview": ["interview", "candidate", "experience", "tell me about", "why do you", "questions for"],
    "brainstorming": ["brainstorm", "ideas", "what if", "we could", "option a", "option b", "let's try"],
    "casual": ["hey", "thanks", "sure", "okay", "got it", "sounds good", "catch you"],
}


def _tokenize_simple(text: str) -> List[str]:
    """Lowercase, alphanumeric + hyphen tokens."""
    t = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t)
    return tokens


def infer_conversation_type(text: str) -> str:
    """Infer meeting / lecture / interview / brainstorming / casual from keywords."""
    if not text or len(text) < 20:
        return "casual"
    lower = text.lower()
    scores = {}
    for ctype, keywords in CONVERSATION_PATTERNS.items():
        scores[ctype] = sum(1 for k in keywords if k in lower)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "meeting"


def extract_tags_simple(text: str, max_tags: int = 12) -> List[str]:
    """
    Lightweight keyphrase extraction: filter stopwords, take frequent meaningful words.
    No TF-IDF corpus; just frequency in this text and length filter.
    """
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "we", "our", "you", "your", "they", "them",
        "i", "me", "my", "he", "she", "so", "if", "as", "by", "from", "up", "out", "about",
    }
    tokens = _tokenize_simple(text)
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    # Prefer longer and more frequent
    scored = [(t, c * (1 + 0.1 * len(t))) for t, c in counts.most_common(max_tags * 2)]
    scored.sort(key=lambda x: -x[1])
    seen = set()
    tags = []
    for t, _ in scored:
        if t not in seen and len(tags) < max_tags:
            seen.add(t)
            tags.append(t)
    return tags[:max_tags]


class Tagger:
    """Produce conversation_type and topic tags for a text."""

    def tag(self, text: str, max_tags: int = 12) -> Tuple[str, List[str]]:
        """Returns (conversation_type, list of tags)."""
        if not text or not text.strip():
            return "casual", []
        ctype = infer_conversation_type(text)
        tags = extract_tags_simple(text, max_tags=max_tags)
        return ctype, tags


def get_tagger() -> Tagger:
    return Tagger()
