"""Tagging for chunks: topics/keywords (heuristic; same as backend tagging)."""
from __future__ import annotations
import re
from collections import Counter
from typing import List

STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
}


def get_tags(text: str, max_tags: int = 12) -> List[str]:
    if not (text or "").strip():
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
    added = set(s[0] for s in scored)
    for w, c in unigrams.most_common(max_tags * 2):
        if any(w in p for p in added if " " in p):
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
