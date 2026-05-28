"""
Context-aware lead phrase picker for full-duplex-style responses.

Inspired by MoshiRAG's lead+body pattern: the assistant speaks a short filler
phrase immediately after the user finishes speaking while the LLM/RAG runs in
the background. This eliminates the dead-silence gap that makes turn-based
systems feel robotic.

The lead phrase is chosen heuristically from the user's text — no LLM call
needed — so it fires in < 50 ms.
"""
from __future__ import annotations

import random
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Phrase pools keyed by intent category
# ---------------------------------------------------------------------------

_KNOWLEDGE_LEADS = [
    "Let me check that for you.",
    "Good question, let me look that up.",
    "Let me pull that up.",
    "One moment while I check.",
    "I'll find that for you.",
    "Let me look into that.",
]

_QUESTION_LEADS = [
    "Good question.",
    "Sure, let me think about that.",
    "Great question.",
    "Of course.",
    "Sure.",
    "Let me think.",
]

_INSTRUCTION_LEADS = [
    "On it.",
    "Of course.",
    "Sure thing.",
    "Absolutely.",
    "Got it.",
    "Sure.",
]

_OPINION_LEADS = [
    "Interesting.",
    "That's a great point.",
    "I see.",
    "Good thinking.",
    "Sure.",
]

_GREETING_LEADS = [
    "Hey!",
    "Hello!",
    "Hi there!",
    "Good to hear from you.",
]

_FALLBACK_LEADS = [
    "Sure.",
    "Of course.",
    "Let me think about that.",
    "One moment.",
    "Alright.",
]

# ---------------------------------------------------------------------------
# Keyword patterns for intent classification
# ---------------------------------------------------------------------------

_RE_KNOWLEDGE = re.compile(
    r"\b(what is|what are|who is|who are|how does|how do|when did|when was|"
    r"where is|where are|why does|why did|explain|tell me about|"
    r"define|definition|document|regulation|section|chapter|fmr|"
    r"according to|based on|in the|from the|does it say|what does)\b",
    re.I,
)

_RE_INSTRUCTION = re.compile(
    r"\b(can you|could you|please|would you|help me|show me|find me|"
    r"give me|list|summarize|calculate|compute|create|generate|write|"
    r"make|translate|convert|fix|check|review|compare)\b",
    re.I,
)

_RE_OPINION = re.compile(
    r"\b(what do you think|what's your opinion|do you think|"
    r"do you agree|what would you|is it better|should I|"
    r"recommend|suggest|advice|advise)\b",
    re.I,
)

_RE_GREETING = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|"
    r"how are you|what'?s up|howdy)\b",
    re.I,
)


def pick_lead_phrase(
    user_text: str,
    needs_rag: bool = False,
    persona: Optional[str] = None,
) -> str:
    """
    Return a short context-aware filler phrase to speak immediately after
    the user finishes, while LLM/RAG generates the real response.

    Args:
        user_text:  The transcribed user utterance.
        needs_rag:  True when RAG retrieval has been detected as needed.
        persona:    Optional persona string (for future persona-specific pools).

    Returns:
        A short string (1–8 words) suitable for immediate TTS synthesis.
    """
    t = (user_text or "").strip()

    if not t:
        return random.choice(_FALLBACK_LEADS)

    if _RE_GREETING.match(t):
        return random.choice(_GREETING_LEADS)

    if needs_rag or _RE_KNOWLEDGE.search(t):
        return random.choice(_KNOWLEDGE_LEADS)

    if _RE_INSTRUCTION.search(t):
        return random.choice(_INSTRUCTION_LEADS)

    if _RE_OPINION.search(t):
        return random.choice(_OPINION_LEADS)

    if "?" in t:
        return random.choice(_QUESTION_LEADS)

    return random.choice(_FALLBACK_LEADS)
