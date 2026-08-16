"""Semantic conversational-intent classifier.

Rules (exact phrases / anchored patterns) catch the obvious cases for free; this layer
catches what rules can't: novel phrasings of small talk ("how's life treating you?") and
topic refusals ("please, no more of this visa stuff") — by MEANING, not by pattern.

Approach: prototype embeddings. A small set of example utterances per intent is embedded
once (lazily, via the same nomic-embed-text model RAG uses). An incoming message is
embedded (single-text embeds hit the module-global LRU cache in embeddings.py, and the
same vector is then REUSED by retrieval for knowledge queries — so this layer adds ~zero
latency to real questions) and classified by max cosine similarity against each intent's
prototypes, with a floor and a margin over the "knowledge-seeking" contrast class so
domain questions are never misrouted.

Fails open: any error or timeout returns None → the caller proceeds to normal RAG.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from ..core.config import settings
from .embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

# ── Prototypes ──────────────────────────────────────────────────────────────
# Small talk: greetings, wellbeing checks, pleasantries — no information need.
_SMALLTALK_PROTOTYPES = (
    "hi, how are you doing today?",
    "hey there! how's it going?",
    "how's life treating you?",
    "good morning, hope you're well",
    "what's up? how have you been?",
    "hello! nice to talk to you",
    "you doing okay?",
    "thanks a lot, that was helpful",
    "haha that's funny",
    "tell me a joke",
    "who are you? tell me about yourself",
    "goodbye, have a nice day",
    "wassup my friend",
    "how was your day?",
    "hey, long time no see!",
    "it's been a while! how have you been?",
)

# Refusal / redirection: the user is declining or steering away from the current topic.
_REFUSAL_PROTOTYPES = (
    "no, I don't want anything related to that",
    "please stop talking about that topic",
    "I'm not interested in that, let's move on",
    "can we change the subject?",
    "enough about that already",
    "why do you keep bringing that up? drop it",
    "forget about that, I'd rather talk about something else",
    "that's not what I asked about",
    "no more of this, please",
    "leave that topic alone",
    "stop with the documents already",
    "I don't want to hear about those files anymore",
)

# Knowledge-seeking contrast class: real questions that MUST reach retrieval. These act
# as anchors — small talk / refusal only wins with a margin over the best knowledge match.
# Includes casually/conversationally phrased domain questions: eval showed those embed
# deceptively close to small-talk prototypes ("Do jumbo CDs earn anything extra?").
_KNOWLEDGE_PROTOTYPES = (
    "what does the document say about this topic?",
    "summarize the key points of the uploaded file",
    "what are the reimbursement rates in the regulation?",
    "how do I file for an extension?",
    "explain the policy on travel expenses",
    "what did we decide in the last meeting?",
    "which section covers compliance requirements?",
    "what is the deadline mentioned in the contract?",
    "how does this procedure work step by step?",
    "find the definition of this term in the glossary",
    "what are the risks identified in the report?",
    "compare the two sections of the manual",
    # conversationally phrased domain questions (still knowledge-seeking):
    "do I get charged anything extra for early withdrawal?",
    "who do I call when there's a problem with a result?",
    "how fast do I need to submit the paperwork?",
    "how long do I have to return it and what's the fee?",
    "what's the penalty if I cancel early?",
    "how often do these reviews happen?",
    "what rate does that account pay and what's the minimum?",
    "what happens after the trial period ends?",
)

_emb = OllamaEmbeddings()  # shares the module-global query LRU cache with retrieval

_proto_lock = asyncio.Lock()
_proto_matrix: Optional[np.ndarray] = None  # rows: all prototypes, L2-normalized
_proto_slices: dict[str, slice] = {}


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


async def _ensure_prototypes() -> bool:
    """Embed prototype utterances once. Returns True when ready."""
    global _proto_matrix
    if _proto_matrix is not None:
        return True
    async with _proto_lock:
        if _proto_matrix is not None:
            return True
        groups = {
            "smalltalk": _SMALLTALK_PROTOTYPES,
            "refusal": _REFUSAL_PROTOTYPES,
            "knowledge": _KNOWLEDGE_PROTOTYPES,
        }
        texts: list[str] = []
        start = 0
        slices: dict[str, slice] = {}
        for name, protos in groups.items():
            texts.extend(protos)
            slices[name] = slice(start, start + len(protos))
            start += len(protos)
        vecs = await _emb.embed(texts, kind="raw")
        _proto_slices.update(slices)
        _proto_matrix = _l2(np.asarray(vecs, dtype=np.float32))
        logger.info("Intent classifier: embedded %d prototypes", len(texts))
        return True


async def classify_conversational(question: str, timeout_s: float = 4.0) -> Optional[str]:
    """Return "smalltalk" or "refusal" when the message is semantically conversational,
    else None (proceed to RAG). Fails open on any error.
    """
    if not getattr(settings, "RAG_INTENT_SEMANTIC", True):
        return None
    q = (question or "").strip()
    nwords = len(q.split())
    # Long messages are almost never pure small talk; don't spend a call on them.
    if not q or nwords > 24:
        return None
    # Real small talk is SHORT. Longer conversational phrasings are usually domain
    # questions ("Do jumbo CDs earn anything extra, and how long ...?") — never
    # classify those as small talk; refusals may legitimately run longer.
    smalltalk_max_words = int(getattr(settings, "RAG_INTENT_SMALLTALK_MAX_WORDS", 12))
    floor = float(getattr(settings, "RAG_INTENT_SIM_FLOOR", 0.68))
    margin = float(getattr(settings, "RAG_INTENT_MARGIN", 0.05))
    try:
        async def _run() -> Optional[str]:
            if not await _ensure_prototypes():
                return None
            qv = await _emb.embed([q], kind="raw")  # raw: thresholds calibrated on un-prefixed vectors
            qn = _l2(np.asarray(qv, dtype=np.float32))[0]
            sims = _proto_matrix @ qn
            best = {name: float(sims[sl].max()) for name, sl in _proto_slices.items()}
            logger.debug(
                "Intent sims: smalltalk=%.3f refusal=%.3f knowledge=%.3f q=%r",
                best["smalltalk"], best["refusal"], best["knowledge"], q[:80],
            )
            if best["refusal"] >= floor and best["refusal"] > best["knowledge"] + margin:
                return "refusal"
            if (
                nwords <= smalltalk_max_words
                and best["smalltalk"] >= floor
                and best["smalltalk"] > best["knowledge"] + margin
            ):
                return "smalltalk"
            return None

        return await asyncio.wait_for(_run(), timeout=timeout_s)
    except Exception as e:  # fail open: never block the answer path
        logger.debug("Intent classifier unavailable (%s) — falling through to RAG", e)
        return None
