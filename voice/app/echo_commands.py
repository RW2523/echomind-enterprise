"""
EchoMind command / intent router. Parses voice commands and returns deterministic
responses for profile, listen mode, and memory queries. Fact-check and open queries
return None so session can use LLM with retrieved context.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# Intent result: (handled: bool, response_text: Optional[str], extra: dict for session use)
IntentResult = Tuple[bool, Optional[str], Dict[str, Any]]


def _normalize(t: str) -> str:
    return (t or "").strip().lower()


def _extract_after(utterance: str, patterns: List[str]) -> Optional[str]:
    """If utterance starts with one of the patterns (after normalization), return the rest (name/value)."""
    u = _normalize(utterance)
    for p in patterns:
        if u.startswith(p):
            rest = u[len(p):].strip()
            if rest:
                return rest
    return None


def _match_any(utterance: str, phrases: List[str]) -> bool:
    u = _normalize(utterance)
    return any(p in u for p in phrases)


def _extract_minutes(utterance: str) -> Optional[float]:
    """Heuristic: 'last 5 minutes' -> 5.0, 'last 10 minutes' -> 10.0."""
    u = _normalize(utterance)
    # "last N minute(s)", "past N minute(s)", "last N min"
    m = re.search(r"(?:last|past)\s+(\d+)\s*(?:minute|min)s?\b", u)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*(?:minute|min)s?\s*(?:ago|back)", u)
    if m:
        return float(m.group(1))
    return None


# Semantic intent: phrases that suggest "user wants to change what they call the assistant"
_WAKE_WORD_CHANGE_INTENT_KEYWORDS = [
    "change wake word", "change the wake word", "change your wake word",
    "change trigger", "change trigger word", "change the trigger",
    "rename you", "rename the assistant", "rename yourself",
    "call you", "call you something", "call you differently",
    "new name", "different name", "change your name", "change name to",
    "what i call you", "what to call you", "how to call you",
    "be called", "be named", "from now on call",
    "wake word to", "trigger word to", "name you",
]

# Extraction patterns: (regex, group_or_last) - group 1 or use last word of match
_WAKE_WORD_EXTRACT_PATTERNS = [
    (r"(?:to|into)\s+([a-zA-Z][a-zA-Z0-9\s\-]{0,30}?)(?:\s+from\s+now|\s*[.?!,]|$)", 1),
    (r"(?:as|like)\s+([a-zA-Z][a-zA-Z0-9\s\-]{0,30}?)(?:\s*[.?!,]|$)", 1),
    (r"call\s+(?:you|yourself)\s+([a-zA-Z][a-zA-Z0-9\s\-]{0,30}?)(?:\s*[.?!,]|$)", 1),
    (r"(?:be\s+)?(?:called|named)\s+([a-zA-Z][a-zA-Z0-9\s\-]{0,30}?)(?:\s+from\s+now|\s*[.?!,]|$)", 1),
    (r"([a-zA-Z][a-zA-Z0-9\-]{1,30})\s+from\s+now\s+on", 1),
    (r"([a-zA-Z][a-zA-Z0-9\-]{1,30})\s*[.?!,]?\s*$", 1),  # last word/phrase
]


def _has_wake_word_change_intent(normalized_utterance: str) -> bool:
    """True if utterance semantically suggests changing the wake word / assistant name."""
    return any(kw in normalized_utterance for kw in _WAKE_WORD_CHANGE_INTENT_KEYWORDS)


def _extract_new_wake_word(utterance: str, current_wake: str, last_utterance: Optional[str] = None) -> Optional[str]:
    """
    Detect wake-word-change intent by semantic meaning, then extract the new word.
    Triggers on: "change wake word to X", "rename you to X", "call you X", "I want to call you Bob", etc.
    If X is "that"/"this"/"it", use last_utterance when provided.
    """
    u = _normalize(utterance)
    current = (current_wake or "").strip().lower()
    if current and u.startswith(current):
        u = u[len(current):].lstrip(" ,;:")
    if not _has_wake_word_change_intent(u):
        return None
    for pat, grp in _WAKE_WORD_EXTRACT_PATTERNS:
        m = re.search(pat, u)
        if m:
            raw = (m.group(grp) if isinstance(grp, int) else m.group(0)).strip()
            raw = re.sub(r"[.?!,;:]+$", "", raw)
            if not raw or len(raw) > 50:
                continue
            if raw in ("that", "this", "it") and last_utterance:
                words = (last_utterance or "").strip().split()
                return words[-1] if words else None
            if raw in ("yes", "no", "change", "name", "word", "you", "me"):
                continue
            return raw
    return None


def parse_and_route(
    user_text: str,
    profile: Dict[str, Any],
    memory_summary: str,
    listen_only: bool,
    trigger_phrases: List[str],
    pending_wake_word_change: Optional[str] = None,
    last_utterance: Optional[str] = None,
) -> IntentResult:
    """
    Run intent routing. Returns (handled, response_text, extra).
    - If handled and response_text is set: session should speak response_text and not call LLM.
    - If handled and response_text is None but extra has e.g. "fact_check": True: session should do fact-check flow.
    - If not handled: session should proceed to LLM/RAG with compiled context.
    """
    u = _normalize(user_text)
    extra: Dict[str, Any] = {}

    # ----- Confirm wake word change (user said "yes" after we proposed a change) -----
    if pending_wake_word_change:
        if _match_any(user_text, ["yes", "yeah", "confirm", "change it", "do it", "sure", "ok", "okay"]):
            extra["confirm_wake_word_change"] = True
            return True, f"Done. Wake word is now {pending_wake_word_change}. Say '{pending_wake_word_change}' when you want me to respond.", extra
        if _match_any(user_text, ["no", "cancel", "never mind", "forget it", "nope"]):
            extra["clear_pending_wake_word_change"] = True
            return True, "Okay, I won't change the wake word.", extra

    # ----- Change wake word (requires confirmation) -----
    new_wake = _extract_new_wake_word(user_text, profile.get("wake_word") or "", last_utterance)
    if new_wake and len(new_wake) < 50:
        extra["pending_wake_word_change"] = new_wake
        return True, f"Do you want to change the wake word to {new_wake}? Say yes to confirm.", extra

    # ----- Assistant name (immediate, no confirmation) -----
    for pattern in ["your name is ", "call yourself ", "wake word is ", "you're called "]:
        name = _extract_after(user_text, [pattern])
        if name and len(name) < 80:
            extra["set_assistant_name"] = name.strip()
            return True, f"Got it. I'll respond to the name {name.strip()}.", extra

    # ----- User name ----- (avoid "i'm"/"i am" so "I'm in X" is not parsed as name)
    for pattern in ["my name is ", "call me "]:
        name = _extract_after(user_text, [pattern])
        if name and len(name) < 80:
            extra["set_user_name"] = name.strip()
            return True, f"Nice to meet you, {name.strip()}.", extra

    # ----- Timezone -----
    if _match_any(user_text, ["set timezone to ", "timezone is ", "my timezone is ", "i'm in timezone "]):
        m = re.search(r"(?:timezone|time zone)\s+(?:is|to)?\s*([\w/\s+-]+?)(?:\s*\.|$)", u, re.I)
        if not m:
            m = re.search(r"(?:set\s+)?timezone\s+to\s+([\w/\s+-]+)", u, re.I)
        if m:
            tz = m.group(1).strip()
            if len(tz) < 60:
                extra["set_timezone"] = tz
                return True, f"Timezone set to {tz}.", extra

    # ----- Location -----
    for pattern in ["i'm in ", "i am in ", "location is ", "i'm at ", "set location to "]:
        loc = _extract_after(user_text, [pattern])
        if loc and len(loc) < 120:
            extra["set_location"] = loc.strip()
            return True, f"Noted. Location: {loc.strip()}.", extra

    # ----- Start listening (enter listen-only) -----
    if _match_any(user_text, ["listen to conversation", "start listening", "just listen", "keep listening"]):
        extra["set_listen_only"] = True
        # The chained `or` of literals always resolved to the wake word and never the exit phrases —
        # resolve the wake word once and name the exit phrase explicitly. (audit M1/L1)
        wake = (profile.get("wake_word") or "EchoMind").strip()
        return True, f"I am starting to listen. Say '{wake}' or 'stop listening' when you want me to respond.", extra

    # # ----- Stop listening (exit listen-only) -----
    # if _match_any(user_text, ["stop listening", "pause listening", "pause", "don't listen"]):
    #     extra["set_listen_only"] = False
    #     return True, "Stopped listening. Say 'start listening' when you want me to listen again.", extra
    # # "stop" alone: only treat as stop listening if it's the only word (clear intent)
    # if u.strip() == "stop":
    #     extra["set_listen_only"] = False
    #     return True, "Stopped listening. Say 'start listening' when you want me to listen again.", extra

    # ----- Resume listening -----
    if _match_any(user_text, ["resume listening", "resume", "start listening again"]):
        extra["set_listen_only"] = True
        return True, "Resuming. I'm listening again.", extra

    # ----- Clear memory -----
    if _match_any(user_text, ["clear memory", "clear conversation", "forget everything", "reset memory"]):
        extra["clear_memory"] = True
        return True, "Memory cleared.", extra

    # ----- Query: what did I say (last N minutes) -----
    mins = _extract_minutes(user_text)
    if mins is not None and _match_any(user_text, ["what did i say", "what did we say", "what was said", "recap", "last minutes"]):
        extra["memory_query_minutes"] = mins
        extra["memory_query_type"] = "recap"
        # Session will fill summary and either speak or pass to LLM
        return True, None, extra

    # ----- Summarize last N minutes -----
    if mins is not None and _match_any(user_text, ["summarize", "summary", "summarise"]):
        extra["memory_query_minutes"] = mins
        extra["memory_query_type"] = "summarize"
        return True, None, extra

    # ----- When did we mention X -----
    if _match_any(user_text, ["when did we", "when did i", "when did you", "when was", "when did we mention", "when did we talk about"]):
        extra["memory_query_type"] = "when_mentioned"
        extra["memory_query_topic"] = u  # session can use for query_topic
        return True, None, extra

    # ----- Timestamps and tags -----
    if _match_any(user_text, ["timestamps and tags", "give timestamps", "list with timestamps", "who said what"]):
        extra["memory_query_type"] = "timestamps_tags"
        return True, None, extra

    # ----- Fact check -----
    if _match_any(user_text, ["fact check", "fact check it", "fact check that", "verify that", "verify it"]):
        extra["fact_check"] = True
        return True, None, extra

    return False, None, extra


def strip_wake_word(utterance: str, wake_word: str) -> str:
    """Remove leading wake word (case-insensitive) and comma/space. Return trimmed rest."""
    if not (wake_word or "").strip():
        return (utterance or "").strip()
    u = (utterance or "").strip()
    w = wake_word.strip().lower()
    if not w:
        return u
    if u.lower().startswith(w):
        rest = u[len(w):].lstrip(" ,;:")
        return rest.strip()
    return u
