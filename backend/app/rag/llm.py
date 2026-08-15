from __future__ import annotations
import json
import logging
import os
import re
import time
import httpx
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Logging full prompts means writing every uploaded-document / transcript excerpt to logs
# in plaintext (PII / confidential content). Off by default; opt in only for debugging.
_LOG_FULL_PROMPTS = os.getenv("ECHOMIND_LOG_FULL_PROMPTS", "0").lower() in ("1", "true", "yes")


def _log_chat_request(url: str, payload: dict, stream: bool) -> None:
    """Log chat/completions request. Logs only metadata at INFO; the full prompt (which
    contains retrieved document/transcript content) is logged at DEBUG and only when
    ECHOMIND_LOG_FULL_PROMPTS is enabled, to avoid leaking sensitive content into logs."""
    messages = payload.get("messages") or []
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    logger.info(
        "LLM request %s -> %s/chat/completions model=%s msgs=%d prompt_chars=%d max_tokens=%s",
        "stream" if stream else "sync",
        url,
        payload.get("model"),
        len(messages),
        total_chars,
        payload.get("max_tokens"),
    )
    if _LOG_FULL_PROMPTS and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "LLM full_payload (%s) %s",
            "stream" if stream else "sync",
            json.dumps(payload, ensure_ascii=False),
        )


# Reasoning models (e.g. Qwen3) emit <think> blocks by default — slow + noisy for chat/voice.
# Disabled by default; set LLM_ENABLE_THINKING=1 to re-enable (e.g. for batch report quality).
_ENABLE_THINKING = (__import__("os").getenv("LLM_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes"))
_EXTRA = {} if _ENABLE_THINKING else {"chat_template_kwargs": {"enable_thinking": False}}

# Greedy-ish decoding (temp 0.2) with no repetition suppression let Qwen3 fall into exact
# repetition loops — observed as an answer tail of "page 22, " repeated hundreds of times
# while emitting inline citations. A gentle frequency penalty makes the loop unlikely;
# the tail-loop guard below removes it when it still happens. TRT-LLM's OpenAI endpoint
# accepts both penalty fields (verified against trtllm-serve 1.2.0rc6).
_FREQ_PENALTY = float(os.getenv("ECHOMIND_LLM_FREQ_PENALTY", "0.2"))
_PRES_PENALTY = float(os.getenv("ECHOMIND_LLM_PRES_PENALTY", "0.0"))
if _FREQ_PENALTY:
    _EXTRA["frequency_penalty"] = _FREQ_PENALTY
if _PRES_PENALTY:
    _EXTRA["presence_penalty"] = _PRES_PENALTY


# ── Repetition-loop guard ───────────────────────────────────────────────────────
# Detects a short unit repeated verbatim at the tail of the text (the signature of a
# decode-time degeneration loop). The unit must contain a letter or digit so that
# legitimate runs of dashes/pipes (tables, rules) never trigger it.

def _find_tail_loop(text: str, min_unit: int = 4, max_unit: int = 60,
                    min_reps: int = 4, min_span: int = 24):
    """Return (unit, reps) if text ends with `unit` repeated >= min_reps times
    covering >= min_span chars, else None."""
    tail = text[-600:]
    n = len(tail)
    for unit_len in range(min_unit, min(max_unit, n // min_reps) + 1):
        unit = tail[n - unit_len:]
        if not any(ch.isalnum() for ch in unit):
            continue
        reps = 1
        i = n - unit_len
        while i - unit_len >= 0 and tail[i - unit_len:i] == unit:
            reps += 1
            i -= unit_len
        if reps >= min_reps and reps * unit_len >= min_span:
            return unit, reps
    return None


_SENT_TAIL_RE = re.compile(r'[.!?:]["\'\u201d\u2019)\]]*\s')


def trim_to_sentence_end(text: str) -> str:
    """If an answer stops mid-sentence (token cap, stream cut), trim back to the last
    complete sentence — provided that loses at most ~15% of the answer. A truncated
    answer that ends on a full stop reads finished; one that stops mid-word reads broken."""
    t = (text or "").rstrip()
    if not t or t[-1] in '.!?:"\'\u201d\u2019)]':
        return text
    last = None
    for m in _SENT_TAIL_RE.finditer(t):
        last = m.end()
    if last and (len(t) - last) <= max(200, int(len(t) * 0.15)):
        cut = t[:last].rstrip()
        logger.warning("LLM: mid-sentence tail trimmed — removed %d chars", len(t) - len(cut))
        return cut
    return text


def finalize_answer(text: str) -> str:
    """Post-generation cleanup applied to every final answer: collapse degeneration
    loops, then land the ending on a sentence boundary."""
    return trim_to_sentence_end(trim_tail_repetition(text))


def trim_tail_repetition(text: str) -> str:
    """Collapse a degeneration loop at the end of an answer to a single occurrence."""
    if not text:
        return text
    hit = _find_tail_loop(text)
    if not hit:
        return text
    unit, _ = hit
    # Unwind EVERY trailing occurrence (the detector's 600-char window under-counts
    # long loops), then keep exactly one so the sentence still reads naturally.
    cut = len(text)
    while cut >= len(unit) and text[cut - len(unit):cut] == unit:
        cut -= len(unit)
    cleaned = (text[:cut] + unit).rstrip(" ,;·-")
    logger.warning(
        "LLM: repetition loop trimmed — unit=%r removed %d trailing chars",
        unit[:40], len(text) - len(cleaned),
    )
    return cleaned


# ── Reasoning-leak guard ────────────────────────────────────────────────────────
# Qwen3-30B on TRT-LLM intermittently (~6% of short answers, sampling-dependent) fails to
# emit EOS: it produces a complete answer, then a stray token ("RefreshLayout", "移除",
# "Feedback", "<think>"), then chain-of-thought about the user's request. thinking is
# already disabled via chat_template_kwargs and finish_reason is usually "stop", so this is
# a decode-time artifact, not a prompt bug. Cut the answer at the leak boundary.
_LEAK_RE = re.compile(
    r"(?:<think>|"
    r"(?:^|\n)\s*(?:Okay|Alright|Hmm|So)\s*,?\s+(?:the user|I need to|I should|let me|we need)\b|"
    r"(?:^|\n)\s*The user (?:asked|wants|is asking)\b)",
    re.IGNORECASE,
)


def strip_reasoning_leak(text: str) -> str:
    """Truncate an answer at leaked chain-of-thought. Returns the clean prefix."""
    if not text:
        return text
    m = _LEAK_RE.search(text)
    if not m or m.start() == 0:
        # start==0 means the WHOLE response is reasoning — keep it rather than blank the answer
        return text
    cleaned = text[: m.start()].rstrip()
    # Drop a trailing stray token left on the last line (e.g. "... 😊\nRefreshLayout")
    lines = cleaned.split("\n")
    if len(lines) > 1 and 0 < len(lines[-1].strip()) <= 24 and " " not in lines[-1].strip():
        lines = lines[:-1]
        cleaned = "\n".join(lines).rstrip()
    logger.warning(
        "LLM: reasoning leak detected — truncated %d trailing chars", len(text) - len(cleaned)
    )
    return cleaned


class OpenAICompatChat:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages, temperature: float, max_tokens: int) -> str:
        payload={"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens,"stream":False, **_EXTRA}
        _log_chat_request(self.base_url, payload, stream=False)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            j=r.json()
            out = finalize_answer(strip_reasoning_leak((j["choices"][0]["message"]["content"] or "").strip()))
        elapsed_ms = (time.monotonic() - t0) * 1000
        usage = j.get("usage") or {}
        logger.info(
            "LLM sync done model=%s total_ms=%.1f http_status=%s prompt_msgs=%d completion_chars=%d "
            "usage_prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            elapsed_ms,
            getattr(r, "status_code", "?"),
            len(messages),
            len(out),
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )
        return out

    async def chat_stream(self, messages, temperature: float, max_tokens: int) -> AsyncIterator[str]:
        """Stream LLM response token-by-token (SSE). Yields content deltas."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True, **_EXTRA}
        _log_chat_request(self.base_url, payload, stream=True)
        t0 = time.monotonic()
        ttft_mono: float | None = None
        sse_data_events = 0
        content_deltas = 0
        _acc = ""
        total_chars = 0

        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or line.strip() != line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        sse_data_events += 1
                        try:
                            j = json.loads(data)
                            delta = (j.get("choices") or [{}])[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                # Reasoning-leak guard (streaming): once chain-of-thought
                                # starts leaking, stop the stream rather than emit it.
                                _acc += content
                                if _LEAK_RE.search(_acc[-400:]):
                                    logger.warning("LLM stream: reasoning leak detected — cutting stream")
                                    break
                                if _find_tail_loop(_acc):
                                    logger.warning("LLM stream: repetition loop detected — cutting stream")
                                    break
                                if ttft_mono is None:
                                    ttft_mono = time.monotonic()
                                content_deltas += 1
                                total_chars += len(content)
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        t1 = time.monotonic()
        ttft_ms = (ttft_mono - t0) * 1000 if ttft_mono is not None else 0.0
        logger.info(
            "LLM stream done model=%s ttft_ms=%.1f stream_total_ms=%.1f sse_data_events=%d "
            "content_deltas=%d output_chars=%d prompt_msgs=%d",
            self.model,
            ttft_ms,
            (t1 - t0) * 1000,
            sse_data_events,
            content_deltas,
            total_chars,
            len(messages),
        )
