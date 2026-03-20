import json
import logging
import time
import requests
from typing import Dict, Iterator, List, Optional

from ..config import SETTINGS

logger = logging.getLogger(__name__)


def _log_chat_request(url: str, payload: dict, stream: bool) -> None:
    """Log full LLM request payload when LLM_LOG_PAYLOAD is enabled. Uses WARNING so it shows with default log level."""
    if not getattr(SETTINGS, "LLM_LOG_PAYLOAD", False):
        return
    mode = "stream" if stream else "sync"
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.warning(
        "[VOICE_LLM_REQUEST] %s -> %s\nfull_payload=%s",
        mode,
        url,
        payload_json,
    )


class OpenAICompatLLMStream:
    def __init__(self, url: str, model: str, temperature: float = 0.7, max_tokens: int = 220):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream_messages(self, messages: List[Dict], request_timeout: int = 120) -> Iterator[str]:
        """OpenAI-compatible chat with ``stream: true`` for low time-to-first-token (e.g. TensorRT-LLM)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        _log_chat_request(self.url, payload, stream=True)
        logger.info(
            "VOICE_LLM stream start url=%s model=%s stream=true prompt_msgs=%d max_tokens=%d temperature=%s",
            self.url,
            self.model,
            len(messages),
            self.max_tokens,
            self.temperature,
        )

        t0 = time.monotonic()
        ttft_mono: Optional[float] = None
        content_chunks = 0
        out_chars = 0
        sse_data_lines = 0
        http_status: Optional[int] = None

        try:
            with requests.post(self.url, json=payload, stream=True, timeout=request_timeout) as r:
                http_status = r.status_code
                r.raise_for_status()
                logger.info("VOICE_LLM stream connected http_status=%s", http_status)
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        data = raw[len("data:") :].strip()
                    else:
                        data = raw.strip()

                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    sse_data_lines += 1
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content")
                    if token:
                        if ttft_mono is None:
                            ttft_mono = time.monotonic()
                        content_chunks += 1
                        out_chars += len(token)
                        yield token
        finally:
            t1 = time.monotonic()
            total_ms = (t1 - t0) * 1000
            ttft_ms = (ttft_mono - t0) * 1000 if ttft_mono is not None else 0.0
            logger.info(
                "VOICE_LLM stream done model=%s stream=true ttft_ms=%.1f stream_total_ms=%.1f "
                "sse_data_events=%d content_chunks=%d output_chars=%d prompt_msgs=%d http_status=%s",
                self.model,
                ttft_ms,
                total_ms,
                sse_data_lines,
                content_chunks,
                out_chars,
                len(messages),
                http_status if http_status is not None else "?",
            )

    def complete_messages(self, messages: List[Dict]) -> str:
        """Non-streaming completion (summaries / tool-style paths). Prefer ``stream_messages`` for dialogue."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        _log_chat_request(self.url, payload, stream=False)
        logger.info(
            "VOICE_LLM sync start url=%s model=%s stream=false prompt_msgs=%d max_tokens=%d",
            self.url,
            self.model,
            len(messages),
            self.max_tokens,
        )
        t0 = time.monotonic()
        r = requests.post(self.url, json=payload, timeout=90)
        http_status = r.status_code
        r.raise_for_status()
        data = r.json()
        out = (data["choices"][0]["message"]["content"] or "").strip()
        elapsed_ms = (time.monotonic() - t0) * 1000
        usage = data.get("usage") or {}
        logger.info(
            "VOICE_LLM sync done model=%s stream=false total_ms=%.1f http_status=%s completion_chars=%d "
            "usage_prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            elapsed_ms,
            http_status,
            len(out),
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )
        return out
