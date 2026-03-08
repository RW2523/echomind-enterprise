import json
import logging
import requests
from typing import Iterator, List, Dict

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
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        _log_chat_request(self.url, payload, stream=True)
        with requests.post(self.url, json=payload, stream=True, timeout=request_timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data:"):
                    data = raw[len("data:"):].strip()
                else:
                    data = raw.strip()

                if data == "[DONE]":
                    return

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    yield token

    def complete_messages(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        _log_chat_request(self.url, payload, stream=False)
        r = requests.post(self.url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
