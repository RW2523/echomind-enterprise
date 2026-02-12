import json
import logging
import requests

logger = logging.getLogger(__name__)


def _log_chat_request(url: str, payload: dict) -> None:
    """Log full prompt (no content cut)."""
    logger.info("LLM request sync -> %s full_payload=%s", url, json.dumps(payload, ensure_ascii=False))


class OpenAICompatLLM:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def chat(self, user_text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a realtime voice assistant. Be concise, helpful, and conversational."},
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.7,
            "max_tokens": 220,
            "stream": False
        }
        _log_chat_request(self.url, payload)
        r = requests.post(self.url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
