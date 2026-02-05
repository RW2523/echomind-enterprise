import json
import requests
from typing import Iterator, List, Dict

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
        r = requests.post(self.url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
