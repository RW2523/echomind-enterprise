"""
Persistent storage for the wake word. Load on startup, save on change.
Uses a JSON file (default: /voices/wake_word.json or WAKE_WORD_FILE env).
"""
from __future__ import annotations
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.getenv("WAKE_WORD_FILE", "/voices/wake_word.json")


def load_wake_word(default: str = "EchoMind") -> str:
    """Load wake word from storage. Returns default if file missing or invalid."""
    for path in [DEFAULT_PATH, "/tmp/wake_word.json"]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                w = (data.get("wake_word") or "").strip()
                if w:
                    return w
        except Exception as e:
            logger.warning("Failed to load wake word from %s: %s", path, e)
    return default


def save_wake_word(wake_word: str) -> bool:
    """Save wake word to storage. Returns True on success."""
    for path in [DEFAULT_PATH, "/tmp/wake_word.json"]:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"wake_word": wake_word.strip()}, f, indent=2)
            return True
        except Exception as e:
            logger.warning("Failed to save wake word to %s: %s", path, e)
    return False
