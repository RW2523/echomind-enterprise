"""
Piper TTS voice download from Hugging Face (rhasspy/piper-voices).
List installed voices and download a voice by id (e.g. en_US-lessac-medium).
"""
from __future__ import annotations
import os
import re
import urllib.request
from typing import List, Tuple

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def get_voices_dir() -> str:
    """Directory where .onnx and .onnx.json files are stored (e.g. /voices)."""
    return os.getenv("VOICES_DIR", "/voices")


def list_installed_voices() -> List[str]:
    """Return list of voice ids that have both .onnx and .onnx.json in the voices dir."""
    root = get_voices_dir()
    if not os.path.isdir(root):
        return []
    seen: set = set()
    for name in os.listdir(root):
        if name.endswith(".onnx") and not name.endswith(".onnx.json"):
            voice_id = name[:-5]  # strip .onnx
            if os.path.isfile(os.path.join(root, f"{voice_id}.onnx.json")):
                seen.add(voice_id)
    return sorted(seen)


def voice_id_to_hf_path(voice_id: str) -> Tuple[str, str]:
    """
    Map Piper voice id (e.g. en_US-lessac-medium) to Hugging Face path components.
    Returns (onnx_path, json_path) relative to repo root, e.g.
    en/en_US/lessac/medium/en_US-lessac-medium.onnx
    """
    voice_id = (voice_id or "").strip()
    if not voice_id or not re.match(r"^[a-z]{2}_[A-Z]{2}-[a-z0-9_]+-(?:low|medium|high)$", voice_id, re.IGNORECASE):
        raise ValueError(f"Invalid Piper voice id: {voice_id}")
    parts = voice_id.split("-")
    if len(parts) < 4:
        raise ValueError(f"Invalid Piper voice id: {voice_id}")
    locale = f"{parts[0]}_{parts[1]}"
    speaker = parts[2]
    quality = parts[3]
    subpath = f"en/{locale}/{speaker}/{quality}/{voice_id}"
    return f"{subpath}.onnx", f"{subpath}.onnx.json"


def download_voice(voice_id: str) -> str:
    """
    Download the given Piper voice (onnx + json) to the voices dir.
    Returns the path to the .onnx file. Raises on failure.
    """
    root = get_voices_dir()
    os.makedirs(root, exist_ok=True)
    onnx_rel, json_rel = voice_id_to_hf_path(voice_id)
    onnx_url = f"{HF_BASE}/{onnx_rel}"
    json_url = f"{HF_BASE}/{json_rel}"
    onnx_path = os.path.join(root, f"{voice_id}.onnx")
    json_path = os.path.join(root, f"{voice_id}.onnx.json")

    def _download(url: str, path: str) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "EchoMind-Voice/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(path, "wb") as f:
                f.write(resp.read())

    try:
        _download(onnx_url, onnx_path)
        _download(json_url, json_path)
    except Exception as e:
        for p in (onnx_path, json_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise RuntimeError(f"Download failed for {voice_id}: {e}") from e
    return onnx_path
