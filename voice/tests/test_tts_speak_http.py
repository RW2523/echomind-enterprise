"""HTTP POST /tts/speak (Speak Now) tests."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

_VOICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VOICE_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_ROOT)

os.environ.setdefault("VOICE_NEMOTRON_STARTUP_LOAD", "0")

pytest.importorskip("httpx")


@pytest.fixture
def app_client():
    from app.server import app

    return app


@pytest.mark.asyncio
async def test_tts_speak_returns_wav(app_client):
    fake_audio = np.zeros(800, dtype=np.float32)

    class FakeTTS:
        sr = 22050

        def synth(self, text: str):
            assert "Hello" in text
            return fake_audio

    with patch("app.tts_oneoff._cached_piper", return_value=FakeTTS()):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post("/tts/speak", json={"text": "Hello from test."})
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("audio/wav")
    assert r.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_tts_speak_whitespace_rejected(app_client):
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/tts/speak", json={"text": "    "})
    assert r.status_code == 400
