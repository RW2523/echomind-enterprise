"""
Streaming ASR service: buffers incoming audio to model chunk size, runs ASR, emits partial/final.
Simple endpointing: finalize after SILENCE_MS_BEFORE_FINAL of no speech (or empty partial).
"""
import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Optional

import numpy as np

from app.asr_model_adapter import ASRModelAdapter
from app.audio_utils import TARGET_SAMPLE_RATE, process_audio_chunk
from app.config import (
    ASR_ATT_CONTEXT_RIGHT,
    ASR_MODEL_NAME,
    AUDIO_CHUNK_MS,
    SILENCE_MS_BEFORE_FINAL,
    MIN_PARTIAL_LENGTH_FOR_FINAL,
)

logger = logging.getLogger(__name__)

# Dedicated executor for GPU ASR so we don't block the event loop or compete with other threads
_ASR_EXECUTOR = None

def _get_asr_executor():
    global _ASR_EXECUTOR
    if _ASR_EXECUTOR is None:
        import concurrent.futures
        _ASR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
    return _ASR_EXECUTOR

# Model expects chunks in 80ms frames. right=0 -> 1 frame = 80ms (low latency)
FRAME_MS = 80
SAMPLES_PER_FRAME = int(TARGET_SAMPLE_RATE * FRAME_MS / 1000)


def _samples_for_chunk_ms(ms: int) -> int:
    return int(TARGET_SAMPLE_RATE * ms / 1000)


class StreamingSession:
    """One WebSocket session: buffer audio, run ASR, track partial/final."""

    def __init__(
        self,
        session_id: str,
        asr: ASRModelAdapter,
        on_partial,
        on_final,
        on_status,
        on_error,
        silence_ms_before_final: int = SILENCE_MS_BEFORE_FINAL,
        min_partial_len_for_final: int = MIN_PARTIAL_LENGTH_FOR_FINAL,
    ):
        self.session_id = session_id
        self.asr = asr
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_status = on_status
        self._on_error = on_error
        self.silence_ms = silence_ms_before_final
        self.min_partial_len = min_partial_len_for_final
        self.state = asr.create_session_state()
        self.buffer: deque = deque()
        self.buffer_samples = 0
        self.last_partial: Optional[str] = None
        self.last_partial_time: float = 0.0
        self._lock = asyncio.Lock()
        self._run_task: Optional[asyncio.Task] = None
        self._closed = False
        self.client_sample_rate: Optional[int] = None  # set from client config
        self._stream_step_num: int = 0

    async def _call_on_partial(self, text: str, ts: float) -> None:
        if asyncio.iscoroutinefunction(self._on_partial):
            await self._on_partial(self.session_id, text, ts)
        else:
            self._on_partial(self.session_id, text, ts)

    async def _call_on_final(self, text: str, ts: float) -> None:
        if asyncio.iscoroutinefunction(self._on_final):
            await self._on_final(self.session_id, text, ts)
        else:
            self._on_final(self.session_id, text, ts)

    async def push_audio(self, payload: bytes, sample_rate: Optional[int] = None) -> None:
        """Append raw PCM and process when we have enough for one model chunk."""
        sr = sample_rate if sample_rate is not None else self.client_sample_rate or TARGET_SAMPLE_RATE
        try:
            samples, _ = process_audio_chunk(payload, sample_rate=sr)
        except Exception as e:
            logger.exception("Audio decode error")
            if asyncio.iscoroutinefunction(self._on_error):
                await self._on_error(self.session_id, str(e))
            else:
                self._on_error(self.session_id, str(e))
            return
        async with self._lock:
            self.buffer.append(samples)
            self.buffer_samples += len(samples)
        await self._maybe_process()

    def _chunk_samples_needed(self) -> int:
        """Samples needed for one model step (e.g. 160ms = 2 frames)."""
        return _samples_for_chunk_ms(AUDIO_CHUNK_MS)

    async def _maybe_process(self) -> None:
        """Process one chunk if buffer has enough samples."""
        needed = self._chunk_samples_needed()
        async with self._lock:
            if self.buffer_samples < needed:
                return
            chunks = []
            while self.buffer and sum(len(c) for c in chunks) < needed:
                c = self.buffer.popleft()
                self.buffer_samples -= len(c)
                chunks.append(c)
            if not chunks:
                return
            combined = np.concatenate(chunks).astype(np.float32)
            if len(combined) > needed:
                audio = combined[:needed]
                remainder = combined[needed:]
                self.buffer.appendleft(remainder)
                self.buffer_samples += len(remainder)
            else:
                audio = combined

        step_num = self._stream_step_num
        self._stream_step_num += 1
        try:
            loop = asyncio.get_event_loop()
            text, self.state = await loop.run_in_executor(
                _get_asr_executor(),
                lambda: self.asr.process_chunk(
                    audio, self.state, keep_all_outputs=False, step_num=step_num
                ),
            )
        except Exception as e:
            logger.exception("ASR process error")
            if asyncio.iscoroutinefunction(self._on_error):
                await self._on_error(self.session_id, str(e))
            else:
                self._on_error(self.session_id, str(e))
            return

        now = time.time()
        if text:
            if self.last_partial and not text.startswith(self.last_partial):
                if text.startswith((" ", "-", ",", ".")) or self.last_partial.endswith((" ", "-")):
                    self.last_partial = self.last_partial + text
                else:
                    self.last_partial = self.last_partial + " " + text
            else:
                self.last_partial = text
            self.last_partial_time = now
            await self._call_on_partial(self.last_partial, now)
        else:
            if (
                self.last_partial
                and len(self.last_partial.strip()) >= self.min_partial_len
                and (now - self.last_partial_time) * 1000 >= self.silence_ms
            ):
                await self._call_on_final(self.last_partial.strip(), now)
                self.last_partial = None
                self.last_partial_time = 0.0

    async def flush(self) -> None:
        """Flush remaining buffer and finalize any last partial."""
        async with self._lock:
            if not self.buffer:
                if self.last_partial and len(self.last_partial.strip()) >= self.min_partial_len:
                    await self._call_on_final(self.last_partial.strip(), time.time())
                self.last_partial = None
                return
            chunks = list(self.buffer)
            self.buffer.clear()
            self.buffer_samples = 0
        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
            step_num = self._stream_step_num
            self._stream_step_num += 1
            try:
                loop = asyncio.get_event_loop()
                text, self.state = await loop.run_in_executor(
                    _get_asr_executor(),
                    lambda: self.asr.process_chunk(
                        audio, self.state, keep_all_outputs=True, step_num=step_num
                    ),
                )
                if text:
                    await self._call_on_partial(text, time.time())
                    if len(text.strip()) >= self.min_partial_len:
                        await self._call_on_final(text.strip(), time.time())
            except Exception as e:
                logger.exception("ASR flush error")
                if asyncio.iscoroutinefunction(self._on_error):
                    await self._on_error(self.session_id, str(e))
                else:
                    self._on_error(self.session_id, str(e))
        elif self.last_partial and len(self.last_partial.strip()) >= self.min_partial_len:
            await self._call_on_final(self.last_partial.strip(), time.time())
        self.last_partial = None

    def close(self) -> None:
        self._closed = True


class StreamingASRService:
    """Creates and tracks streaming sessions; holds single ASR model."""

    def __init__(self):
        self.asr = ASRModelAdapter(
            model_name=ASR_MODEL_NAME,
            att_context_right=ASR_ATT_CONTEXT_RIGHT,
        )
        self._sessions: dict[str, StreamingSession] = {}

    def ensure_loaded(self) -> None:
        self.asr.load()

    def create_session(self, on_partial, on_final, on_status, on_error) -> StreamingSession:
        session_id = str(uuid.uuid4())
        session = StreamingSession(
            session_id=session_id,
            asr=self.asr,
            on_partial=on_partial,
            on_final=on_final,
            on_status=on_status,
            on_error=on_error,
        )
        self._sessions[session_id] = session
        return session

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
