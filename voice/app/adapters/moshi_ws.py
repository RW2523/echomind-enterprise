import asyncio
import base64
import json
import websockets

class MoshiWsAdapter:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self._out_q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._closed = False
        self._recv_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=20)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self):
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _recv_loop(self):
        while not self._closed and self.ws:
            raw = await self.ws.recv()
            if isinstance(raw, bytes):
                await self._out_q.put({"type": "audio_out_bin", "data": raw})
                continue
            try:
                await self._out_q.put(json.loads(raw))
            except Exception:
                continue

    async def recv(self) -> dict:
        return await self._out_q.get()

    async def send_audio(self, pcm16: bytes, sr: int):
        if not self.ws:
            return
        msg = {"type": "audio", "sample_rate": sr, "pcm16_b64": base64.b64encode(pcm16).decode("utf-8")}
        await self.ws.send(json.dumps(msg))

    async def text_inject(self, text: str, generation_id: int):
        if not self.ws:
            return
        await self.ws.send(json.dumps({"type": "text_inject", "text": text, "generation_id": generation_id}))

    async def cancel(self, generation_id: int):
        if not self.ws:
            return
        await self.ws.send(json.dumps({"type": "cancel", "generation_id": generation_id}))
