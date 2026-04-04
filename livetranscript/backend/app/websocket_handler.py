"""WebSocket endpoint: accept audio binary frames, send JSON transcript events."""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.config import WS_PATH
from app.streaming_service import StreamingASRService

logger = logging.getLogger(__name__)


def _message(
    type: str,
    text: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    detail: Optional[str] = None,
) -> dict:
    o = {
        "type": type,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    if session_id is not None:
        o["session_id"] = session_id
    if text is not None:
        o["text"] = text
    if start_ms is not None:
        o["start_ms"] = start_ms
    if end_ms is not None:
        o["end_ms"] = end_ms
    if detail is not None:
        o["detail"] = detail
    return o


async def handle_transcribe_ws(websocket: WebSocket, service: StreamingASRService) -> None:
    await websocket.accept()

    async def _send(ws: WebSocket, typ: str, text: str, sid: str) -> None:
        try:
            await ws.send_json(_message(type=typ, text=text, session_id=sid))
        except Exception as e:
            logger.warning("Send %s failed: %s", typ, e)

    async def _send_status(ws: WebSocket, msg: str, sid: str) -> None:
        try:
            await ws.send_json(_message(type="status", detail=msg, session_id=sid))
        except Exception as e:
            logger.warning("Send status failed: %s", e)

    async def _send_error(ws: WebSocket, msg: str, sid: str) -> None:
        try:
            await ws.send_json(_message(type="error", detail=msg, session_id=sid))
        except Exception as e:
            logger.warning("Send error failed: %s", e)

    session: Optional[object] = None
    try:
        async def on_partial(sid: str, text: str, _ts: float) -> None:
            await _send(websocket, "partial", text, sid)
        async def on_final(sid: str, text: str, _ts: float) -> None:
            await _send(websocket, "final", text, sid)
        async def on_status_cb(sid: str, msg: str) -> None:
            await _send_status(websocket, msg, sid)
        async def on_error_cb(sid: str, msg: str) -> None:
            await _send_error(websocket, msg, sid)

        session = service.create_session(
            on_partial=on_partial,
            on_final=on_final,
            on_status=on_status_cb,
            on_error=on_error_cb,
        )
        sid = session.session_id

        await websocket.send_json(
            _message(type="status", detail="connected", session_id=sid)
        )

        while True:
            data = await websocket.receive()
            if "bytes" in data:
                payload = data["bytes"]
                if payload:
                    await session.push_audio(payload, sample_rate=session.client_sample_rate)
            elif "text" in data:
                try:
                    cmd = json.loads(data["text"])
                    if cmd.get("type") == "end_utterance":
                        if session:
                            await session.flush()
                        continue
                    if cmd.get("type") == "config" and "sample_rate" in cmd:
                        session.client_sample_rate = int(cmd["sample_rate"])
                except (json.JSONDecodeError, ValueError):
                    pass
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_json(
                _message(type="error", detail=str(e))
            )
        except Exception:
            pass
    finally:
        if session:
            await session.flush()
            service.remove_session(session.session_id)
        try:
            await websocket.close()
        except Exception:
            pass
