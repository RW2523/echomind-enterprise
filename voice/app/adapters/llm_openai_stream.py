import json
import logging
import re
import socket
import threading
import time
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

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


# Reasoning models (e.g. Qwen3) emit <think> blocks by default — kills voice TTFT.
# Disabled by default; set LLM_ENABLE_THINKING=1 to re-enable.
_ENABLE_THINKING = (__import__("os").getenv("LLM_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes"))
_EXTRA = {} if _ENABLE_THINKING else {"chat_template_kwargs": {"enable_thinking": False}}


# ── Tool-call events ───────────────────────────────────────────────────────────
# trtllm-serve runs without --tool_parser, so a Qwen3 tool call arrives as TEXT:
#   "<tool_call>\n{\"name\": ..., \"arguments\": {...}}\n</tool_call>"
# "<tool_call>" is a single token and is the very first content token when the model
# decides to call a tool (measured: ~30 ms after send), so tool-vs-answer is knowable
# from token one. If the server is ever started with a tool parser, delta.tool_calls
# is handled too. Text between the tags is never yielded as speakable text.
@dataclass
class ToolStart:
    """First token was '<tool_call>': a tool call is being generated (name unknown yet)."""


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


ToolEvent = Union[str, ToolStart, ToolCall]


class StreamHandle:
    """Lets the owner abort an in-flight stream from another thread/task: sets the stop
    event AND closes the HTTP response, so a producer blocked in a socket read wakes up
    immediately instead of at the next SSE line."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.response: Optional[requests.Response] = None
        self.first_kind: Optional[str] = None      # "tool" | "text" once the first content token is seen

    def abort(self) -> None:
        """Idempotent. The socket close runs on a helper thread: closing a streaming response can
        wait for the peer, and abort() is called from the event-loop thread (barge-in, close)."""
        self.stop.set()
        r = self.response
        if r is None:
            return
        self.response = None

        def _close(resp=r):
            # shutdown() is what actually wakes a reader blocked in recv(); close() alone waits
            # for the next SSE line (measured: 1.2-2.7 s on an idle stream).
            try:
                conn = getattr(resp.raw, "_connection", None)
                sock = getattr(conn, "sock", None)
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                resp.close()
            except Exception:
                pass
        threading.Thread(target=_close, name="llm-abort", daemon=True).start()

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
# Tag variants seen live: <tool_call>, <call>, <function_call>; the closing tag is sometimes absent.
_TOOL_JSON_RE = re.compile(r"<(?:tool_|function_)?call>\s*(\{.*?\})\s*(?:</(?:tool_|function_)?call>|$)", re.S)
_TOOL_CLOSE_RE = re.compile(r"</(?:tool_|function_)?call>")
_TOOL_OPEN_RE = re.compile(r"^\s*<(?:tool_|function_)?call>")
_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def parse_tool_call_text(text: str) -> Optional[ToolCall]:
    """Parse the first complete <tool_call>{json}</tool_call> in ``text``; None if absent/invalid."""
    m = _TOOL_JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        # Model occasionally emits single quotes / trailing commas; try a lenient repair.
        try:
            obj = json.loads(re.sub(r",\s*}", "}", m.group(1).replace("'", '"')))
        except Exception:
            return None
    name = str(obj.get("name") or "").strip()
    if not name:
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"query": args}
    return ToolCall(name=name, arguments=args if isinstance(args, dict) else {}, raw=m.group(0))


def early_tool_name(text: str) -> Optional[str]:
    """Tool name as soon as it appears in a partial <tool_call> (≈250 ms in), before the JSON closes."""
    m = _TOOL_NAME_RE.search(text or "")
    return m.group(1) if m else None


def strip_tool_markup(text: str) -> str:
    """Safety net: remove any tool-call XML that leaked into speakable text."""
    out = _TOOL_JSON_RE.sub("", text or "")
    return re.sub(r"</?(?:tool_|function_)?call>", "", out)


class OpenAICompatLLMStream:
    def __init__(self, url: str, model: str, temperature: float = 0.7, max_tokens: int = 220):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── streaming ─────────────────────────────────────────────────────────────
    def stream_messages(self, messages: List[Dict], request_timeout: int = 120,
                        stop: Optional[threading.Event] = None) -> Iterator[str]:
        """Text-only stream (no tools). Kept for the summary / fact-check / fallback paths."""
        for ev in self.stream_events(messages, request_timeout=request_timeout, stop=stop):
            if isinstance(ev, str):
                yield ev

    def stream_events(self, messages: List[Dict], *, tools: Optional[List[Dict]] = None,
                      request_timeout: int = 120, stop: Optional[threading.Event] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      handle: Optional[StreamHandle] = None) -> Iterator[ToolEvent]:
        """OpenAI-compatible chat with ``stream: true``.

        Yields ``str`` text deltas, or — when ``tools`` are offered and the model calls one —
        a ``ToolStart`` on the first token followed by exactly one ``ToolCall`` once the
        call is complete (nothing in between reaches the caller as text).

        ``stop``: a threading.Event; when set, the HTTP response is closed so the server
        actually aborts generation (an abandoned speculative reply must not keep the
        engine busy until max_tokens)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
            **_EXTRA,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"      # 'required' -> HTTP 400, 'none' is not honoured
        _log_chat_request(self.url, payload, stream=True)
        try:  # debug tap: `touch /tmp/dump_payload` inside the container to record the last requests
            import os as _os
            if _os.path.exists("/tmp/dump_payload"):
                with open("/tmp/last_payloads.jsonl", "a") as _f:
                    _f.write(json.dumps(payload) + "\n")
        except Exception:
            pass
        logger.info(
            "VOICE_LLM stream start url=%s model=%s stream=true prompt_msgs=%d max_tokens=%d temperature=%s tools=%d",
            self.url, self.model, len(messages), payload["max_tokens"], payload["temperature"], len(tools or []),
        )

        t0 = time.monotonic()
        ttft_mono: Optional[float] = None
        content_chunks = 0
        out_chars = 0
        sse_data_lines = 0
        http_status: Optional[int] = None
        aborted = False

        # Tool-call state machine
        gate_open = False            # first non-empty content token seen
        in_tool = False
        tool_buf = ""
        tool_emitted = False
        # structured deltas (only if the server has a tool parser)
        struct_calls: Dict[int, Dict[str, str]] = {}

        r = None
        if handle is not None:
            stop = handle.stop if stop is None else stop
        try:
            if stop is not None and stop.is_set():
                return
            r = requests.post(self.url, json=payload, stream=True, timeout=request_timeout)
            if handle is not None:
                handle.response = r
                if handle.stop.is_set():          # abort raced the connect: don't occupy the engine
                    aborted = True
                    return
            http_status = r.status_code
            r.raise_for_status()
            logger.info("VOICE_LLM stream connected http_status=%s", http_status)
            for raw in r.iter_lines(decode_unicode=True):
                if stop is not None and stop.is_set():
                    aborted = True
                    break
                if not raw:
                    continue
                data = raw[len("data:"):].strip() if raw.startswith("data:") else raw.strip()
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
                ch0 = choices[0] or {}
                delta = ch0.get("delta") or {}

                # Structured tool-call deltas (server-side parser present).
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index") or 0)
                    slot = struct_calls.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    if not gate_open:
                        gate_open = True
                        in_tool = True
                        if ttft_mono is None:
                            ttft_mono = time.monotonic()
                        if handle is not None:
                            handle.first_kind = "tool"
                        yield ToolStart()

                token = delta.get("content")
                if not token:
                    # Legacy / alternate OpenAI-compatible streaming (e.g. some TensorRT-LLM builds)
                    token = ch0.get("text")
                if not token:
                    continue
                if ttft_mono is None:
                    ttft_mono = time.monotonic()
                content_chunks += 1
                out_chars += len(token)

                if not gate_open:
                    if not token.strip():
                        continue                      # leading whitespace: keep the gate closed
                    gate_open = True
                    if token.lstrip().startswith(_TOOL_OPEN) or _TOOL_OPEN.startswith(token.strip()):
                        in_tool = True
                        tool_buf = token
                        if handle is not None:
                            handle.first_kind = "tool"
                        yield ToolStart()
                        continue
                    if handle is not None:
                        handle.first_kind = "text"
                if in_tool:
                    tool_buf += token
                    if _TOOL_CLOSE_RE.search(tool_buf) and not tool_emitted:
                        call = parse_tool_call_text(tool_buf)
                        if call is not None:
                            tool_emitted = True
                            yield call
                            # One tool call per turn: stop reading (and let the server abort).
                            aborted = True
                            break
                    continue
                # A tool call after some prose ("Let me check. <tool_call>…"): switch modes.
                if _TOOL_OPEN in token:
                    pre, _, post = token.partition(_TOOL_OPEN)
                    if pre:
                        yield pre
                    in_tool = True
                    tool_buf = _TOOL_OPEN + post
                    yield ToolStart()
                    continue
                yield token

            # Structured call completed without a text close tag
            if in_tool and not tool_emitted and struct_calls:
                slot = struct_calls[min(struct_calls)]
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except Exception:
                    args = {}
                if slot["name"]:
                    tool_emitted = True
                    yield ToolCall(name=slot["name"], arguments=args if isinstance(args, dict) else {},
                                   raw=json.dumps(slot))
            elif in_tool and not tool_emitted:
                # Text tool call that never closed (or used a variant tag): parse what we have.
                call = parse_tool_call_text(tool_buf)
                if call is None:
                    m = re.search(r"\{.*\}", tool_buf, re.S)
                    if m:
                        try:
                            obj = json.loads(m.group(0))
                            if obj.get("name"):
                                args = obj.get("arguments")
                                call = ToolCall(name=str(obj["name"]), arguments=args if isinstance(args, dict) else {}, raw=m.group(0))
                        except Exception:
                            call = None
                if call is not None:
                    tool_emitted = True
                    yield call
                else:
                    logger.warning("VOICE_LLM tool call did not parse: %r", tool_buf[:200])
        except Exception as e:
            if stop is not None and stop.is_set():
                aborted = True            # our own abort closed the socket: not an error
            else:
                raise
        finally:
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
            t1 = time.monotonic()
            total_ms = (t1 - t0) * 1000
            ttft_ms = (ttft_mono - t0) * 1000 if ttft_mono is not None else 0.0
            logger.info(
                "VOICE_LLM stream done model=%s stream=true ttft_ms=%.1f stream_total_ms=%.1f "
                "sse_data_events=%d content_chunks=%d output_chars=%d prompt_msgs=%d http_status=%s "
                "tool=%s aborted=%s",
                self.model, ttft_ms, total_ms, sse_data_lines, content_chunks, out_chars, len(messages),
                http_status if http_status is not None else "?",
                "yes" if in_tool else "no", aborted,
            )

    # ── non-streaming ─────────────────────────────────────────────────────────
    def complete_messages(self, messages: List[Dict]) -> str:
        """Non-streaming completion (summaries / tool-style paths). Prefer ``stream_messages`` for dialogue."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **_EXTRA,
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
        return strip_tool_markup(out).strip()
