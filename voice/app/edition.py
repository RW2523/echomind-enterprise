"""Product edition switch for the voice service (EchoMind Law, decisions D-003 and D-016).

`ECHOMIND_EDITION=law` makes the voice service refuse to start without WebSocket auth, a strong
shared secret and offline mode, or with LLM payload logging or an evaluation ablation arm on.
In that edition spoken content is also kept out of the logs: free-text log arguments are
replaced by their length. Any other value (or unset) keeps today's behaviour.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

LAW = "law"
_MIN_SECRET_BYTES = 32
_TRUE = ("1", "true", "yes")


def is_law() -> bool:
    return (os.getenv("ECHOMIND_EDITION") or "").strip().lower() == LAW


class PostureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


def posture_violations(env: Optional[Mapping[str, str]] = None) -> list[Violation]:
    e = os.environ if env is None else env
    flag = lambda k, d="0": (e.get(k, d) or "").strip().lower() in _TRUE
    out: list[Violation] = []
    if not flag("VOICE_AUTH_ENABLED"):
        out.append(Violation("auth_disabled", "VOICE_AUTH_ENABLED must be 1"))
    if len((e.get("VOICE_AUTH_SECRET") or "").encode()) < _MIN_SECRET_BYTES:
        out.append(Violation("weak_secret", f"VOICE_AUTH_SECRET must equal the backend AUTH_SECRET (at least {_MIN_SECRET_BYTES} bytes)"))
    if not flag("VOICE_OFFLINE", "1"):
        out.append(Violation("cloud_egress", "VOICE_OFFLINE must be 1 (no runtime voice downloads)"))
    if flag("LLM_LOG_PAYLOAD"):
        out.append(Violation("content_logging", "LLM_LOG_PAYLOAD must be off"))
    if (e.get("VOICE_ABLATION_MODE") or "dual_i1").strip() != "dual_i1":
        out.append(Violation("eval_mode", "VOICE_ABLATION_MODE is an evaluation-only switch and must be unset"))
    return out


def validate_law_posture(env: Optional[Mapping[str, str]] = None) -> None:
    if not is_law():
        return
    violations = posture_violations(env)
    if violations:
        detail = "; ".join(f"[{v.code}] {v.message}" for v in violations)
        raise PostureError(f"EchoMind Law voice service refuses to start: {detail}")


def _redact(value):
    # Free text (anything with whitespace, or long) is spoken content or derived from it.
    # Short single tokens — tool names, ids, reasons, model names — stay readable.
    if isinstance(value, str) and (any(c.isspace() for c in value) or len(value) > 64):
        return f"<{len(value)} chars>"
    return value


class ContentRedactingFilter(logging.Filter):
    """In the Law edition, replace free-text log arguments with their length."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not is_law() or not record.args:
            return True
        if isinstance(record.args, Mapping):
            record.args = {k: _redact(v) for k, v in record.args.items()}
        else:
            record.args = tuple(_redact(a) for a in record.args)
        return True


def install_log_redaction() -> None:
    """Attach the redacting filter to every root handler (child loggers propagate through them)."""
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, ContentRedactingFilter) for f in handler.filters):
            handler.addFilter(ContentRedactingFilter())
