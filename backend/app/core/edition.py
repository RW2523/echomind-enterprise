"""Product edition switch (EchoMind Law, decision D-003).

`ECHOMIND_EDITION=law` turns on the Law edition. In that edition the backend refuses to start with
a configuration that could expose client matter data: authentication off, a weak or missing signing
secret, wildcard CORS, a cloud egress path, or prompt-content logging. Any other value (or unset)
keeps the existing EchoMind behaviour unchanged.

The edition is read from the environment at call time (not cached at import) so tests and
long-running workers always see the current value.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

LAW = "law"
_MIN_SECRET_BYTES = 32
_TRUE = ("1", "true", "yes")


def current_edition() -> str:
    return (os.getenv("ECHOMIND_EDITION") or "").strip().lower()


def is_law() -> bool:
    return current_edition() == LAW


class PostureError(RuntimeError):
    """Raised at startup when the Law edition is configured insecurely."""


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


def posture_inputs(settings=None, env: Optional[Mapping[str, str]] = None) -> dict:
    """Snapshot of every setting the posture check looks at."""
    if settings is None:
        from .config import settings  # local import: config reads the environment at import time
    e = os.environ if env is None else env
    return {
        "AUTH_ENABLED": bool(settings.AUTH_ENABLED),
        "AUTH_SECRET": settings.AUTH_SECRET or "",
        "CORS_ALLOW_ORIGINS": settings.CORS_ALLOW_ORIGINS or "",
        "DOCGEN_IMAGE_BACKEND": e.get("DOCGEN_IMAGE_BACKEND", "none"),
        "ECHOMIND_LOG_FULL_PROMPTS": e.get("ECHOMIND_LOG_FULL_PROMPTS", "0"),
    }


def posture_violations(inputs: Mapping) -> list[Violation]:
    out: list[Violation] = []
    if not inputs.get("AUTH_ENABLED"):
        out.append(Violation("auth_disabled", "AUTH_ENABLED must be 1"))
    if len(str(inputs.get("AUTH_SECRET") or "").encode()) < _MIN_SECRET_BYTES:
        out.append(Violation("weak_secret", f"AUTH_SECRET must be set explicitly (at least {_MIN_SECRET_BYTES} bytes)"))
    origins = [o.strip() for o in str(inputs.get("CORS_ALLOW_ORIGINS") or "").split(",") if o.strip()]
    if not origins or any("*" in o for o in origins):
        out.append(Violation("cors_open", "CORS_ALLOW_ORIGINS must list explicit origins (no '*')"))
    elif any(not o.startswith(("https://", "http://")) for o in origins):
        out.append(Violation("cors_invalid", "every CORS origin must start with https:// or http://"))
    if str(inputs.get("DOCGEN_IMAGE_BACKEND") or "").strip().lower() == "nim":
        out.append(Violation("cloud_egress", "DOCGEN_IMAGE_BACKEND=nim sends prompts to a cloud API"))
    if str(inputs.get("ECHOMIND_LOG_FULL_PROMPTS") or "").strip().lower() in _TRUE:
        out.append(Violation("content_logging", "ECHOMIND_LOG_FULL_PROMPTS must be off"))
    return out


def validate_law_posture(inputs: Optional[Mapping] = None) -> None:
    """No-op outside the Law edition; otherwise raise PostureError listing every violation."""
    if not is_law():
        return
    violations = posture_violations(inputs if inputs is not None else posture_inputs())
    if violations:
        detail = "; ".join(f"[{v.code}] {v.message}" for v in violations)
        raise PostureError(f"EchoMind Law edition refuses to start: {detail}")


def content_for_log(text: Optional[str], limit: int = 80) -> str:
    """Text safe to put in a log line: a truncated excerpt normally, only its length in the Law edition."""
    text = text or ""
    if is_law():
        return f"<{len(text)} chars>"
    return text[:limit]


def api_docs_enabled() -> bool:
    """FastAPI /docs and /openapi.json are off in the Law edition unless LEGAL_DEV_OPENAPI=1 (dev only)."""
    if not is_law():
        return True
    return (os.getenv("LEGAL_DEV_OPENAPI") or "").strip().lower() in _TRUE


def _redact(value):
    # Free text (anything with whitespace, or long) is a question, quote, title or path derived
    # from matter content. Short single tokens — ids, URLs, model names, reasons — stay readable.
    if isinstance(value, str) and (any(c.isspace() for c in value) or len(value) > 64):
        return f"<{len(value)} chars>"
    return value


class ContentRedactingFilter(logging.Filter):
    """In the Law edition, replace free-text log arguments with their length (THREAT_MODEL T-11)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not is_law() or not record.args:
            return True
        if isinstance(record.args, Mapping):
            record.args = {k: _redact(v) for k, v in record.args.items()}
        else:
            record.args = tuple(_redact(a) for a in record.args)
        return True


def install_log_redaction() -> None:
    """Attach the filter to every root handler; child loggers propagate through them."""
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, ContentRedactingFilter) for f in handler.filters):
            handler.addFilter(ContentRedactingFilter())
