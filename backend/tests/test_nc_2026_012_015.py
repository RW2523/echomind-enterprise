"""Regression tests for the defects found by the 2026-10-06 acceptance tests (REG-04 NC-2026-012…015)."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


# ── NC-2026-012: logout must revoke the session token ────────────────────────────
def test_nc012_revoked_token_is_rejected():
    from app.core import auth
    from app.core.db import init_db
    init_db()
    token = auth.make_token("u1", "user", "qa_user")
    assert auth.decode_token(token)["username"] == "qa_user"
    assert auth.revoke_token(token) is True
    assert auth.decode_token(token) is None            # replay after logout → rejected
    other = auth.make_token("u2", "user", "qa_other")
    assert auth.decode_token(other) is not None        # other sessions unaffected


def test_nc012_revoking_garbage_is_a_noop():
    from app.core import auth
    assert auth.revoke_token("") is False
    assert auth.revoke_token("not.a.token") is False


# ── NC-2026-013: uploads limited to parsed formats, checked by content ───────────
def test_nc013_upload_rejects_executables_and_binary_text():
    from app.rag.parse import upload_rejection
    assert upload_rejection("tool.exe", b"MZ" + bytes(range(256)))
    assert upload_rejection("notes.txt", b"MZ\x00\x01\x02binary")
    assert upload_rejection("fake.pdf", b"MZ not a pdf")
    assert upload_rejection("fake.docx", b"plain text pretending")


def test_nc013_upload_accepts_supported_files():
    from app.rag.parse import upload_rejection
    assert upload_rejection("notes.txt", "Calibration interval is fourteen days.\n".encode()) == ""
    assert upload_rejection("README.md", b"# Title\n\nSome text\twith a tab.\r\n") == ""
    assert upload_rejection("report.pdf", b"%PDF-1.7\n...") == ""
    assert upload_rejection("deck.pptx", b"PK\x03\x04rest-of-zip") == ""


# ── NC-2026-014: small talk never triggers retrieval; no chapter guessing ────────
def test_nc014_greeting_plus_small_talk_is_conversational():
    from app.rag.advanced import _is_general_conversation
    for q in ("Hello! How are you today?", "Hi there, how's it going?", "Good morning, how are you?",
              "hey how are you doing", "Hello, thanks!"):
        assert _is_general_conversation(q) is True, q


def test_nc014_greeting_plus_real_question_still_retrieves():
    from app.rag.advanced import _is_general_conversation
    for q in ("Hello, what is the SLA for P1 incidents?", "Hi, how are invoices processed?",
              "Good morning, what does clause 7 say about termination?"):
        assert _is_general_conversation(q) is False, q


def test_nc014_financial_advisor_does_not_guess_chapters():
    from app.rag import advanced
    for d in (advanced._PERSONA_RAG_PROMPTS, advanced._PERSONA_GENERAL_PROMPTS, advanced._PERSONA_STRICT_CITATION_PROMPTS):
        assert "likely covers it" not in d.get("Financial Advisor", "")


# ── NC-2026-015: export gateway key formats and overlap-safe redaction ───────────
def test_nc015_key_formats_are_detected_and_redacted():
    from app.core.export_gateway import evaluate_export
    keys = ["sk-test-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c", "sk_live_51HxYzAbCdEfGhIjKlMnOp", "sk-proj-AbCdEfGhIjKlMnOpQrStUv",
            "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx", "pk_test_AbCdEfGhIjKlMnOpQr", "sk-AbCdEfGhIjKlMnOpQrSt"]
    for k in keys:
        out = evaluate_export(f"Deploy key: {k} — keep safe.")
        assert any(f["type"] == "api_key" for f in out["findings"]), k
        assert k not in out["redacted_text"], k
        assert out["safe_to_export"] is False


def test_nc015_assignment_and_overlap_redact_cleanly():
    from app.core.export_gateway import evaluate_export
    text = "api_key = sk-test-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c and mail a@example.com"
    out = evaluate_export(text)
    red = out["redacted_text"]
    assert "sk-test" not in red and "4f9a8b7c" not in red
    assert red.count("[REDACTED:") == 2                     # key (merged overlap) + email
    assert red.endswith("[REDACTED:email]")


def test_nc015_ordinary_text_is_not_flagged():
    from app.core.export_gateway import evaluate_export
    out = evaluate_export("The risk desk reviewed the skeleton plan and the task list on Monday.")
    assert out["finding_count"] == 0
