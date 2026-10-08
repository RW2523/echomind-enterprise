"""S1.0 / D-003, D-016 — voice service in the EchoMind Law edition.

- refuses to start without voice auth, a strong secret, offline mode, or with payload logging;
- the evaluation ablation arm that speaks invented "facts" is unreachable;
- the /tmp/dump_payload tap never writes LLM payloads;
- spoken content never reaches the logs (only its length does).
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys

import pytest

VOICE_DIR = pathlib.Path(__file__).resolve().parents[1]
SECURE = {"VOICE_AUTH_ENABLED": "1", "VOICE_AUTH_SECRET": "s" * 48, "VOICE_OFFLINE": "1", "LLM_LOG_PAYLOAD": "0"}


def _codes(env):
    from app.edition import posture_violations
    return {v.code for v in posture_violations(env)}


def test_secure_voice_configuration_passes():
    assert _codes(SECURE) == set()


@pytest.mark.parametrize("override, code", [
    ({"VOICE_AUTH_ENABLED": "0"}, "auth_disabled"),
    ({"VOICE_AUTH_SECRET": "short"}, "weak_secret"),
    ({"VOICE_OFFLINE": "0"}, "cloud_egress"),
    ({"LLM_LOG_PAYLOAD": "1"}, "content_logging"),
    ({"VOICE_ABLATION_MODE": "dual_no_i1"}, "eval_mode"),
])
def test_each_insecure_voice_setting_is_a_violation(override, code):
    assert code in _codes({**SECURE, **override})


def test_ablation_arm_is_unreachable_in_law(monkeypatch, tmp_path):
    from app import session
    monkeypatch.setenv("VOICE_ABLATION_MODE", "dual_no_i1")
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    assert session._ablation_mode() == "dual_no_i1"      # research builds keep the experiment
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    assert session._ablation_mode() == "dual_i1"


def test_payload_dump_tap_is_inert_in_law(monkeypatch, tmp_path):
    from app.adapters import llm_openai_stream as mod
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    flag, out = tmp_path / "dump_payload", tmp_path / "last_payloads.jsonl"
    flag.touch()
    monkeypatch.setattr(mod, "_DUMP_FLAG", str(flag))
    monkeypatch.setattr(mod, "_DUMP_FILE", str(out))
    mod._maybe_dump_payload({"messages": [{"role": "user", "content": "privileged question"}]})
    assert not out.exists()
    monkeypatch.delenv("ECHOMIND_EDITION")
    mod._maybe_dump_payload({"messages": []})
    assert json.loads(out.read_text().strip()) == {"messages": []}


def test_spoken_content_is_redacted_from_logs_in_law(monkeypatch):
    from app.edition import ContentRedactingFilter
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    f = ContentRedactingFilter()
    rec = logging.LogRecord("app.session.turn", logging.INFO, __file__, 1,
                            "[ROUTE] tool=%s arg=%r topic=%r ms=%.0f",
                            ("search_knowledge_base", "what is the liability cap for Voss", "liability cap", 812.0), None)
    assert f.filter(rec) is True
    line = rec.getMessage()
    assert "liability" not in line and "Voss" not in line
    assert "search_knowledge_base" in line and "812" in line


def test_log_filter_is_a_noop_outside_law(monkeypatch):
    from app.edition import ContentRedactingFilter
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "said %r", ("hello there",), None)
    ContentRedactingFilter().filter(rec)
    assert rec.getMessage() == "said 'hello there'"


def _import_server(**env):
    child = {k: v for k, v in os.environ.items() if not k.startswith(("VOICE_", "ECHOMIND_", "LLM_LOG"))}
    child.update({"PYTHONWARNINGS": "ignore"}, **env)
    return subprocess.run([sys.executable, "-c", "import app.server; print('STARTED')"], cwd=VOICE_DIR,
                          env=child, capture_output=True, text=True, timeout=180)


def test_law_voice_refuses_to_start_without_auth():
    r = _import_server(ECHOMIND_EDITION="law", VOICE_AUTH_ENABLED="0", VOICE_OFFLINE="1")
    assert r.returncode != 0 and "refuses to start" in r.stderr and "auth_disabled" in r.stderr


def test_law_voice_starts_when_secure():
    r = _import_server(ECHOMIND_EDITION="law", **SECURE)
    assert r.returncode == 0 and "STARTED" in r.stdout, r.stderr[-2000:]
