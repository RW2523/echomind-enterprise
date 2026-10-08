"""S1.0 / D-003 — the Law edition refuses to start with an insecure configuration.

Unit tests drive the pure posture check; subprocess tests prove `import app.main` really aborts
(uvicorn imports the module, so an import-time refusal is a refusal to serve).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[2]
STRONG_SECRET = "x" * 48

SECURE = {
    "AUTH_ENABLED": True,
    "AUTH_SECRET": STRONG_SECRET,
    "CORS_ALLOW_ORIGINS": "https://law.firm.example,http://localhost:5174",
    "DOCGEN_IMAGE_BACKEND": "none",
    "ECHOMIND_LOG_FULL_PROMPTS": "0",
}


def _codes(inputs):
    from app.core.edition import posture_violations
    return {v.code for v in posture_violations(inputs)}


def test_secure_configuration_has_no_violations():
    assert _codes(SECURE) == set()


@pytest.mark.parametrize(
    "override, code",
    [
        ({"AUTH_ENABLED": False}, "auth_disabled"),
        ({"AUTH_SECRET": ""}, "weak_secret"),
        ({"AUTH_SECRET": "short-secret"}, "weak_secret"),
        ({"CORS_ALLOW_ORIGINS": "*"}, "cors_open"),
        ({"CORS_ALLOW_ORIGINS": ""}, "cors_open"),
        ({"CORS_ALLOW_ORIGINS": "https://law.firm.example,*"}, "cors_open"),
        ({"CORS_ALLOW_ORIGINS": "law.firm.example"}, "cors_invalid"),
        ({"DOCGEN_IMAGE_BACKEND": "nim"}, "cloud_egress"),
        ({"DOCGEN_IMAGE_BACKEND": " NIM "}, "cloud_egress"),
        ({"ECHOMIND_LOG_FULL_PROMPTS": "1"}, "content_logging"),
        ({"ECHOMIND_LOG_FULL_PROMPTS": "true"}, "content_logging"),
    ],
)
def test_each_insecure_setting_is_a_violation(override, code):
    assert code in _codes({**SECURE, **override})


def test_validate_raises_only_in_law_edition(monkeypatch):
    from app.core.edition import PostureError, validate_law_posture
    insecure = {**SECURE, "AUTH_ENABLED": False}
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    validate_law_posture(insecure)  # legacy editions keep today's behaviour
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    with pytest.raises(PostureError) as exc:
        validate_law_posture(insecure)
    assert "auth_disabled" in str(exc.value)
    validate_law_posture(SECURE)


def test_edition_is_read_at_call_time(monkeypatch):
    from app.core.edition import is_law
    monkeypatch.setenv("ECHOMIND_EDITION", " Law ")
    assert is_law()
    monkeypatch.setenv("ECHOMIND_EDITION", "health")
    assert not is_law()


def test_content_for_log_hides_text_in_law(monkeypatch):
    from app.core.edition import content_for_log
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    assert content_for_log("What is the liability cap?", 10) == "What is th"
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    shown = content_for_log("What is the liability cap?", 10)
    assert "liability" not in shown and "26" in shown


# ── Startup behaviour in a real interpreter ──────────────────────────────────────

def _import_main(tmp_path, **env):
    child_env = {k: v for k, v in os.environ.items() if not k.startswith(("AUTH_", "CORS_", "ECHOMIND_", "DOCGEN_", "LEGAL_"))}
    child_env.update({"ECHOMIND_DATA_DIR": str(tmp_path), "PYTHONWARNINGS": "ignore"})
    child_env.update(env)
    code = "import app.main as m; print('DOCS', m.app.docs_url, m.app.openapi_url)"
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR, env=child_env,
                          capture_output=True, text=True, timeout=120)


def test_law_backend_refuses_to_start_when_insecure(tmp_path):
    r = _import_main(tmp_path, ECHOMIND_EDITION="law", AUTH_ENABLED="0", CORS_ALLOW_ORIGINS="*")
    assert r.returncode != 0
    assert "refuses to start" in r.stderr and "auth_disabled" in r.stderr and "cors_open" in r.stderr
    assert not (tmp_path / "echomind.sqlite").exists(), "nothing may be written before the posture check"


def test_law_backend_starts_when_secure_and_hides_api_docs(tmp_path):
    r = _import_main(tmp_path, ECHOMIND_EDITION="law", AUTH_ENABLED="1", AUTH_SECRET=STRONG_SECRET,
                     AUTH_ADMIN_PASSWORD="test-admin-password-1", CORS_ALLOW_ORIGINS="http://localhost:5174")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DOCS None None" in r.stdout


def test_legacy_backend_is_unchanged(tmp_path):
    r = _import_main(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DOCS /docs /openapi.json" in r.stdout
