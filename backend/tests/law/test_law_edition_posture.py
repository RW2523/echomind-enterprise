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

# D-014: a valid master key outside the data folder, readable by its owner only.
import tempfile  # noqa: E402

from conftest import write_master_key  # noqa: E402

_KEYS = pathlib.Path(tempfile.mkdtemp(prefix="law_keys_"))
_DATA = pathlib.Path(tempfile.mkdtemp(prefix="law_data_"))
GOOD_KEY = write_master_key(_KEYS / "master.key")


def _key_file(name: str, content: str, mode: int = 0o600, folder: pathlib.Path = _KEYS) -> str:
    p = folder / name
    p.write_text(content)
    os.chmod(p, mode)
    return str(p)


SECURE = {
    "AUTH_ENABLED": True,
    "AUTH_SECRET": STRONG_SECRET,
    "CORS_ALLOW_ORIGINS": "https://law.firm.example,http://localhost:5174",
    "DOCGEN_IMAGE_BACKEND": "none",
    "ECHOMIND_LOG_FULL_PROMPTS": "0",
    "LEGAL_MASTER_KEY_FILE": GOOD_KEY,
    "DATA_DIR": str(_DATA),
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
        ({"AUTH_ADMIN_PASSWORD": "admin12345"}, "weak_admin_password"),
        ({"AUTH_ADMIN_PASSWORD": "password1234"}, "weak_admin_password"),
        ({"LEGAL_MASTER_KEY_FILE": ""}, "master_key_missing"),
        ({"LEGAL_MASTER_KEY_FILE": str(_KEYS / "absent.key")}, "master_key_missing"),
        ({"LEGAL_MASTER_KEY_FILE": _key_file("open.key", open(GOOD_KEY).read(), 0o644)}, "master_key_insecure"),
        ({"LEGAL_MASTER_KEY_FILE": _key_file("short.key", "c2hvcnQ=\n")}, "master_key_invalid"),
        ({"LEGAL_MASTER_KEY_FILE": _key_file("in-data.key", open(GOOD_KEY).read(), folder=_DATA)}, "master_key_in_data"),
        ({"LEGAL_DEMO_SIGNIN": True}, "demo_signin"),
        ({"LEGAL_DEMO_SIGNIN": True, "LEGAL_PROFILE": "production"}, "demo_signin"),
        ({"LEGAL_DEMO_SIGNIN": True, "LEGAL_PROFILE": "demo"}, "demo_not_seeded"),
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
    (tmp_path / "data").mkdir(exist_ok=True)
    child_env.update({"ECHOMIND_DATA_DIR": str(tmp_path / "data"), "PYTHONWARNINGS": "ignore",
                      "LEGAL_MASTER_KEY_FILE": write_master_key(tmp_path / "master.key")})
    child_env.update(env)
    code = "import app.main as m; print('DOCS', m.app.docs_url, m.app.openapi_url)"
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR, env=child_env,
                          capture_output=True, text=True, timeout=120)


def test_law_backend_refuses_to_start_when_insecure(tmp_path):
    r = _import_main(tmp_path, ECHOMIND_EDITION="law", AUTH_ENABLED="0", CORS_ALLOW_ORIGINS="*")
    assert r.returncode != 0
    assert "refuses to start" in r.stderr and "auth_disabled" in r.stderr and "cors_open" in r.stderr
    assert not (tmp_path / "data" / "echomind.sqlite").exists(), "nothing may be written before the posture check"


def test_law_backend_starts_when_secure_and_hides_api_docs(tmp_path):
    r = _import_main(tmp_path, ECHOMIND_EDITION="law", AUTH_ENABLED="1", AUTH_SECRET=STRONG_SECRET,
                     AUTH_ADMIN_PASSWORD="seven brave otters swim", CORS_ALLOW_ORIGINS="http://localhost:5174")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DOCS None None" in r.stdout


def test_legacy_backend_is_unchanged(tmp_path):
    r = _import_main(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DOCS /docs /openapi.json" in r.stdout


# ── Logs never carry matter content in the Law edition (THREAT_MODEL T-11) ──────
def test_free_text_log_arguments_are_redacted_in_law(monkeypatch):
    import logging
    from app.core.edition import ContentRedactingFilter
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    rec = logging.LogRecord("app.rag.advanced", logging.INFO, __file__, 1,
                            "RAG semantic-first: document wins (score=%.3f) question=%s",
                            (0.691, "What is the liability cap in the MSA between Harrowgate and Voss-Lindqvist?"), None)
    ContentRedactingFilter().filter(rec)
    line = rec.getMessage()
    assert "Harrowgate" not in line and "liability" not in line
    assert "0.691" in line and "chars>" in line


def test_identifiers_and_urls_stay_readable(monkeypatch):
    import logging
    from app.core.edition import ContentRedactingFilter
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    rec = logging.LogRecord("httpx", logging.INFO, __file__, 1, 'HTTP Request: %s %s "%s %d %s"',
                            ("POST", "http://localhost:11435/api/embeddings", "HTTP/1.1", 200, "OK"), None)
    ContentRedactingFilter().filter(rec)
    assert rec.getMessage() == 'HTTP Request: POST http://localhost:11435/api/embeddings "HTTP/1.1 200 OK"'


def test_secrets_in_request_paths_are_masked_in_access_logs(monkeypatch):
    """S4.1 (D-054): an enquirer's intake link is a credential, as is a sign-in enrolment token. The server's
    access log line (a short path with no spaces, which the free-text rule keeps) must not carry them."""
    import logging
    from app.core.edition import ContentRedactingFilter
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    token = "Q3vR9_xYk2-PzLm8NwTa4cHj7UeBs1oGfDi6KqWt0Ve"
    for path in (f"/api/intake-link/{token}", f"/api/intake-link/{token}/submit", f"/intake/{token}", f"/signin?enrol={token}"):
        rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                                ("127.0.0.1:5000", "GET", path, "1.1", 200), None)
        ContentRedactingFilter().filter(rec)
        line = rec.getMessage()
        assert token not in line and "<secret>" in line, line
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:5000", "GET", "/api/legal/matters", "1.1", 200), None)
    ContentRedactingFilter().filter(rec)
    assert "/api/legal/matters" in rec.getMessage()


def test_the_law_backend_masks_link_tokens_in_its_access_log():
    import logging
    from app.core.edition import ContentRedactingFilter, install_log_redaction
    install_log_redaction()
    assert any(isinstance(f, ContentRedactingFilter) for f in logging.getLogger("uvicorn.access").filters)


def test_log_filter_is_inert_outside_law(monkeypatch):
    import logging
    from app.core.edition import ContentRedactingFilter
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "question=%s", ("what is the cap",), None)
    ContentRedactingFilter().filter(rec)
    assert rec.getMessage() == "question=what is the cap"


def test_law_backend_installs_the_log_filter(tmp_path):
    r = _import_main(tmp_path, ECHOMIND_EDITION="law", AUTH_ENABLED="1", AUTH_SECRET=STRONG_SECRET,
                     AUTH_ADMIN_PASSWORD="seven brave otters swim", CORS_ALLOW_ORIGINS="http://localhost:5174")
    assert r.returncode == 0, r.stderr[-2000:]
    probe = subprocess.run(
        [sys.executable, "-c",
         "import logging, app.main; logging.getLogger('app.rag.advanced').info('question=%s', 'Voss-Lindqvist liability cap')"],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120,
        env={**{k: v for k, v in os.environ.items() if not k.startswith(("AUTH_", "CORS_", "ECHOMIND_", "LEGAL_"))},
             "ECHOMIND_DATA_DIR": str(tmp_path / "data"), "LEGAL_MASTER_KEY_FILE": str(tmp_path / "master.key"),
             "ECHOMIND_EDITION": "law", "AUTH_ENABLED": "1",
             "AUTH_SECRET": STRONG_SECRET, "AUTH_ADMIN_PASSWORD": "seven brave otters swim",
             "CORS_ALLOW_ORIGINS": "http://localhost:5174", "PYTHONWARNINGS": "ignore"})
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert "question=<28 chars>" in probe.stdout and "Voss" not in probe.stdout + probe.stderr


def test_unset_admin_password_is_not_a_violation():
    assert "weak_admin_password" not in _codes({**SECURE, "AUTH_ADMIN_PASSWORD": ""})


def test_first_admin_password_goes_to_a_private_file_not_the_log(fresh_db, law_edition, monkeypatch, caplog):
    import logging
    import os
    import stat
    from app.core import auth
    from app.core.config import settings
    from app.core.db import init_db
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", STRONG_SECRET)
    monkeypatch.setattr(settings, "AUTH_ADMIN_PASSWORD", "")
    init_db()
    with caplog.at_level(logging.DEBUG):
        auth.seed_admin()
    path = os.path.join(settings.DATA_DIR, "legal", "bootstrap_admin_password")
    pw = open(path).read().strip()
    assert len(pw) >= 20 and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert pw not in caplog.text and "legal/bootstrap_admin_password" in caplog.text
    assert auth.authenticate("admin", pw)
    from app.legal.identity import accounts
    assert "firm_admin" in accounts.roles_of(auth.get_user_by_username("admin")["id"])


def test_demo_sign_in_runs_only_on_a_demo_instance(tmp_path):
    """Commercial-readiness review F1 (2026-10-28): one-click entry needs LEGAL_PROFILE=demo and a data folder the demo seed
    made; never beside the firm's single sign-on. Without demo sign-in, none of this is asked."""
    data = tmp_path / "data"
    (data / "legal").mkdir(parents=True)
    demo = {**SECURE, "DATA_DIR": str(data), "LEGAL_DEMO_SIGNIN": True, "LEGAL_PROFILE": "demo"}
    assert _codes(demo) == {"demo_not_seeded"}
    (data / "legal" / "DEMO_INSTANCE").write_text("seed-demo\n")
    assert _codes(demo) == set()
    assert _codes({**demo, "LEGAL_OIDC_ISSUER": "https://login.firm.example"}) == {"demo_with_sso"}
    assert _codes({**SECURE, "LEGAL_PROFILE": "demo"}) == set()


def test_the_running_settings_feed_the_demo_check(monkeypatch):
    from app.core.config import settings
    from app.core.edition import posture_inputs
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN", True)
    got = posture_inputs(settings, env={"LEGAL_PROFILE": "demo"})
    assert got["LEGAL_DEMO_SIGNIN"] is True and got["LEGAL_PROFILE"] == "demo"


def test_tracebacks_keep_their_frames_but_not_the_message_in_law(monkeypatch):
    """Review R12 (2026-10-28): an exception's message can quote matter content; its type and frames stay for diagnosis."""
    import logging
    import sys
    from app.core.edition import ContentRedactingFilter
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    quoted = " ".join(["clause 9.2 of the", "Harrow" + "gate", "MSA caps liability"])      # content arrives at run time, not in source
    try:
        raise ValueError(quoted)
    except ValueError:
        rec = logging.LogRecord("app.legal", logging.ERROR, __file__, 1, "answer failed", None, sys.exc_info())
    ContentRedactingFilter().filter(rec)
    text = logging.Formatter().format(rec)
    assert "Harrowgate" not in text and "ValueError: <" in text and "test_tracebacks_keep_their_frames" in text
