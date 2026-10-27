"""Pilot configuration (D-059): the owner turned on every Phase 4 feature for the pilot on 2026-10-11.

The code keeps each feature off by default (the per-feature tests check that). The pilot stack gets its
switches from `.env.law` through `docker-compose.law.yml`, so the overlay must pass each one to the backend,
off when unset, and the pilot settings template must turn each one on.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

# Backend switch → the .env.law name that sets it, and its value when .env.law leaves it out.
PHASE4 = {
    "LEGAL_INTAKE": ("LAW_INTAKE", "0"),
    "LEGAL_INTAKE_LINKS": ("LAW_INTAKE_LINKS", "0"),
    "LEGAL_SUMMARIES": ("LAW_SUMMARIES", "0"),
    "LEGAL_TIME_ENTRIES": ("LAW_TIME_ENTRIES", "0"),
    "LEGAL_PANEL": ("LAW_PANEL", "0"),
    "LEGAL_CONVERSATION": ("LAW_CONVERSATION", "0"),     # S5.1 (D-062), switched the same way
}


class _Overlay(yaml.SafeLoader):
    """Reads compose's `!reset` and `!override` tags as plain values."""


for _tag in ("!reset", "!override"):
    _Overlay.add_constructor(_tag, lambda loader, node: (
        loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
        else loader.construct_mapping(node) if isinstance(node, yaml.MappingNode)
        else None if node.value in ("", "null", "~") else loader.construct_scalar(node)))


def _backend_env() -> dict[str, str]:
    compose = yaml.load((ROOT / "docker-compose.law.yml").read_text(), Loader=_Overlay)
    return dict(item.split("=", 1) for item in compose["services"]["backend"]["environment"])


def _template() -> dict[str, str]:
    pairs = {}
    for line in (ROOT / ".env.law.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            pairs[k] = v
    return pairs


def test_the_pilot_stack_passes_every_phase4_switch_to_the_backend_off_when_unset():
    env = _backend_env()
    for switch, (name, default) in PHASE4.items():
        assert env.get(switch) == f"${{{name}:-{default}}}", switch
    assert env.get("LEGAL_INTAKE_LINK_DAYS") == "${LAW_INTAKE_LINK_DAYS:-14}"


def test_the_pilot_settings_turn_on_every_phase4_feature():
    t = _template()
    for switch, (name, _) in PHASE4.items():
        assert t.get(name) == "1", f"{name} ({switch}) is not on in .env.law.example"
    assert t.get("LAW_INTAKE_LINK_DAYS") == "14"


# D-060: the pilot runs Law images built on the 1.4.1 images with one commit's code baked in, so switching
# branches in this checkout never changes what the pilot runs, and it is capped so it cannot crowd out production.
SERVICES = {"backend": "LAW_BACKEND_IMAGE", "voice": "LAW_VOICE_IMAGE", "frontend": "LAW_FRONTEND_IMAGE"}


def _services() -> dict:
    return yaml.load((ROOT / "docker-compose.law.yml").read_text(), Loader=_Overlay)["services"]


def test_the_pilot_runs_prebuilt_law_images_with_no_code_mounted_from_the_checkout():
    svc = _services()
    for name, var in SERVICES.items():
        s = svc[name]
        assert s.get("image", "").startswith(f"${{{var}:-echomind-law-{name}:"), name
        assert s.get("pull_policy") == "never" and "build" in s and s["build"] is None, name
        mounts = [m for m in s.get("volumes") or [] if isinstance(m, str)]
        assert not any(m.split(":")[0] in ("./backend/app", "./voice/app", "./frontend") for m in mounts), name
    # Compose ignores a `!reset` that arrives through a merged anchor, so each service resets its own build.
    assert (ROOT / "docker-compose.law.yml").read_text().count("build: !reset null") >= len(SERVICES)


def test_each_pilot_container_is_capped_and_is_killed_before_production_under_memory_pressure():
    svc = _services()
    for name in SERVICES:
        s = svc[name]
        assert s.get("mem_limit") and s.get("memswap_limit") == s["mem_limit"], name
        assert int(s.get("oom_score_adj", 0)) >= 500, name


def test_the_law_backend_image_adds_what_the_1_4_1_image_lacks():
    d = (ROOT / "backend" / "Dockerfile.law").read_text()
    assert "cryptography==" in d and "tesseract-ocr" in d and "ms-marco-MiniLM-L-6-v2" in d
    assert "COPY backend/app /app/app" in d
    assert "COPY voice/app /workspace/app" in (ROOT / "voice" / "Dockerfile.law").read_text()
    f = (ROOT / "frontend" / "Dockerfile.law").read_text()
    assert "COPY frontend/dist/ /usr/share/nginx/html/" in f and "openssl req" in f   # its own certificate


def test_the_image_build_refuses_uncommitted_changes_and_tags_by_commit():
    s = (ROOT / "scripts" / "law" / "build_images.sh").read_text()
    assert "git status --porcelain" in s and "git archive" in s and "rev-parse --short" in s


def test_the_pilot_never_shares_the_development_backends_master_key():
    dev = re.search(r'SECRETS/master\.key|\.law-secrets/master\.key', (ROOT / "scripts" / "law" / "dev_backend.sh").read_text())
    assert dev, "the development key path moved; update this test"
    assert _template()["LAW_MASTER_KEY_FILE"] == "./.law-secrets/pilot-master.key"


# D-060: on the shared host the pilot loads speech recognition on first use, not at every start (a CPU load of
# the model pushed the backend past its cap). Production keeps preloading: the switch defaults on.
def _lifespan_threads(monkeypatch, preload: str | None) -> list:
    import asyncio
    import app.main as m
    if preload is None:
        monkeypatch.delenv("ECHOMIND_ASR_PRELOAD", raising=False)
    else:
        monkeypatch.setenv("ECHOMIND_ASR_PRELOAD", preload)
    started = []

    class _Thread:
        def __init__(self, target=None, daemon=None, **_):
            self.target = target

        def start(self):
            started.append(self.target)

    async def _nothing():
        return None

    monkeypatch.setattr(m.threading, "Thread", _Thread)
    monkeypatch.setattr(m, "_warm_llm_and_embeddings", _nothing)
    monkeypatch.setattr(m, "_stt_fatal_watchdog", _nothing)
    from app.legal.jobs import worker
    monkeypatch.setattr(worker, "start", lambda *a, **k: None)
    monkeypatch.setattr(worker, "stop", lambda *a, **k: None)

    async def _run():
        async with m.lifespan(m.app):
            pass
    asyncio.run(_run())
    return [t.__name__ for t in started]


def test_speech_recognition_preloads_by_default_and_the_pilot_can_load_it_on_first_use(monkeypatch):
    assert "_warm_nemotron_stt" in _lifespan_threads(monkeypatch, None)
    assert "_warm_nemotron_stt" not in _lifespan_threads(monkeypatch, "0")


def test_the_pilot_loads_speech_models_on_first_use_only():
    env = _backend_env()
    assert env.get("ECHOMIND_ASR_PRELOAD") == "${LAW_ASR_PRELOAD:-0}"
    voice = dict(i.split("=", 1) for i in _services()["voice"]["environment"])
    assert voice.get("VOICE_USE_PARAKEET") == "${LAW_VOICE_USE_PARAKEET:-0}"
    assert voice.get("VOICE_NEMOTRON_STARTUP_LOAD") == "${LAW_VOICE_STARTUP_LOAD:-0}"


# D-062: the pilot's nginx passed /voice/* to the voice service, whose HTTP routes (/speak, /voices/installed, its test
# page) answer without a session. The Law interface never calls them; speech goes through the signed-in backend.
def test_the_law_frontend_refuses_the_voice_service_and_otherwise_matches_the_standard_config():
    conf = (ROOT / "frontend" / "nginx.law.conf").read_text()
    assert "proxy_pass http://voice" not in conf
    blocks = re.findall(r"location /voice/ \{([^}]*)\}", conf)
    assert len(blocks) == 2 and all("return 404;" in b for b in blocks)                 # the HTTP and the HTTPS server
    # EchoMind Legal adds a Content-Security-Policy to the Law app only (docs/echomind-legal/08): its comment and two
    # add_header lines are the one other allowed difference; everything else must stay identical.
    legal_csp = re.compile(r"\n[ ]*# EchoMind Legal \(docs/echomind-legal/08\)[^\n]*|\n[ ]*add_header (?:Content-Security-Policy|X-Content-Type-Options) [^\n]*")
    strip = lambda t: legal_csp.sub("", re.sub(r"\n\s*location /voice/ \{[^}]*\}", "", t))     # noqa: E731
    assert strip(conf) == strip((ROOT / "frontend" / "nginx.conf").read_text())           # cannot drift apart
    assert conf.count("add_header Content-Security-Policy \"default-src 'self'") == 4      # every location serving the app
    assert "COPY frontend/nginx.law.conf /etc/nginx/conf.d/default.conf" in (ROOT / "frontend" / "Dockerfile.law").read_text()
    assert "frontend/nginx.law.conf" in (ROOT / "scripts" / "law" / "build_images.sh").read_text()
