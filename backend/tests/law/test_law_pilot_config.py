"""Pilot configuration (D-059): the owner turned on every Phase 4 feature for the pilot on 2026-10-11.

The code keeps each feature off by default (the per-feature tests check that). The pilot stack gets its
switches from `.env.law` through `docker-compose.law.yml`, so the overlay must pass each one to the backend,
off when unset, and the pilot settings template must turn each one on.
"""
from __future__ import annotations

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
