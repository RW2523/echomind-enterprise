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
        else loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else loader.construct_scalar(node)))


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
