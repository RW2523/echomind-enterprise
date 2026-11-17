"""The firm's legal vocabulary (LEGAL_FIRM_STYLE, 2026-11-16): a US firm sees US names for the same stages and presets;
the default (uk) is unchanged, and the keys the data uses never change."""
import re

from app.legal.counsel import drafting as chat_drafting
from app.legal.drafting import presets
from app.legal.matters import pipeline

_UK_WORDS = re.compile(r"\b(disclosure|brief counsel|exchange|solicitor|letter before claim|dear sirs|yours faithfully|completion)\b", re.I)


def test_the_default_vocabulary_is_unchanged():
    uk = pipeline.tracks_for("uk")
    assert pipeline.stage_label("disputes", "disclosure") in ("Disclosure & evidence", "Discovery")   # whatever this process runs
    assert [s.label for s in uk["disputes"].stages][3] == "Disclosure & evidence"
    assert [s.label for s in uk["transactions"].stages][4] == "Completion"
    assert "Brief counsel" in uk["disputes"].stages[5].tasks
    assert pipeline.tracks_for("anything else") is uk


def test_us_names_the_same_stages_in_us_terms_with_the_same_keys_and_order():
    uk, us = pipeline.tracks_for("uk"), pipeline.tracks_for("us")
    for key in uk:
        assert [s.key for s in us[key].stages] == [s.key for s in uk[key].stages]
        assert us[key].label == uk[key].label
    disputes = {s.key: s for s in us["disputes"].stages}
    assert disputes["disclosure"].label == "Discovery"
    assert "Take and defend depositions" in disputes["disclosure"].tasks
    assert disputes["hearing"].label == "Trial"
    assert {s.key: s.label for s in us["transactions"].stages}["completion"] == "Closing"
    assert {s.key: s.label for s in us["advisory"].stages}["instruction"] == "Engagement"


def test_no_english_terms_are_left_in_the_us_vocabulary():
    for track in pipeline.tracks_for("us").values():
        for st in track.stages:
            for text in (st.label, *st.tasks):
                assert not _UK_WORDS.search(text), (track.key, st.key, text)


def test_us_presets_carry_a_us_demand_letter_with_the_same_key_kind_and_fields():
    uk = {p["key"]: p for p in presets.presets_for("uk")}
    us = {p["key"]: p for p in presets.presets_for("us")}
    assert uk["demand_letter"]["name"] == "Letter before claim"
    assert us["demand_letter"]["name"] == "Demand letter"
    assert us["demand_letter"]["kind"] == uk["demand_letter"]["kind"] == "demand_letter"
    assert us["demand_letter"]["fields"] == uk["demand_letter"]["fields"]
    assert not _UK_WORDS.search(us["demand_letter"]["body"])
    # every field the body uses is still defined, and the model's passage is still a [[write: …]] instruction
    used = set(re.findall(r"\{\{(\w+)\}\}", us["demand_letter"]["body"]))
    assert used <= {f["key"] for f in us["demand_letter"]["fields"]}
    assert "[[write:" in us["demand_letter"]["body"]
    assert set(uk) == set(us) and all(us[k] is uk[k] for k in uk if k != "demand_letter")


def test_a_demand_letter_request_picks_the_demand_letter_preset_in_either_vocabulary():
    for style in ("uk", "us"):
        options = [{"id": p["key"], "kind": p["kind"], "name": p["name"]} for p in presets.presets_for(style)]
        assert chat_drafting._guess(options, "Draft a demand letter to Brennan Freight Lines") == "demand_letter"
    uk = [{"id": p["key"], "kind": p["kind"], "name": p["name"]} for p in presets.presets_for("uk")]
    assert chat_drafting._guess(uk, "Draft a letter before claim to Apex") == "demand_letter"
