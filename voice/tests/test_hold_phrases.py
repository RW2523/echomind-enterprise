"""
Tests for voice/app/hold_phrases.py (pure stdlib module, no app imports).

Run from the repo root:  python3 -m pytest voice/tests/test_hold_phrases.py -q
"""
import importlib.util
import os
import random
import re

import pytest

_MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "hold_phrases.py")
_spec = importlib.util.spec_from_file_location("hold_phrases", os.path.abspath(_MODULE_PATH))
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

TOOLS = hp.TOOLS
TONES = hp.TONES
ALL_POOLS = [(tool, tone) for tool in TOOLS for tone in TONES]
_ID_RE = re.compile(r"^(search_knowledge_base|search_transcripts|lookup_record)/(%s)/\d+$" % "|".join(TONES))


def _words(text):
    return len(text.split())


def _no_window_repeats(ids, window):
    for i in range(len(ids)):
        chunk = ids[i:i + window]
        assert len(set(chunk)) == len(chunk), f"repeat inside window {window} at {i}: {chunk}"


# ---------------------------------------------------------------------------
# tone_key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    # PersonaType labels seen in production
    ("Lawyer", "legal"),
    ("Legal Advisor", "legal"),
    ("Financial Advisor", "bank"),
    ("Banking Advisor", "bank"),
    ("Clinical Assistant", "health"),
    ("Meeting Facilitator", "meetings"),
    ("Retail Assistant", "retail"),
    ("Customer Care", "customer_care"),
    ("General Assistant", "neutral"),
    ("EchoMind Guide", "neutral"),
    ("Teacher / Professor", "neutral"),
    ("Funny & Calming Assistant", "neutral"),
    ("AI Expert & Manager", "neutral"),
    # vertical pack ids
    ("law", "legal"),
    ("legal", "legal"),
    ("bank", "bank"),
    ("banking", "bank"),
    ("health", "health"),
    ("meetings", "meetings"),
    ("retail", "retail"),
    ("customer_care", "customer_care"),
    # empties
    (None, "neutral"),
    ("", "neutral"),
    ("   ", "neutral"),
])
def test_tone_key_mapping(label, expected):
    assert hp.tone_key(label) == expected


def test_tone_key_is_case_insensitive_and_substring_tolerant():
    assert hp.tone_key("LAWYER") == "legal"
    assert hp.tone_key("  the banking advisor  ") == "bank"
    assert hp.tone_key("clinical-assistant") == "health"
    assert hp.tone_key("Health Care") == "health"          # domain wins over generic "care"
    assert hp.tone_key("Banking Customer Support") == "bank"
    for tone in TONES:
        assert hp.tone_key(tone) == tone
        assert hp.tone_key(tone.upper()) == tone


# ---------------------------------------------------------------------------
# topic_from_query
# ---------------------------------------------------------------------------

def test_topic_stops_at_preposition_then_caps():
    # stop at "after" -> "refund policy cancellations" (3 words) -> common noun so "the " is prepended
    assert hp.topic_from_query("refund policy cancellations after 14 days") == "the refund policy cancellations"


def test_topic_keeps_identifiers_verbatim_and_unstripped():
    assert hp.topic_from_query("customer ID 48213") == "customer ID 48213"
    assert hp.topic_from_query("48213") == "48213"
    assert hp.topic_from_query("case 4471") == "case 4471"
    assert hp.topic_from_query("policy P-1102") == "policy P-1102"
    assert hp.topic_from_query("look up customer 48213") == "customer 48213"
    assert hp.topic_from_query("(P-1102)") == "P-1102"


def test_topic_prepends_the_for_common_nouns_only():
    assert hp.topic_from_query("refund policy") == "the refund policy"
    assert hp.topic_from_query("tell me about our refund policy") == "the refund policy"
    assert hp.topic_from_query("What's the Refund policy?") == "the Refund policy"
    # proper nouns / honorifics keep their case and get no article
    assert hp.topic_from_query("Ms. Patel") == "Ms. Patel"
    assert hp.topic_from_query("Ms Patel") == "Ms. Patel"
    assert hp.topic_from_query("John Smith") == "John Smith"
    # surviving possessive -> no article
    assert hp.topic_from_query("their refund policy") == "their refund policy"


def test_topic_dod_fmr_example():
    out = hp.topic_from_query("What is the DoD FMR volume 7A?")
    assert out is not None
    # case is preserved for acronyms/mixed case (better for TTS); compare case-insensitively
    assert "dod fmr" in out.lower()
    assert out.lower() == "the dod fmr volume"
    assert out == "the DoD FMR volume"


def test_topic_termination_for_convenience():
    # Documented choice: a phrase that already fits within max_words is kept whole
    # (including its preposition) rather than cut to "termination"; as a common-noun
    # phrase it gets the article -> "the termination for convenience".
    out = hp.topic_from_query("termination for convenience")
    assert out == "the termination for convenience"
    # ... but a longer phrase is cut before the last preposition that still fits
    assert hp.topic_from_query("interest rate on savings accounts") == "the interest rate"
    assert hp.topic_from_query("policy on refunds after 14 days") == "the policy on refunds"


def test_topic_strips_leading_question_words_but_not_domain_nouns():
    assert hp.topic_from_query("what does the policy say about refunds") == "the policy about refunds"
    assert hp.topic_from_query("can you check the early repayment fee on my mortgage") == "the early repayment fee"
    assert hp.topic_from_query("terms of service") == "the terms of service"      # "of" is not a stop
    assert hp.topic_from_query("in-network providers") == "the in-network providers"
    assert hp.topic_from_query("US tax policy") == "the US tax policy"           # acronym is not "us"
    assert hp.topic_from_query("IT policy") == "the IT policy"                   # acronym is not "it"


def test_topic_max_words_cap():
    assert hp.topic_from_query("refund policy cancellations after 14 days", max_words=2) == "the refund policy"
    assert hp.topic_from_query("customer refund policy details", max_words=3) == "the customer refund policy"
    assert hp.topic_from_query("refund policy", max_words=1) == "the refund"


def test_topic_returns_none_when_nothing_usable():
    assert hp.topic_from_query(None) is None
    assert hp.topic_from_query("") is None
    assert hp.topic_from_query("   ") is None
    assert hp.topic_from_query("what is") is None
    assert hp.topic_from_query("check") is None
    assert hp.topic_from_query("please look up") is None
    assert hp.topic_from_query("?!,") is None
    # > 40 chars
    assert hp.topic_from_query("supercalifragilisticexpialidocious antidisestablishmentarianism") is None


def test_topic_never_exceeds_limits():
    for q in [
        "refund policy cancellations after 14 days and other things",
        "the quick brown fox jumps over the lazy dog",
        "customer ID 48213 with an unusually long trailing description attached",
        "What is the DoD FMR volume 7A?",
    ]:
        out = hp.topic_from_query(q)
        if out is not None:
            assert len(out) <= hp.TOPIC_MAX_CHARS
            body = out[4:] if out.startswith("the ") else out
            assert _words(body) <= 3
            assert not out.endswith((".", ",", "?"))
            assert "{" not in out


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,tone", ALL_POOLS)
def test_pool_shape_and_style(tool, tone):
    pool = hp.POOLS[(tool, tone)]
    assert len(pool) >= 6
    assert len(set(pool)) == len(pool), "duplicate template in pool"
    slotted = [t for t in pool if "{topic}" in t]
    slotless = [t for t in pool if "{topic}" not in t]
    assert len(slotted) >= 2 and len(slotless) >= 4, "pool must mix slot and slot-less templates"
    for tmpl in pool:
        assert tmpl.endswith("."), tmpl
        assert "!" not in tmpl, tmpl
        assert "great question" not in tmpl.lower(), tmpl
        assert tmpl != "One moment while I check.", tmpl
        assert tmpl.isascii(), tmpl
        assert not tmpl.startswith("{topic}"), tmpl
        assert tmpl.count("{") == tmpl.count("}") <= 1, tmpl
        assert _words(tmpl) <= hp.HOLD_PHRASE_MAX_WORDS, tmpl
        rendered = tmpl.replace("{topic}", "the refund policy")
        assert _words(rendered) <= hp.HOLD_PHRASE_MAX_WORDS, rendered


def test_pool_coverage():
    assert hp.HOLD_PHRASE_MAX_WORDS == 9
    assert set(hp.POOLS.keys()) == set(ALL_POOLS)


# ---------------------------------------------------------------------------
# pick_hold_phrase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,tone", ALL_POOLS)
def test_pick_hold_phrase_anti_repeat_with_topic(tool, tone):
    rng = random.Random(1234)
    ids, texts = [], []
    for _ in range(40):
        text, tid = hp.pick_hold_phrase(tool, tone, "the refund policy", ids[-3:], rng=rng)
        ids.append(tid)
        texts.append(text)
        assert _ID_RE.match(tid), tid
        assert tid.startswith(f"{tool}/"), tid
        assert text.endswith(".") and text[0].isupper(), text
        assert _words(text) <= hp.HOLD_PHRASE_MAX_WORDS, text
        assert "{" not in text and "}" not in text
    _no_window_repeats(ids, 4)


@pytest.mark.parametrize("tool,tone", ALL_POOLS)
def test_pick_hold_phrase_without_topic(tool, tone):
    rng = random.Random(99)
    ids = []
    for _ in range(40):
        text, tid = hp.pick_hold_phrase(tool, tone, None, ids[-3:], rng=rng)
        ids.append(tid)
        assert "{" not in text and "None" not in text, text
        assert text.endswith(".") and text[0].isupper()
        assert _words(text) <= hp.HOLD_PHRASE_MAX_WORDS
    _no_window_repeats(ids, 4)


def test_pick_hold_phrase_uses_topic_and_keeps_its_case():
    rng = random.Random(7)
    seen_topic = False
    for _ in range(30):
        text, tid = hp.pick_hold_phrase("lookup_record", "bank", "customer ID 48213", [], rng=rng)
        if "48213" in text:
            assert "customer ID 48213" in text, text
            seen_topic = True
    assert seen_topic
    text, _ = hp.pick_hold_phrase("lookup_record", "legal", "Ms. Patel", [], rng=random.Random(3))
    assert text.endswith(".") and text[0].isupper()


def test_pick_hold_phrase_prefers_tone_pool_and_falls_back_to_neutral():
    # With every tone template in `recent`, the neutral pool for that tool is used.
    tone_ids = [f"search_knowledge_base/legal/{n}" for n in range(1, len(hp.POOLS[("search_knowledge_base", "legal")]) + 1)]
    _, tid = hp.pick_hold_phrase("search_knowledge_base", "legal", "the refund policy", tone_ids, rng=random.Random(0))
    assert tid.startswith("search_knowledge_base/neutral/")
    # With both exhausted, anything but the very last id is returned.
    neutral_ids = [f"search_knowledge_base/neutral/{n}" for n in range(1, len(hp.POOLS[("search_knowledge_base", "neutral")]) + 1)]
    recent = neutral_ids + tone_ids
    _, tid = hp.pick_hold_phrase("search_knowledge_base", "legal", "the refund policy", recent, rng=random.Random(0))
    assert tid != recent[-1]


def test_pick_hold_phrase_tolerates_unknown_tool_tone_and_long_topic():
    text, tid = hp.pick_hold_phrase("something_else", "Teacher / Professor", "the refund policy", [], rng=random.Random(1))
    assert tid.startswith("search_knowledge_base/neutral/")
    text, tid = hp.pick_hold_phrase("lookup_record", "Lawyer", None, [], rng=random.Random(1))
    assert tid.startswith("lookup_record/legal/")
    # An over-long topic can never push the rendered text past the word cap.
    long_topic = "the very long topic phrase that goes on and on"
    for _ in range(20):
        text, _ = hp.pick_hold_phrase("search_knowledge_base", "bank", long_topic, [], rng=random.Random(5))
        assert _words(text) <= hp.HOLD_PHRASE_MAX_WORDS
    # Topic that already carries a period does not produce ".."
    text, _ = hp.pick_hold_phrase("lookup_record", "retail", "order 1187.", [], rng=random.Random(2))
    assert ".." not in text


def test_pick_hold_phrase_is_deterministic_with_seeded_rng():
    a = [hp.pick_hold_phrase("search_transcripts", "meetings", "the budget", [], rng=random.Random(42)) for _ in range(5)]
    b = [hp.pick_hold_phrase("search_transcripts", "meetings", "the budget", [], rng=random.Random(42)) for _ in range(5)]
    assert a == b


# ---------------------------------------------------------------------------
# pick_ack
# ---------------------------------------------------------------------------

def test_ack_pool_shape():
    assert len(hp.ACKS) >= 6
    for a in hp.ACKS:
        assert a.endswith(".") and "!" not in a and a.isascii()
        assert 1 <= _words(a) <= 3


@pytest.mark.parametrize("tone", TONES)
def test_pick_ack_anti_repeat(tone):
    rng = random.Random(2024)
    ids = []
    for _ in range(40):
        text, tid = hp.pick_ack(tone, ids[-2:], rng=rng)
        ids.append(tid)
        assert re.match(r"^ack/\d+$", tid), tid
        assert text in hp.ACKS
    _no_window_repeats(ids, 3)
    # Also holds with a wider recent window.
    ids = []
    for _ in range(40):
        _, tid = hp.pick_ack(tone, ids[-3:], rng=rng)
        ids.append(tid)
    _no_window_repeats(ids, 4)


def test_pick_ack_formal_tones_skip_chatty_openers():
    for tone in ("legal", "bank", "health"):
        texts = {hp.pick_ack(tone, [], rng=random.Random(i))[0] for i in range(200)}
        assert "So." not in texts and "Well." not in texts
    texts = {hp.pick_ack("neutral", [], rng=random.Random(i))[0] for i in range(200)}
    assert {"So.", "Well."} <= texts
