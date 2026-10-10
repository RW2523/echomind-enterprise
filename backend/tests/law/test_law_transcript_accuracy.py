"""S2.4 — scoring a legal transcript against its script (LIVE-01; D-040).

Word error rate verbatim and normalised; every name, amount, date and reference present (spoken
forms count); every negation kept; speakers attributed. No model is involved: these are the
yardsticks the transcription harness reports with.
"""
from __future__ import annotations

from app.legal.meetings import accuracy as acc

SCRIPT = [
    {"speaker": "Helena Marsh", "text": "Harrowgate Systems Inc. proposes a fee of USD 19,750 per month from 2 March 2026."},
    {"speaker": "Petra Voss", "text": "We will not accept a cap below USD 400,000. Clause 8.1 must change."},
]
ENTITIES = [acc.Entity("name", "Harrowgate Systems"), acc.Entity("amount", "USD 19,750"), acc.Entity("date", "2 March 2026"),
            acc.Entity("amount", "USD 400,000"), acc.Entity("reference", "Clause 8.1")]
NEGATIONS = ["We will not accept a cap below USD 400,000"]


def test_a_perfect_transcript_scores_perfectly():
    s = acc.score(SCRIPT, SCRIPT, ENTITIES, NEGATIONS).as_dict()
    assert s["wer_verbatim"] == 0 and s["wer_normalised"] == 0
    assert s["entity_accuracy"] == 1 and s["negation_accuracy"] == 1 and s["speaker_accuracy"] == 1


def test_spoken_numbers_and_dates_count_as_the_written_ones():
    spoken = [
        {"speaker": "Helena Marsh", "text": "harrowgate systems inc proposes a fee of nineteen thousand seven hundred and fifty dollars "
                                            "per month from the second of march twenty twenty six"},
        {"speaker": "Petra Voss", "text": "we will not accept a cap below four hundred thousand dollars clause eight point one must change"},
    ]
    s = acc.score(SCRIPT, spoken, ENTITIES, NEGATIONS)
    assert s.missed_entities == ["Clause 8.1"]            # "eight point one" is not "8.1": reported, not hidden
    assert s.negations_kept == 1 and s.wer_verbatim > s.wer_normalised


def test_a_dropped_negation_and_a_wrong_amount_are_listed():
    heard = [
        {"speaker": "Helena Marsh", "text": "Harrowgate Systems Inc. proposes a fee of USD 19,570 per month from 2 March 2026."},
        {"speaker": "Petra Voss", "text": "We will accept a cap below USD 400,000. Clause 8.1 must change."},
    ]
    s = acc.score(SCRIPT, heard, ENTITIES, NEGATIONS)
    assert s.missed_entities == ["USD 19,750"] and s.lost_negations == NEGATIONS
    assert 0 < s.wer_normalised < 0.15


def test_words_given_to_the_wrong_speaker_lower_speaker_accuracy():
    swapped = [{"speaker": "Petra Voss", "text": SCRIPT[0]["text"]}, {"speaker": "Petra Voss", "text": SCRIPT[1]["text"]}]
    s = acc.score(SCRIPT, swapped, ENTITIES, NEGATIONS)
    first = len(acc.normalised(SCRIPT[0]["text"]))
    total = first + len(acc.normalised(SCRIPT[1]["text"]))
    assert abs(s.speaker_accuracy - (total - first) / total) < 1e-9
    assert acc.score(SCRIPT, [{"speaker": None, "text": t["text"]} for t in SCRIPT], ENTITIES, NEGATIONS).speaker_accuracy is None


def test_normalisation_is_what_a_lawyer_would_accept():
    assert acc.normalised("Um, the fee is USD 18,500 — payable net thirty (30) days.") == \
        ["the", "fee", "is", "18500", "usd", "payable", "net", "30", "30", "days"]
    assert acc.normalised("$400,000") == acc.normalised("four hundred thousand dollars") == ["400000", "usd"]
    assert acc.normalised("nineteen ninety-nine and one hundred and twenty days") == ["1999", "and", "120", "days"]
    assert acc.wer(["a", "b", "c"], ["a", "x", "c", "d"]) == 2 / 3
