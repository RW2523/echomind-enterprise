"""Term and notice dates calculated from a document's words (2026-10-29 QA RAG-003): calendar arithmetic, leap years, month
ends, and the cases where nothing is calculated."""
from __future__ import annotations

from datetime import date

from app.legal.research import dates


def calc(text: str) -> dict:
    c = dates.calculate(text)
    return {"derived": {d["label"]: d["display"] for d in c.derived}, "facts": {f.label: f.words(text) for f in c.facts},
            "assumptions": c.assumptions, "not": c.not_calculated} if c else {}


def test_the_testers_agreement():
    text = ("1. TERM\nThis agreement begins February 1, 2026 and remains effective for 12 months.\n\n6. RENEWAL\nThe agreement "
            "automatically renews for an additional 12 months unless either party provides written notice of non-renewal at least "
            "45 days before the current term ends.")
    c = calc(text)
    assert c["derived"] == {"Current term ends": "31 January 2027", "Last day to give notice of non-renewal": "17 December 2026",
                            "If no notice is given, the next term": "1 February 2027 to 31 January 2028"}
    assert c["facts"] == {"Start": "begins February 1, 2026", "Term": "remains effective for 12 months",
                          "Notice before the term ends": "at least 45 days before the current term ends",
                          "Renewal": "renews for an additional 12 months"}
    assert any("12 calendar months from and including 1 February 2026" in a for a in c["assumptions"])
    assert any("45 calendar days counted back from 31 January 2027" in a for a in c["assumptions"])


def test_leap_years_and_month_ends():
    assert dates.term_end(date(2028, 2, 29), 12, "month") == date(2029, 2, 28)      # no 29 February in 2029
    assert dates.term_end(date(2028, 3, 1), 12, "month") == date(2029, 2, 28)
    assert dates.term_end(date(2027, 3, 1), 12, "month") == date(2028, 2, 29)       # the term crosses 29 February 2028
    assert dates.term_end(date(2026, 1, 31), 1, "month") == date(2026, 2, 28)       # no 31 February: the month's last day
    assert dates.term_end(date(2028, 1, 31), 1, "month") == date(2028, 2, 29)
    assert dates.term_end(date(2026, 8, 31), 6, "month") == date(2027, 2, 28)
    assert dates.term_end(date(2026, 1, 15), 1, "month") == date(2026, 2, 14)
    assert dates.term_end(date(2026, 3, 2), 3, "year") == date(2029, 3, 1)
    assert dates.term_end(date(2026, 12, 1), 90, "day") == date(2027, 2, 28)
    c = calc("The lease begins on 29 February 2028 and remains in force for 12 months. Notice must be given at least 30 days "
             "before the end of the term.")
    assert c["derived"]["Current term ends"] == "28 February 2029" and c["derived"]["Last day to give notice of non-renewal"] == "29 January 2029"


def test_a_notice_period_in_months_and_a_start_defined_as_the_effective_date():
    c = calc('This Agreement is entered into as of 2 March 2026 (the "Effective Date"). It begins on the Effective Date and continues '
             "for an initial term of thirty-six (36) months. It renews automatically for successive twelve (12) month renewal terms "
             "unless either party gives notice at least three (3) months before the end of the then-current term.")
    assert c["derived"]["Current term ends"] == "1 March 2029"
    assert c["derived"]["Last day to give notice of non-renewal"] == "1 December 2028"
    assert c["facts"]["Start"] == 'as of 2 March 2026 (the "Effective Date")'


def test_a_stated_end_date_is_used_as_stated():
    c = calc("The term is ten (10) years starting on 1 April 2025 and ending on 31 March 2035. Notice of non-renewal must be given "
             "no later than 60 days before the term ends.")
    assert "Current term ends" not in c["derived"] and c["facts"]["End (stated)"] == "ending on 31 March 2035"
    assert c["derived"]["Last day to give notice of non-renewal"] == "30 January 2035"


def test_nothing_is_calculated_when_the_document_is_ambiguous_or_needs_a_holiday_calendar():
    two = calc("This agreement begins February 1, 2026 and remains effective for 12 months. Schedule 2 begins March 1, 2026.")
    assert two["derived"] == {} and "more than once, differently" in two["not"][0]
    business = calc("This agreement begins February 1, 2026 and remains effective for 12 months. Notice is due at least 10 business "
                    "days before the current term ends.")
    assert "Last day to give notice of non-renewal" not in business["derived"] and "holiday calendar" in business["not"][0]
    assert calc("Either party may terminate this agreement on 60 days' written notice.") == {}
    no_term = calc("This agreement begins February 1, 2026.")
    assert no_term["derived"] == {} and "no term length" in no_term["not"][0]
    contradicts = calc("This agreement begins February 1, 2026 and remains effective for twelve (18) months.")
    assert contradicts["derived"] == {}


def test_a_survival_period_is_not_the_term():
    c = calc('This Agreement is made on 10 January 2026 (the "Effective Date"). This Agreement lasts for three (3) years from the '
             "Effective Date. The obligations in Clause 3 continue for five (5) years after this Agreement ends.")
    assert c["derived"]["Current term ends"] == "9 January 2029"


def test_which_questions_ask_for_term_dates():
    for q in ("When does the current term end?", "What is the deadline to give notice of non-renewal?", "When does the lease expire?",
              "By when must notice of non-renewal be given?", "When does the agreement renew?"):
        assert dates.asks_term_dates(q), q
    for q in ("What is the monthly payment?", "When are invoices due?", "Who are the parties?", "What is the termination notice period?"):
        assert not dates.asks_term_dates(q), q
