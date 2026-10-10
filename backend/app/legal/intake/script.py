"""The intake script and disclaimer (S4.1, D-054).

Administrative only: fixed questions in a fixed order, fixed acknowledgements, and no model. The
enquirer accepts the disclaimer (this exact version) before the first question.
"""
from __future__ import annotations

import re
from datetime import date

SCRIPT_VERSION = "intake-script-1"
DISCLAIMER = {
    "version": "intake-disclaimer-1",
    "text": ("These questions collect administrative details only. This is not legal advice, and contacting us does not make "
             "us your lawyers. Please do not send confidential information until we confirm in writing that we can act for you. "
             "Before we can help, we check that we have no conflict of interest with other clients."),
}
MATTER_TYPES = {"corporate": "Corporate", "commercial": "Commercial contracts", "employment": "Employment", "dispute": "A dispute",
                "property": "Property", "regulatory": "Regulatory", "private_client": "Private client", "other": "Something else"}
HEARD_FROM = {"referral": "A referral", "existing_client": "I am an existing client", "website": "The firm's website",
              "event": "An event", "other": "Elsewhere"}
QUESTIONS = [
    {"key": "your_name", "label": "Your name", "kind": "text", "required": True, "max": 120,
     "ask": "What is your full name?", "thanks": "Thank you."},
    {"key": "organisation", "label": "Organisation", "kind": "text", "required": False, "max": 160,
     "ask": "If you are contacting us for a company or organisation, what is its name?", "thanks": "Noted."},
    {"key": "email", "label": "Email", "kind": "email", "required": False, "max": 160,
     "ask": "What email address should we use?", "thanks": "Noted."},
    {"key": "phone", "label": "Telephone", "kind": "phone", "required": False, "max": 40,
     "ask": "And a telephone number? (We need an email address or a telephone number.)", "thanks": "Noted."},
    {"key": "matter_type", "label": "Kind of matter", "kind": "choice", "required": True, "choices": MATTER_TYPES,
     "ask": "Which of these best describes what you need help with?", "thanks": "Thank you."},
    {"key": "other_parties", "label": "Other parties", "kind": "names", "required": False, "max": 160, "max_items": 10,
     "ask": "Who else is involved, such as the other side? Names only, one per line.", "thanks": "Thank you. We check these names before we act."},
    {"key": "summary", "label": "Brief description", "kind": "longtext", "required": True, "max": 600,
     "ask": "In a few sentences, what do you need help with? Please leave out confidential details for now.", "thanks": "Thank you."},
    {"key": "deadline", "label": "Date that matters", "kind": "date", "required": False,
     "ask": "Is there a date by which something must happen? (Leave blank if not.)", "thanks": "Noted."},
    {"key": "heard_from", "label": "How you heard of us", "kind": "choice", "required": False, "choices": HEARD_FROM,
     "ask": "How did you hear of us?", "thanks": "Thank you. That is everything we need for now."},
]
SCRIPT = {"version": SCRIPT_VERSION, "questions": QUESTIONS}
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,120}\.[^@\s.]{2,24}$")
_PHONE = re.compile(r"^\+?[\d ()\-.]{6,40}$")


class Invalid(ValueError):
    pass


def _text(v, q: dict) -> str:
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise Invalid(f"{q['label']}: write it as text.")
    v = " ".join(v.split()) if q["kind"] != "longtext" else v.strip()
    if len(v) > q.get("max", 600):
        raise Invalid(f"{q['label']}: keep it to {q.get('max', 600)} characters.")
    return v


def validate(answers: dict) -> dict:
    """The answers, cleaned, or Invalid with what to correct."""
    if not isinstance(answers, dict):
        raise Invalid("Answer the questions.")
    known = {q["key"] for q in QUESTIONS}
    extra = sorted(set(answers) - known)
    if extra:
        raise Invalid(f"Unknown question(s): {', '.join(extra)}.")
    out: dict = {}
    for q in QUESTIONS:
        v = answers.get(q["key"])
        if q["kind"] == "names":
            items = v if isinstance(v, list) else [s for s in str(v or "").splitlines()]
            names = [" ".join(str(s).split()) for s in items if str(s).strip()]
            if len(names) > q["max_items"]:
                raise Invalid(f"{q['label']}: give up to {q['max_items']} names.")
            if any(len(n) > q["max"] for n in names):
                raise Invalid(f"{q['label']}: keep each name to {q['max']} characters.")
            out[q["key"]] = list(dict.fromkeys(names))
            continue
        if q["kind"] == "choice":
            v = (v or "").strip() if isinstance(v, str) else v
            if v in (None, ""):
                if q["required"]:
                    raise Invalid(f"{q['label']}: choose one.")
                out[q["key"]] = ""
                continue
            if v not in q["choices"]:
                raise Invalid(f"{q['label']}: choose one of the options.")
            out[q["key"]] = v
            continue
        v = _text(v, q)
        if q["required"] and not v:
            raise Invalid(f"{q['label']}: this is needed.")
        if v and q["kind"] == "email" and not _EMAIL.match(v):
            raise Invalid("Email: this does not look like an email address.")
        if v and q["kind"] == "phone" and not _PHONE.match(v):
            raise Invalid("Telephone: use digits, spaces and + ( ) - only.")
        if v and q["kind"] == "date":
            try:
                date.fromisoformat(v)
            except ValueError:
                raise Invalid("Date: use the form 2026-11-30.") from None
        out[q["key"]] = v
    if not out["email"] and not out["phone"]:
        raise Invalid("Give an email address or a telephone number.")
    return out


def names(answers: dict) -> list[str]:
    """The names an intake gives: the enquirer, their organisation and the other parties."""
    return [n for n in [answers.get("your_name", ""), answers.get("organisation", "")] + list(answers.get("other_parties") or []) if n]
