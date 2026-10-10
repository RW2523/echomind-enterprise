"""Replies to an enquirer (S4.1, D-054): fixed templates, approved by a lawyer before the enquirer sees them.

A reply never gives the reason the firm cannot act, and never names another client or party of the firm:
a name the enquirer did not give, found in the firm's register or the intake's screening, refuses the text.
"""
from __future__ import annotations

import hashlib
import re

from . import screening

TEMPLATES = {
    "acknowledge": ("Dear {name},\n\nThank you for contacting us. We have received your enquiry (reference {reference}), and a member of "
                    "our team will be in touch.\n\nThis message is not legal advice, and we do not act for you until we confirm in "
                    "writing that we can."),
    "can_help": ("Dear {name},\n\nThank you for your enquiry (reference {reference}). We would be glad to help. A lawyer from our team "
                 "will contact you to discuss the next steps and our terms of engagement.\n\nWe do not act for you until you have "
                 "received and accepted our terms."),
    "cannot_act": ("Dear {name},\n\nThank you for contacting us about your enquiry (reference {reference}). We are sorry that we are "
                   "not able to help with this matter. This is not a view on its merits. Time limits may apply, so please seek advice "
                   "elsewhere promptly."),
    "more_info": ("Dear {name},\n\nThank you for your enquiry (reference {reference}). Before we can go further, we need the "
                  "following:\n\n{request}\n\nPlease do not send confidential documents until we confirm that we can act for you."),
}
NEEDS_CLEARANCE = {"can_help"}


class Invalid(ValueError):
    pass


def sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def render(template: str, reference: str, answers: dict, request: str | None) -> str:
    if template not in TEMPLATES:
        raise Invalid(f"Use one of the reply templates: {', '.join(TEMPLATES)}.")
    request = " ".join((request or "").split())
    if template == "more_info" and not request:
        raise Invalid("Say what the firm needs from the enquirer.")
    if len(request) > 600:
        raise Invalid("Keep the request to 600 characters.")
    return TEMPLATES[template].format(name=answers.get("your_name") or "Sir or Madam", reference=reference, request=request)


def _contains(body: str, name: str) -> bool:
    k = screening.key(name)
    if not k:
        return False
    if " " in k:                                         # several words: folded, as a run of whole words
        return re.search(rf"(?<!\w){re.escape(k)}(?!\w)", screening.key(body)) is not None
    word = re.sub(r"[^\w]", "", name.split()[0]) if name.split() else ""
    return bool(word) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", body) is not None    # one word: as the firm writes it


def leaks(body: str, others: list[str], own: list[str]) -> bool:
    """Whether the text names another client or party of the firm (a name the enquirer did not give)."""
    mine = {screening.key(n) for n in own if screening.key(n)}
    return any(_contains(body, n) for n in others if screening.key(n) and screening.key(n) not in mine)
