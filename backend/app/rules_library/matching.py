"""Simple local rule matching: phrase / line containment (no policy engine)."""
from __future__ import annotations

import re
from typing import Optional


def transcript_matches_rule(rule_text: str, transcript: str) -> bool:
    """
    True if a substantive fragment of the rule text appears in the transcript (case-insensitive).
    Splits rule body on newlines/semicolons; each segment must be 8–400 chars to match.
    """
    t = (transcript or "").lower()
    rt = (rule_text or "").strip()
    if not rt or len(t) < 12:
        return False
    for part in re.split(r"[\n\r;]+", rt):
        s = part.strip().lower()
        if 8 <= len(s) <= 400 and s in t:
            return True
    whole = rt.lower()
    if 8 <= len(whole) <= 120 and whole in t:
        return True
    return False


def first_matching_rule(
    transcript: str,
    rule_rows: list[dict],
) -> Optional[dict]:
    """rule_rows: dicts with at least id, title, text, rule_set_id, rule_set_name, severity, category."""
    for r in rule_rows:
        if transcript_matches_rule(str(r.get("text") or ""), transcript):
            return r
    return None
