"""The live line of a meeting transcript: the recogniser's pieces joined the way the original EchoMind joins them
(transcribe/session_state.py `_repair_chunk_boundary`, whose word lists are reused here).

The streaming model ends most ~0.5 s chunk hypotheses with a full stop and capitalises the next chunk ("When does the
initial term. Of the master. Services agreement"). For the line shown while someone is still speaking, such a stop is
dropped when the fragment before it cannot end a sentence, or the next piece plainly continues it, or both came in the
same breath. Display only: the transcript's record keeps exactly what the model produced (segments are immutable;
corrections are dated revisions), so nothing here is ever stored.
"""
from __future__ import annotations

import re

from ...transcribe import session_state as ss

SAME_BREATH_MS = 900


def join(text: str, piece: str, same_breath: bool) -> tuple[str, str]:
    """(text, piece) to concatenate, with an artefact '.'/'?' at the end of `text` removed and the piece's capital lowered."""
    t = text.rstrip()
    if not t or t[-1] not in ".?" or t.endswith(".."):
        return text, piece
    m_prev = re.search(r"([A-Za-z0-9']+)[.?]$", t)
    m_next = re.match(r"^(\s*)([A-Za-z0-9']+)", piece)
    if not m_prev or not m_next:
        return text, piece
    prev_w = m_prev.group(1).lower()
    lead, next_w = m_next.group(1), m_next.group(2)
    frag_words = ss._words(ss._last_fragment(t[:-1]))
    short_complete = " ".join(w.lower() for w in frag_words) in ss._SHORT_COMPLETE
    short_frag = len(frag_words) < ss._MIN_SENTENCE_WORDS and not short_complete
    continues = next_w[0].islower() or next_w.lower() in ss._CONTINUATION_START or next_w.isdigit()
    if t[-1] == "?":
        strip = continues
    else:
        strip = prev_w in ss._DANGLING_END or continues or (short_frag and next_w.lower() not in ss._SENTENCE_OPENERS)
    if same_breath and not strip:
        strip = not short_complete                       # no pause in the audio: not a real sentence end
    if not strip:
        return text, piece
    text = t[:-1] + text[len(t):]
    space = "" if text[-1:].isspace() else (lead or " ")
    keep_capital = next_w == "I" or next_w.isupper() or any(c.isdigit() for c in next_w) or next_w in ss._PROPER_NOUNS
    word = next_w if keep_capital or not next_w[:1].isupper() else next_w[0].lower() + next_w[1:]
    return text, space + word + piece[m_next.end():]
