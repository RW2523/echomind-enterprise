"""
PDF/extracted text normalization before chunking and indexing.
Fixes spaced-out letters (C H A P T E R), hyphen line breaks (ac-\ncomplish), and whitespace.
Unit-testable; apply right after parsing, before sanitize/detect/chunking.
"""
from __future__ import annotations
import re


def dehyphenate(text: str) -> str:
    """
    Join hyphenated line breaks: "pat-\ntern" -> "pattern", "ac-\ncomplish" -> "accomplish".
    Matches: optional space before hyphen, newline, optional spaces, then continuation.
    """
    if not text:
        return text
    # Pattern: word chars + optional space + hyphen + newline + optional space -> join (remove hyphen and newline)
    return re.sub(r"(\w)\s*-\s*\n\s*", r"\1", text)


def collapse_spaced_letters(text: str) -> str:
    """
    Collapse single-letter spacing into words: "C H A P T E R" -> "CHAPTER", "O u t l i e r s" -> "Outliers".
    Matches runs of single alphanumeric chars separated by whitespace; replaces with concatenated word.
    """
    if not text:
        return text

    def replace_spaced(match: re.Match) -> str:
        s = match.group(0)
        parts = re.split(r"\s+", s)
        if len(parts) < 2:
            return s
        if all(len(p) == 1 and p.isalnum() for p in parts):
            return "".join(parts)
        return s

    # (single \w + whitespace)+ then single \w; \s+ allows newlines between letters
    return re.sub(r"(?<!\w)(?:\w\s+)+\w(?!\w)", replace_spaced, text)


def remove_repeated_headers_footers(text: str, min_repeats: int = 3, max_line_len: int = 80) -> str:
    """
    Heuristic: remove lines that repeat many times (likely headers/footers).
    Only considers lines with length <= max_line_len. Requires min_repeats occurrences.
    """
    if not text or min_repeats < 2:
        return text or ""
    lines = text.split("\n")
    from collections import Counter
    line_counts = Counter()
    for line in lines:
        s = line.strip()
        if 0 < len(s) <= max_line_len:
            line_counts[s] += 1
    repeated = {s for s, c in line_counts.items() if c >= min_repeats}
    if not repeated:
        return text
    out = []
    for line in lines:
        s = line.strip()
        if s in repeated:
            continue
        out.append(line)
    return "\n".join(out)


def normalize_whitespace_preserve_paragraphs(text: str) -> str:
    """
    Normalize whitespace: collapse runs of spaces/tabs to single space, preserve paragraph breaks (\n\n).
    """
    if not text:
        return text
    # Replace \n\n (paragraph) with a sentinel, collapse other whitespace, restore \n\n
    PARAGRAPH = "\x00PARA\x00"
    t = text.replace("\n\n", PARAGRAPH)
    t = re.sub(r"[ \t\r\n]+", " ", t)
    t = t.replace(PARAGRAPH, "\n\n")
    return t.strip()


def normalize_extracted_text(text: str, remove_headers_footers: bool = False) -> str:
    """
    Full normalization pipeline for PDF/extracted text. Apply after parse, before chunking.
    Order: dehyphenate -> collapse spaced letters -> [optional] remove repeated headers/footers -> normalize whitespace.
    """
    if not (text or "").strip():
        return text or ""
    t = dehyphenate(text)
    t = collapse_spaced_letters(t)
    if remove_headers_footers:
        t = remove_repeated_headers_footers(t)
    t = normalize_whitespace_preserve_paragraphs(t)
    return t
