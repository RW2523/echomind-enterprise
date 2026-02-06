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


def normalize_extracted_text(text: str) -> str:
    """
    Full normalization pipeline for PDF/extracted text. Apply after parse, before chunking.
    Order: dehyphenate -> collapse spaced letters -> normalize whitespace.
    """
    if not (text or "").strip():
        return text or ""
    t = dehyphenate(text)
    t = collapse_spaced_letters(t)
    t = normalize_whitespace_preserve_paragraphs(t)
    return t
