"""Unit tests for PDF/extracted text normalization (dehyphenate, collapse spaced letters, whitespace)."""
import pytest
from app.rag.normalize import (
    dehyphenate,
    collapse_spaced_letters,
    normalize_whitespace_preserve_paragraphs,
    normalize_extracted_text,
)


def test_dehyphenate():
    assert dehyphenate("") == ""
    assert dehyphenate("pat-\ntern") == "pattern"
    assert dehyphenate("ac-\ncomplish") == "accomplish"
    assert dehyphenate("some-\n  thing") == "something"
    assert dehyphenate("no hyphen here") == "no hyphen here"
    assert dehyphenate("end-\n") == "end"


def test_collapse_spaced_letters():
    assert collapse_spaced_letters("") == ""
    assert collapse_spaced_letters("C H A P T E R") == "CHAPTER"
    assert collapse_spaced_letters("O u t l i e r s") == "Outliers"
    assert collapse_spaced_letters("CHAPTER ONE  The Matthew Effect") == "CHAPTER ONE  The Matthew Effect"  # no change for normal words
    assert collapse_spaced_letters("A B C") == "ABC"
    assert collapse_spaced_letters("normal text") == "normal text"


def test_normalize_whitespace_preserve_paragraphs():
    assert normalize_whitespace_preserve_paragraphs("") == ""
    assert normalize_whitespace_preserve_paragraphs("a   b\t\nc") == "a b c"
    assert normalize_whitespace_preserve_paragraphs("para one\n\npara two") == "para one\n\npara two"
    assert normalize_whitespace_preserve_paragraphs("  \n\n  x  \n\n  y  ") == "x\n\ny"


def test_normalize_extracted_text():
    assert normalize_extracted_text("") == ""
    assert normalize_extracted_text("  ") == ""
    # Combined: dehyphenate + collapse + whitespace
    raw = "C H A P T E R  One\n\nThe  Mat-\nthew  Effect"
    out = normalize_extracted_text(raw)
    assert "CHAPTER" in out
    assert "Matthew" in out
    assert "Effect" in out
    assert "Mat-\n" not in out
