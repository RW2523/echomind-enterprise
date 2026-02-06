#!/usr/bin/env python3
"""
Backfill doc_headings for existing documents after adding the doc_headings table.
Run from backend root: python -m scripts.backfill_doc_headings
Or: PYTHONPATH=. python scripts/backfill_doc_headings.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import init_db
from app.rag.structure import backfill_doc_headings

if __name__ == "__main__":
    init_db()
    n = backfill_doc_headings()
    print(f"Backfilled doc_headings for {n} documents.")
