"""Regulatory updates (S4.4, OPT-04; decision D-068). Off unless LEGAL_REGULATORY is on.

New public legislation and rules arrive as a bundle fetched on the host (scripts/law/fetch_regulatory.py): the
Law containers never reach the internet. `python -m app.legal.regulatory import < bundle.json` checks every field
again (the bundle is untrusted text), stores what is new and matches it to open matters by jurisdiction and the
terms of their practice area (topics.py), so a lawyer sees which updates may concern which matter and why.
"""
