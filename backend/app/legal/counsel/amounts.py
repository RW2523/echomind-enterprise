"""Amounts the matter's documents state, as a chart (docs/echomind-legal/03 `numeric_chart`: "verified numerical data +
units + provenance"; owner's request 2026-11-02).

Read by rule from the documents the person may read (compare/facts.py, the conflict scan's reader), never by the model:
each bar is one amount as a document writes it, with its sentence and a way to open it. One currency per chart (the most
stated one; others are named, never converted or added). Nothing is summed: two documents stating the same fee are two
bars, because the chart shows what the documents say, not what is owed.
"""
from __future__ import annotations

import re
from collections import Counter

from ...core.db import get_conn
from ..compare import facts, scan
from . import discrepancies

MAX_BARS = 24
_VALUE = re.compile(r"^([A-Z]{3})\s+(\d+(?:\.\d+)?)$")


def _same_digits(shown: str, value: float) -> bool:
    """The amount as written has exactly the digits of its value ("$12,000" and 12000): one written another way ("EUR 5.000",
    "USD 1.5 million") is left out of the chart rather than drawn at a length that may be wrong."""
    want = re.sub(r"\D", "", f"{value:.2f}".rstrip("0").rstrip("."))
    return re.sub(r"\D", "", shown) == want


CAUTION = ("Each bar is an amount as a document states it, found by reading the documents' text; amounts are not added, "
           "converted or checked against each other. Open a bar to read its sentence.")


def build(principal: dict, matter: dict, document_ids: list[str]) -> tuple[dict, list[str]]:
    docs = scan.documents(matter["id"], principal["sub"], "signed")
    if document_ids:
        docs = [d for d in docs if d["document_id"] in set(document_ids)]
    found, skipped = [], 0
    for d in docs:
        doc = facts.document(d["document_id"], d["document_title"], d["version_id"], d["text"], d["spans"])
        seen = set()
        for f in doc.facts:
            m = _VALUE.match(f.value) if f.kind == "money" else None
            if not m or (f.value, f.sentence) in seen:
                continue
            if not _same_digits(f.shown, float(m.group(2))):
                skipped += 1
                continue
            seen.add((f.value, f.sentence))
            found.append((d, f, m.group(1), float(m.group(2))))
    units = Counter(u for _, _, u, _ in found)
    if not units:
        return {"summary": f"No amounts are stated in the {len(docs)} document{'s' if len(docs) != 1 else ''} you can read in {matter['number']}.",
                "unit": "—", "bars": [], "more": 0, "other_units": [], "caution": CAUTION}, []
    unit = units.most_common(1)[0][0]
    mine = [x for x in found if x[2] == unit]
    shown = mine[:MAX_BARS]
    bars, cache = [], {}
    with get_conn() as conn:
        for i, (d, f, _, value) in enumerate(shown):
            side = discrepancies._side(conn, matter["number"], scan._side(d, f, conn, cache, []))
            bars.append({"id": f"am{i + 1}", "shown": f.shown[:80], "value": value, "document_title": d["document_title"][:300],
                         "clause": side.get("clause"), "source": side})
    n_docs = len({d["document_id"] for d, *_ in mine})
    summary = (f"{len(mine)} amount{'s' if len(mine) != 1 else ''} in {unit} stated in {n_docs} document{'s' if n_docs != 1 else ''} "
               f"of {matter['number']}.")
    if skipped:
        summary += f" {skipped} amount{'s' if skipped != 1 else ''} written in another form {'are' if skipped != 1 else 'is'} not charted."
    return {"summary": summary[:400], "unit": unit, "bars": bars, "more": max(0, len(mine) - len(shown)),
            "other_units": [u for u, _ in units.most_common() if u != unit][:6], "caution": CAUTION}, sorted({d["document_id"] for d, *_ in shown})
