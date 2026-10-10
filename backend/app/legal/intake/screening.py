"""Conflict screening (S4.1, D-054): names an intake gives against the firm's clients and matter parties.

Names are compared after folding case, punctuation, a leading "the" and legal forms (Ltd, GmbH, Inc…):
- equal names match;
- names with the same words in another order match ("Smith, John" and "John Smith");
- names of five or more letters match when they are at least 88% alike ("Alfa Holdings" and "Alpha Holdings").

Former clients and the parties of closed matters count; a party removed from a matter's register does not.
The result goes to the conflicts team. Front-desk staff and the enquirer learn only whether to refer.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from ...core.db import get_conn

THRESHOLD = 0.88
_LEGAL_FORM = {"ltd", "limited", "plc", "llp", "lp", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ab", "as",
               "asa", "oy", "gmbh", "ag", "sa", "sas", "srl", "spa", "bv", "nv", "pty", "pte", "kk", "se"}


def key(name: str) -> str:
    words = re.sub(r"[^\w\s]", " ", (name or "").casefold().replace("&", " and ")).split()
    if words and words[0] == "the":
        words = words[1:]
    while words and words[-1] in _LEGAL_FORM:
        words = words[:-1]
    return " ".join(words)


def score(a: str, b: str) -> float:
    """How alike two folded names are, 0 to 1."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if sorted(a.split()) == sorted(b.split()):
        return 0.99
    if min(len(a), len(b)) < 5:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 3)


def register(org_id: str) -> list[dict]:
    """Every name the firm screens against: its clients, and the parties on its matters' registers."""
    with get_conn() as conn:
        clients = conn.execute("SELECT id, name FROM clients WHERE org_id=?", (org_id,)).fetchall()
        parties = conn.execute(
            "SELECT p.name, p.role, m.id, m.number, c.id, c.name FROM legal_matter_parties p JOIN matters m ON m.id = p.matter_id "
            "JOIN clients c ON c.id = m.client_id WHERE m.org_id=? AND p.removed_at IS NULL", (org_id,)).fetchall()
    out = [{"name": n, "kind": "client", "client_id": cid, "client": n, "matter_id": None, "matter_number": None} for cid, n in clients]
    out += [{"name": n, "kind": role, "client_id": cid, "client": cname, "matter_id": mid, "matter_number": num}
            for n, role, mid, num, cid, cname in parties]
    return out


def screen(org_id: str, names: list[str]) -> list[dict]:
    """The register entries each name matches, best first."""
    entries = [(e, key(e["name"])) for e in register(org_id)]
    hits = []
    for searched in dict.fromkeys(n for n in names if n and n.strip()):
        k = key(searched)
        if not k:
            continue
        for e, ek in entries:
            s = score(k, ek)
            if s >= THRESHOLD:
                hits.append({"searched": searched, "matched": e["name"], "kind": e["kind"], "client_id": e["client_id"], "client": e["client"],
                             "matter_id": e["matter_id"], "matter_number": e["matter_number"], "score": s})
    return sorted(hits, key=lambda h: (-h["score"], h["searched"], h["matched"]))
