"""The parties and the documents that name them, as a small graph (docs/echomind-legal/03 `graph_excerpt`: "bounded
nodes/edges and source refs"; owner's request 2026-11-02).

There is no relationship store: the graph shows only what can be shown with a source. Companies are read by rule from
the documents the person may read (compare/facts.py, the conflict scan's reader); an edge means "party to" when the
document names the company as one of its parties (after "between", a defined role, the letterhead…), otherwise "named
in", and every edge opens the sentence it rests on. A party's role (client, adverse…) comes only from the matter's own
records. Nothing is inferred about how parties relate to each other.
"""
from __future__ import annotations

from ...core.db import get_conn
from ..compare import facts, scan
from ..compare.align import fold
from . import discrepancies

MAX_PARTIES, MAX_DOCUMENTS = 8, 8
ROLE = {"adverse": "Adverse", "related": "Related", "client_group": "Client group", "other": "Other"}
CAUTION = ("Companies found by reading each document's text, and where each is named. “Party to” means the document names it "
           "as a party; it says nothing else about how parties relate. Roles come from the matter's records.")


def _records(matter_id: str) -> dict[str, tuple[str, str]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT name, role FROM legal_matter_parties WHERE matter_id=? AND removed_at IS NULL", (matter_id,)).fetchall()
        client = conn.execute("SELECT c.name FROM matters m JOIN clients c ON c.id = m.client_id WHERE m.id=?", (matter_id,)).fetchone()
    out = {fold(n): (n, ROLE.get(r, "Other")) for n, r in rows}
    if client and client[0]:
        out.setdefault(fold(client[0]), (client[0], "Client"))
    return out


def build(principal: dict, matter: dict, document_ids: list[str]) -> tuple[dict, list[str]]:
    docs = scan.documents(matter["id"], principal["sub"], "signed")
    if document_ids:
        docs = [d for d in docs if d["document_id"] in set(document_ids)]
    records = _records(matter["id"])
    mentions: dict[str, dict] = {}                       # folded name -> {label, docs: {document_id: (doc, fact, party)}}
    for d in docs:
        doc = facts.document(d["document_id"], d["document_title"], d["version_id"], d["text"], d["spans"])
        named = set(facts.parties(doc))
        for f in doc.facts:
            if f.kind != "entity" or not (f.subject or f.shown):
                continue
            key = fold(f.subject or f.shown)
            if not key:
                continue
            entry = mentions.setdefault(key, {"label": f.shown, "docs": {}})
            prior = entry["docs"].get(d["document_id"])
            if prior is None or (f.shown in named and not prior[2]):
                entry["docs"][d["document_id"]] = (d, f, f.shown in named)
    ranked = sorted(mentions.items(), key=lambda kv: (kv[0] not in records, -len(kv[1]["docs"]), kv[1]["label"].lower()))
    parties = ranked[:MAX_PARTIES]
    doc_count: dict[str, int] = {}
    for _, e in parties:
        for did in e["docs"]:
            doc_count[did] = doc_count.get(did, 0) + 1
    kept_docs = sorted(doc_count, key=lambda did: -doc_count[did])[:MAX_DOCUMENTS]
    by_id = {d["document_id"]: d for d in docs}
    nodes, edges, cache = [], [], {}
    with get_conn() as conn:
        for i, (key, e) in enumerate(parties):
            pid = f"p{i + 1}"
            label, role = records.get(key, (e["label"], None))
            nodes.append({"id": pid, "label": label[:160], "kind": "party", **({"role": role} if role else {})})
            for did, (d, f, party) in e["docs"].items():
                if did not in kept_docs or len(edges) >= 40:
                    continue
                side = discrepancies._side(conn, matter["number"], scan._side(d, f, conn, cache, []))
                edges.append({"id": f"e{len(edges) + 1}", "source": pid, "target": f"d{kept_docs.index(did) + 1}",
                              "label": "party to" if party else "named in", "evidence": side})
    for j, did in enumerate(kept_docs):
        nodes.append({"id": f"d{j + 1}", "label": by_id[did]["document_title"][:160], "kind": "document"})
    if not parties:
        summary = f"No companies are named in the {len(docs)} document{'s' if len(docs) != 1 else ''} you can read in {matter['number']}."
    else:
        summary = (f"{len(parties)} part{'ies' if len(parties) != 1 else 'y'} named across {len(kept_docs)} document{'s' if len(kept_docs) != 1 else ''} "
                   f"of {matter['number']}.")
    return ({"summary": summary[:400], "nodes": nodes, "edges": edges, "more": max(0, len(ranked) - len(parties)), "caution": CAUTION},
            sorted(kept_docs))
