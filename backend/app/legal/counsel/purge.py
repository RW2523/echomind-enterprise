"""What the chat keeps about a document, removed when the document is purged (records/deletion.py, SEC-07).

Called inside the purge's own transaction, so either everything goes or nothing does:
- citations pointing into the document are deleted (they hold offsets and a hash, never words);
- potential inconsistencies quoting it go with their review history (a scan can find them again);
- answers keep their wording, but citation chips naming the document lose its title and are marked removed, and their notes
  (which name the documents a conflict or an amendment is about) are dropped;
- a briefing that named the document is replaced by "no longer available";
- evidence comparisons need nothing: they are resolved from their discrepancy, which is gone.
"""
from __future__ import annotations

import json

from . import citations, discrepancies

REMOVED = "This briefing referred to a document that has since been deleted."


def document(conn, matter_id: str, document_id: str) -> None:
    citations.purge_document(conn, matter_id, document_id)
    discrepancies.purge_document(conn, matter_id, document_id)
    rows = conn.execute("SELECT c.id, c.type, c.props_json FROM legal_counsel_components c WHERE c.matter_id=? AND EXISTS "
                        "(SELECT 1 FROM json_each(c.document_ids_json) j WHERE j.value=?)", (matter_id, document_id)).fetchall()
    for cid, type_, pj in rows:
        props = json.loads(pj)
        if type_ == "source_answer":
            props["citations"] = [({**c, "document_title": "", "removed": True, "clause": None, "page": None}
                                   if c.get("document_id") == document_id else c) for c in props.get("citations", [])]
            props["notes"] = []                    # notes name the documents they are about (conflicts, amendments)
            conn.execute("UPDATE legal_counsel_components SET props_json=? WHERE id=?", (json.dumps(props, ensure_ascii=False), cid))
        elif type_ == "priority_list":
            conn.execute("UPDATE legal_counsel_components SET type='unavailable', props_json=? WHERE id=?",
                         (json.dumps({"text": REMOVED}), cid))


def remaining(conn, matter_id: str, document_id: str) -> int:
    """Rows that would still disclose the document (0 after a purge): the purge's verification counts them."""
    n = conn.execute("SELECT COUNT(*) FROM legal_counsel_citations WHERE matter_id=? AND document_id=?", (matter_id, document_id)).fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM legal_discrepancies WHERE matter_id=? AND (document_a_id=? OR document_b_id=?)",
                      (matter_id, document_id, document_id)).fetchone()[0]
    for type_, pj in conn.execute("SELECT c.type, c.props_json FROM legal_counsel_components c WHERE c.matter_id=? AND EXISTS "
                                  "(SELECT 1 FROM json_each(c.document_ids_json) j WHERE j.value=?)", (matter_id, document_id)).fetchall():
        props = json.loads(pj)
        if type_ == "source_answer":
            n += sum(1 for c in props.get("citations", []) if c.get("document_id") == document_id and c.get("document_title"))
            n += 1 if props.get("notes") else 0
        elif type_ == "priority_list":
            n += 1
    return n
