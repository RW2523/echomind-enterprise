"""SEC-02 — every matter route, swept as people the matter must stay closed to (external review, 2026-11-07).

test_law_route_table.py proves that every route refuses an anonymous caller. This file signs in as real people of the firm
who must still not reach a matter, and calls every matter route as each of them:
- an outsider: bob, who has never been on M-101;
- a walled lawyer: alice, the lead of M-101, once an ethical wall screens her from it (requested and approved by two
  managing partners through the API);
- a revoked member: carol, who worked on M-215 (her own conversation, cited answer, cards and time) before her membership
  was revoked;
- an outsider using his own matter: bob, through M-215 (which he leads), naming M-101's records; a record must be bound to
  the matter it is reached through.

The route list is read from the live app, so a new route is swept without anyone remembering to add it: every HTTP route
and WebSocket whose path names a matter, every route whose path names a per-matter record reached without one (a
conversation, card, turn, citation, deletion or time entry), and every route whose query or body names a matter. Any other
route with a path parameter must have that parameter in FIRM_LEVEL, with the reason it holds no matter content.

Each call uses real ids (one of each kind of record, seeded in both matters) and a body built from the route's own schema
(required fields filled with the first enum value, short strings and small numbers; id fields given the real ids), with
every feature switch on and the per-person model quota off, so that neither a 422, a "feature off" 404 nor a 429 can hide a
missing check. Every answer must be 401, 403 or 404 (the capture socket: 4401, 4403 or 4404), and no answer may contain the
matter's content markers: its title, and its marker word, which is in every document, title and record seeded in it.

Also here: a closed matter refuses every write the policy blocks on closed matters (the writes kept open are named, with
why); no list shows the matter to these people; targeted checks for potential inconsistencies, the briefing, Council and
panel results, cited passages, conversation cards and background work that finishes after access is gone; and what the
query-embedding cache holds. All names and documents are fictional (DEMO / SYNTHETIC).
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from typing import Any, Optional

import pytest
from fastapi import params as fparams
from fastapi.routing import APIRoute, APIWebSocketRoute
from pydantic import TypeAdapter
from starlette.websockets import WebSocketDisconnect

from conftest import as_user, csrf_headers, index_all, upload

WSS = "wss://testserver"
TAGS = {"M-101": "SWEEPCANARY101", "M-215": "SWEEPCANARY215"}
REFUSED = (401, 403, 404)
SOCKET_REFUSED = (4401, 4403, 4404)
CLOSED = "This matter is closed. Reopen it to make changes."


# ── Fictional matter content (DEMO / SYNTHETIC) ──────────────────────────────────
def agreement(tag):
    return (f"SERVICES AGREEMENT {tag} (fictional, DEMO / SYNTHETIC)\n\n"
            f"This Services Agreement is made between Quillfeather {tag} LLC and Marrow Peak Industries, Inc.\n\n"
            f"4.2 Delivery. Marrow Peak Industries, Inc. shall complete delivery by May 15, 2026.\n\n"
            f"9.1 Fees. Quillfeather {tag} LLC shall pay USD 120,000 within thirty days of delivery.\n")


def witness(tag, letter, day):
    return (f"INTERVIEW MEMORANDUM {tag}, Witness {letter} (fictional, DEMO / SYNTHETIC)\n\n"
            f"Prepared in the dispute between Quillfeather {tag} LLC and Marrow Peak Industries, Inc.\n\n"
            f"Witness {letter} remembers that the delivery under the Services Agreement took place on May {day}, 2026.")


VERDICT = ('Same subject: yes, both describe the delivery under the Services Agreement.\n'
           'Verdict: CONFLICT\n'
           'Explanation: Source 1 gives [1: "{a}"] while source 2 gives [2: "{b}"].')
QUOTE = "complete delivery by May 15, 2026"


def _source_with(user: str, quote: str) -> Optional[str]:
    return next((m.group(1) for m in re.finditer(r'<source n="(\d+)"[^>]*>(.*?)</source>', user, re.S) if quote in m.group(2)), None)


class Model:
    """Scripted stand-in for the model (no network): answers cite the source holding QUOTE, conflict checks answer
    VERDICT, panel perspectives cite the same clause, and anything else gets an empty JSON result."""

    async def chat_stream(self, messages, temperature, max_tokens):
        n = _source_with(messages[-1]["content"], QUOTE)
        text = f'Delivery is due by that date [{n}: "{QUOTE}"].' if n else "The documents in this matter do not answer that."
        for piece in re.findall(r"\S+\s*|\n+", text):
            yield piece

    async def chat(self, messages, temperature, max_tokens):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "contradict" in system:
            a, b = re.search(r'source 1 says "(.*?)" and source 2 says "(.*?)"', user).groups()
            return VERDICT.format(a=a, b=b)
        if re.search(r"(CHALLENGE|REBUTTAL): \w+", system):
            return "NO OBJECTION"
        if re.search(r"PERSPECTIVE: \w+", system):
            return f'Delivery is due by that date [{_source_with(user, QUOTE) or 1}: "{QUOTE}"].\n\nMISSING FACTS: none'
        return json.dumps({"issues": [], "missing_facts": [], "stop": False, "points": [], "items": []})

    chat_raw = chat


# ── Requests built from each route's own declarations ────────────────────────────
def _string(s: dict) -> str:
    pat, lo, hi, fmt = s.get("pattern"), s.get("minLength", 0), s.get("maxLength", 10 ** 6), s.get("format")
    if fmt == "date":
        return "2026-01-15"
    if fmt == "date-time":
        return "2026-01-15T10:00:00Z"
    cands = []
    if pat:                                              # an id pattern such as ^cmp_[0-9a-f]{32}$
        m = re.fullmatch(r"\^?([A-Za-z]+_)\[0-9a-f\]\{(\d+)\}\$?", pat)
        if m:
            cands.append(m.group(1) + "0" * int(m.group(2)))
    cands += ["sweep", "x", "abcdefgh", "sweep-value-0001", "2026-01-15", "a" * max(lo, 1), "A1", "1"]
    return next((c for c in cands if lo <= len(c) <= hi and (not pat or re.search(pat, c))), "x" * max(lo, 1))


def _sample(s: dict, defs: dict) -> Any:
    """The smallest value the JSON schema accepts: required properties only, first enum value, short strings, 1."""
    if "$ref" in s:
        return _sample(defs[s["$ref"].split("/")[-1]], defs)
    if "const" in s:
        return s["const"]
    if "enum" in s:
        return s["enum"][0]
    for k in ("anyOf", "oneOf"):
        if k in s:
            return _sample(([o for o in s[k] if o.get("type") != "null"] or s[k])[0], defs)
    if "allOf" in s:
        return _sample(s["allOf"][0], defs)
    t = s.get("type")
    if t == "object" or "properties" in s:
        req = set(s.get("required", []))
        return {k: _sample(v, defs) for k, v in (s.get("properties") or {}).items() if k in req}
    if t == "array":
        return [_sample(s.get("items") or {"type": "string"}, defs) for _ in range(s.get("minItems", 0))]
    if t == "string":
        return _string(s)
    if t == "integer":
        return max(int(s.get("minimum", 1)), 1)
    if t == "number":
        return float(max(s.get("minimum", 1), 1))
    if t == "boolean":
        return False
    return None if t == "null" else "sweep"


def _schema(ann) -> dict:
    return TypeAdapter(ann).json_schema(ref_template="#/$defs/{model}")


def sample(ann) -> Any:
    js = _schema(ann)
    return _sample(js, js.get("$defs", {}))


# Body fields that carry a record id under another name.
_ALIASES = {"left_version_id": "version_id", "right_version_id": "other_version_id", "object_id": "document_id",
            "documents": "document_id", "focus_discrepancy_id": "discrepancy_id", "after_turn_id": "counsel_turn_id"}
# Routes whose validators a schema cannot describe (the 422s the generated body still met), handled explicitly.
BODY = {
    ("POST", "/api/legal/matters/{matter_id}/panel"): {"question": "Can the client insist on delivery by the agreed date?",
                                                       "specialist": "commercial"},
    ("POST", "/api/legal/counsel/actions"): {"action_id": "review", "expected_version": 1, "idempotency_key": "sweep-key-0001",
                                             "parameters": {"resolution": "explained", "note": "Dispatch and arrival (fictional)."}},
}


def _id_for(field: str, ids: dict):
    key = _ALIASES.get(field, field)
    if key in ids:
        return [ids[key]] if field == "documents" else ids[key]
    if field.endswith("_ids") and field[:-1] in ids:
        return [ids[field[:-1]]]
    return None


def _key(path: str, name: str) -> str:
    """Path parameters whose record depends on where they appear."""
    for frag, n, key in (("/regulatory/", "update_id", "regulatory_update_id"), ("/meetings/", "artifact_id", "minutes_id"),
                         ("/drafts/", "artifact_id", "draft_id"), ("/redlines/", "artifact_id", "redline_id"),
                         ("/summaries/", "artifact_id", "summary_id"), ("/artifacts/", "artifact_id", "draft_id"),
                         ("/playbook-checks/", "check_id", "playbook_check_id"), ("/citation-checks/", "check_id", "citation_check_id"),
                         ("/definitions-checks/", "check_id", "definitions_check_id"), ("/research/", "turn_id", "research_turn_id"),
                         ("/counsel/turns/", "turn_id", "counsel_turn_id"), ("/labels/", "object_id", "document_id")):
        if n == name and frag in path:
            return key
    return name


def request_for(method: str, route: APIRoute, ids: dict) -> tuple[str, dict, set, set, Optional[str]]:
    """(url, request kwargs, record ids named in the path, in the query or body, problem) from the route's declarations."""
    in_path, in_body, values = set(), set(), {}
    for f in route.dependant.path_params:
        key = _key(route.path, f.name)
        if key in ids:
            values[f.name] = ids[key]
            in_path.add(key)
        else:
            values[f.name] = sample(f.field_info.annotation)
    url = re.sub(r"\{([^}]+)\}", lambda m: str(values[m.group(1)]), route.path)
    query = {}
    for f in route.dependant.query_params:
        if f.name in ids:
            query[f.alias] = ids[f.name]
            in_body.add(f.name)
        elif f.required:
            query[f.alias] = sample(f.field_info.annotation)
    kw: dict = {"params": query}
    problem = None
    bf = route.body_field
    if bf is not None and isinstance(bf.field_info, (fparams.Form, fparams.File)):
        data, files = {}, {}
        for f in route.dependant.body_params:
            if isinstance(f.field_info, fparams.File):
                files[f.alias] = ("sweep.txt", b"Sweep upload (fictional).", "text/plain")
            elif f.required:
                data[f.alias] = sample(f.field_info.annotation)
        kw.update(data=data, files=files)
    elif bf is not None:
        ann = bf.field_info.annotation
        base = dict(BODY.get((method, route.path)) or sample(ann) or {})
        named = {k: v for k in (_schema(ann).get("properties") or {}) if (v := _id_for(k, ids)) is not None}
        for body in ({**base, **named}, {**base, **{k: v for k, v in named.items() if k in base}}):
            try:
                TypeAdapter(ann).validate_python(body)
                problem = None
                break
            except Exception as e:  # noqa: BLE001
                problem = str(e).splitlines()[0]
        in_body |= {_ALIASES.get(k, k) for k in body if k in named}
        kw["json"] = body
    return url, kw, in_path - {"matter_id"}, in_body - {"matter_id", "matter_ids"}, problem


# ── The route table ──────────────────────────────────────────────────────────────
def legal_routes(app):
    for r in app.routes:
        if isinstance(r, APIRoute) and r.path.startswith("/api/legal/"):
            for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
                yield m, r


# Per-matter records that some routes reach without the matter in the path.
PER_MATTER = {"conversation_id", "component_id", "citation_id", "turn_id", "deletion_id", "entry_id"}
# Path parameters of firm-level records, with why they hold no matter content.
FIRM_LEVEL = {
    "client_id": "a client record (name, code); client walls are covered in test_law_walls.py",
    "username": "firm administration of people and their firm roles",
    "wall_id": "ethical walls, for the walls authority (managing partners)", "exc_id": "wall exceptions, as above",
    "gid": "break-glass grants, for firm administrators and managing partners",
    "hold_id": "legal holds, for records staff; a hold names whole matters, never their content",
    "policy_id": "retention policies (records staff)",
    "connector_id": "document connectors (firm administrators); a matter's own links are swept",
    "template_id": "the firm's template library", "playbook_id": "the firm's playbooks", "position_id": "playbook positions",
    "checklist_id": "the firm's due-diligence checklists",
    "version_id": "a template version (/templates/...); document versions sit under a matter and are swept",
    "iid": "an enquiry at the front desk, before any matter exists (converting it creates a new matter)",
    "rid": "a reply to such an enquiry",
}
_LAST = ("/delete", "/remove", "/end", "/withdraw", "/finalise", "/discard", "/reject")


def _named(route: APIRoute) -> set:
    names = {f.name for f in route.dependant.path_params} | {f.name for f in route.dependant.query_params}
    bf = route.body_field
    if bf is not None and not isinstance(bf.field_info, (fparams.Form, fparams.File)):
        names |= set(_schema(bf.field_info.annotation).get("properties") or {})
    return names


def swept_routes(app) -> list[tuple[str, APIRoute]]:
    """Every matter route: reads first, anything that removes or ends a record last."""
    out = []
    for method, r in legal_routes(app):
        path_names = {f.name for f in r.dependant.path_params}
        if "matter_id" in path_names or path_names & PER_MATTER or {"matter_id", "matter_ids"} & _named(r):
            out.append((method, r))
    rank = lambda m, p: 0 if m == "GET" else 2 if m == "DELETE" or p.endswith(_LAST) else 1  # noqa: E731
    return sorted(out, key=lambda x: rank(x[0], x[1].path))


def object_ids(db, matter_id: str, owner_id: str) -> dict:
    """Real ids of one record of each kind in the matter (the owner's own for chat records and time entries)."""
    one = lambda sql, *a: (lambda r: r[0] if r else None)(db.execute(sql, a).fetchone())  # noqa: E731
    ids = {"matter_id": matter_id, "matter_ids": [matter_id]}
    for key, sql in {
        "membership_id": "SELECT id FROM matter_memberships WHERE matter_id=? AND revoked_at IS NULL ORDER BY granted_at DESC",
        "document_id": "SELECT id FROM legal_documents WHERE matter_id=? AND deleted_at IS NULL ORDER BY created_at",
        "meeting_id": "SELECT id FROM legal_meetings WHERE matter_id=?",
        "participant_id": "SELECT id FROM legal_meeting_participants WHERE matter_id=?",
        "segment_id": "SELECT id FROM legal_transcript_segments WHERE matter_id=?",
        "alert_id": "SELECT id FROM legal_assist_alerts WHERE matter_id=?",
        "item_id": "SELECT id FROM legal_agenda_items WHERE matter_id=?",
        "export_id": "SELECT id FROM legal_exports WHERE matter_id=?",
        "comparison_id": "SELECT id FROM legal_comparisons WHERE matter_id=?",
        "obligation_id": "SELECT id FROM legal_obligations WHERE matter_id=?",
        "run_id": "SELECT id FROM legal_dd_runs WHERE matter_id=?",
        "issue_id": "SELECT id FROM legal_dd_issues WHERE matter_id=?",
        "party_id": "SELECT id FROM legal_matter_parties WHERE matter_id=?",
        "review_id": "SELECT id FROM legal_panel_reviews WHERE matter_id=?",
        "question_id": "SELECT id FROM legal_council_questions WHERE matter_id=?",
        "task_id": "SELECT id FROM legal_matter_tasks WHERE matter_id=?",
        "update_id": "SELECT id FROM legal_matter_updates WHERE matter_id=?",
        "link_id": "SELECT id FROM legal_connector_links WHERE matter_id=?",
        "deletion_id": "SELECT id FROM legal_deletions WHERE matter_id=?",
        "regulatory_update_id": "SELECT update_id FROM legal_regulatory_matches WHERE matter_id=?",
        "minutes_id": "SELECT id FROM legal_ai_artifacts WHERE matter_id=? AND kind='minutes'",
        "draft_id": "SELECT id FROM legal_ai_artifacts WHERE matter_id=? AND kind='legal_draft'",
        "redline_id": "SELECT id FROM legal_ai_artifacts WHERE matter_id=? AND kind='redline'",
        "summary_id": "SELECT id FROM legal_ai_artifacts WHERE matter_id=? AND kind='summary'",
        "playbook_check_id": "SELECT id FROM legal_playbook_checks WHERE matter_id=?",
        "citation_check_id": "SELECT id FROM legal_citation_checks WHERE matter_id=?",
        "definitions_check_id": "SELECT id FROM legal_definition_checks WHERE matter_id=?",
        "research_turn_id": "SELECT id FROM legal_research_turns WHERE matter_id=?",
        "discrepancy_id": "SELECT id FROM legal_discrepancies WHERE matter_id=?",
    }.items():
        ids[key] = one(sql, matter_id)
    ids["version_id"] = one("SELECT current_version_id FROM legal_documents WHERE id=?", ids["document_id"])
    ids["other_version_id"] = one("SELECT current_version_id FROM legal_documents WHERE matter_id=? AND id<>? AND deleted_at IS NULL",
                                  matter_id, ids["document_id"])
    ids["entry_id"] = one("SELECT id FROM legal_time_entries WHERE matter_id=? AND user_id=?", matter_id, owner_id)
    for key, sql in (("conversation_id", "SELECT id FROM legal_conversations WHERE matter_id=? AND owner_id=? ORDER BY created_at"),
                     ("component_id", "SELECT id FROM legal_counsel_components WHERE matter_id=? AND owner_id=? "
                                      "ORDER BY type <> 'evidence_comparison', created_at"),
                     ("citation_id", "SELECT id FROM legal_counsel_citations WHERE matter_id=? AND owner_id=?"),
                     ("counsel_turn_id", "SELECT id FROM legal_conversation_turns WHERE matter_id=? AND owner_id=? "
                                         "AND status='completed' ORDER BY created_at")):
        ids[key] = one(sql, matter_id, owner_id)
    missing = [k for k, v in ids.items() if v is None]
    assert not missing, f"no seeded record for {missing}"
    return ids


# ── Seeding: one record of each kind in each matter ──────────────────────────────
def artifact_content(title, document_id):
    """One body each artifact reader can open (draft, minutes, summary, redline)."""
    return {"title": title, "text": title, "gaps": [], "fields": [], "blocks": [], "summary": {}, "changes": [], "points": [], "items": [],
            "template": {"id": "tpl_sweep", "version_id": "tplv_sweep", "kind": "engagement_letter"},
            "document": {"id": document_id, "filename": "agreement.txt", "title": title}}


def insert(db, table, **values):
    """A row with these values; the other NOT NULL columns without a default get a placeholder of their type."""
    for _, col, typ, notnull, default, pk in db.execute(f"PRAGMA table_info({table})").fetchall():
        if col in values or not notnull or default is not None or pk:
            continue
        values[col] = ("{}" if col.endswith("_json") else 0 if "INT" in (typ or "").upper() else
                       "2026-11-07T00:00:00.000Z" if col.endswith("_at") else "sweep")
    db.execute(f"INSERT INTO {table} ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})", tuple(values.values()))


def ask(c, conv, text, **extra):
    r = c.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(c), json={"text": text, **extra})
    assert r.status_code == 200, r.text
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events[-1]["event"] == "turn.completed", (text, events[-1])
    return next(e["data"] for e in events if e["event"] == "ui.final")


def new_conversation(c, *mids):
    r = c.post("/api/legal/counsel/conversations", headers=csrf_headers(c), json={"matter_ids": list(mids)})
    assert r.status_code == 201, r.text
    return r.json()["conversation"]["id"]


def counsel_work(c, mid, tag) -> dict:
    """A conversation with a cited answer, a potential-inconsistency card and a briefing, owned by `c`."""
    conv = new_conversation(c, mid)
    assert c.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(c),
                   json={"title": f"Delivery questions {tag}"}).status_code == 200
    cards = {"conversation": conv}
    for text in (f"When must Marrow Peak complete delivery under the {tag} agreement?", "Find delivery date conflicts",
                 "What matters today?"):
        for comp in ask(c, conv, text)["components"]:
            cards.setdefault(comp["type"], comp)
    assert {"source_answer", "evidence_comparison", "priority_list"} <= set(cards), sorted(cards)
    return cards


def seed(w, number, lead) -> dict:
    """Records of each kind in one matter, made by its lead, each carrying the matter's marker word. Records the API makes
    only through the model, a microphone, a firm library or a public feed are written to the database directly."""
    from app.legal.council import discovery
    tag, mid, lead_id = TAGS[number], w["matters"][number]["id"], w["users"][lead]["id"]
    c = as_user(w, lead)
    h, base = csrf_headers(c), f"/api/legal/matters/{mid}"

    def must(r):
        assert r.status_code < 300, r.text
        return r

    docs = []
    for name, text, title in (("agreement.txt", agreement(tag), f"Services Agreement {tag}"),
                              ("witness-a.txt", witness(tag, "A", 12), f"Witness A memo {tag}"),
                              ("witness-b.txt", witness(tag, "B", 15), f"Witness B memo {tag}"),
                              ("old-draft.txt", f"Superseded draft {tag} (fictional).", f"Old draft {tag}")):
        docs.append(must(upload(c, mid, name, text, status="executed", title=title)).json()["document"]["id"])
    index_all()
    db = sqlite3.connect(w["db"])
    org = db.execute("SELECT org_id FROM matters WHERE id=?", (mid,)).fetchone()[0]
    vers = [db.execute("SELECT current_version_id FROM legal_documents WHERE id=?", (d,)).fetchone()[0] for d in docs]
    meeting = must(c.post(f"{base}/meetings", headers=h, json={"title": f"Client call {tag}", "kind": "client_call"})).json()["meeting"]["id"]
    must(c.post(f"{base}/meetings/{meeting}/participants", headers=h, json={"name": f"Pat Ellery {tag}", "role": "host"}))
    must(c.post(f"{base}/meetings/{meeting}/agenda", headers=h, json={"title": f"Delivery dates {tag}"}))
    must(c.post(f"{base}/exports", headers=h, json={"object_type": "document", "object_id": docs[0], "format": "pdf", "purpose": f"Client {tag}"}))
    must(c.post(f"{base}/obligations", headers=h, json={"document_id": docs[0], "start": 0, "end": 30, "action": f"deliver {tag}"}))
    must(c.post(f"{base}/parties", headers=h, json={"name": f"Marrow Peak {tag}", "role": "adverse"}))
    must(c.post(f"{base}/updates", headers=h, json={"body": f"Called the client about delivery {tag}"}))
    must(c.post(f"{base}/tasks", headers=h, json={"title": f"Reply to Marrow Peak {tag}"}))
    must(c.post(f"{base}/compare", headers=h, json={"left_version_id": vers[1], "right_version_id": vers[2]}))
    must(c.post(f"{base}/documents/{docs[0]}/citation-check", headers=h, json={}))
    must(c.post(f"{base}/documents/{docs[0]}/definitions-check", headers=h, json={}))
    must(c.post(f"{base}/research/query", headers=h, json={"question": f"When must delivery be complete under the {tag} agreement?"}))
    must(c.post(f"{base}/panel", headers=h, json={"question": f"Can Quillfeather {tag} insist on delivery by May 15, 2026?",
                                                  "specialist": "commercial"}))
    must(c.post(f"{base}/documents/{docs[3]}/delete", headers=h, json={"reason": f"Superseded {tag}"}))
    index_all()
    work = counsel_work(c, mid, tag)

    cap = "cap_" + number
    insert(db, "legal_captures", id=cap, meeting_id=meeting, matter_id=mid, started_by=lead_id, consent_snapshot="[]", transcription="available")
    insert(db, "legal_transcript_segments", id="seg_" + number, capture_id=cap, meeting_id=meeting, matter_id=mid, seq=1,
           text=f"We expect delivery by the fifteenth {tag}.", created_at="2026-11-07T10:01:00.000Z")
    insert(db, "legal_assist_alerts", id="alr_" + number, meeting_id=meeting, matter_id=mid, capture_id=cap, segment_id="seg_" + number,
           pack="legal_client_call", rule="date", verdict="differs", severity="critical", method="rule",
           title=f"Date said differs {tag}", said_json=json.dumps({"text": tag}), source_json="{}", status="open")
    for kind, subject_type, subject in (("legal_draft", "draft", "drf_" + number), ("minutes", "meeting", meeting),
                                        ("summary", "document", docs[0]), ("redline", "redline", "rdl_" + number)):
        insert(db, "legal_ai_artifacts", id=f"art_{kind}_{number}", matter_id=mid, kind=kind, subject_type=subject_type, subject_id=subject,
               version=1, status="draft", content_json=json.dumps(artifact_content(f"{kind} {tag}", docs[0])), content_sha256="0" * 64,
               inputs_json="{}", created_by=lead_id, origin="model", document_id=docs[0])
    insert(db, "legal_playbooks", id="pb_" + number, org_id=org, name=f"Playbook {number}", created_by=lead_id)
    insert(db, "legal_playbook_checks", id="pbc_" + number, matter_id=mid, document_id=docs[0], version_id=vers[0],
           playbook_id="pb_" + number, results_json="[]", inputs_json="{}", created_by=lead_id)
    insert(db, "legal_dd_checklists", id="ddc_" + number, org_id=org, family_id="ddf_" + number, version=1, name="Checklist",
           content_json=json.dumps({"items": []}), content_sha256="x", status="approved", created_by=lead_id)
    insert(db, "legal_dd_runs", id="ddr_" + number, matter_id=mid, checklist_id="ddc_" + number, status="done", created_by=lead_id)
    insert(db, "legal_dd_issues", id="ddi_" + number, run_id="ddr_" + number, matter_id=mid, item_id="i1", document_id=docs[0],
           version_id=vers[0], kind="missing", severity="high", title=f"Missing consent {tag}", detail=f"Detail {tag}",
           evidence_json="[]", status="preliminary", version=1)
    insert(db, "legal_regulatory_updates", id="reg_" + number, org_id=org, source="federalregister.gov", external_id="ext_" + number,
           jurisdiction="US", kind="rule", title=f"Delivery rule update {number}", summary="A public update (fictional).",
           published="2026-11-01", subjects_json="[]")
    insert(db, "legal_regulatory_matches", update_id="reg_" + number, matter_id=mid, terms_json=json.dumps([tag]),
           matched_at="2026-11-07T00:00:00.000Z", status="new", version=1)
    insert(db, "legal_connectors", id="con_" + number, org_id=org, kind="local_folder", name=f"Share {number}", config_sealed="x",
           status="active", created_by=lead_id)
    insert(db, "legal_connector_links", id="lnk_" + number, connector_id="con_" + number, matter_id=mid, remote_path=f"/{tag}",
           created_by=lead_id)
    db.commit()
    db.close()
    time_entry(w, number, lead)
    discovery.run_for_matter(org, mid)
    return {"documents": docs, "versions": vers, "meeting": meeting, "work": work}


def time_entry(w, number, who):
    mid, uid = w["matters"][number]["id"], w["users"][who]["id"]
    with sqlite3.connect(w["db"]) as db:
        org = db.execute("SELECT org_id FROM matters WHERE id=?", (mid,)).fetchone()[0]
        insert(db, "legal_time_entries", id=f"te_{number}_{who}", org_id=org, user_id=uid, matter_id=mid, work_date="2026-11-06",
               source="suggested", minutes_suggested=30, narrative_suggested=f"Reviewed {TAGS[number]}", minutes=30,
               narrative=f"Reviewed {TAGS[number]}", status="draft", version=1)


SWITCHES = ("LEGAL_PANEL", "LEGAL_SUMMARIES", "LEGAL_TRANSLATION", "LEGAL_TIME_ENTRIES", "LEGAL_CONVERSATION", "LEGAL_REGULATORY",
            "LEGAL_INTAKE", "LEGAL_OBLIGATIONS_ICS", "LEGAL_COUNCIL_DISCOVERY")


@pytest.fixture
def sw(docs, fresh_db, monkeypatch):
    """Both matters seeded; carol worked on M-215 (her own conversation, citation, cards and time) until she was revoked."""
    from app.core.config import settings
    from app.legal.matters import repo
    from app.legal.research import llm
    for name in SWITCHES:                     # every feature on: "this feature is off" must not stand in for a refusal
        monkeypatch.setattr(settings, name, True)
    monkeypatch.setattr(settings, "LEGAL_MODEL_REQUESTS_PER_MIN", 0)       # nor may "too many requests"
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    monkeypatch.setattr(llm, "_override", Model())
    docs["db"] = fresh_db
    docs["seeded"] = {"M-101": seed(docs, "M-101", "alice"), "M-215": seed(docs, "M-215", "bob")}
    m215, carol, bob = docs["matters"]["M-215"]["id"], docs["users"]["carol"]["id"], docs["users"]["bob"]["id"]
    back = repo.add_membership(m215, carol, "paralegal", granted_by=bob)
    docs["seeded"]["M-215"]["carol"] = counsel_work(as_user(docs, "carol"), m215, TAGS["M-215"])
    time_entry(docs, "M-215", "carol")
    repo.revoke_membership(m215, back["id"], by=bob, reason="left the team")
    return docs


def wall_alice(w, number="M-101"):
    """An ethical wall screening alice from the matter, requested by one managing partner and approved by another."""
    mp, mp2 = as_user(w, "mp"), as_user(w, "mp2")
    r = mp.post("/api/legal/walls", headers=csrf_headers(mp), json={
        "name": "Conflict screen", "rationale": "Alice advised the counterparty before (fictional).", "screened": ["alice"],
        "protected_matters": [w["matters"][number]["id"]], "protected_clients": []})
    assert r.status_code == 200, r.text
    ok = mp2.post(f"/api/legal/walls/{r.json()['wall']['id']}/approve", headers=csrf_headers(mp2))
    assert ok.status_code == 200 and ok.json()["wall"]["status"] == "active", ok.text


def leaks(w, number: str, text: str) -> bool:
    low = text.lower()
    return TAGS[number].lower() in low or w["matters"][number]["title"].lower() in low


def ids_of(w, number, owner):
    with sqlite3.connect(w["db"]) as db:
        return object_ids(db, w["matters"][number]["id"], w["users"][owner]["id"])


def _org(w):
    with sqlite3.connect(w["db"]) as db:
        return db.execute("SELECT org_id FROM matters WHERE id=?", (w["matters"]["M-101"]["id"],)).fetchone()[0]


# ── The route table is fully classified ──────────────────────────────────────────
def test_every_route_with_a_path_parameter_is_swept_or_named_firm_level(law_app):
    swept = {(m, r.path) for m, r in swept_routes(law_app)}
    unclassified = [f"{m} {r.path}" for m, r in legal_routes(law_app)
                    if (names := {f.name for f in r.dependant.path_params}) and (m, r.path) not in swept and not names <= set(FIRM_LEVEL)]
    assert not unclassified, "sweep these routes, or add their parameter to FIRM_LEVEL with why:\n" + "\n".join(unclassified)
    assert len(swept) > 150, len(swept)
    sockets = [r.path for r in law_app.routes if isinstance(r, APIWebSocketRoute) and "{matter_id}" in r.path]
    assert sockets == ["/api/legal/matters/{matter_id}/meetings/{meeting_id}/capture"]


def test_every_generated_body_passes_its_route_s_own_validation(sw):
    """A 422 answers before the authorization check, so it would hide a missing one."""
    ids = ids_of(sw, "M-101", "alice")
    problems = [f"{m} {r.path}: {p}" for m, r in swept_routes(sw["app"]) if (p := request_for(m, r, ids)[4])]
    assert not problems, "\n".join(problems)


# Reads the lead cannot make with these records, and why (placeholders written straight to the database).
LEAD_CANNOT_READ = {"/api/legal/matters/{matter_id}/exports/{export_id}/file": "an export downloads once it is approved",
                    "/api/legal/matters/{matter_id}/drafts/{artifact_id}/preview.{fmt}": "the placeholder draft has no firm template",
                    "/api/legal/matters/{matter_id}/redlines/{artifact_id}/word": "the placeholder redline has no Word original"}


def test_the_sweep_uses_real_records_the_lead_can_open(sw):
    """The same GET requests made by the matter's lead succeed: the ids are real and every feature is on, so the
    refusals below come from the authorization checks, not from a missing record or a switch."""
    ids = ids_of(sw, "M-101", "alice")
    alice = as_user(sw, "alice")
    not_ok = {}
    for method, r in swept_routes(sw["app"]):
        if method != "GET":
            continue
        url, kw, _, _, _ = request_for(method, r, ids)
        try:
            status = alice.get(url, **kw).status_code
        except Exception as e:  # noqa: BLE001  (the placeholder records above)
            status = type(e).__name__
        if status != 200:
            not_ok[r.path] = status
    assert set(not_ok) <= set(LEAD_CANNOT_READ), not_ok


# ── The sweep ────────────────────────────────────────────────────────────────────
CASES = {   # who, the matter in the path, the matter whose records are named, whose own chat records are named
    "outsider": ("bob", "M-101", "M-101", "alice"),
    "walled": ("alice", "M-101", "M-101", "alice"),
    "revoked": ("carol", "M-215", "M-215", "carol"),
    "through_own_matter": ("bob", "M-215", "M-101", "alice"),
}
# Through his own matter, bob may name another matter's document in these bodies: each keeps only the documents of the
# matter in the path, so the answer is about his matter alone (and the content check below still applies).
FILTERS_FOREIGN_IDS = {"POST /api/legal/matters/{matter_id}/research/query", "POST /api/legal/matters/{matter_id}/panel"}


def _setup(w, case):
    if case == "walled":
        wall_alice(w)
    who, path_matter, obj_matter, owner = CASES[case]
    ids = ids_of(w, obj_matter, owner)
    ids.update(matter_id=w["matters"][path_matter]["id"], matter_ids=[w["matters"][path_matter]["id"]])
    return as_user(w, who), ids, obj_matter


@pytest.mark.parametrize("case", list(CASES))
def test_no_matter_route_answers_them(sw, case):
    c, ids, number = _setup(sw, case)
    checked, wrong = 0, []
    for method, r in swept_routes(sw["app"]):
        url, kw, in_path, in_body, _ = request_for(method, r, ids)
        if case == "through_own_matter" and not (in_path or in_body):
            continue                                # bob's own matter, naming none of M-101's records: his to use
        resp = c.request(method, url, headers=csrf_headers(c), **kw)
        checked += 1
        key = f"{method} {r.path}"
        if case != "through_own_matter" or in_path or r.path.startswith("/api/legal/counsel/"):
            if resp.status_code not in REFUSED:
                wrong.append(f"{key} -> {resp.status_code} {resp.text[:120]}")
        elif resp.status_code < 400 and key not in FILTERS_FOREIGN_IDS:
            wrong.append(f"{key} accepted another matter's record -> {resp.status_code} {resp.text[:120]}")
        if leaks(sw, number, resp.text):
            wrong.append(f"{key} -> {resp.status_code} shows {number} content: {resp.text[:160]}")
    assert checked > (40 if case == "through_own_matter" else 150), checked
    assert not wrong, f"{case} reached {number}:\n" + "\n".join(wrong)


@pytest.mark.parametrize("case", list(CASES))
def test_the_capture_socket_refuses_them(sw, case):
    c, ids, _ = _setup(sw, case)
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect(f"{WSS}/api/legal/matters/{ids['matter_id']}/meetings/{ids['meeting_id']}/capture",
                                 headers={"Origin": "https://testserver"}) as ws:
            ws.receive_text()
    assert exc.value.code in SOCKET_REFUSED, exc.value.code


@pytest.mark.parametrize("case", ["outsider", "walled", "revoked"])
def test_no_list_shows_them_the_matter(sw, case):
    """Every GET without a path parameter (matters, conversations, reviews, activity, time, pipeline, regulatory updates,
    deletions...), read as each person: whatever it answers, nothing of the matter is in it."""
    c, _, number = _setup(sw, case)
    shown, read = [], 0
    for method, r in legal_routes(sw["app"]):
        if method != "GET" or r.dependant.path_params or any(f.required for f in r.dependant.query_params):
            continue
        resp = c.get(r.path)
        read += 1
        if leaks(sw, number, resp.text):
            shown.append(f"GET {r.path} -> {resp.status_code}: {resp.text[:200]}")
    assert read >= 25, read
    assert not shown, "\n".join(shown)
    member = as_user(sw, "carol" if number == "M-101" else "bob")     # someone who may see it does: the markers work
    assert leaks(sw, number, member.get("/api/legal/matters").text)


# ── A closed matter ──────────────────────────────────────────────────────────────
# Writes the policy keeps open on a closed matter (policy.BLOCKED_WHEN_CLOSED: "reading, membership, export and records
# continue"), each with why. Every other write must answer 409 "This matter is closed".
OPEN_WHEN_CLOSED = {
    "PATCH /api/legal/matters/{matter_id}": "the matter itself: reopening it (only its status may change)",
    "POST /api/legal/matters/{matter_id}/memberships": "membership continues",
    "PATCH /api/legal/matters/{matter_id}/memberships/{membership_id}": "membership continues",
    "DELETE /api/legal/matters/{matter_id}/memberships/{membership_id}": "membership continues",
    "POST /api/legal/matters/{matter_id}/documents/search": "reading (research.query is not blocked)",
    "POST /api/legal/matters/{matter_id}/research/query": "reading",
    "POST /api/legal/matters/{matter_id}/compare": "reading: comparing two versions",
    "POST /api/legal/matters/{matter_id}/compare/{comparison_id}/explain": "reading",
    "POST /api/legal/matters/{matter_id}/compare/scan": "reading; what it finds cannot be resolved while the matter is closed",
    "POST /api/legal/matters/{matter_id}/documents/{document_id}/citation-check": "reading: a rules-only check (research.query, D-048)",
    "POST /api/legal/matters/{matter_id}/documents/{document_id}/definitions-check": "reading: a rules-only check (research.query, D-049)",
    "POST /api/legal/matters/{matter_id}/documents/{document_id}/translate": "reading; nothing is stored (D-070)",
    "POST /api/legal/matters/{matter_id}/summaries/{artifact_id}/translate": "reading; nothing is stored (D-070)",
    "POST /api/legal/counsel/conversations": "reading: a conversation about the matter",
    "PATCH /api/legal/counsel/conversations/{conversation_id}": "the person's own conversation title",
    "DELETE /api/legal/counsel/conversations/{conversation_id}": "off the person's own list (the record stays)",
    "POST /api/legal/counsel/conversations/{conversation_id}/turns": "reading: asking",
    "PUT /api/legal/counsel/turns/{turn_id}/feedback": "the person's own rating of an answer",
    "POST /api/legal/matters/{matter_id}/conversation/transcribe": "speech to text for a question; stores nothing",
    "POST /api/legal/matters/{matter_id}/conversation/speak": "an answer read aloud; stores nothing",
    "POST /api/legal/matters/{matter_id}/meetings/{meeting_id}/participants/{participant_id}/consent/withdraw":
        "withdrawing consent to recording always works",
    "POST /api/legal/matters/{matter_id}/documents/{document_id}/delete": "records: deletion requests (S1.9)",
    "POST /api/legal/matters/{matter_id}/meetings/{meeting_id}/transcript/delete": "records: deletion requests",
    "POST /api/legal/deletions/{deletion_id}/restore": "records",
    "POST /api/legal/matters/{matter_id}/exports": "export continues (S1.10)",
    "PUT /api/legal/matters/{matter_id}/exports/{export_id}/plan": "export continues",
    "POST /api/legal/matters/{matter_id}/exports/{export_id}/submit": "export continues",
    "POST /api/legal/matters/{matter_id}/exports/{export_id}/approve": "export continues (a second person approves)",
    "POST /api/legal/matters/{matter_id}/exports/{export_id}/reject": "export continues",
    "POST /api/legal/matters/{matter_id}/parties": "the party register, which conflict searches read, is kept on closed matters (D-054)",
    "POST /api/legal/matters/{matter_id}/parties/{party_id}/remove": "the party register (D-054)",
    "POST /api/legal/matters/{matter_id}/regulatory/{update_id}": "regulatory.decide is not blocked on closed matters (D-068)",
    "DELETE /api/legal/matters/{matter_id}/connector-links/{link_id}": "stops syncing a folder into the matter",
    "PATCH /api/legal/time/{entry_id}": "the person's own time records",
    "POST /api/legal/time/{entry_id}/approve": "the person's own time records",
    "POST /api/legal/time/{entry_id}/discard": "the person's own time records",
    "POST /api/legal/time/{entry_id}/reopen": "the person's own time records",
}
# Firm-level decisions that also name a matter: refused to a lead without the firm role whatever the matter's state.
FIRM_DECISIONS = {"POST /api/legal/deletions/{deletion_id}/approve", "POST /api/legal/deletions/{deletion_id}/reject",
                  "POST /api/legal/holds", "POST /api/legal/walls/{wall_id}/exceptions"}


def _set_status(c, mid, status):
    v = c.get(f"/api/legal/matters/{mid}").json()["matter"]["version"]
    r = c.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(c), json={"status": status, "version": v})
    assert r.status_code == 200, r.text


def test_a_closed_matter_refuses_every_write_the_policy_blocks(sw):
    """bob, the lead of M-215 (every matter right), closes it; each write route is then called with his real records."""
    bob, mid = as_user(sw, "bob"), sw["matters"]["M-215"]["id"]
    ids = {**ids_of(sw, "M-215", "bob"), "connector_id": "con_M-215"}     # connectors are the firm's; this one is active
    _set_status(bob, mid, "closed")
    wrong, kept_open = [], []
    for method, r in swept_routes(sw["app"]):
        key = f"{method} {r.path}"
        if method == "GET" or key in FIRM_DECISIONS:
            continue
        url, kw, _, _, _ = request_for(method, r, ids)
        resp = bob.request(method, url, headers=csrf_headers(bob), **kw)
        if key in OPEN_WHEN_CLOSED:
            kept_open.append(key)
            assert resp.status_code != 500, (key, resp.text)
        elif resp.status_code != 409 or resp.json().get("detail") != CLOSED:
            wrong.append(f"{key} -> {resp.status_code} {resp.text[:140]}")
    assert not wrong, "writes a closed matter must refuse:\n" + "\n".join(wrong)
    assert sorted(kept_open) == sorted(OPEN_WHEN_CLOSED)


def test_a_folder_cannot_be_linked_to_a_closed_matter(sw):
    """Found by the sweep: linking a folder imports its documents into the matter, yet it needed only matter.update,
    which a closed matter keeps (to reopen it). It now also needs the right to import, which a closed matter refuses."""
    bob, mid = as_user(sw, "bob"), sw["matters"]["M-215"]["id"]
    body = {"connector_id": "con_M-215", "remote_path": "Clients/Lease"}
    _set_status(bob, mid, "closed")
    r = bob.post(f"/api/legal/matters/{mid}/connector-links", headers=csrf_headers(bob), json=body)
    assert r.status_code == 409 and r.json()["detail"] == CLOSED
    _set_status(bob, mid, "open")
    assert bob.post(f"/api/legal/matters/{mid}/connector-links", headers=csrf_headers(bob), json=body).status_code == 200
    m101 = sw["matters"]["M-101"]["id"]
    carol = as_user(sw, "carol")                                    # read-only on M-101: neither right
    assert carol.post(f"/api/legal/matters/{m101}/connector-links", headers=csrf_headers(carol), json=body).status_code == 403


def test_a_document_of_another_matter_has_no_redlines_here(sw):
    """Found by the sweep: the redline list of a document answered 200 (empty) for a document of another matter, and for
    one the person may not open; it now answers 404, like the document's other lists."""
    bob = as_user(sw, "bob")
    m215, foreign = sw["matters"]["M-215"]["id"], sw["seeded"]["M-101"]["documents"][0]
    assert bob.get(f"/api/legal/matters/{m215}/documents/{foreign}/redlines").status_code == 404
    own = sw["seeded"]["M-215"]["documents"][0]
    assert bob.get(f"/api/legal/matters/{m215}/documents/{own}/redlines").status_code == 200


def test_a_person_s_own_deletion_request_leaves_their_queue_with_the_matter(sw):
    """Found by the sweep: the reviews queue kept listing a person's own deletion requests (the document's title and the
    matter's) after a wall or a revoked membership; they are listed only while the person can open the matter."""
    alice = as_user(sw, "alice")
    mine = lambda: [i for i in alice.get("/api/legal/reviews").json()["mine"] if i["kind"] == "deletion"]  # noqa: E731
    assert any(TAGS["M-101"] in i["title"] for i in mine())
    wall_alice(sw)
    assert mine() == []
    mp = as_user(sw, "mp")                                          # records reviewers still see it, to decide it (D-027)
    assert any(TAGS["M-101"] in i["title"] for i in mp.get("/api/legal/reviews").json()["waiting"])


# ── Targeted: what the chat shows ────────────────────────────────────────────────
def _card(w, number, type_, owner=None):
    return w["seeded"][number]["carol" if owner == "carol" else "work"][type_]


def test_potential_inconsistencies_stay_inside_their_matter(sw):
    card = _card(sw, "M-101", "evidence_comparison")
    did, mid, m215 = card["props"]["discrepancy_id"], sw["matters"]["M-101"]["id"], sw["matters"]["M-215"]["id"]
    carol = as_user(sw, "carol")                                    # a member still sees it: the record is there
    assert did in [d["id"] for d in carol.get(f"/api/legal/counsel/discrepancies?matter_id={mid}").json()["discrepancies"]]
    bob = as_user(sw, "bob")
    assert bob.get(f"/api/legal/counsel/discrepancies?matter_id={mid}&state=all").status_code == 404
    env = ask(bob, new_conversation(bob, m215), "Review", focus_discrepancy_id=did)    # named from his own matter
    assert not leaks(sw, "M-101", json.dumps(env)) and did not in json.dumps(env)
    action = {**BODY[("POST", "/api/legal/counsel/actions")], "component_id": card["id"], "expected_version": card["version"]}
    for matter in (mid, m215):
        assert bob.post("/api/legal/counsel/actions", headers=csrf_headers(bob), json={**action, "matter_id": matter}).status_code == 404
    wall_alice(sw)
    alice = as_user(sw, "alice")
    assert alice.get(f"/api/legal/counsel/discrepancies?matter_id={mid}").status_code == 404
    assert alice.get(f"/api/legal/counsel/components/{card['id']}").status_code == 404
    r = alice.post("/api/legal/counsel/actions", headers=csrf_headers(alice), json={**action, "matter_id": mid, "idempotency_key": "sweep-0002"})
    assert r.status_code == 404
    with sqlite3.connect(sw["db"]) as db:                           # nobody resolved it
        assert db.execute("SELECT status FROM legal_discrepancies WHERE id=?", (did,)).fetchone()[0] == "potential"


def test_the_briefing_never_names_a_matter_the_person_may_not_open(sw):
    mid, m114 = sw["matters"]["M-101"]["id"], sw["matters"]["M-114"]["id"]
    alice = as_user(sw, "alice")
    both = new_conversation(alice, m114, mid)                       # about two matters, started before the wall
    own = sw["seeded"]["M-101"]["work"]["conversation"]
    wall_alice(sw)
    env = ask(alice, new_conversation(alice, m114), "What matters today?")
    assert not leaks(sw, "M-101", json.dumps(env))
    assert env["components"][0]["props"]["summary"].startswith("Nothing new in the M-114 records")
    for conv in (own, both):                                         # the conversations that include M-101 are gone for her
        assert alice.get(f"/api/legal/counsel/conversations/{conv}").status_code == 404
        r = alice.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(alice), json={"text": "What matters today?"})
        assert r.status_code == 404 and not leaks(sw, "M-101", r.text)
    assert alice.post("/api/legal/counsel/conversations", headers=csrf_headers(alice), json={"matter_ids": [m114, mid]}).status_code == 404
    bob = as_user(sw, "bob")                                         # another matter's documents named in a briefing
    conv = new_conversation(bob, sw["matters"]["M-215"]["id"])
    r = bob.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(bob),
                 json={"text": "What matters today?", "document_ids": sw["seeded"]["M-101"]["documents"][:3]})
    assert r.status_code in (200, 404) and not leaks(sw, "M-101", r.text)


def test_council_and_panel_results_reach_only_the_matter_s_people(sw):
    mid, m215 = sw["matters"]["M-101"]["id"], sw["matters"]["M-215"]["id"]
    ids = ids_of(sw, "M-101", "alice")
    carol = as_user(sw, "carol")                                     # a member reads the panel and the Council's questions
    listing = carol.get(f"/api/legal/matters/{mid}/panel").json()
    assert ids["review_id"] in [r["id"] for r in listing["reviews"]]
    assert ids["question_id"] in [q["id"] for q in listing["questions"]]
    assert carol.get(f"/api/legal/matters/{mid}/panel/{ids['review_id']}").status_code == 200
    wall_alice(sw)
    for who in ("bob", "alice"):
        c = as_user(sw, who)
        for path_matter in (mid, m215):
            base = f"/api/legal/matters/{path_matter}"
            assert c.get(f"{base}/panel/{ids['review_id']}").status_code == 404
            for verb, body in (("convene", {"specialist": "commercial"}), ("dismiss", {"reason": "Not needed (fictional)."})):
                r = c.post(f"{base}/council/questions/{ids['question_id']}/{verb}", headers=csrf_headers(c), json=body)
                assert r.status_code == 404, (who, verb, r.text)
        assert c.get(f"/api/legal/matters/{mid}/panel").status_code == 404
        for path in ("/api/legal/reviews", "/api/legal/counsel/activity"):
            assert not leaks(sw, "M-101", c.get(path).text), (who, path)
    with sqlite3.connect(sw["db"]) as db:                            # nothing was convened or dismissed by them
        assert db.execute("SELECT review_id FROM legal_council_questions WHERE id=?", (ids["question_id"],)).fetchone()[0] is None


def test_a_cited_passage_opens_only_for_its_owner_while_they_may_read_it(sw):
    cit = _card(sw, "M-101", "source_answer")["props"]["citations"][0]
    alice, bob = as_user(sw, "alice"), as_user(sw, "bob")
    r = alice.get(f"/api/legal/counsel/citations/{cit['id']}")
    assert r.status_code == 200 and r.json()["citation"]["quote"] == QUOTE
    vid = r.json()["citation"]["version"]["id"]
    reader = f"/api/legal/matters/{sw['matters']['M-101']['id']}/documents/{cit['document_id']}/versions/{vid}/reader"
    assert alice.get(reader).status_code == 200
    for path in (f"/api/legal/counsel/citations/{cit['id']}", reader):
        assert bob.get(path).status_code == 404
    wall_alice(sw)
    for path in (f"/api/legal/counsel/citations/{cit['id']}", reader):
        r = alice.get(path)
        assert r.status_code == 404 and QUOTE not in r.text
    carols = _card(sw, "M-215", "source_answer", owner="carol")["props"]["citations"][0]["id"]
    r = as_user(sw, "carol").get(f"/api/legal/counsel/citations/{carols}")
    assert r.status_code == 404 and QUOTE not in r.text


def test_conversation_cards_open_only_in_a_conversation_the_owner_can_still_open(sw):
    alice = as_user(sw, "alice")
    work = sw["seeded"]["M-101"]["work"]
    kinds = ("source_answer", "evidence_comparison", "priority_list")
    cards = [work[t]["id"] for t in kinds]
    for cid in cards:
        assert alice.get(f"/api/legal/counsel/components/{cid}").status_code == 200
    turn = ids_of(sw, "M-101", "alice")["counsel_turn_id"]
    for who in ("bob", "carol"):                                     # carol is on M-101, but these are alice's
        c = as_user(sw, who)
        for cid in cards:
            assert c.get(f"/api/legal/counsel/components/{cid}").status_code == 404
        assert c.put(f"/api/legal/counsel/turns/{turn}/feedback", headers=csrf_headers(c), json={"rating": "up"}).status_code == 404
    wall_alice(sw)
    for cid in cards:
        assert alice.get(f"/api/legal/counsel/components/{cid}").status_code == 404
    assert alice.put(f"/api/legal/counsel/turns/{turn}/feedback", headers=csrf_headers(alice), json={"rating": "up"}).status_code == 404
    assert work["conversation"] not in [c["id"] for c in alice.get("/api/legal/counsel/conversations").json()["conversations"]]
    carol = as_user(sw, "carol")
    for t in kinds:
        assert carol.get(f"/api/legal/counsel/components/{_card(sw, 'M-215', t, owner='carol')['id']}").status_code == 404


# ── Targeted: background work that finishes after access is gone ─────────────────
def _latest_job(db, kind):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT status, error FROM legal_jobs WHERE kind=? ORDER BY created_at DESC, rowid DESC", (kind,)).fetchone()


def test_a_summary_queued_before_a_wall_is_refused_when_it_runs(sw):
    alice, mid = as_user(sw, "alice"), sw["matters"]["M-101"]["id"]
    doc = sw["seeded"]["M-101"]["documents"][1]
    r = alice.post(f"/api/legal/matters/{mid}/documents/{doc}/summaries", headers=csrf_headers(alice), json={})
    assert r.status_code < 300, r.text
    wall_alice(sw)
    index_all()
    assert _latest_job(sw["db"], "summary.run") == ("failed", "not_allowed")
    with sqlite3.connect(sw["db"]) as db:
        assert db.execute("SELECT COUNT(*) FROM legal_ai_artifacts WHERE kind='summary' AND subject_id=?", (doc,)).fetchone()[0] == 0


def test_a_panel_queued_before_a_wall_writes_nothing_when_it_runs(sw):
    alice, mid = as_user(sw, "alice"), sw["matters"]["M-101"]["id"]
    r = alice.post(f"/api/legal/matters/{mid}/panel", headers=csrf_headers(alice),
                   json={"question": "Must Marrow Peak deliver by May 15, 2026 under clause 4.2?", "specialist": "commercial"})
    assert r.status_code == 200, r.text
    rid = r.json()["review"]["id"]
    wall_alice(sw)
    index_all()
    with sqlite3.connect(sw["db"]) as db:
        status, artifact = db.execute("SELECT status, artifact_id FROM legal_panel_reviews WHERE id=?", (rid,)).fetchone()
    assert status in ("failed", "not_allowed") and artifact is None


def test_a_queued_diligence_run_is_refused_once_the_requester_is_walled(sw):
    from app.legal.diligence import run
    with sqlite3.connect(sw["db"]) as db:
        insert(db, "legal_dd_runs", id="ddr_walled", matter_id=sw["matters"]["M-101"]["id"], checklist_id="ddc_M-101", status="queued",
               created_by=sw["users"]["alice"]["id"])
    wall_alice(sw)
    assert asyncio.run(run.run({"object_id": "ddr_walled", "org_id": _org(sw)})) == "not_allowed"


def test_an_answer_in_progress_stops_when_a_wall_goes_up(sw):
    from app.legal.retrieval import scope
    s = scope.authorize({"sub": sw["users"]["alice"]["id"], "org_id": _org(sw)}, sw["matters"]["M-101"]["id"])
    scope.recheck(s)
    wall_alice(sw)
    with pytest.raises(scope.ScopeRevoked):
        scope.recheck(s)


# ── What the query-embedding cache holds ─────────────────────────────────────────
def test_the_query_embedding_cache_holds_vectors_keyed_by_the_query_text_only(monkeypatch):
    """app/rag/embeddings.py keeps, per process, a vector for each single text embedded as a question: the key is the
    question as sent to the model (which may itself use words of a matter), the value a list of numbers. No matter,
    document or person id is kept with it, a hit returns only the vector, and a passage of a document (embedded at
    indexing, kind="document") is never kept in it, even when the document has a single chunk (found by this sweep:
    a one-chunk document's text used to be cached as a key)."""
    from app.rag import embeddings

    async def fake(self, client, text, original_len):
        return [0.25, 0.5, 0.75]
    monkeypatch.setattr(embeddings.OllamaEmbeddings, "_embed_one", fake)
    monkeypatch.setattr(embeddings, "_QUERY_EMBED_CACHE", type(embeddings._QUERY_EMBED_CACHE)())
    e = embeddings.OllamaEmbeddings()
    passage = f"4.2 Delivery. Marrow Peak shall complete delivery by May 15, 2026 {TAGS['M-101']}."
    asyncio.run(e.embed([passage], kind="document"))                   # a one-chunk document being indexed
    asyncio.run(e.embed([passage, "A second chunk."], kind="document"))
    asyncio.run(e.embed(["When must delivery be complete?"], kind="query"))
    [(key, vec)] = embeddings._QUERY_EMBED_CACHE.items()
    assert key.endswith("When must delivery be complete?") and TAGS["M-101"] not in key
    assert isinstance(vec, list) and all(isinstance(x, float) for x in vec)
