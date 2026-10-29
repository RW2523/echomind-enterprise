#!/usr/bin/env python3
"""Reproduce the 2026-10-29 functional QA findings (EchoMind Legal fix brief: RAG-001, RAG-002, RAG-003, UX-001, CHAT-001).

Builds a throwaway Law data folder like run_law_eval.py, imports the testers' fictional sample_msa_test.txt into one
matter (QA-101, as M-101) and nothing into a second (QA-114, as M-114), and asks the testers' questions through the chat
endpoint they used, with the real answer and embedding models on this host. For every question it prints the answer
card (state, answer, citations with clause and verification, unsupported sentences, notes) and the model's raw text, and
writes them to eval/law/qa_20261029/runs/<time>.json. Nothing leaves this host; the data folder is deleted at the end.

    cd backend && python3 ../eval/law/qa_20261029/repro.py [--repeat N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_law_eval as base                                   # noqa: E402  (the evaluation's throwaway setup)

QUESTIONS = [
    ("RAG-001", "QA-101", "What is the monthly payment and when must invoices be paid?"),
    ("RAG-002", "QA-101", "What is the termination notice period, and can either party terminate? Cite the supporting clause."),
    ("RAG-003", "QA-101", "When does the current term end, what is the deadline to give notice of non-renewal, what happens if no "
                          "notice is given, and what assumptions does that rely on?"),
    ("UX-001", "QA-101", "What is the early termination penalty under the agreement?"),
    ("CHAT-001", "QA-101", "Which documents are available in this matter?"),
    ("CHAT-001", "QA-114", "Which documents are available in this matter?"),
    ("ISOLATION", "QA-114", "What is the monthly payment and when must invoices be paid?"),
]


class Recording:
    """The real answer model, with every raw reply kept."""
    def __init__(self, real):
        self.real, self.raw = real, []

    async def chat_stream(self, messages, temperature, max_tokens):
        out = []
        async for piece in self.real.chat_stream(messages, temperature=temperature, max_tokens=max_tokens):
            out.append(piece)
            yield piece
        self.raw.append("".join(out))

    def __getattr__(self, name):
        return getattr(self.real, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated issue ids")
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--embed", default="http://localhost:11435/api/embeddings")
    args = ap.parse_args()
    data = base.setup_env(args)
    try:
        return run(args)
    finally:
        import shutil
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def run(args) -> int:
    import logging
    from fastapi.testclient import TestClient
    logging.disable(logging.INFO)
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from app.legal.research import llm
    from conftest import csrf_headers, make_user, sign_in

    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Harrowgate Technologies LLC (fictional)", "HTL", alice["id"])
    matters = {"QA-101": repo.create_matter(org, client["id"], "QA-101", "Sample MSA renewal (QA fixture)", alice["id"]),
               "QA-114": repo.create_matter(org, client["id"], "QA-114", "Second matter, no documents (QA fixture)", alice["id"])}
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    r = c.post(f"/api/legal/matters/{matters['QA-101']['id']}/documents/import", headers=csrf_headers(c),
               files={"file": ("sample_msa_test.txt", (HERE / "sample_msa_test.txt").read_bytes(), "text/plain")},
               data={"kind": "contract", "status": "draft", "title": "sample_msa_test"})
    assert r.status_code == 202, r.text
    worker.run_pending()
    rec = Recording(llm.get())
    llm._override = rec
    out = []
    only = set(filter(None, args.only.split(",")))
    for k in range(1, args.repeat + 1):
        for issue, number, question in QUESTIONS:
            if only and issue not in only:
                continue
            conv = c.post("/api/legal/counsel/conversations", headers=csrf_headers(c), json={"matter_id": matters[number]["id"]})
            assert conv.status_code == 201, conv.text
            cid = conv.json()["conversation"]["id"]
            before = len(rec.raw)
            t0 = time.monotonic()
            resp = c.post(f"/api/legal/counsel/conversations/{cid}/turns", headers=csrf_headers(c), json={"text": question})
            assert resp.status_code == 200, resp.text
            events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
            final = next((e["data"] for e in events if e["event"] == "ui.final"), {})
            comps = final.get("components") or []
            card = next((x["props"] for x in comps if x.get("type") == "source_answer"), None)
            row = {"run": k, "issue": issue, "matter": number, "question": question, "seconds": round(time.monotonic() - t0, 1),
                   "status": final.get("status"), "summary": final.get("summary"),
                   "components": [x.get("type") for x in comps], "card": card,
                   "other": [x.get("props") for x in comps if x.get("type") != "source_answer"],
                   "raw": rec.raw[before:]}
            out.append(row)
            print(f"\n=== {issue} {number} run {k} — {row['status']} — {row['summary']}  components={row['components']}")
            if card:
                print(f"  state={card['state']}\n  answer={card['answer']!r}")
                for ci in card["citations"]:
                    print(f"  cite n={ci['n']} clause={ci.get('clause')!r} verified={ci['verified']} doc={ci['document_title']!r}")
                print(f"  unsupported={card.get('unsupported')}\n  notes={card.get('notes')}")
                if card.get("absence"):
                    print(f"  absence={card['absence']!r}")
                if card.get("calculation"):
                    calc = card["calculation"]
                    print("  calculated=" + "; ".join(f"{d['label']}: {d['display']}" for d in calc["derived"])
                          + "  from " + ", ".join(f"{x['label']} cl. {x.get('clause')}" for x in calc["facts"]))
            else:
                print("  " + json.dumps(row["other"])[:600])
            for raw in row["raw"]:
                print(f"  raw={raw!r}")
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
