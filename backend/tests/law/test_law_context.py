"""Context added around what research finds (research/routes._context, 2026-11-07, external review point 3): the clause a
passage is "subject to" and the definition of a defined term it uses, each the document's own passage, numbered and cited
like any other source, within a cap. Fictional documents only."""
from __future__ import annotations

import json
import re

import pytest

from conftest import as_user, csrf_headers, index_all, upload

FILL = " ".join(["The parties shall keep records of the matters in this clause and make them available on reasonable request."] * 6)
CONTRACT = (
    "SUPPLY AGREEMENT (fictional)\n\n"
    f"1.1 \"Deliverables\" means the monthly reports and the quarterly forecasts described in Schedule 2. {FILL}\n\n"
    f"4.1 Records. {FILL}\n\n"
    f"7.1 Acceptance. {FILL}\n\n"
    "9.1 Termination. The Customer may terminate this Agreement on 60 days' written notice, subject to clause 9.4. "
    "The Supplier shall hand over all Deliverables within 10 days of termination.\n\n"
    f"9.2 Effect. {FILL}\n\n"
    f"9.3 Survival. {FILL}\n\n"
    "9.4 Minimum term. No notice of termination may be given before 1 January 2027.\n")


class Recorder:
    def __init__(self):
        self.prompts = []

    async def chat_stream(self, messages, temperature, max_tokens):
        self.prompts.append(messages[-1]["content"])
        for piece in re.findall(r"\S+\s*|\n+", "The documents in this matter do not answer that."):
            yield piece

    async def chat(self, messages, temperature=0.0, max_tokens=200):
        return "?"


@pytest.fixture
def ctx(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    monkeypatch.setattr(settings, "LEGAL_RERANK_MIN_SCORE", -100.0)
    monkeypatch.setattr(settings, "LEGAL_RESEARCH_PASSAGES", 1)        # one passage found, so what is added is visible
    monkeypatch.setattr(settings, "LEGAL_RESEARCH_CONTEXT", "on")
    model = Recorder()
    monkeypatch.setattr(llm, "_override", model)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    upload(alice, mid, "supply.txt", CONTRACT, status="executed", title="Supply Agreement")
    index_all()
    docs.update(mid=mid, model=model)
    return docs


def sources(c, mid, q):
    r = c.post(f"/api/legal/matters/{mid}/research/query", headers=csrf_headers(c), json={"question": q})
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()][0]["citations"]


def test_the_clause_a_passage_is_subject_to_and_a_definition_it_uses_are_added_as_sources(ctx):
    alice = as_user(ctx, "alice")
    got = sources(alice, ctx["mid"], "On what notice may the Customer terminate the agreement, and what happens to the Deliverables?")
    texts = [s["text"] for s in got]
    assert "60 days' written notice" in texts[0]
    assert any("No notice of termination may be given before 1 January 2027" in t for t in texts[1:])
    assert any('"Deliverables" means the monthly reports' in t for t in texts[1:])
    prompt = ctx["model"].prompts[-1]
    assert 'context="subject to"' in prompt and 'context="definition"' in prompt


def test_nothing_is_added_when_the_setting_is_off_and_never_beyond_the_cap(ctx, monkeypatch):
    from app.core.config import settings
    from app.legal.research import routes
    alice = as_user(ctx, "alice")
    q = "On what notice may the Customer terminate the agreement, and what happens to the Deliverables?"
    monkeypatch.setattr(settings, "LEGAL_RESEARCH_CONTEXT", "off")
    assert len(sources(alice, ctx["mid"], q)) == 1
    monkeypatch.setattr(settings, "LEGAL_RESEARCH_CONTEXT", "on")
    monkeypatch.setattr(routes, "MAX_CONTEXT", 1)
    assert len(sources(alice, ctx["mid"], q)) == 2
