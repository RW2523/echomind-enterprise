"""EchoMind Legal voice (2026-11-02): spoken questions are turns of the open chat conversation.

Each question goes to /api/legal/counsel/conversations/{id}/turns with the person's on-behalf-of token, so it is answered
from the matter's documents with checked quotes and saved with its sources. Only the checked paragraphs are spoken
(without their source marks), or the server's summary of a card; nothing is asked twice behind the person's back, and
there is no ambient listening, no speculative answer and no "legal advisor" greeting.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest


def test_a_question_is_a_turn_of_the_chat_conversation():
    from app.session import OmniSessionA
    law = SimpleNamespace(backend_auth=object(), law_matter_id="m-101", law_conversation_id="conv_" + "a" * 32)
    url, body = OmniSessionA._rag_request(law, "http://backend:8000/", {"message": "When is delivery due?"})
    assert url == "http://backend:8000/api/legal/counsel/conversations/conv_" + "a" * 32 + "/turns"
    assert body["text"] == "When is delivery due?" and len(body["client_request_id"]) == 24 and set(body) == {"text", "client_request_id"}


def test_what_is_said_for_a_card_or_a_decline_is_the_servers_own_wording():
    from app.session import law_card_speech
    decline = {"summary": "No source found.", "components": [{"type": "source_answer", "props": {"state": "abstained",
               "absence": "The penalty is not mentioned in the documents."}}]}
    assert law_card_speech(decline) == "The penalty is not mentioned in the documents."
    assert law_card_speech({"summary": "", "components": [{"type": "source_answer", "props": {"state": "abstained"}}]}) == \
        "The documents in this matter do not answer that."
    assert law_card_speech({"summary": "14 open obligations in M-101.", "components": [{"type": "checklist", "props": {}}]}) == \
        "14 open obligations in M-101. The details are on your screen."
    assert law_card_speech(None) == "The answer is on your screen."


class _Resp:
    def __init__(self, lines, status=200):
        self.lines, self.status_code = lines, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _session(monkeypatch, responses):
    from app import session as mod
    calls = []

    def post(url, json=None, stream=False, timeout=None, headers=None):
        calls.append((url, json, headers))
        return responses.pop(0)
    monkeypatch.setattr(mod.requests, "post", post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod.SETTINGS, "LEAD_PHRASE_ENABLED", False, raising=False)
    s = object.__new__(mod.OmniSessionA)
    sent, spoken = [], []

    async def send(msg):
        sent.append(msg)

    async def pipeline(q, gen):
        while True:
            item = await q.get()
            if item is None:
                return
            spoken.append(item[1] if isinstance(item, tuple) else item)
    s.send, s._phrase_pipeline = send, pipeline
    s.backend_auth = SimpleNamespace(headers=lambda: {"Authorization": "Bearer obo.x.y"})
    s.law_matter_id, s.law_conversation_id = "m-101", "conv_" + "b" * 32
    s.generation_id, s._law_turn, s._recent_hold, s.history = 1, {}, [], []
    s._backend_handle = s._llm_prod_task = s._tts_phrase_task = None
    s.conversation_memory = SimpleNamespace(add_text=lambda *a, **k: None)
    s._assistant_active_gen, s.turn_id = None, 0
    s._discard_spec = lambda reason: None
    return s, calls, sent, spoken


def _lines(*events):
    return [json.dumps({"event": e, "data": d}) for e, d in events]


def test_only_checked_paragraphs_are_spoken_without_their_marks_and_the_turn_is_shown(monkeypatch):
    s, calls, sent, spoken = _session(monkeypatch, [_Resp(_lines(
        ("turn.started", {"turn_id": "turn_1"}),
        ("answer.delta", {"text": "Delivery is due by May 15, 2026 [1]. "}),
        ("answer.delta", {"text": "Fees are USD 120,000 [1, 2].\n\n"}),
        ("ui.final", {"summary": "Answered from 2 sources.", "components": [{"type": "source_answer", "props": {"state": "answered"}}]}),
        ("turn.completed", {"turn_id": "turn_1"})))])
    asyncio.run(s._law_reply(1, "When is delivery due?"))
    said = " ".join(spoken)
    assert "May 15, 2026" in said and "USD 120,000" in said and "[" not in said
    assert calls[0][0].endswith("/api/legal/counsel/conversations/conv_" + "b" * 32 + "/turns")
    assert calls[0][2]["Authorization"] == "Bearer obo.x.y"
    assert {"type": "turn_saved", "generation_id": 1, "turn_id": "turn_1", "conversation_id": s.law_conversation_id} in sent
    assert sent[-1]["event"] == "BACK_TO_LISTENING"


def test_a_card_is_announced_by_its_summary(monkeypatch):
    s, calls, sent, spoken = _session(monkeypatch, [_Resp(_lines(
        ("turn.started", {"turn_id": "turn_2"}),
        ("ui.final", {"summary": "14 open obligations in M-101.", "components": [{"type": "checklist", "props": {}}]}),
        ("turn.completed", {"turn_id": "turn_2"})))])
    asyncio.run(s._law_reply(1, "What are the open obligations?"))
    assert spoken == ["14 open obligations in M-101. The details are on your screen."]


def test_a_failed_turn_is_said_once_and_never_asked_again(monkeypatch):
    s, calls, sent, spoken = _session(monkeypatch, [_Resp(_lines(("turn.started", {"turn_id": "turn_3"}),
                                                                ("turn.failed", {"message": "x"})))])
    s._speak_phrase = None

    async def speak(gen, text, **kw):
        spoken.append(text)
    s._speak_phrase = speak
    asyncio.run(s._law_reply(1, "When is delivery due?"))
    assert len(calls) == 1 and spoken[-1] == "The answer could not be completed. Please ask again."
    s, calls, sent, spoken = _session(monkeypatch, [_Resp(_lines(("scope_revoked", {"turn_id": "turn_4"})))])
    s._speak_phrase = speak
    asyncio.run(s._law_reply(1, "When is delivery due?"))
    assert len(calls) == 1 and spoken[-1] == "You no longer have access to this matter."


def test_a_question_right_after_an_interruption_waits_for_the_stopped_turn(monkeypatch):
    ok = _Resp(_lines(("turn.started", {"turn_id": "turn_5"}), ("answer.delta", {"text": "Yes."}),
                      ("ui.final", {"summary": "s", "components": []}), ("turn.completed", {})))
    s, calls, sent, spoken = _session(monkeypatch, [_Resp([], status=409), _Resp([], status=409), ok])
    asyncio.run(s._law_reply(1, "And the fee?"))
    assert len(calls) == 3 and spoken == ["Yes."]


def test_no_speculative_answer_and_no_ambient_listening_in_law(monkeypatch):
    from app import session as mod
    s = object.__new__(mod.OmniSessionA)
    s.backend_auth, s._spec = object(), None
    s._discard_spec = lambda reason: None
    assert mod.OmniSessionA._maybe_start_speculation(s, 1, "When is delivery due under the agreement?") is None
    assert "legal advisor" not in mod.LAW_INTRO and "documents" in mod.LAW_INTRO


@pytest.fixture
def law_server(monkeypatch):
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    monkeypatch.setenv("VOICE_AUTH_ENABLED", "1")
    monkeypatch.setenv("VOICE_OFFLINE", "1")
    monkeypatch.setenv("VOICE_ALLOWED_ORIGINS", "https://law.example")
    from app import server
    seen = {}

    class _Stub:
        def __init__(self, ws):
            self.ws, self.backend_auth = ws, None

        async def start(self, sid):
            seen["conversation"] = getattr(self, "law_conversation_id", "unset")
            await self.ws.send_text(json.dumps({"type": "hello"}))

        async def close(self):
            pass

    async def gate(ws):
        return SimpleNamespace(headers=lambda: {}, still_valid=lambda: True)
    monkeypatch.setattr(server, "OmniSessionA", _Stub)
    monkeypatch.setattr(server, "_law_ws_gate", gate)
    from fastapi.testclient import TestClient
    return TestClient(server.app), seen


def test_the_socket_takes_only_a_well_formed_conversation_id(law_server):
    from starlette.websockets import WebSocketDisconnect
    client, seen = law_server
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws?matter=m-101&conversation=../../x", headers={"origin": "https://law.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008
    conv = "conv_" + "c" * 32
    with client.websocket_connect(f"/ws?matter=m-101&conversation={conv}", headers={"origin": "https://law.example"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
    assert seen["conversation"] == conv


def test_calculated_dates_are_said_as_calculated(monkeypatch):
    from app.session import law_calculation_speech
    env = {"components": [{"type": "source_answer", "props": {"state": "answered", "calculation": {
        "document_title": "Master Services Agreement", "derived": [{"label": "Current term ends", "display": "1 March 2029"},
                                                              {"label": "Last day to give notice of non-renewal", "display": "1 December 2028"},
                                                              {"label": "If no notice is given, the next term", "display": "x"}]}}}]}
    assert law_calculation_speech(env) == ("Calculated from Master Services Agreement, not stated in it: current term ends: 1 March 2029; "
                                           "last day to give notice of non-renewal: 1 December 2028.")
    assert law_calculation_speech({"components": [{"type": "source_answer", "props": {"state": "answered"}}]}) == ""
    s, calls, sent, spoken = _session(monkeypatch, [_Resp(_lines(
        ("turn.started", {"turn_id": "turn_9"}), ("answer.delta", {"text": "The term is 36 months [1]."}),
        ("ui.final", env), ("turn.completed", {})))])
    asyncio.run(s._law_reply(1, "When does the term end?"))
    assert "1 March 2029" in " ".join(spoken) and "not stated in it" in " ".join(spoken)
