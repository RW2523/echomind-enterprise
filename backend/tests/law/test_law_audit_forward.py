"""S1.5 / SEC-04 — forwarding the audit trail to the firm's SIEM as CEF over syslog (decision D-069).

Off unless LEGAL_AUDIT_SYSLOG names a destination (udp://host:port or tcp://host:port). Each event goes once, in
chain order, as an RFC 5424 line carrying a CEF record with the event's seq and hash, so the SIEM's copy can be
checked against the chain. A cursor in the database moves on only after a send succeeds: a restart or an
unreachable SIEM skips nothing. A new destination starts at the current end of the trail unless
LEGAL_AUDIT_SYSLOG_FROM_START asks for the history. The SIEM here is a real socket on 127.0.0.1.
"""
from __future__ import annotations

import logging
import re
import socket
import sqlite3
import time
from datetime import datetime, timezone

import pytest

from conftest import as_user


@pytest.fixture(autouse=True)
def _forwarder_stopped():
    yield
    from app.legal.audit import forward
    forward.stop()


@pytest.fixture
def siem(law_app, monkeypatch):
    """siem("udp") or siem("tcp", port): a socket on this machine that LEGAL_AUDIT_SYSLOG then points at."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG_INTERVAL_SEC", 0.05)
    opened = []

    def point(scheme="udp", port=0):
        if scheme == "udp":
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", port))
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(8)
        opened.append(s)
        monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG", f"{scheme}://127.0.0.1:{s.getsockname()[1]}")
        return s
    yield point
    for s in opened:
        s.close()


def datagrams(sock, wait=0.3) -> list[str]:
    """Every datagram that arrives until none has for `wait` seconds."""
    sock.settimeout(wait)
    out = []
    while True:
        try:
            out.append(sock.recv(65535).decode("utf-8"))
        except socket.timeout:
            return out


def tcp_frames(listener) -> list[str]:
    """Accept one connection and split what it carried by RFC 6587 octet counting."""
    listener.settimeout(5)
    conn, _ = listener.accept()
    with conn:
        conn.settimeout(5)
        data = b""
        while chunk := conn.recv(65536):
            data += chunk
    out = []
    while data:
        n, space, rest = data.partition(b" ")
        assert n.isdigit() and space and len(rest) >= int(n), data[:40]
        out.append(rest[:int(n)].decode("utf-8"))
        data = rest[int(n):]
    return out


def parse(line: str) -> dict:
    """One RFC 5424 line carrying CEF, split into its parts with the CEF escapes undone."""
    m = re.match(r"<(\d+)>1 (\S+) (\S+) (\S+) - - - (CEF:.*)\Z", line, re.S)
    assert m, line
    pri, ts, host, app, cef = m.groups()
    header, cur, i = [], "", 0
    while len(header) < 7:
        if cef[i] == "\\":
            cur, i = cur + cef[i + 1], i + 2
        elif cef[i] == "|":
            header, cur, i = header + [cur], "", i + 1
        else:
            cur, i = cur + cef[i], i + 1
    ext, rest, i = {}, cef[i:], 0
    while i < len(rest):
        k = re.match(r"([A-Za-z0-9]+)=", rest[i:])
        assert k, rest[i:]
        i += k.end()
        val = ""
        while i < len(rest):
            if rest[i] == "\\":
                val, i = val + {"n": "\n", "r": "\r"}.get(rest[i + 1], rest[i + 1]), i + 2
            elif rest[i] == " " and re.match(r" [A-Za-z0-9]+=", rest[i:]):
                i += 1
                break
            else:
                val, i = val + rest[i], i + 1
        ext[k.group(1)] = val
    return {"pri": int(pri), "ts": ts, "host": host, "app": app, "cef": header, "ext": ext}


def events(db) -> list[dict]:
    with sqlite3.connect(db) as c:
        return [dict(zip(("seq", "ts", "hash", "org_id"), r)) for r in c.execute("SELECT seq, ts, hash, org_id FROM audit_events ORDER BY seq")]


def cursor(db) -> list[tuple]:
    with sqlite3.connect(db) as c:
        return c.execute("SELECT destination, last_seq FROM legal_audit_forward_state").fetchall()


def epoch_ms(ts: str) -> int:
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1000 + dt.microsecond // 1000


# ── Off by default ───────────────────────────────────────────────────────────────
def test_forwarding_is_off_by_default(law_app, fresh_db, monkeypatch):
    from app.core.config import Settings, settings
    from app.legal.audit import forward, record
    for name in ("LEGAL_AUDIT_SYSLOG", "LEGAL_AUDIT_SYSLOG_FROM_START", "LEGAL_AUDIT_SYSLOG_INTERVAL_SEC"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert (s.LEGAL_AUDIT_SYSLOG, s.LEGAL_AUDIT_SYSLOG_FROM_START, s.LEGAL_AUDIT_SYSLOG_INTERVAL_SEC) == ("", False, 2.0)
    assert settings.LEGAL_AUDIT_SYSLOG == ""
    record.system_event("test.event")
    assert forward.start() is False and forward._thread is None
    assert forward.forward_pending() == 0
    assert cursor(fresh_db) == []


def _forwarder_running_during_lifespan(monkeypatch) -> bool:
    """Run the backend's start-up and shut-down (models and the job worker left out): was the forwarder running?"""
    import asyncio
    import app.main as m
    from app.legal.audit import forward
    from app.legal.jobs import worker
    monkeypatch.setenv("ECHOMIND_ASR_PRELOAD", "0")

    async def _nothing():
        return None
    monkeypatch.setattr(m, "_warm_llm_and_embeddings", _nothing)
    monkeypatch.setattr(m, "_stt_fatal_watchdog", _nothing)
    monkeypatch.setattr(worker, "start", lambda *a, **k: None)
    monkeypatch.setattr(worker, "stop", lambda *a, **k: None)
    seen = []

    async def _run():
        async with m.lifespan(m.app):
            seen.append(forward._thread is not None and forward._thread.is_alive())
    asyncio.run(_run())
    assert forward._thread is None                      # stopped at shut-down
    return seen[0]


def test_the_backend_starts_the_forwarder_only_in_law_with_a_destination(siem, monkeypatch):
    assert _forwarder_running_during_lifespan(monkeypatch) is False
    siem("udp")
    assert _forwarder_running_during_lifespan(monkeypatch) is True
    monkeypatch.delenv("ECHOMIND_EDITION")
    assert _forwarder_running_during_lifespan(monkeypatch) is False


@pytest.mark.parametrize("url", ["https://siem.example:514", "syslog://127.0.0.1:514", "udp://127.0.0.1", "tcp://:6514"])
def test_an_unknown_destination_does_not_start(law_app, monkeypatch, caplog, capsys, url):
    from app.core.config import settings
    from app.legal.audit import forward
    from app.legal.audit.__main__ import main
    monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG", url)
    caplog.set_level(logging.INFO)
    assert forward.start() is False and forward._thread is None
    assert "LEGAL_AUDIT_SYSLOG must be udp://host:port or tcp://host:port" in caplog.text
    with pytest.raises(ValueError):
        forward.forward_pending()
    assert main(["forward", "--once"]) == 1
    assert "udp://host:port or tcp://host:port" in capsys.readouterr().err


# ── The message ──────────────────────────────────────────────────────────────────
def _event(**kw) -> dict:
    e = {"seq": 7, "ts": "2026-10-12T08:00:00.123Z", "org_id": "org-1", "actor_user_id": "u-1", "actor_username": "alice",
         "actor_roles": "attorney", "auth": "session", "session_ref": "s-1", "ip": "10.0.0.5", "action": "matter.read",
         "object_type": "document", "object_id": "d-9", "matter_id": "m-3", "outcome": "allow", "reason": None,
         "details_json": None, "prev_hash": "0" * 64, "hash": "ab" * 32}
    return {**e, **kw}


def test_cef_fields_severity_and_escaping():
    from app.legal.audit import forward
    line = forward.cef(_event(action="test|pipe\\x", outcome="deny", reason="a=b\\c\r\nd", details_json='{"k":"v=w"}'), "1.2|3")
    assert line == ("CEF:0|EchoMind|EchoMind Law|1.2\\|3|test\\|pipe\\\\x|test\\|pipe\\\\x|6|"
                    "rt=1791792000123 act=test|pipe\\\\x outcome=deny suid=u-1 suser=alice src=10.0.0.5 "
                    "cs1Label=matter cs1=m-3 cs2Label=object cs2=document:d-9 cs3Label=reason cs3=a\\=b\\\\c\\r\\nd "
                    '''cs4Label=details cs4={"k":"v\\=w"} cs5Label=hash cs5=''' + "ab" * 32 + " "
                    "cs6Label=org cs6=org-1 cn1Label=seq cn1=7")
    bare = _event(org_id=None, actor_user_id=None, actor_username=None, ip=None, object_type=None, object_id=None, matter_id=None)
    assert forward.cef(bare, "1.0").endswith(f"|matter.read|matter.read|3|rt=1791792000123 act=matter.read outcome=allow "
                                             f"cs5Label=hash cs5={'ab' * 32} cn1Label=seq cn1=7")

    def severity(**kw):
        return parse(forward.syslog_line(_event(**kw), "1.0"))["cef"][6]
    assert [severity(outcome=o) for o in ("allow", "deny", "error")] == ["3", "6", "8"]
    assert [severity(action="breakglass.approve", outcome=o) for o in ("allow", "deny", "error")] == ["7", "7", "8"]
    assert severity(action="documents.read", reason="break_glass") == "7"            # a read under emergency access stands out too


def test_syslog_framing():
    from app.legal.audit import forward
    for outcome, pri in (("allow", 166), ("deny", 164), ("error", 163)):             # local4 (20): info, warning, err
        p = parse(forward.syslog_line(_event(outcome=outcome), "1.0"))
        assert (p["pri"], p["ts"], p["host"], p["app"]) == (pri, "2026-10-12T08:00:00.123Z", socket.gethostname(), "echomind-law")
        assert p["cef"][:6] == ["CEF:0", "EchoMind", "EchoMind Law", "1.0", "matter.read", "matter.read"]


# ── Delivery ─────────────────────────────────────────────────────────────────────
def test_an_audited_action_arrives_as_one_cef_line(world, siem, fresh_db, monkeypatch):
    from app.legal.audit import forward
    monkeypatch.setenv("BUILD_VERSION", "1.4.0")
    alice = as_user(world, "alice")
    sock = siem("udp")
    assert forward.forward_pending() == 0                # a new destination starts at the current end
    mid = world["matters"]["M-101"]["id"]
    assert alice.get(f"/api/legal/matters/{mid}").status_code == 200
    assert forward.forward_pending() == 1
    got = datagrams(sock)
    assert len(got) == 1
    p, last = parse(got[0]), events(fresh_db)[-1]
    assert p["pri"] == 166 and p["app"] == "echomind-law" and p["ts"] == last["ts"]
    assert p["cef"] == ["CEF:0", "EchoMind", "EchoMind Law", "1.4.0", "matter.read", "matter.read", "3"]
    assert p["ext"] == {"rt": str(epoch_ms(last["ts"])), "act": "matter.read", "outcome": "allow",
                        "suid": world["users"]["alice"]["id"], "suser": "alice", "src": "testclient",
                        "cs1Label": "matter", "cs1": mid, "cs2Label": "object", "cs2": f"matter:{mid}",
                        "cs5Label": "hash", "cs5": last["hash"], "cs6Label": "org", "cs6": last["org_id"],
                        "cn1Label": "seq", "cn1": str(last["seq"])}
    assert "MSA renewal" not in got[0]                    # the matter's title stays home
    assert forward.forward_pending() == 0 and datagrams(sock) == []
    assert cursor(fresh_db) == [(f"udp://127.0.0.1:{sock.getsockname()[1]}", last["seq"])]


def test_tcp_frames_each_message_by_octet_counting(siem, fresh_db):
    from app.legal.audit import forward, record
    listener = siem("tcp")
    forward.forward_pending()
    tricky = ["café", "a|b=c\\d\r\ne"]                    # the count is in bytes; escapes survive the trip
    for reason in tricky:
        record.system_event("test.event", reason=reason)
    assert forward.forward_pending() == 2
    frames, last_two = tcp_frames(listener), events(fresh_db)[-2:]
    assert [parse(f)["ext"]["cs3"] for f in frames] == tricky
    assert [(parse(f)["ext"]["cn1"], parse(f)["ext"]["cs5"]) for f in frames] == [(str(e["seq"]), e["hash"]) for e in last_two]


def test_a_restart_resumes_where_it_stopped_without_repeats(siem, fresh_db):
    from app.legal.audit import forward, record
    sock = siem("udp")
    forward.forward_pending()
    for i in range(3):
        record.system_event("test.event", details={"n": i})
    assert forward.start()
    first = datagrams(sock, wait=1.0)
    forward.stop()
    record.system_event("test.event", details={"n": 3})   # written while the forwarder is down
    assert forward.start()
    second = datagrams(sock, wait=1.0)
    forward.stop()
    seqs = [int(parse(x)["ext"]["cn1"]) for x in first + second]
    tail = [e["seq"] for e in events(fresh_db)[-4:]]
    assert (len(first), len(second)) == (3, 1) and seqs == tail
    assert cursor(fresh_db)[0][1] == tail[-1]


def test_a_failed_send_keeps_the_cursor_and_the_next_run_delivers(siem, fresh_db):
    from app.legal.audit import forward, record
    listener = siem("tcp")
    port = listener.getsockname()[1]
    forward.forward_pending()
    record.system_event("test.event", details={"n": 1})
    assert forward.forward_pending() == 1 and len(tcp_frames(listener)) == 1
    sent = events(fresh_db)[-1]["seq"]
    listener.close()                                      # the SIEM goes away
    record.system_event("test.event", details={"n": 2})
    with pytest.raises(OSError):
        forward.forward_pending()
    assert cursor(fresh_db)[0][1] == sent
    listener = siem("tcp", port)                          # and comes back
    assert forward.forward_pending() == 1
    assert [int(parse(f)["ext"]["cn1"]) for f in tcp_frames(listener)] == [events(fresh_db)[-1]["seq"]]
    assert cursor(fresh_db)[0][1] == events(fresh_db)[-1]["seq"]


def test_the_background_loop_backs_off_and_logs_only_the_error_type(siem, fresh_db, monkeypatch, caplog):
    from app.core.config import settings
    from app.legal.audit import forward, record
    siem("tcp").close()                                   # nothing listens there
    forward.forward_pending()
    record.system_event("test.event", reason="QUOKKA-7731 privileged")
    caplog.set_level(logging.WARNING, logger="app.legal.audit.forward")
    assert forward.start()
    deadline = time.monotonic() + 5
    while "ConnectionRefusedError" not in caplog.text and time.monotonic() < deadline:
        time.sleep(0.05)
    forward.stop()
    assert "ConnectionRefusedError" in caplog.text and "QUOKKA" not in caplog.text
    monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG_INTERVAL_SEC", 2)
    assert [forward._backoff(n) for n in (0, 1, 3, 10)] == [2, 4, 16, 60]


def test_a_new_destination_starts_at_the_end_unless_asked_for_the_history(siem, fresh_db, monkeypatch):
    from app.core.config import settings
    from app.legal.audit import forward, record
    for i in range(3):
        record.system_event("test.event", details={"n": i})
    a = siem("udp")
    assert forward.forward_pending() == 0 and datagrams(a) == []
    record.system_event("test.event", details={"n": 3})
    assert forward.forward_pending() == 1 and len(datagrams(a)) == 1
    b = siem("udp")
    monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG_FROM_START", True)
    everything = [e["seq"] for e in events(fresh_db)]
    assert forward.forward_pending() == len(everything)
    assert [int(parse(x)["ext"]["cn1"]) for x in datagrams(b)] == everything
    assert datagrams(a) == []


def test_the_cli_sends_everything_pending(siem, fresh_db, monkeypatch, capsys):
    import functools
    from app.core.config import settings
    from app.legal.audit import forward, record
    from app.legal.audit.__main__ import main
    sock = siem("udp")
    monkeypatch.setattr(settings, "LEGAL_AUDIT_SYSLOG_FROM_START", True)
    for i in range(5):
        record.system_event("test.event", details={"n": i})
    monkeypatch.setattr(forward, "forward_pending", functools.partial(forward.forward_pending, limit=2))   # several batches
    total = len(events(fresh_db))
    assert main(["forward", "--once"]) == 0
    assert capsys.readouterr().out.strip() == f"forwarded: {total}"
    assert len(datagrams(sock)) == total
    assert main(["forward", "--once"]) == 0 and capsys.readouterr().out.strip() == "forwarded: 0"
