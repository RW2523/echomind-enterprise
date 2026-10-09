"""T11 — the outbound-connection recorder used by scripts/law/egress_test.sh (S1.12).

The recorder runs in a separate interpreter (audit hooks cannot be removed), and the events are
raised with sys.audit, so these tests never touch the network.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("host,local", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", True), ("/run/llm.sock", True), (None, True),
    ("127.8.0.1", True), ("[::1]", True), ("8.8.8.8", False), ("api.openai.com", False), ("huggingface.co", False),
    ("10.0.0.5", False), ("192.168.1.10", False), ("2606:4700::1111", False),
])
def test_t11_only_this_machine_counts_as_local(host, local):
    from app.legal import egress
    assert egress.is_local(host) is local


def test_t11_every_lookup_and_connect_attempt_is_recorded(tmp_path):
    log = tmp_path / "egress.jsonl"
    code = (
        "import sys, threading\n"
        "from app.legal import egress\n"
        f"egress.install({str(log)!r})\n"
        "sys.audit('socket.getaddrinfo', b'localhost', 8355, 0, 0, 0)\n"
        "sys.audit('socket.connect', None, ('127.0.0.1', 8355))\n"
        "sys.audit('socket.getaddrinfo', 'huggingface.co', 443, 0, 0, 0)\n"
        "t = threading.Thread(target=lambda: sys.audit('socket.connect', None, ('203.0.113.7', 443)), name='worker')\n"
        "t.start(); t.join()\n"
        "sys.audit('socket.sendto', None, ('198.51.100.9', 53))\n"
        "sys.audit('open', '/etc/hostname', 'r', 0)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=".")
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(e["event"], e["host"], e["port"], e["local"]) for e in events] == [
        ("socket.getaddrinfo", "localhost", 8355, True),
        ("socket.connect", "127.0.0.1", 8355, True),
        ("socket.getaddrinfo", "huggingface.co", 443, False),
        ("socket.connect", "203.0.113.7", 443, False),
        ("socket.sendto", "198.51.100.9", 53, False),
    ]
    assert events[3]["thread"] == "worker"
