#!/usr/bin/env python3
"""Watch one process's sockets from outside for the network-denied test (S1.12, spec test T11).

Every 50 ms, maps the process's open socket descriptors (/proc/<pid>/fd) to the kernel's socket
tables (/proc/net/tcp, tcp6, udp, udp6) and records every remote endpoint that is not this machine.
This sees connections made by native code too, which the backend's Python recorder cannot; a
connection that opens and closes between two looks can be missed, which is why both run together.
Reading our own processes' descriptors needs no special permission.

    egress_watch.py --pid PID --out FILE [--seconds N]     # stops when the process exits
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import struct
import time


def _addr(hexaddr: str) -> tuple[str, int]:
    ip_hex, port_hex = hexaddr.split(":")
    raw = bytes.fromhex(ip_hex)
    if len(raw) == 4:
        ip = socket.inet_ntop(socket.AF_INET, struct.pack("<I", int(ip_hex, 16)))
    else:
        ip = socket.inet_ntop(socket.AF_INET6, b"".join(struct.pack("<I", int(ip_hex[i:i + 8], 16)) for i in range(0, 32, 8)))
    return ip, int(port_hex, 16)


def _local(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if getattr(a, "ipv4_mapped", None):
        a = a.ipv4_mapped
    return a.is_loopback or a.is_unspecified


def _socket_inodes(pid: int) -> set[str]:
    out = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            out.add(target[8:-1])
    return out


def _table(pid: int) -> dict[str, tuple[str, str, int]]:
    rows = {}
    for proto in ("tcp", "tcp6", "udp", "udp6"):
        try:
            with open(f"/proc/{pid}/net/{proto}") as f:
                next(f)
                for line in f:
                    parts = line.split()
                    remote, inode = parts[2], parts[9]
                    ip, port = _addr(remote)
                    rows[inode] = (proto, ip, port)
        except OSError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=7200)
    args = ap.parse_args()
    seen, looks, end = set(), 0, time.monotonic() + args.seconds
    with open(args.out, "a", buffering=1) as out:
        out.write(json.dumps({"ts": time.time(), "watching": args.pid}) + "\n")
        while time.monotonic() < end and os.path.exists(f"/proc/{args.pid}"):
            try:
                inodes = _socket_inodes(args.pid)
                table = _table(args.pid)
            except OSError:
                break
            looks += 1
            for inode in inodes:
                if inode in table:
                    proto, ip, port = table[inode]
                    if port and not _local(ip) and (proto, ip, port) not in seen:
                        seen.add((proto, ip, port))
                        out.write(json.dumps({"ts": time.time(), "proto": proto, "remote": ip, "port": port}) + "\n")
            time.sleep(0.05)
        out.write(json.dumps({"ts": time.time(), "done": True, "looks": looks, "remote_endpoints": len(seen)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
