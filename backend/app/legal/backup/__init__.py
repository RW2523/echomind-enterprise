"""Encrypted backup and restore of an EchoMind Law data folder (S1.12, spec test T13; OPS-01).

create
    A consistent snapshot of the database (SQLite's online backup API, safe while the app runs) and
    every other file of the data folder — encrypted documents and exports, audit anchors, credentials
    files — except the live database files and the re-downloadable model caches. One tar stream,
    sealed with AES-256-GCM in 1 MiB chunks under a backup key kept apart from the data and from the
    master key. Each chunk's nonce and associated data carry its position and whether it is the last,
    so an edited, reordered or truncated archive does not open. A manifest at the end lists every
    file's SHA-256 and size, the applied migrations, the audit head and record counts.

restore
    Authenticates the whole archive before writing anything, refuses a target folder that is not
    empty, extracts only plain files and folders inside the target, then checks every file against
    the manifest.

The master key is never in a backup. A restore needs three things kept apart: the archive, the backup
key and the master key (without it, restored documents cannot be opened).

    python -m app.legal.backup keygen --out /secure/backup.key
    python -m app.legal.backup create --out /backups/law-20261009.emlb --key-file /secure/backup.key
    python -m app.legal.backup verify --archive /backups/law-20261009.emlb --key-file /secure/backup.key
    python -m app.legal.backup restore --archive … --key-file … --target /restore/data
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import sqlite3
import stat
import struct
import tarfile
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from typing import BinaryIO, Iterator, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

FORMAT = "echomind-law-backup/1"
MAGIC = b"EMLBACKUP1\n"
CHUNK = 1 << 20
MANIFEST = "MANIFEST.json"
SKIP_DIRS = {"hf_cache", "docgen_models"}                     # re-downloadable model caches (prepare_offline.sh)
SKIP_SUFFIXES = ("-wal", "-shm", "-journal", ".partial")
_LEN = struct.Struct(">I")


class BackupError(Exception):
    """The backup or restore cannot be completed safely; nothing usable was written."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ── Keys ─────────────────────────────────────────────────────────────────────────
def generate_key(path: str) -> str:
    """Write a new random 256-bit backup key (base64), readable only by its owner. Never overwrites."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise BackupError(f"{path} already exists; a backup key is never overwritten")
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
    return path


def read_key(path: str) -> bytes:
    try:
        st = os.stat(path)
    except OSError:
        raise BackupError("the backup key file cannot be read")
    if st.st_mode & 0o077:
        raise BackupError("the backup key file must be readable only by its owner (chmod 600)")
    with open(path, "rb") as f:
        try:
            key = base64.b64decode(f.read().strip(), validate=True)
        except ValueError:
            raise BackupError("the backup key file is not a base64 key")
    if len(key) != 32:
        raise BackupError("the backup key must be 32 bytes")
    return key


def key_id(key: bytes) -> str:
    return hashlib.sha256(b"echomind-law-backup-key" + key).hexdigest()[:16]


# ── Sealed stream ────────────────────────────────────────────────────────────────
def _nonce(prefix: bytes, n: int) -> bytes:
    return prefix + n.to_bytes(4, "big")


def _aad(header_hash: bytes, n: int, final: bool) -> bytes:
    return header_hash + n.to_bytes(4, "big") + (b"\x01" if final else b"\x00")


class _SealingWriter(io.RawIOBase):
    """A write-only stream that seals what it is given in fixed chunks."""

    def __init__(self, out: BinaryIO, key: bytes):
        self._out, self._aead, self._buf, self._n = out, AESGCM(key), bytearray(), 0
        self._prefix = secrets.token_bytes(8)
        header = json.dumps({"format": FORMAT, "alg": "AES-256-GCM", "chunk": CHUNK, "nonce_prefix": self._prefix.hex(),
                             "key_id": key_id(key), "created_at": _now()}, sort_keys=True).encode()
        self._hash = hashlib.sha256(header).digest()
        out.write(MAGIC + _LEN.pack(len(header)) + header)
        self.bytes_out = len(MAGIC) + 4 + len(header)

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self._buf += data
        while len(self._buf) > CHUNK:                         # keep the last chunk back: it is sealed as final
            self._seal(bytes(self._buf[:CHUNK]), final=False)
            del self._buf[:CHUNK]
        return len(data)

    def _seal(self, data: bytes, final: bool) -> None:
        if self._n >= 2 ** 32 - 1:
            raise BackupError("archive too large")
        ct = self._aead.encrypt(_nonce(self._prefix, self._n), data, _aad(self._hash, self._n, final))
        self._out.write(_LEN.pack(len(ct)) + ct)
        self.bytes_out += 4 + len(ct)
        self._n += 1

    def finish(self) -> None:
        self._seal(bytes(self._buf), final=True)
        self._buf.clear()


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise BackupError("the archive is truncated")
    return data


def _open_sealed(f: BinaryIO, key: bytes) -> Iterator[bytes]:
    """Yield the plaintext chunks of a sealed archive, refusing anything altered, reordered or cut short."""
    if f.read(len(MAGIC)) != MAGIC:
        raise BackupError("not an EchoMind Law backup")
    (hlen,) = _LEN.unpack(_read_exact(f, 4))
    if hlen > 4096:
        raise BackupError("the archive header is damaged")
    header_bytes = _read_exact(f, hlen)
    try:
        header = json.loads(header_bytes)
        prefix = bytes.fromhex(header["nonce_prefix"])
    except (ValueError, KeyError):
        raise BackupError("the archive header is damaged")
    if header.get("format") != FORMAT or len(prefix) != 8:
        raise BackupError("unsupported backup format")
    if header.get("key_id") != key_id(key):
        raise BackupError("this archive was made with a different backup key")
    aead, h, n = AESGCM(key), hashlib.sha256(header_bytes).digest(), 0
    while True:
        raw = f.read(4)
        if len(raw) != 4:
            raise BackupError("the archive is truncated")
        (clen,) = _LEN.unpack(raw)
        if clen > CHUNK + 16:
            raise BackupError("the archive is damaged")
        ct = _read_exact(f, clen)
        for final in (False, True):
            try:
                pt = aead.decrypt(_nonce(prefix, n), ct, _aad(h, n, final))
                break
            except InvalidTag:
                pt = None
        if pt is None:
            raise BackupError("the archive does not authenticate: it was altered or the key is wrong")
        n += 1
        yield pt
        if final:
            if f.read(1):
                raise BackupError("unexpected data after the end of the archive")
            return


# ── Create ───────────────────────────────────────────────────────────────────────
class _Hashing(io.RawIOBase):
    def __init__(self, f: BinaryIO):
        self._f, self.sha = f, hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self._f.read(len(b))
        self.sha.update(data)
        b[: len(data)] = data
        return len(data)


def _summary(db: str) -> dict:
    with closing(sqlite3.connect(db)) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def count(sql: str) -> Optional[int]:
            try:
                return c.execute(sql).fetchone()[0]
            except sqlite3.Error:
                return None
        schema = [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version")] if "schema_migrations" in names else []
        head = c.execute("SELECT seq, hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone() if "audit_events" in names else None
        return {"schema": schema,
                "audit": {"events": count("SELECT COUNT(*) FROM audit_events"), "head_seq": head[0] if head else None,
                          "head_hash": head[1] if head else None},
                "counts": {"users": count("SELECT COUNT(*) FROM users"), "matters": count("SELECT COUNT(*) FROM matters"),
                           "documents": count("SELECT COUNT(*) FROM legal_documents"),
                           "document_versions": count("SELECT COUNT(*) FROM legal_document_versions"),
                           "active_holds": count("SELECT COUNT(*) FROM legal_holds WHERE status IN ('active','release_requested')"),
                           "exports": count("SELECT COUNT(*) FROM legal_exports")}}


def _files(data_dir: str, live_db: str) -> Iterator[tuple[str, str]]:
    live = {os.path.realpath(live_db)} | {os.path.realpath(live_db + s) for s in ("-wal", "-shm", "-journal")}
    for root, dirs, files in os.walk(data_dir, followlinks=False):
        if os.path.realpath(root) == os.path.realpath(data_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(root, d)))
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.realpath(path) in live or name.endswith(SKIP_SUFFIXES) or name.endswith(".sqlite.backup"):
                continue
            st = os.lstat(path)
            if stat.S_ISREG(st.st_mode):
                yield path, os.path.relpath(path, data_dir)


def create(out_path: str, key_file: str, data_dir: Optional[str] = None, db_path: Optional[str] = None,
           audit: bool = True) -> dict:
    from ...core.config import settings
    data_dir = os.path.realpath(data_dir or settings.DATA_DIR)
    db_path = os.path.realpath(db_path or settings.DB_PATH)
    out_path = os.path.realpath(out_path)
    key = read_key(key_file)
    if os.path.commonpath([out_path, data_dir]) == data_dir:
        raise BackupError("write the backup outside the data folder (and preferably to another disk)")
    if os.path.commonpath([db_path, data_dir]) != data_dir:
        raise BackupError("the database must be inside the data folder")
    if os.path.exists(out_path):
        raise BackupError(f"{out_path} already exists")
    db_rel = os.path.relpath(db_path, data_dir)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    partial = out_path + ".partial"
    with tempfile.TemporaryDirectory(prefix=".emlb-", dir=out_dir) as tmp:   # 0700
        snap = os.path.join(tmp, "snapshot.sqlite")
        with closing(sqlite3.connect(db_path)) as src, closing(sqlite3.connect(snap)) as dst:
            src.backup(dst)                                  # consistent while the app keeps writing
        manifest = {"format": FORMAT, "created_at": _now(), "database": db_rel, **_summary(snap), "files": {}}
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as raw:
                sealer = _SealingWriter(raw, key)
                with tarfile.open(fileobj=sealer, mode="w|gz", format=tarfile.PAX_FORMAT) as tar:
                    for path, rel in [(snap, db_rel)] + list(_files(data_dir, db_path)):
                        with open(path, "rb") as f:
                            info = tar.gettarinfo(fileobj=f, arcname=f"data/{rel}")
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.mode = 0o600
                            h = _Hashing(f)
                            tar.addfile(info, h)
                            manifest["files"][rel] = {"sha256": h.sha.hexdigest(), "size": info.size}
                    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
                    info = tarfile.TarInfo(MANIFEST)
                    info.size, info.mode, info.mtime = len(body), 0o600, int(datetime.now().timestamp())
                    tar.addfile(info, io.BytesIO(body))
                sealer.finish()
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(partial, out_path)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
    result = {"archive": out_path, "bytes": os.path.getsize(out_path), "files": len(manifest["files"]),
              "schema": manifest["schema"], "audit": manifest["audit"], "counts": manifest["counts"], "key_id": key_id(key)}
    if audit and data_dir == os.path.realpath(settings.DATA_DIR):
        from ..audit.record import record
        record("backup.create", "allow", actor_username="operator:cli", object_type="backup",
               details={"files": result["files"], "bytes": result["bytes"], "sha256": hashlib.sha256(
                   json.dumps(manifest, sort_keys=True).encode()).hexdigest()})
    return result


# ── Verify and restore ───────────────────────────────────────────────────────────
def _decrypt_to(archive: str, key: bytes, out: BinaryIO) -> None:
    with open(archive, "rb") as f:
        for chunk in _open_sealed(f, key):
            out.write(chunk)


def _safe_member(m: tarfile.TarInfo) -> Optional[str]:
    """The path under the target for an archive entry, or None for the manifest. Refuses anything else."""
    if m.name == MANIFEST and m.isfile():
        return None
    if not (m.isfile() or m.isdir()) or not m.name.startswith("data/"):
        raise BackupError(f"the archive holds an entry that is not allowed: {m.name!r}")
    rel = m.name[len("data/"):]
    norm = os.path.normpath(rel)
    if not rel or os.path.isabs(rel) or norm == ".." or norm.startswith("../") or "\x00" in rel:
        raise BackupError(f"the archive holds an entry outside the data folder: {m.name!r}")
    return norm


def _read_manifest(tar_path: str) -> dict:
    with tarfile.open(tar_path, mode="r:gz") as tar:
        manifest = None
        for m in tar:
            _safe_member(m)
            if m.name == MANIFEST:
                manifest = json.loads(tar.extractfile(m).read())
    if not manifest or manifest.get("format") != FORMAT:
        raise BackupError("the archive has no valid manifest")
    return manifest


def verify(archive: str, key_file: str) -> dict:
    """Authenticate the whole archive and return its manifest, without writing anything outside a temp folder."""
    key = read_key(key_file)
    with tempfile.TemporaryDirectory(prefix=".emlb-verify-") as tmp:
        tar_path = os.path.join(tmp, "archive.tar.gz")
        with open(tar_path, "wb") as out:
            _decrypt_to(archive, key, out)
        return _read_manifest(tar_path)


def restore(archive: str, key_file: str, target: str) -> dict:
    key = read_key(key_file)
    target = os.path.realpath(target)
    if os.path.exists(target) and (not os.path.isdir(target) or os.listdir(target)):
        raise BackupError("restore into a new or empty folder; nothing was changed")
    os.makedirs(target, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".emlb-restore-", dir=os.path.dirname(target)) as tmp:
        tar_path = os.path.join(tmp, "archive.tar.gz")
        with open(tar_path, "wb") as out:
            _decrypt_to(archive, key, out)                   # the whole archive authenticates first
        manifest = _read_manifest(tar_path)
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for m in tar:
                rel = _safe_member(m)
                if rel is None:
                    continue
                dest = os.path.join(target, rel)
                if m.isdir():
                    os.makedirs(dest, mode=0o700, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(dest), mode=0o700, exist_ok=True)
                src = tar.extractfile(m)
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as out:
                    while chunk := src.read(1 << 16):
                        out.write(chunk)
    problems = []
    for rel, meta in manifest["files"].items():
        path = os.path.join(target, rel)
        if not os.path.isfile(path):
            problems.append(f"missing {rel}")
            continue
        with open(path, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != meta["sha256"]:
                problems.append(f"changed {rel}")
    if problems:
        raise BackupError("restored files do not match the manifest: " + ", ".join(problems[:5]))
    db = os.path.join(target, manifest["database"])
    restored = _summary(db)
    if restored["schema"] != manifest["schema"] or restored["audit"] != manifest["audit"] or restored["counts"] != manifest["counts"]:
        raise BackupError("the restored database does not match the manifest")
    return {"target": target, "database": db, "files": len(manifest["files"]), "created_at": manifest["created_at"],
            "schema": restored["schema"], "audit": restored["audit"], "counts": restored["counts"]}
