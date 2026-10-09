"""Encrypted storage for uploaded files (S1.6; decision D-014, application level).

Each file gets its own random 256-bit key and is sealed with AES-256-GCM. The associated data
binds the ciphertext to its version id, so a stored file swapped for another one (or edited)
fails to open. The file key is wrapped with the deployment's master key (AES key wrap, RFC 3394)
and the wrapped form is stored with the version row; the master key itself lives outside the
data folder (LEGAL_MASTER_KEY_FILE) and is never written anywhere by the application.

On disk: DATA_DIR/legal/<folder>/<matter_id>/<id>.bin = MAGIC | nonce(12) | ciphertext+tag, where the
folder is "blobs" for uploaded versions and "exports" for released exports (S1.10).
"""
from __future__ import annotations

import base64
import os
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import InvalidUnwrap, aes_key_unwrap, aes_key_wrap

from ...core.config import settings

MAGIC = b"EMLB1"


class IntegrityError(Exception):
    """The stored file does not open with its key: it was altered, swapped or truncated."""


def _master_key() -> bytes:
    path = settings.LEGAL_MASTER_KEY_FILE
    with open(path, "rb") as f:
        key = base64.b64decode(f.read().strip(), validate=True)
    if len(key) != 32:
        raise ValueError("master key must be 32 bytes")
    return key


def _path(matter_id: str, version_id: str, folder: str = "blobs") -> str:
    return os.path.join(settings.DATA_DIR, "legal", folder, matter_id, f"{version_id}.bin")


def store(matter_id: str, version_id: str, data: bytes, folder: str = "blobs") -> str:
    """Encrypt and write the file; returns the wrapped file key (hex) to keep with the version."""
    file_key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    sealed = AESGCM(file_key).encrypt(nonce, data, version_id.encode())
    path = _path(matter_id, version_id, folder)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(MAGIC + nonce + sealed)
    os.replace(tmp, path)
    return aes_key_wrap(_master_key(), file_key).hex()


def load(matter_id: str, version_id: str, wrapped_key: str, folder: str = "blobs") -> bytes:
    try:
        file_key = aes_key_unwrap(_master_key(), bytes.fromhex(wrapped_key))
        with open(_path(matter_id, version_id, folder), "rb") as f:
            raw = f.read()
        if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 12 + 16:
            raise IntegrityError
        nonce, sealed = raw[len(MAGIC):len(MAGIC) + 12], raw[len(MAGIC) + 12:]
        return AESGCM(file_key).decrypt(nonce, sealed, version_id.encode())
    except (InvalidTag, InvalidUnwrap, ValueError, FileNotFoundError) as e:
        raise IntegrityError from e


def remove(matter_id: str, version_id: str, folder: str = "blobs") -> None:
    try:
        os.remove(_path(matter_id, version_id, folder))
    except FileNotFoundError:
        pass
