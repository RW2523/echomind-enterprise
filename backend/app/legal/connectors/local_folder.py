"""A folder on this server as a document source (S1.11; for tests, pilots and air-gapped imports).

Permissions come from a sidecar next to each file: `<file>.acl.json` = {"users": ["alice", ...]}.
No sidecar, or one that is not valid, means no permissions: the file is not imported. The root
must be inside LEGAL_CONNECTOR_LOCAL_ROOTS; links (symlinks) are never followed, and nothing
outside the root is read.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
from typing import Iterable, Optional

from ...core.config import settings
from .base import ConnectorError, RemoteItem

SIDECAR = ".acl.json"


def allowed_roots() -> list[str]:
    return [os.path.realpath(r) for r in settings.LEGAL_CONNECTOR_LOCAL_ROOTS.split(":") if r.strip()]


def _inside(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.realpath(path), root]) == root


def root_allowed(root: str) -> bool:
    if not root or not os.path.isabs(root) or not os.path.isdir(root):
        return False
    return any(_inside(root, r) for r in allowed_roots())


def clean_folder(folder: str) -> Optional[str]:
    """A folder path relative to the root, or None if it is absolute or climbs out of it."""
    folder = (folder or "").strip().replace("\\", "/")
    if not folder or folder.startswith("/"):
        return None
    norm = posixpath.normpath(folder)
    if norm == ".." or norm.startswith("../") or "/../" in f"/{norm}/":
        return None
    return norm


class LocalFolderConnector:
    kind = "local_folder"

    def __init__(self, root: str):
        self.root = os.path.realpath(root)

    def check(self) -> None:
        if not root_allowed(self.root):
            raise ConnectorError("root_unavailable")

    def _dir(self, folder: str) -> str:
        f = clean_folder(folder)
        path = os.path.join(self.root, f) if f else None
        if not path or not os.path.isdir(path) or os.path.islink(path) or not _inside(path, self.root):
            raise ConnectorError("folder_unavailable")
        return path

    @staticmethod
    def _acl(path: str) -> Optional[list[str]]:
        side = path + SIDECAR
        if not os.path.isfile(side) or os.path.islink(side):
            return None
        try:
            with open(side, encoding="utf-8") as f:
                users = json.load(f).get("users")
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(users, list) or not all(isinstance(u, str) and u.strip() for u in users):
            return None
        return sorted({u.strip() for u in users})

    def list(self, folder: str) -> Iterable[RemoteItem]:
        self.check()
        base = self._dir(folder)
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not os.path.islink(os.path.join(dirpath, d))]
            for name in sorted(filenames):
                if name.startswith(".") or name.endswith(SIDECAR):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, base).replace(os.sep, "/")
                if os.path.islink(path) or not _inside(path, self.root):
                    yield RemoteItem(rel, name, "", None, unsafe=True)
                    continue
                with open(path, "rb") as f:
                    version = hashlib.sha256(f.read()).hexdigest()
                yield RemoteItem(rel, name, version, self._acl(path))

    def fetch(self, folder: str, remote_id: str) -> bytes:
        base = self._dir(folder)
        rel = clean_folder(remote_id)
        path = os.path.join(base, rel) if rel else ""
        if not rel or os.path.islink(path) or not _inside(path, self.root) or not os.path.isfile(path):
            raise ConnectorError("item_unavailable")
        with open(path, "rb") as f:
            return f.read()
