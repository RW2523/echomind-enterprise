"""The connector contract (S1.11).

A connector lists the documents under a folder of the source system and fetches their bytes.
Each item carries the people the source says may see it (`acl`); None means the source gave no
usable permissions — and then the item is not imported (fail closed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol


class ConnectorError(Exception):
    """The source could not be read (unreachable, misconfigured, path missing)."""


@dataclass(frozen=True)
class RemoteItem:
    remote_id: str                 # stable id within the linked folder (for folders: relative path)
    name: str                      # file name, for the title and type checks
    version: str                   # changes whenever the content changes
    acl: Optional[list[str]]       # usernames allowed by the source; None = unknown → not imported
    unsafe: bool = False           # e.g. a link pointing outside the configured root → skipped


class DmsConnector(Protocol):
    kind: str

    def check(self) -> None: ...

    def list(self, folder: str) -> Iterable[RemoteItem]: ...

    def fetch(self, folder: str, remote_id: str) -> bytes: ...
