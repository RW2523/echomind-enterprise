"""Loaded partitions, least recently used first out (LEGAL_PARTITION_CACHE)."""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict

from ...core.config import settings
from .partition_index import PartitionIndex

_lock = threading.Lock()
_cache: "OrderedDict[tuple[str, str], PartitionIndex]" = OrderedDict()


def partition_for(matter_id: str) -> str:
    return f"m:{matter_id}"


def dirname(partition: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", partition)


def _root() -> str:
    return os.path.join(settings.DATA_DIR, "legal", "partitions")


def for_matter(matter_id: str) -> PartitionIndex:
    partition = partition_for(matter_id)
    key = (_root(), partition)
    with _lock:
        idx = _cache.get(key)
        if idx is not None and not idx.files_changed():
            _cache.move_to_end(key)
            return idx
        _cache.pop(key, None)
    idx = PartitionIndex.open(partition, matter_id, os.path.join(key[0], dirname(partition)))
    with _lock:
        idx = _cache.setdefault(key, idx)
        _cache.move_to_end(key)
        while len(_cache) > max(1, settings.LEGAL_PARTITION_CACHE):
            _cache.popitem(last=False)
    return idx


def clear() -> None:
    with _lock:
        _cache.clear()
