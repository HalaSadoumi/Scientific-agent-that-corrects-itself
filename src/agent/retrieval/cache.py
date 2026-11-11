"""Small process-local TTL cache for retrieval results."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any
from uuid import UUID

_MAX_ENTRIES = 512
_TTL_SECONDS = 5
_lock = Lock()
_items: OrderedDict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = OrderedDict()


def get(session_id: UUID, query: str, limit: int) -> list[dict[str, Any]] | None:
    key = (str(session_id), query.strip().casefold(), limit)
    with _lock:
        entry = _items.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires <= time.monotonic():
            _items.pop(key, None)
            return None
        _items.move_to_end(key)
        return [dict(row) for row in value]


def put(session_id: UUID, query: str, limit: int, hits: list[dict[str, Any]]) -> None:
    key = (str(session_id), query.strip().casefold(), limit)
    with _lock:
        _items[key] = (time.monotonic() + _TTL_SECONDS, [dict(row) for row in hits])
        _items.move_to_end(key)
        while len(_items) > _MAX_ENTRIES:
            _items.popitem(last=False)


def invalidate(session_id: UUID) -> None:
    prefix = str(session_id)
    with _lock:
        for key in [key for key in _items if key[0] == prefix]:
            _items.pop(key, None)
