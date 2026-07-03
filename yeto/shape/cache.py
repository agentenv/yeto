"""Disk-backed TTL cache for slow cloud-API lookups.

Quota and spot-placement-score queries take seconds each and their answers
change on the order of hours, so `yeto shape` caches them on disk: repeated
invocations (and the interactive re-shaping loop) stay fast without hammering
the AWS APIs. The cache is a single JSON file so it survives across processes
and is trivially inspectable/deletable by the user.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

_DEFAULT_PATH = Path.home() / ".cache" / "yeto" / "shape-cache.json"


def _now() -> float:
    """Indirection over time.time() so tests can monkeypatch the clock."""
    return time.time()


class TTLCache:
    """JSON-file cache mapping string keys to (timestamp, value) pairs.

    Values must be JSON-serializable (callers guarantee this). Concurrent
    misses on the same key are deduplicated: only one thread runs the fetch,
    the rest block on a per-key lock and read the freshly cached result —
    important because callers fan out over a thread pool and the underlying
    fetches are expensive API calls.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        ttl: float = 3600.0,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self.ttl = ttl
        self.enabled = enabled
        self._lock = threading.Lock()  # guards _data, _loaded, _key_locks, file I/O
        self._key_locks: dict[str, threading.Lock] = {}
        self._data: dict[str, list] = {}  # key -> [unix_ts, value]
        self._loaded = False

    def get_or(self, key: str, fetch: Callable[[], Any]) -> Any:
        """Return the cached value for `key`, or fetch, cache, and return it.

        A raising fetch propagates and caches nothing, so transient API
        failures are retried on the next call rather than pinned for a TTL.
        """
        if not self.enabled:
            return fetch()

        with self._lock:
            self._ensure_loaded()
            hit, value = self._lookup(key)
            if hit:
                return value
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # Run the fetch outside the global lock so unrelated keys proceed in
        # parallel; the per-key lock ensures a single fetch per missing key.
        with key_lock:
            with self._lock:
                hit, value = self._lookup(key)
                if hit:  # another thread fetched while we waited
                    return value
            value = fetch()
            with self._lock:
                self._data[key] = [_now(), value]
                self._save()
            return value

    def clear(self) -> None:
        """Drop all entries in memory and delete the file on disk."""
        with self._lock:
            self._data = {}
            self._loaded = True
            try:
                self.path.unlink()
            except OSError:
                pass

    def _lookup(self, key: str) -> tuple[bool, Any]:
        """Return (hit, value) for a fresh entry. Caller holds `_lock`."""
        entry = self._data.get(key)
        if entry is not None and _now() - entry[0] < self.ttl:
            return True, entry[1]
        return False, None

    def _ensure_loaded(self) -> None:
        """Lazily read the file once; a corrupt/missing file starts empty."""
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                self._data = raw
        except (OSError, ValueError):
            self._data = {}

    def _save(self) -> None:
        """Atomically persist fresh entries. Caller holds `_lock`.

        Expired entries are dropped at save time so the file does not grow
        without bound; the tmp-then-replace dance keeps a concurrent reader
        from ever seeing a half-written file.
        """
        now = _now()
        fresh = {k: v for k, v in self._data.items() if now - v[0] < self.ttl}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(fresh))
        os.replace(tmp, self.path)
