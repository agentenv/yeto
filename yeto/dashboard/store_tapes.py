"""Island tapes uploaded to the run store (``yeto.rl.engine.tape_upload``).

``--store s3://bucket/runs/<run>`` (or a local / shared store directory) makes
serve and export read every ``tapes/<island>/<incarnation>/<seq>.jsonl`` chunk:
islands on leased nodes appear without a per-node ``dashboard mirror``, and a
replacement node shows up with its first upload. Each (island, incarnation) is one
source; chunks are read once, in sequence order, and the offsets passed to the
reducer keep a re-read from double counting.

Request budget. An island adds a chunk every 30 s and none is deleted, so a listing
of all of ``tapes/`` grows by one request (1000 keys) every few hours of a run, on
every poll. :class:`StoreTapes` keeps, per (island, incarnation), the last chunk it
consumed and lists from there (``start_after``): a poll walks ``tapes/`` one page
at a time and, when a page ends inside a tape it already knows, jumps to that
tape's last consumed chunk. That is one request plus one per tape with more than a
page of chunks, whatever the age of the run; a new island or incarnation is found
by the same walk. ``interval_s`` spaces the polls of the store (the uploads are
30 s apart; ``yeto dashboard serve --store-poll``).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import time
from typing import Any, Callable

_log = logging.getLogger(__name__)

PREFIX = "tapes/"
_SEQ_SPAN = 1 << 32  # offset = seq * span + byte position inside the chunk
PAGE_KEYS = 1000  # keys of one listing request (S3 ListObjectsV2)
DEFAULT_STORE_POLL_S = 10.0  # ``serve``: seconds between two polls of the store


def tape_of(key: str, prefix: str = PREFIX) -> str | None:
    """``<prefix><island>/<incarnation>/`` of a chunk key; None for any other key."""
    if not key.startswith(prefix):
        return None
    parts = key[len(prefix):].split("/")
    return prefix + "/".join(parts[:2]) + "/" if len(parts) == 3 else None


# a trainer launch's tape folder ends with its start (tape_upload: ``l<k>-<host>-<YYYYmmddTHHMMSSZ>``)
_STARTED = re.compile(r"-(\d{8}T\d{6}Z)$")
_UNDATED_LOGGED: set[str] = set()


def launch_started(incarnation: str) -> float | None:
    """Epoch seconds of the start in a launch's tape folder name (``...-20261125T230755Z``);
    None when the name carries none (a sampler's folder) or a malformed one."""
    m = _STARTED.search(str(incarnation))
    if m is None:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def _launch_ordered(keys: list[str], prefix: str) -> list[str]:
    """``keys`` sorted as before (by name), except that an island's incarnations come in the
    order they started (their folder names' UTC stamps; one second apart or less: by name), so
    the newest launch is read last and its records are the island's current state: by name, a
    move to another host can sort its new folder first (``l0-computeinstance-...`` before
    ``l0-modal-...``). An island with one incarnation (or none dated) reads exactly as before."""
    by_island: dict[str, set[str]] = {}
    for k in keys:
        parts = k[len(prefix):].split("/") if k.startswith(prefix) else []
        if len(parts) == 3:
            by_island.setdefault(parts[0], set()).add(parts[1])
    rank: dict[tuple[str, str], tuple] = {}
    for island, incs in by_island.items():
        if len(incs) < 2:
            continue
        started = {i: launch_started(i) for i in incs}
        if any(v is None for v in started.values()):
            if island not in _UNDATED_LOGGED:
                _UNDATED_LOGGED.add(island)
                _log.debug("store tapes: island %s has incarnation folders without a start stamp (%s): read by name",
                           island, sorted(i for i, v in started.items() if v is None))
            continue
        for pos, inc in enumerate(sorted(incs, key=lambda i: (started[i], i))):
            rank[(island, inc)] = (pos,)
    if not rank:
        return sorted(keys)

    def key(k: str) -> tuple:
        parts = k[len(prefix):].split("/") if k.startswith(prefix) else []
        if len(parts) == 3 and (parts[0], parts[1]) in rank:
            return (prefix + parts[0] + "/", rank[(parts[0], parts[1])], parts[2])
        return (k, (), "")
    return sorted(keys, key=key)


def open_objects(store: str, environ: Any = None) -> Any:
    """The object API for ``store``: an ``s3://`` URI (credentials from the
    environment, as for the launcher) or a directory path."""
    if "://" in store:
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        return head_objects(store, environ=environ)
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

    return DirectoryObjectStore(os.path.expanduser(store))


class StoreTapes:
    def __init__(self, objects: Any, *, label: str = "store", interval_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, page_keys: int = PAGE_KEYS,
                 prefix: str = PREFIX) -> None:
        self.objects = objects
        self.label = label
        # ``tapes/`` of the run store; ``trainer<k>/tapes/`` for the samplers of a fleet island
        # (they upload under their island's store lane, fleet_islands.PlanFile)
        self.prefix = prefix
        self.interval_s = float(interval_s)
        self.page_keys = int(page_keys)
        self._clock = clock
        self._next_poll: float | None = None
        # tape prefix -> its last consumed chunk key: every chunk up to it is done with
        self.cursors: dict[str, str] = {}
        self.seen: set[str] = set()  # consumed keys that belong to no tape (never read)
        self.sources: set[str] = set()
        self.errors = 0
        self.last_error: str | None = None
        self.requests = 0  # listing requests so far
        # health of the store reads, for the page and the alerts (``status``): failures since
        # the last poll that read the store cleanly, and the wall clock of that poll
        self.failing = 0
        self.last_ok_at: float | None = None
        self.last_error_at: float | None = None

    def _done(self, key: str) -> bool:
        tape = tape_of(key, self.prefix)
        return key in self.seen if tape is None else key <= self.cursors.get(tape, "")

    def _new_keys(self) -> list[str]:
        """The chunk keys not consumed yet, sorted (module doc, request budget)."""
        page = getattr(self.objects, "list_page", None)
        if not callable(page):  # an object API without paged listings: everything, every poll
            self.requests += 1
            return _launch_ordered([k for k in self.objects.list(self.prefix) if k.endswith(".jsonl") and not self._done(k)],
                                   self.prefix)
        new: list[str] = []
        cursor = ""
        while True:
            keys, last = page(self.prefix, start_after=cursor, limit=self.page_keys)
            self.requests += 1
            new += [k for k in keys if k.endswith(".jsonl") and not self._done(k)]
            if last is None:
                return _launch_ordered(new, self.prefix)
            # the page ended inside a known tape: the rest of it up to its cursor is consumed
            cursor = max(last, self.cursors.get(tape_of(last, self.prefix) or "", ""))

    def pump(self, reducer) -> int:
        now = self._clock()
        if self._next_poll is not None and now < self._next_poll:
            return 0
        self._next_poll = now + self.interval_s
        try:
            keys = self._new_keys()
        except Exception as error:  # noqa: BLE001 - a store hiccup must not stop the page
            self._failed(error)
            return 0
        n, blocked = 0, set()
        for key in keys:
            parts = key[len(self.prefix):].split("/")
            if len(parts) != 3:
                self.seen.add(key)
                continue
            island, inc, chunk = parts
            if (island, inc) in blocked:
                continue  # an earlier chunk of this tape failed: keep the order
            try:
                seq = int(chunk.split(".")[0])
                data = self.objects.read(key)
            except Exception as error:  # noqa: BLE001
                self._failed(error)
                blocked.add((island, inc))
                continue  # retried at the next poll
            self.cursors[tape_of(key, self.prefix)] = key  # keys come sorted: the cursor only moves forward
            if data is None:
                continue
            source = f"{self.label}:{island}/{inc}"
            self.sources.add(source)
            pos = 0
            for line in data.splitlines(keepends=True):
                pos += len(line)
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(rec, dict) and reducer.feed(rec, source=source, offset=seq * _SEQ_SPAN + pos):
                    n += 1
        if not blocked:
            self.failing = 0
            self.last_ok_at = time.time()
        return n

    def _failed(self, error: BaseException) -> None:
        self.errors += 1
        self.failing += 1
        self.last_error = f"{type(error).__name__}: {error}"[:300]
        self.last_error_at = time.time()

    def status(self) -> dict:
        """What the page and the alerts show about the store reads: ``failing`` counts the
        failures since the last clean poll (0: the store reads fine, or was never polled)."""
        return {"label": self.label, "uri": getattr(self.objects, "uri", None), "sources": len(self.sources),
                "requests": self.requests, "errors": self.errors, "failing": self.failing,
                "last_error": self.last_error, "last_error_at": self.last_error_at,
                "last_ok_at": self.last_ok_at, "polled": self._next_poll is not None}
