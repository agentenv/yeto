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

import json
import os
import time
from typing import Any, Callable

PREFIX = "tapes/"
_SEQ_SPAN = 1 << 32  # offset = seq * span + byte position inside the chunk
PAGE_KEYS = 1000  # keys of one listing request (S3 ListObjectsV2)
DEFAULT_STORE_POLL_S = 10.0  # ``serve``: seconds between two polls of the store


def tape_of(key: str) -> str | None:
    """``tapes/<island>/<incarnation>/`` of a chunk key; None for any other key."""
    parts = key.split("/")
    return "/".join(parts[:3]) + "/" if len(parts) == 4 and key.startswith(PREFIX) else None


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
                 clock: Callable[[], float] = time.monotonic, page_keys: int = PAGE_KEYS) -> None:
        self.objects = objects
        self.label = label
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

    def _done(self, key: str) -> bool:
        tape = tape_of(key)
        return key in self.seen if tape is None else key <= self.cursors.get(tape, "")

    def _new_keys(self) -> list[str]:
        """The chunk keys not consumed yet, sorted (module doc, request budget)."""
        page = getattr(self.objects, "list_page", None)
        if not callable(page):  # an object API without paged listings: everything, every poll
            self.requests += 1
            return sorted(k for k in self.objects.list(PREFIX) if k.endswith(".jsonl") and not self._done(k))
        new: list[str] = []
        cursor = ""
        while True:
            keys, last = page(PREFIX, start_after=cursor, limit=self.page_keys)
            self.requests += 1
            new += [k for k in keys if k.endswith(".jsonl") and not self._done(k)]
            if last is None:
                return sorted(new)
            # the page ended inside a known tape: the rest of it up to its cursor is consumed
            cursor = max(last, self.cursors.get(tape_of(last) or "", ""))

    def pump(self, reducer) -> int:
        now = self._clock()
        if self._next_poll is not None and now < self._next_poll:
            return 0
        self._next_poll = now + self.interval_s
        try:
            keys = self._new_keys()
        except Exception as error:  # noqa: BLE001 - a store hiccup must not stop the page
            self.errors += 1
            self.last_error = f"{type(error).__name__}: {error}"[:300]
            return 0
        n, blocked = 0, set()
        for key in keys:
            parts = key[len(PREFIX):].split("/")
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
                self.errors += 1
                self.last_error = f"{type(error).__name__}: {error}"[:300]
                blocked.add((island, inc))
                continue  # retried at the next poll
            self.cursors[tape_of(key)] = key  # keys come sorted: the cursor only moves forward
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
        return n
