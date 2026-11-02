"""Incarnation lease of a sampler island: fence a superseded incarnation.

A spot sampler island can outlive its own replacement: a VM that was taken for
preempted (stopped, unreachable, deleted from the console too late) may still
run while a new incarnation of the same island already writes. Before this
module the 120B path (the engine sampler, no syncer) had no fence at all; the
only isolation was the incarnation prefix of the sample ids (defects doc,
"Old sampler" of the D23 run).

The lease is one JSON record per sampler island,
``sample-buffer/leases/<sampler>.json``, in a store every incarnation and the
trainer read (default: the buffer store; ``spec["incarnation_lease"]["store"]``
names another one, e.g. a shared bucket when every VM has its own local
buffer store)::

    {"sampler", "incarnation", "claimed_at", "generation",
     "superseded": [incarnations this lease replaced, oldest first],
     "sealed": {incarnation: groups of it the claimer saw in its buffer}}

Sampler side (:class:`IncarnationLease`):

* :meth:`claim` at join: reads the previous lease, writes a new one naming this
  incarnation; the previous holder, every earlier holder and every incarnation
  with groups in this incarnation's buffer are ``superseded``;
* :meth:`poll` reads the lease; another holder -> this incarnation is fenced
  (sticky). The engine sampler polls it on a watcher thread (``poll_s``) and
  once more right before every group write (:meth:`check_before_write`, at most
  every ``write_check_s``), so a streaming round that writes several groups per
  call never writes one after it lost the lease by more than that interval;
  the island then leaves as ``LeaveReason.FENCED``.

Buffer side (:class:`IncarnationGate`, the trainer's buffer feed): an entry of a
*superseded* incarnation is stale unless it was written before the claim, i.e.
it is among the first ``sealed[incarnation]`` entries of that incarnation in
buffer order, or its ``appended_at`` is older than ``claimed_at`` minus a clock
margin. Entries of the holder, of an incarnation the lease does not know (a
newer one whose lease has not reached the trainer yet), and entries without a
round record are accepted. No lease: everything is accepted (older runs).

A read failure never fences (fail open on the sampler; the gate is the second
line). Nothing here talks to a cloud.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

LEASE_DIR = "sample-buffer/leases"
FENCED_EVENT = "rl_sampler_fenced"
LEASE_EVENT = "rl_sampler_lease"
STALE_REASON = "stale_incarnation"
DEFAULT_POLL_S = 30.0
DEFAULT_WRITE_CHECK_S = 1.0
DEFAULT_CLOCK_MARGIN_S = 60.0


def lease_key(sampler: str) -> str:
    return f"{LEASE_DIR}/{sampler}.json"


def _put(store: Any, key: str, record: Mapping[str, Any]) -> None:
    from .sampler_resume import _put_record  # noqa: PLC0415 - one writer for both record kinds

    _put_record(store, key, dict(record))


def read_record(store: Any, key: str) -> dict[str, Any] | None:
    """The JSON record at ``key`` under the store root (an object store or a
    directory store, after its reload), None when absent or unreadable."""
    objects = getattr(store, "objects", None)
    try:
        if objects is not None:
            data = objects.read(key)
        elif callable(getattr(store, "read", None)):  # a bare ObjectStore
            data = store.read(key)
        else:
            reload = getattr(store, "reload", None)
            if callable(reload):
                reload()
            path = Path(store.root) / key
            data = path.read_bytes() if path.exists() else None
        if data is None:
            return None
        doc = json.loads(data)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def read_lease(store: Any, sampler: str) -> dict[str, Any] | None:
    return read_record(store, lease_key(sampler))


def sealed_counts(entries: Iterable[Any], sampler: str) -> dict[str, int]:
    """Groups per incarnation of ``sampler`` among buffer ``entries``."""
    counts: Counter[str] = Counter()
    for entry in entries:
        record = getattr(entry, "sampler_round", None)
        if record and record.get("sampler") == sampler and record.get("incarnation") is not None:
            counts[str(record["incarnation"])] += 1
    return dict(counts)


class IncarnationLease:
    """One incarnation's side of the lease (module doc)."""

    def __init__(self, store: Any, sampler: str, incarnation: str, *,
                 write_check_s: float = DEFAULT_WRITE_CHECK_S,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.sampler = str(sampler)
        self.incarnation = str(incarnation)
        self.write_check_s = float(write_check_s)
        self.clock = clock
        self.monotonic = monotonic
        self.holder: str | None = None  # who the last read named
        self.fenced_by: str | None = None
        self.read_failures = 0
        self.reads = 0
        self._checked_at = -float("inf")
        self._lock = threading.Lock()

    @property
    def fenced(self) -> bool:
        return self.fenced_by is not None

    def claim(self, entries: Iterable[Any] = (), earlier: Iterable[str] = ()) -> dict[str, Any]:
        """Write the lease naming this incarnation (module doc); returns the record."""
        previous = read_lease(self.store, self.sampler) or {}
        superseded: list[str] = []
        for inc in [*previous.get("superseded", ()), previous.get("incarnation"), *earlier]:
            if inc is not None and str(inc) != self.incarnation and str(inc) not in superseded:
                superseded.append(str(inc))
        sealed = {k: v for k, v in sealed_counts(entries, self.sampler).items() if k != self.incarnation}
        for inc in sealed:
            if inc not in superseded:
                superseded.append(inc)
        record = {"sampler": self.sampler, "incarnation": self.incarnation, "claimed_at": self.clock(),
                  "generation": int(previous.get("generation") or 0) + 1,
                  "superseded": superseded, "sealed": sealed}
        _put(self.store, lease_key(self.sampler), record)
        self.holder = self.incarnation
        return record

    def poll(self) -> bool:
        """Read the lease; True once another incarnation holds it (sticky)."""
        with self._lock:
            self._checked_at = self.monotonic()
            if self.fenced_by is not None:
                return True
            self.reads += 1
            lease = read_lease(self.store, self.sampler)
            if lease is None:
                self.read_failures += 1
                return False
            self.holder = str(lease.get("incarnation"))
            if self.holder != self.incarnation:
                self.fenced_by = self.holder
            return self.fenced_by is not None

    def check_before_write(self) -> bool:
        """Before a group write: the cached verdict when read in the last
        ``write_check_s`` seconds, else a fresh :meth:`poll`."""
        if self.fenced_by is not None:
            return True
        if self.monotonic() - self._checked_at < self.write_check_s:
            return False
        return self.poll()

    def detail(self) -> dict[str, Any]:
        return {"fenced": "superseded", "superseded_by": self.fenced_by, "lease_reads": self.reads,
                "lease_read_failures": self.read_failures}


class IncarnationGate:
    """The buffer reader's stale-incarnation verdict (module doc).

    ``entries()``: the reader's entries in buffer order (``FileSampleBuffer.entries``);
    ``refresh_s``: how often the leases are read again (an unknown incarnation
    forces a read at most that often too)."""

    def __init__(self, store: Any, entries: Callable[[], list[Any]], *, refresh_s: float = 30.0,
                 clock_margin_s: float = DEFAULT_CLOCK_MARGIN_S,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.entries = entries
        self.refresh_s = float(refresh_s)
        self.clock_margin_s = float(clock_margin_s)
        self.monotonic = monotonic
        self._leases: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._seen: Counter[tuple[str, str]] = Counter()  # (sampler, incarnation) over positions < _upto
        self._upto = 0
        self.stale = 0

    def _lease(self, sampler: str, *, force: bool = False) -> dict[str, Any] | None:
        now = self.monotonic()
        cached = self._leases.get(sampler)
        if cached is None or now - cached[0] >= self.refresh_s or (force and now - cached[0] >= 1.0):
            cached = (now, read_lease(self.store, sampler))
            self._leases[sampler] = cached
        return cached[1]

    def _ordinal(self, position: int, key: tuple[str, str]) -> int:
        """1-based rank of the entry at ``position`` among its incarnation's entries."""
        if position < self._upto:  # read again (a reset cursor): count from the start
            self._seen.clear()
            self._upto = 0
        entries = self.entries()
        for entry in entries[self._upto:position + 1]:
            record = getattr(entry, "sampler_round", None) or {}
            if record.get("incarnation") is not None:
                self._seen[(str(record.get("sampler")), str(record["incarnation"]))] += 1
        self._upto = max(self._upto, min(position + 1, len(entries)))
        return self._seen[key]

    def verdict(self, entry: Any, position: int) -> str | None:
        """None: train it; else the reason it is stale (a detail string)."""
        record = getattr(entry, "sampler_round", None) or {}
        sampler, inc = record.get("sampler"), record.get("incarnation")
        if sampler is None or inc is None:
            return None
        sampler, inc = str(sampler), str(inc)
        ordinal = self._ordinal(int(position), (sampler, inc))
        lease = self._lease(sampler)
        if lease is None:
            return None
        if str(lease.get("incarnation")) != inc and inc not in lease.get("superseded", ()):
            lease = self._lease(sampler, force=True) or lease  # maybe a newer claim
        holder = str(lease.get("incarnation"))
        if holder == inc or inc not in [str(s) for s in lease.get("superseded", ())]:
            return None
        sealed = int((lease.get("sealed") or {}).get(inc, 0))
        if ordinal <= sealed:
            return None
        claimed = float(lease.get("claimed_at") or 0.0)
        appended = float(getattr(entry, "appended_at", 0.0) or 0.0)
        if appended and appended < claimed - self.clock_margin_s:
            return None
        self.stale += 1
        return (f"incarnation {inc} of {sampler} superseded by {holder} (generation "
                f"{lease.get('generation')}); entry {ordinal} of it, {sealed} sealed")
