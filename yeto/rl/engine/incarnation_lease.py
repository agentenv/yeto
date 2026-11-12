"""Incarnation lease of a sampler island: fence a superseded incarnation.

A spot sampler island can outlive its own replacement: a VM that was taken for
preempted (stopped, unreachable, deleted from the console too late) may still
run while a new incarnation of the same island already writes. Before this
module the 120B path (the engine sampler, no syncer) had no fence at all; the
only isolation was the incarnation prefix of the sample ids (an earlier spot
run saw an old sampler keep writing).

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
round record are accepted. No lease: everything is accepted (older runs). A
lease read that fails is not "no lease": the gate keeps the lease it read last
and reads again at the next verdict; a sampler whose lease it never read is
read again (``unread_retry_s`` apart, for at most ``unread_wait_s``) before any
of its entries gets a verdict, then :class:`LeaseUnreadable` is raised (nothing
of a possibly fenced incarnation is let through on a failed read).

A read failure never fences (fail open on the sampler; the gate is the second
line): the watcher and the write check count it and read again at their next
turn. Two bounds on that: a lease that could not be read for ``max_unread_s``
(default 60 s, two watcher polls) no longer allows a group write
(:meth:`IncarnationLease.check_before_write` refuses until a read succeeds
again: an incarnation that cannot know whether it was fenced does not write).
Inside that window a fenced incarnation whose reads fail can still write; only
the gate's clock comparison drops those groups. This is the weak point of a
fence that is a record the writer must read; a shard ownership record that the
write itself is conditioned on removes it. And
:meth:`IncarnationLease.claim` never writes over a lease it could not read (the
new record would name nothing ``superseded`` and the gate would train what the
fenced incarnation writes afterwards): it reads again a few times, then raises
:class:`LeaseUnreadable`. Nothing here talks to a cloud.
"""

from __future__ import annotations

import json
import os
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
DEFAULT_MAX_UNREAD_S = 60.0  # a lease unread for longer allows no group write (two watcher polls)
CLAIM_READ_TRIES = 4  # reads of the previous lease before a claim gives up
CLAIM_READ_BACKOFF_S = 0.5  # doubled after every failed read
GATE_UNREAD_RETRY_S = 2.0  # the trainer's gate: a lease never read so far is read again this often
GATE_UNREAD_WAIT_S = 600.0  # ... for at most this long, then LeaseUnreadable


class LeaseUnreadable(RuntimeError):
    """The lease record could not be read (a store error or a torn record), which
    is not the same as "there is no lease"."""


def lease_key(sampler: str) -> str:
    return f"{LEASE_DIR}/{sampler}.json"


def _put(store: Any, key: str, record: Mapping[str, Any]) -> None:
    from .sampler_resume import _put_record  # noqa: PLC0415 - one writer for both record kinds

    _put_record(store, key, dict(record))


def _load(store: Any, key: str) -> dict[str, Any] | None:
    objects = getattr(store, "objects", None)
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


def read_record(store: Any, key: str) -> dict[str, Any] | None:
    """The JSON record at ``key`` under the store root (an object store or a
    directory store, after its reload), None when absent or unreadable."""
    try:
        return _load(store, key)
    except (OSError, ValueError):
        return None


def load_record(store: Any, key: str) -> dict[str, Any] | None:
    """:func:`read_record` that tells "absent" (None) from "could not be read"
    (:class:`LeaseUnreadable`, whatever the store raised: an S3 client's errors
    are not ``OSError`` subclasses: a 5xx after its retries, an expired credential, no
    route to the endpoint, a body cut short)."""
    try:
        return _load(store, key)
    except Exception as error:  # noqa: BLE001 - any store error is "unread", never "absent"
        raise LeaseUnreadable(f"{key}: {type(error).__name__}: {error}") from error


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
                 monotonic: Callable[[], float] = time.monotonic,
                 max_unread_s: float | None = DEFAULT_MAX_UNREAD_S) -> None:
        """``max_unread_s``: the longest the lease may stay unread before a group
        write is refused (module doc); None: no bound."""
        self.store = store
        self.sampler = str(sampler)
        self.incarnation = str(incarnation)
        self.write_check_s = float(write_check_s)
        self.max_unread_s = None if max_unread_s is None else float(max_unread_s)
        self.clock = clock
        self.monotonic = monotonic
        self.holder: str | None = None  # who the last read named
        self.fenced_by: str | None = None
        self.read_failures = 0
        self.reads = 0
        self.unread_refusals = 0  # group writes refused because the lease stayed unread
        self._checked_at = -float("inf")
        self._read_at: float | None = None  # the last time the lease was read (or written by claim)
        self._lock = threading.Lock()

    @property
    def fenced(self) -> bool:
        return self.fenced_by is not None

    def _previous(self, tries: int, sleep: Callable[[float], None]) -> dict[str, Any]:
        """The lease a claim replaces ({}: none). Never {} for one that could not be read."""
        delay = CLAIM_READ_BACKOFF_S
        for attempt in range(max(1, int(tries))):
            try:
                return load_record(self.store, lease_key(self.sampler)) or {}
            except LeaseUnreadable as error:
                self.read_failures += 1
                if attempt + 1 >= max(1, int(tries)):
                    raise LeaseUnreadable(
                        f"lease of {self.sampler} not claimed for {self.incarnation}: the previous lease "
                        f"could not be read in {attempt + 1} tries ({error})") from error
                sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def claim(self, entries: Iterable[Any] = (), earlier: Iterable[str] = (), *,
              read_tries: int = CLAIM_READ_TRIES, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
        """Write the lease naming this incarnation (module doc); returns the record.
        :class:`LeaseUnreadable` when the previous lease could not be read in
        ``read_tries`` reads: nothing is written then."""
        previous = self._previous(read_tries, sleep)
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
        self._read_at = self.monotonic()
        return record

    def poll(self) -> bool:
        """Read the lease; True once another incarnation holds it (sticky)."""
        with self._lock:
            self._checked_at = self.monotonic()
            if self.fenced_by is not None:
                return True
            self.reads += 1
            try:
                lease = load_record(self.store, lease_key(self.sampler))
            except LeaseUnreadable:
                lease = None
            if lease is None:
                self.read_failures += 1
                return False
            self._read_at = self._checked_at
            self.holder = str(lease.get("incarnation"))
            if self.holder != self.incarnation:
                self.fenced_by = self.holder
            return self.fenced_by is not None

    def unread_s(self) -> float:
        """Seconds since the lease was last read (0.0 before the first read or claim)."""
        return 0.0 if self._read_at is None else max(0.0, self.monotonic() - self._read_at)

    def unknown(self) -> bool:
        """Not known to be fenced, but the lease stayed unread for longer than
        ``max_unread_s``: this incarnation cannot tell whether it still holds it."""
        return (self.fenced_by is None and self.max_unread_s is not None
                and self.unread_s() > self.max_unread_s)

    def check_before_write(self) -> bool:
        """Before a group write: the cached verdict when read in the last
        ``write_check_s`` seconds, else a fresh :meth:`poll`. True: do not write
        (fenced, sticky; or :meth:`unknown`, until a read succeeds again)."""
        if self.fenced_by is not None:
            return True
        if self.monotonic() - self._checked_at >= self.write_check_s and self.poll():
            return True
        if self.unknown():
            self.unread_refusals += 1
            return True
        return False

    def detail(self) -> dict[str, Any]:
        return {"fenced": "superseded", "superseded_by": self.fenced_by, "lease_reads": self.reads,
                "lease_read_failures": self.read_failures}

    def health(self) -> dict[str, Any]:
        """The read counters (the sampler's ``rl_sampler_store_read_failed`` / ``_ok`` events, ``what: lease``)."""
        return {"lease_reads": self.reads, "lease_read_failures": self.read_failures,
                "lease_unread_s": round(self.unread_s(), 1), "lease_unread_refusals": self.unread_refusals}


class IncarnationGate:
    """The buffer reader's stale-incarnation verdict (module doc).

    ``entries()``: the reader's entries in buffer order (``FileSampleBuffer.entries``);
    ``refresh_s``: how often the leases are read again (an unknown incarnation
    forces a read at most that often too). ``unread_retry_s`` / ``unread_wait_s``:
    a lease never read so far is read again that often, for at most that long
    (module doc)."""

    def __init__(self, store: Any, entries: Callable[[], list[Any]], *, refresh_s: float = 30.0,
                 clock_margin_s: float = DEFAULT_CLOCK_MARGIN_S,
                 monotonic: Callable[[], float] = time.monotonic,
                 unread_retry_s: float = GATE_UNREAD_RETRY_S, unread_wait_s: float = GATE_UNREAD_WAIT_S,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.store = store
        self.entries = entries
        self.refresh_s = float(refresh_s)
        self.clock_margin_s = float(clock_margin_s)
        self.monotonic = monotonic
        self.unread_retry_s, self.unread_wait_s = float(unread_retry_s), float(unread_wait_s)
        self.sleep = sleep
        self.read_failures = 0
        self._leases: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._seen: Counter[tuple[str, str]] = Counter()  # (sampler, incarnation) over positions < _upto
        self._upto = 0
        self.stale = 0

    def _lease(self, sampler: str, *, force: bool = False) -> dict[str, Any] | None:
        now = self.monotonic()
        cached = self._leases.get(sampler)
        if cached is None or now - cached[0] >= self.refresh_s or (force and now - cached[0] >= 1.0):
            waited = 0.0
            while True:
                try:
                    cached = (now, load_record(self.store, lease_key(sampler)))
                    break
                except LeaseUnreadable as error:
                    self.read_failures += 1
                    if cached is not None:  # the lease read last stands; read again unread_retry_s later
                        self._leases[sampler] = (now - max(0.0, self.refresh_s - self.unread_retry_s), cached[1])
                        return cached[1]
                    if waited >= self.unread_wait_s:
                        raise LeaseUnreadable(f"lease of {sampler} not read in {waited:.0f} s; its entries "
                                              f"get no verdict on an unread lease ({error})") from error
                    self.sleep(self.unread_retry_s)
                    waited += self.unread_retry_s
                    now = self.monotonic()
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


INCARNATION_FENCE_ENV = "YETO_BUFFER_INCARNATION_FENCE"  # "0": no stale-incarnation gate
LEASE_ROOT_ENV = "YETO_INCARNATION_LEASE_ROOT"  # the store holding the sampler leases (default: the buffer's)


def default_incarnation_gate(buffer: Any, environ: Mapping[str, str] | None = None) -> Any:
    """The feed's stale-incarnation gate over the buffer's store (or
    :data:`LEASE_ROOT_ENV`); None when off (:data:`INCARNATION_FENCE_ENV` ``0``) or
    the buffer has no store / entries (test doubles)."""
    env = os.environ if environ is None else environ
    if str(env.get(INCARNATION_FENCE_ENV, "1")).strip() == "0":
        return None
    entries = getattr(buffer, "entries", None)
    store = getattr(buffer, "store", None)
    if env.get(LEASE_ROOT_ENV):
        from yeto.rl.engine.resume import store_for

        store = store_for(str(env[LEASE_ROOT_ENV]), environ=env)
    if store is None or not callable(entries):
        return None
    return IncarnationGate(store, entries)
