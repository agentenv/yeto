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
round record are accepted. No lease: everything is accepted (older runs).

A lease that is *absent* (the store answers "no such key") and one that could not
be *read* (a timeout, a 5xx, a truncated or garbled body) are told apart
(:func:`read_record_strict`, :class:`LeaseReadError`):

* on the sampler a read failure never fences (fail open; the gate is the second
  line), and :attr:`IncarnationLease.unknown` says the last read failed;
* :meth:`IncarnationLease.claim` does not write a lease over a previous one it
  could not read (it would forget whom it supersedes): it retries, then raises;
* the gate fails closed: :meth:`IncarnationGate.verdict` raises
  :class:`LeaseUnknown` for an entry whose sampler's lease it cannot read (the
  reader defers that group and judges it again later; it is neither trained nor
  counted stale), and a failed read is not cached as "no lease". Reads that keep
  failing are logged at most every ``warn_s`` seconds.

Nothing here talks to a cloud.
"""

from __future__ import annotations

import http.client
import json
import logging
import random
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
DEFAULT_RETRY_S = 1.0  # the gate reads a lease that failed again at most this often
DEFAULT_WARN_S = 60.0  # ... and logs a lease it keeps failing to read at most this often
CLAIM_READ_ATTEMPTS = 3
CLAIM_PAUSE_S = 2.0  # the first pause between the claim's reads (doubled after each)

# what a store read raises when it could not read (not: the key is absent). An
# object store over S3 / R2 raises botocore's own types (ClientError: a 5xx, a
# 403; BotoCoreError: EndpointConnectionError, ConnectionClosedError,
# IncompleteReadError) and urllib3's / http.client's, most of them no OSError
_READ_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError, http.client.HTTPException)
try:
    from botocore.exceptions import BotoCoreError, ClientError

    _READ_ERRORS += (BotoCoreError, ClientError)
except ImportError:  # pragma: no cover - a store without boto3
    pass
try:
    from urllib3.exceptions import HTTPError as _Urllib3Error

    _READ_ERRORS += (_Urllib3Error,)
except ImportError:  # pragma: no cover
    pass
# the error codes of an absent key (a store that answers with a bare ClientError)
_ABSENT_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def _error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return ""
    return str((response.get("Error") or {}).get("Code", ""))


def _claim_pause(attempt: int, sleep: Callable[[float], None] | None = None) -> None:
    """Wait before read ``attempt + 2`` of the previous lease (exponential, jittered)."""
    (sleep or time.sleep)(random.uniform(0.5, 1.0) * CLAIM_PAUSE_S * 2.0 ** attempt)

log = logging.getLogger(__name__)


class LeaseReadError(OSError):
    """A lease record that exists or may exist but could not be read (a timeout,
    a 5xx, a truncated or garbled body): unlike an absent one it is no verdict."""


class LeaseUnknown(Exception):
    """:meth:`IncarnationGate.verdict` could not read the entry's lease: the entry
    is neither trainable nor stale yet (the caller defers it)."""


def lease_key(sampler: str) -> str:
    return f"{LEASE_DIR}/{sampler}.json"


def _put(store: Any, key: str, record: Mapping[str, Any]) -> None:
    from .sampler_resume import _put_record  # noqa: PLC0415 - one writer for both record kinds

    _put_record(store, key, dict(record))


def read_record_strict(store: Any, key: str) -> dict[str, Any] | None:
    """The JSON record at ``key`` under the store root (an object store or a
    directory store, after its reload); None only when it is absent. Raises
    :class:`LeaseReadError` when it could not be read: the store raised (a
    timeout, a 5xx, a reset; botocore's types included, :data:`_READ_ERRORS`) or
    the body is not a JSON object (truncated, garbled). An object store answers a
    missing key with None; a bare ``ClientError`` naming 404 / NoSuchKey /
    NotFound is absent too."""
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
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                data = None
        if data is None:
            return None
        doc = json.loads(data)
    except _READ_ERRORS as error:
        if _error_code(error) in _ABSENT_CODES:
            return None
        raise LeaseReadError(f"{key}: {type(error).__name__}: {error}") from error
    if not isinstance(doc, dict):
        raise LeaseReadError(f"{key}: not a JSON object ({type(doc).__name__})")
    return doc


def read_record(store: Any, key: str) -> dict[str, Any] | None:
    """:func:`read_record_strict`, None also when unreadable (callers that treat
    both alike)."""
    try:
        return read_record_strict(store, key)
    except LeaseReadError:
        return None


def read_lease(store: Any, sampler: str, *, strict: bool = False) -> dict[str, Any] | None:
    """The sampler's lease; None when absent (or unreadable, unless ``strict``:
    then :class:`LeaseReadError`)."""
    return (read_record_strict if strict else read_record)(store, lease_key(sampler))


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
                 sleep: Callable[[float], None] | None = None) -> None:
        self.store = store
        self.sampler = str(sampler)
        self.incarnation = str(incarnation)
        self.write_check_s = float(write_check_s)
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep  # the claim's pause between its reads (None: time.sleep)
        self.holder: str | None = None  # who the last read named
        self.fenced_by: str | None = None
        self.read_failures = 0
        self.reads = 0
        self.unknown = False  # the last read of the lease failed (no verdict)
        self._checked_at = -float("inf")
        self._lock = threading.Lock()

    @property
    def fenced(self) -> bool:
        return self.fenced_by is not None

    def claim(self, entries: Iterable[Any] = (), earlier: Iterable[str] = ()) -> dict[str, Any]:
        """Write the lease naming this incarnation (module doc); returns the record.
        A previous lease that cannot be read (:class:`LeaseReadError` after
        :data:`CLAIM_READ_ATTEMPTS` reads, paused by :func:`_claim_pause`) is not overwritten: the new record would
        forget the incarnations it supersedes, so the error is raised."""
        for attempt in range(CLAIM_READ_ATTEMPTS):
            try:
                previous = read_lease(self.store, self.sampler, strict=True) or {}
                break
            except LeaseReadError:
                if attempt + 1 == CLAIM_READ_ATTEMPTS:
                    raise
                _claim_pause(attempt, self.sleep)
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
            try:
                lease = read_lease(self.store, self.sampler, strict=True)
            except LeaseReadError:
                lease = None
            self.unknown = lease is None
            if lease is None:  # unreadable, or absent (deleted): no verdict, fail open
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


DEFAULT_CLAIM_PATIENCE_S = 600.0


def claim_patiently(lease: IncarnationLease, entries: Callable[[], Iterable[Any]], earlier: Iterable[str] = (),
                    *, patience_s: float = DEFAULT_CLAIM_PATIENCE_S, should_stop: Callable[[], bool] = lambda: False,
                    monotonic: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any] | None:
    """:meth:`IncarnationLease.claim` for a sampler at its join: a previous lease the
    store cannot read for a while (an R2 outage) is waited for, up to ``patience_s``
    seconds, rather than ending the process at once (a sampler that dies at its
    start is a failed node to the pool). None when ``should_stop`` (a reclaim, a
    stop) ends the wait; :class:`LeaseReadError` once the patience is spent."""
    earlier = list(earlier)
    started = monotonic()
    attempt = 0
    while True:
        try:
            return lease.claim(entries(), earlier=earlier)
        except LeaseReadError as error:
            waited = monotonic() - started
            if waited >= float(patience_s):
                raise
            log.warning("incarnation lease of %s unreadable at the claim (%.0f s): %s", lease.sampler, waited, error)
        if should_stop():
            return None
        sleep(min(30.0, CLAIM_PAUSE_S * 2.0 ** min(attempt, 4)))
        attempt += 1
        if should_stop():
            return None


class IncarnationGate:
    """The buffer reader's stale-incarnation verdict (module doc).

    ``entries()``: the reader's entries in buffer order (``FileSampleBuffer.entries``);
    ``refresh_s``: how often the leases are read again (an unknown incarnation
    forces a read at most that often too). A lease read that failed is not cached:
    :meth:`verdict` raises :class:`LeaseUnknown` and the lease is read again on a
    later verdict, at most every ``retry_s`` seconds; while the reads keep failing
    a warning is logged at most every ``warn_s`` seconds."""

    def __init__(self, store: Any, entries: Callable[[], list[Any]], *, refresh_s: float = 30.0,
                 clock_margin_s: float = DEFAULT_CLOCK_MARGIN_S, retry_s: float = DEFAULT_RETRY_S,
                 warn_s: float = DEFAULT_WARN_S,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.entries = entries
        self.refresh_s = float(refresh_s)
        self.clock_margin_s = float(clock_margin_s)
        self.retry_s = float(retry_s)
        self.warn_s = float(warn_s)
        self.monotonic = monotonic
        self._leases: dict[str, tuple[float, dict[str, Any] | None]] = {}
        # sampler -> (first failure, last failed read, last warning) while its reads fail
        self._failing: dict[str, list[float]] = {}
        self._seen: Counter[tuple[str, str]] = Counter()  # (sampler, incarnation) over positions < _upto
        self._upto = 0
        self.stale = 0
        self.deferred = 0  # verdicts that raised LeaseUnknown
        self.read_failures = 0

    def _read(self, sampler: str, now: float) -> dict[str, Any] | None:
        """Read the lease (cached on success); :class:`LeaseUnknown` on a failure."""
        failing = self._failing.get(sampler)
        if failing is not None and now - failing[1] < self.retry_s:
            raise LeaseUnknown(f"lease of {sampler} unreadable (retried every {self.retry_s:g} s)")
        try:
            lease = read_lease(self.store, sampler, strict=True)
        except LeaseReadError as error:
            self.read_failures += 1
            if failing is None:
                failing = self._failing[sampler] = [now, now, -float("inf")]
            failing[1] = now
            if now - failing[2] >= self.warn_s:
                failing[2] = now
                log.warning("incarnation gate: lease of %s unreadable for %.0f s (%d failed reads); its "
                            "groups wait in the buffer until it is read: %s", sampler, now - failing[0],
                            self.read_failures, error)
            raise LeaseUnknown(str(error)) from error
        if failing is not None:
            del self._failing[sampler]
            log.warning("incarnation gate: lease of %s read again after %.0f s", sampler, now - failing[0])
        self._leases[sampler] = (now, lease)
        return lease

    def _lease(self, sampler: str, *, force: bool = False) -> dict[str, Any] | None:
        now = self.monotonic()
        cached = self._leases.get(sampler)
        if cached is None or now - cached[0] >= self.refresh_s or (force and now - cached[0] >= 1.0):
            return self._read(sampler, now)
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
        """None: train it; else the reason it is stale (a detail string). Raises
        :class:`LeaseUnknown` when the entry's lease cannot be read (fail closed:
        the caller neither trains the entry nor counts it stale, and asks again)."""
        record = getattr(entry, "sampler_round", None) or {}
        sampler, inc = record.get("sampler"), record.get("incarnation")
        if sampler is None or inc is None:
            return None
        sampler, inc = str(sampler), str(inc)
        ordinal = self._ordinal(int(position), (sampler, inc))
        try:
            lease = self._lease(sampler)
            if lease is not None and str(lease.get("incarnation")) != inc and inc not in lease.get("superseded", ()):
                lease = self._lease(sampler, force=True) or lease  # maybe a newer claim
        except LeaseUnknown:
            self.deferred += 1
            raise
        if lease is None:
            return None
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
