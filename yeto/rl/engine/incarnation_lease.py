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
  reader defers that group, and in strict buffer order every group behind it,
  and judges them again later; they are neither trained nor counted stale: the
  training pauses), and a failed read is not cached as "no lease". Reads that
  keep failing are logged ("training paused: lease of shard X unreadable since
  T") at most every ``warn_s`` seconds.

Shard ownership records (``sample_buffer`` module doc) replace the lease JSON as
the fence wherever the sampler's shard can hold them - a shard on an object
store, in a run whose trainer reads them (``TrainerLease.current(objects) >= 1``).
There :meth:`IncarnationLease.claim` appends the shard's ownership record
(``FileSampleBuffer.claim_owner``), :meth:`~IncarnationLease.poll` is an
incremental read of the shard index, and the write itself refuses
(``ShardFenced``) once another owner's record is in the shard: no clock and no
check-then-write window. The lease JSON is still written (observation, older
readers) but decides nothing, and the trainer's :class:`IncarnationGate` judges
only the records of a shard's legacy part (before its first ownership record).
Everywhere else (a shared-directory buffer, the syncer data path, a run whose
trainer is older) the lease JSON and the gate work as described above. A sampler
that joined before the run's trainer claimed its generation lease starts under
the lease JSON and moves to ownership records at a later :meth:`~IncarnationLease.poll`
(``adopted``), unless its shard was fenced meanwhile.

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
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .object_store import is_absent_store_error

LEASE_DIR = "sample-buffer/leases"
FENCED_EVENT = "rl_sampler_fenced"
LEASE_EVENT = "rl_sampler_lease"
STALE_REASON = "stale_incarnation"
DEFAULT_POLL_S = 30.0
DEFAULT_WRITE_CHECK_S = 1.0
DEFAULT_CLOCK_MARGIN_S = 60.0
# a lease record field: its incarnation's shard writer reads ownership records
READS_OWNER_RECORDS = "reads_owner_records"
DEFAULT_RETRY_S = 1.0  # the gate reads a lease that failed again at most this often
DEFAULT_WARN_S = 60.0  # ... and logs a lease it keeps failing to read at most this often
CLAIM_READ_ATTEMPTS = 3
DEFAULT_MAX_UNREAD_S = 60.0  # a lease unread for longer allows no group write (two watcher polls)
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

# consecutive failed reads of the shard index after which an owning incarnation
# stops trusting its last verdict (the write itself reads the index and fails closed)
DEFAULT_MAX_READ_FAILURES = 5


def owner_records_enabled(buffer: Any) -> bool:
    """Whether a sampler writes shard ownership records into ``buffer``: its shard
    can hold them (``FileSampleBuffer.can_own``) and the run's trainer reads them
    (a trainer generation lease exists; an older trainer fails on the record)."""
    if buffer is None or not getattr(buffer, "can_own", False):
        return False
    from .trainer_lease import TrainerLease  # noqa: PLC0415

    return TrainerLease.current(buffer.index.objects) >= 1


def fence_shard_owner(store: Any, shard: str, incarnation: str, *, by: str = "pool",
                      clock: Callable[[], float] = time.time) -> tuple[Any, bool] | None:
    """Fence ``shard`` from outside its sampler (the pool, an operator): append an
    ownership record naming ``incarnation`` (``pool-fence:<ticket>``) to its index,
    so every group its sampler created before stays valid and nothing it writes
    after is (``sample_buffer`` module doc). ``(owner, created)`` as
    ``sample_buffer.claim_shard_owner``; None where the shard holds no ownership
    records: the store has no object API, the shard has no index of its own, or
    the run's trainer is older (no trainer lease), or the shard holds no ownership
    record yet and the lease record's holder did not say that it reads them
    (:data:`READS_OWNER_RECORDS`; older code would stop on the record instead of
    continuing the shard) - the lease record alone fences then.

    The shard's end is found from the lease record's ``owner`` hint (the position of
    the last ownership record a claim wrote, checked against the shard): one LIST
    after it, not of the whole shard."""
    from .sample_buffer import (  # noqa: PLC0415
        BUFFER_DIR,
        SHARD_NAMES,
        ObjectIndex,
        ShardTail,
        _owner_of,
        _shard_name,
        claim_shard_owner,
        is_owner_record,
        shard_prefix,
    )
    from .trainer_lease import TrainerLease  # noqa: PLC0415

    objects = getattr(store, "objects", None)
    if objects is None or not callable(getattr(objects, "create", None)):
        return None
    if not objects.exists(f"{BUFFER_DIR}/{SHARD_NAMES}/{_shard_name(shard)}"):
        return None
    if TrainerLease.current(objects) < 1:
        return None
    tail = ShardTail()
    lease = read_lease(store, shard) or {}
    hint = lease.get("owner")
    if isinstance(hint, Mapping) and hint.get("pos") is not None:
        try:
            raw = objects.read(ObjectIndex(objects, prefix=shard_prefix(shard)).key(int(hint["pos"])))
            record = None if raw is None else json.loads(raw)
            if isinstance(record, dict) and is_owner_record(record) and int(record["pos"]) == int(hint["pos"]):
                tail = ShardTail(next=int(hint["pos"]) + 1, owner=_owner_of(record))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a hint only: the whole shard is listed
    return claim_shard_owner(objects, shard, incarnation, by=by, clock=clock, tail=tail,
                             only_owned=not lease.get(READS_OWNER_RECORDS))


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
        if is_absent_store_error(error):
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
    """One incarnation's side of the lease (module doc).

    ``buffer``: the sampler's shard writer. When it can hold ownership records
    (:func:`owner_records_enabled` at :meth:`claim`), the shard index is the fence
    (``owned``) and the lease JSON is written for observation only."""

    def __init__(self, store: Any, sampler: str, incarnation: str, *,
                 write_check_s: float = DEFAULT_WRITE_CHECK_S,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] | None = None,
                 max_unread_s: float | None = DEFAULT_MAX_UNREAD_S,
                 buffer: Any = None, by: str = "sampler",
                 max_read_failures: int = DEFAULT_MAX_READ_FAILURES) -> None:
        """``max_unread_s``: the longest the lease may stay unread before a group
        write is refused (:meth:`unread_too_long`); None: no bound. Owned (shard
        ownership records), the bound is ``max_read_failures`` failed reads of the
        shard index in a row instead."""
        self.store = store
        self.max_unread_s = None if max_unread_s is None else float(max_unread_s)
        self.unread_refusals = 0  # group writes refused because the lease stayed unread
        self._read_at: float | None = None  # the last time the lease was read (or written by claim)
        self.sampler = str(sampler)
        self.incarnation = str(incarnation)
        self.write_check_s = float(write_check_s)
        self.clock = clock
        self.monotonic = monotonic
        self.buffer = buffer
        self.by = str(by)
        self.max_read_failures = int(max_read_failures)
        self.owned = False  # the shard's ownership records decide, not the lease JSON
        self.adopted = False  # owned since a poll after the join (the trainer's lease came later)
        self._record: dict[str, Any] | None = None  # the lease record this incarnation wrote
        self.owner: Any = None  # this incarnation's ownership record (owned)
        self.sleep = sleep  # the claim's pause between its reads (None: time.sleep)
        self.holder: str | None = None  # who the last read named
        self.fenced_by: str | None = None
        self.read_failures = 0
        self.failures_in_a_row = 0
        self.reads = 0
        self.unknown = False  # the last read of the lease failed (no verdict)
        self._checked_at = -float("inf")
        self._lock = threading.Lock()

    @property
    def fenced(self) -> bool:
        self._sync()
        return self.fenced_by is not None

    def _sync(self) -> None:
        """Owned: take the verdict of the shard writer's last read (a write that
        raised ``ShardFenced`` has fenced it without a poll)."""
        if self.owned and self.fenced_by is None and self.buffer.fenced:
            self.holder = self.fenced_by = self.buffer.owner.incarnation

    def take_shard(self) -> Any:
        """At the sampler's start, before it reads its resume point: append the shard's
        ownership record (where the shard can hold one; module doc) and return it, else
        None. Every valid write of an earlier incarnation then precedes the record and
        is in the shard index read after it, so the resume point read after this call
        is final: an earlier incarnation that still runs (a second start, a process
        taken for dead) can add no round the new one does not see - its later writes
        are refused, or void for the merge. :meth:`claim` then writes the lease record
        with this owner and appends no second record."""
        if self.owned or not owner_records_enabled(self.buffer):
            return self.owner
        self.owner = self.buffer.claim_owner(self.incarnation, by=self.by)
        self.owned = True
        return self.owner

    def claim(self, entries: Iterable[Any] = (), earlier: Iterable[str] = (), *,
              owner: Any = None) -> dict[str, Any]:
        """Take the shard for this incarnation: its ownership record where the shard
        can hold one (module doc), and the lease record; returns the lease record
        (owned: with ``owner``, the ownership record's epoch and position).
        ``owner``: the ownership record a fence already wrote (:func:`fence_shard_owner`).
        A previous lease that cannot be read (:class:`LeaseReadError` after
        :data:`CLAIM_READ_ATTEMPTS` reads, paused by :func:`_claim_pause`) is not
        overwritten: the new record would forget the incarnations it supersedes, so
        the error is raised - before any ownership record is written, so a claim
        that is tried again (:func:`claim_patiently`) appends at most one."""
        for attempt in range(CLAIM_READ_ATTEMPTS):
            try:
                previous = read_lease(self.store, self.sampler, strict=True) or {}
                break
            except LeaseReadError:
                if attempt + 1 == CLAIM_READ_ATTEMPTS:
                    raise
                _claim_pause(attempt, self.sleep)
        if owner is None and self.owned and self.owner is not None:  # taken by take_shard
            owner = self.owner
        elif owner is None and owner_records_enabled(self.buffer):
            owner = self.buffer.claim_owner(self.incarnation, by=self.by)
            self.owned, self.owner = True, owner
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
        if getattr(self.buffer, "can_own", False):
            # this incarnation's writer reads ownership records: a pool fence may write
            # one into its shard even before it took the shard with one (fence_shard_owner)
            record[READS_OWNER_RECORDS] = True
        if owner is not None:
            record["owner"] = {"epoch": owner.epoch, "pos": owner.pos, "by": owner.by}
        try:
            _put(self.store, lease_key(self.sampler), record)
        except Exception:  # noqa: BLE001 - owned: the lease record is observation only
            if owner is None:
                raise
        self.holder = self.incarnation
        self._record = record
        self._read_at = self.monotonic()
        return record

    def _adopt(self) -> None:
        """A sampler under the lease record: once the run's trainer reads ownership
        records, take the shard with one (only a shard that has none: a fence that
        got there first stays, and fences this incarnation)."""
        from .sample_buffer import ShardFenced  # noqa: PLC0415

        try:
            if not owner_records_enabled(self.buffer):
                return
            self.owner = self.buffer.claim_owner(self.incarnation, by=self.by, first_only=True)
        except ShardFenced as fenced:
            self.owned = True
            self.holder = self.fenced_by = fenced.owner.incarnation
            return
        except Exception:  # noqa: BLE001 - not this time; the lease record still fences
            return
        self.owned = self.adopted = True
        # a pool fence that read the shard before this record wrote the lease record
        # only (the shard was not owned then; ``fence_shard_owner``): it wrote the lease
        # before that read, so a read now shows it
        try:
            lease = read_lease(self.store, self.sampler, strict=True)
        except LeaseReadError:
            lease = None
        if lease is not None and str(lease.get("incarnation")) != self.incarnation:
            self.holder = self.fenced_by = str(lease.get("incarnation"))
            return
        if self._record is not None:  # the fence's hint of the shard's end (observation)
            self._record["owner"] = {"epoch": self.owner.epoch, "pos": self.owner.pos, "by": self.owner.by}
            try:
                _put(self.store, lease_key(self.sampler), self._record)
            except Exception:  # noqa: BLE001
                pass

    def poll(self) -> bool:
        """Read the fence; True once another incarnation holds the shard (sticky).
        Owned: an incremental read of the shard index; else the lease record."""
        with self._lock:
            self._checked_at = self.monotonic()
            self._sync()
            if self.fenced_by is not None:
                return True
            self.reads += 1
            if self.owned:
                try:
                    self.buffer.refresh()
                except Exception:  # noqa: BLE001 - no verdict this time; the write reads again
                    self.read_failures += 1
                    self.failures_in_a_row += 1
                    self.unknown = True
                    return False
                self.failures_in_a_row = 0
                self.unknown = False
                self._read_at = self._checked_at
                self._sync()
                return self.fenced_by is not None
            try:
                lease = read_lease(self.store, self.sampler, strict=True)
            except LeaseReadError:
                lease = None
            self.unknown = lease is None
            if lease is None:  # unreadable, or absent (deleted): no verdict, fail open
                self.read_failures += 1
                return False
            self._read_at = self._checked_at
            self.holder = str(lease.get("incarnation"))
            if self.holder != self.incarnation:
                self.fenced_by = self.holder
            elif self._record is not None and getattr(self.buffer, "can_own", False):
                self._adopt()
            return self.fenced_by is not None

    @property
    def unreadable(self) -> bool:
        """Owned: the shard index could not be read ``max_read_failures`` times in a
        row; no group is written until a read succeeds (not a fence: nothing is sticky)."""
        return self.owned and self.failures_in_a_row >= self.max_read_failures

    def unread_s(self) -> float:
        """Seconds since the lease (owned: the shard index) was last read (0.0 before the first read or claim)."""
        return 0.0 if self._read_at is None else max(0.0, self.monotonic() - self._read_at)

    def unread_too_long(self) -> bool:
        """Not known to be fenced, but this incarnation cannot tell whether it still
        holds the shard: the lease stayed unread for longer than ``max_unread_s``, or
        (owned) the shard index was :attr:`unreadable` (:attr:`unknown` says only
        that the last read failed)."""
        if self.fenced_by is not None:
            return False
        if self.owned:
            return self.unreadable
        return self.max_unread_s is not None and self.unread_s() > self.max_unread_s

    def check_before_write(self) -> bool:
        """Before a group write: True when it must not be written. Owned: the shard
        writer's verdict, no request here - the write reads the index itself and
        raises ``ShardFenced`` (so there is no window between this check and the
        create) - or the index was :attr:`unreadable` (read again at most every
        ``write_check_s``). Else the cached lease verdict when read in the last
        ``write_check_s`` seconds, or a fresh :meth:`poll`. True: do not write
        (fenced, sticky; or :meth:`unread_too_long`, until a read succeeds again)."""
        if self.owned:
            self._sync()
            if self.fenced_by is None and self.unreadable and self.monotonic() - self._checked_at >= self.write_check_s:
                self.poll()
            if self.fenced_by is None and self.unreadable:
                self.unread_refusals += 1
            return self.fenced_by is not None or self.unreadable
        if self.fenced_by is not None:
            return True
        if self.monotonic() - self._checked_at >= self.write_check_s and self.poll():
            return True
        if self.unread_too_long():
            self.unread_refusals += 1
            return True
        return False

    def health(self) -> dict[str, Any]:
        """The read counters (the sampler's ``rl_sampler_store_read_failed`` / ``_ok`` events, ``what: lease``)."""
        return {"lease_reads": self.reads, "lease_read_failures": self.read_failures,
                "lease_unread_s": round(self.unread_s(), 1), "lease_unread_refusals": self.unread_refusals}

    def detail(self) -> dict[str, Any]:
        out = {"fenced": "superseded", "superseded_by": self.fenced_by, "lease_reads": self.reads,
               "lease_read_failures": self.read_failures}
        if self.owned:
            owner = self.buffer.owner
            out["shard_owner"] = None if owner is None else owner.to_dict()
        return out


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


_ORDINALS_KEPT = 4096  # IncarnationGate: the positions whose ordinal it keeps


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
                 monotonic: Callable[[], float] = time.monotonic,
                 owner_epoch: Callable[[int], int] | None = None) -> None:
        """``owner_epoch(position)``: the shard owner epoch of a merged position
        (``FileSampleBuffer.shard_epoch_at``); a position above 0 was merged under the
        shard's ownership records and is never judged here (no lease read either)."""
        self.store = store
        self.entries = entries
        self.owner_epoch = owner_epoch
        self.refresh_s = float(refresh_s)
        self.clock_margin_s = float(clock_margin_s)
        self.retry_s = float(retry_s)
        self.warn_s = float(warn_s)
        self.monotonic = monotonic
        self._leases: dict[str, tuple[float, dict[str, Any] | None]] = {}
        # sampler -> (first failure, last failed read, last warning, first failure wall time)
        # while its reads fail
        self._failing: dict[str, list[float]] = {}
        self._seen: Counter[tuple[str, str]] = Counter()  # (sampler, incarnation) over positions < _upto
        self._upto = 0
        # position -> its ordinal among its incarnation's entries (a verdict asked again)
        self._ordinals: OrderedDict[int, int] = OrderedDict()
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
                failing = self._failing[sampler] = [now, now, -float("inf"), time.time()]
            failing[1] = now
            if now - failing[2] >= self.warn_s:
                failing[2] = now
                since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(failing[3]))
                log.warning("incarnation gate: training paused: lease of shard %s unreadable since %s "
                            "(%.0f s, %d failed reads); its groups and every group behind them wait in "
                            "the buffer until it is read: %s", sampler, since, now - failing[0],
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
        # while its reads fail an older read does not answer (fail closed)
        if (cached is None or sampler in self._failing or now - cached[0] >= self.refresh_s
                or (force and now - cached[0] >= 1.0)):
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
        if self.owner_epoch is not None and self.owner_epoch(int(position)) > 0:
            return None  # valid by its shard's ownership records: the merge decided
        # judged before (the index is append-only: its ordinal stands), or counted now
        ordinal = self._ordinals.get(int(position)) if int(position) < self._upto else None
        if ordinal is None:
            ordinal = self._ordinal(int(position), (sampler, inc))
            self._ordinals[int(position)] = ordinal
            while len(self._ordinals) > _ORDINALS_KEPT:
                self._ordinals.popitem(last=False)
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

    def recheck(self, items: list[tuple[Any, int]], read: dict[str, float], *,
                fresh_s: float = 1.0) -> list[str | None | LeaseUnknown]:
        """:meth:`verdict` on ``items`` (``(entry, position)`` pairs it let through
        earlier) with their leases read again now: one read per shard, except a
        shard that ``read`` (shard -> monotonic time of this caller's last recheck
        read; updated) shows read less than ``fresh_s`` seconds ago. A result is
        None (train it), the stale detail, or a :class:`LeaseUnknown` (unreadable:
        the caller waits and asks again)."""
        now = self.monotonic()
        failed: dict[str, LeaseUnknown] = {}
        out: list[str | None | LeaseUnknown] = []
        for entry, position in items:
            sampler = (getattr(entry, "sampler_round", None) or {}).get("sampler")
            if sampler is not None and now - read.get(str(sampler), -float("inf")) >= fresh_s:
                read[str(sampler)] = now
                try:
                    self._read(str(sampler), now)
                except LeaseUnknown as error:
                    failed[str(sampler)] = error
            if sampler is not None and str(sampler) in failed:
                self.deferred += 1  # not read again by the verdict
                out.append(failed[str(sampler)])
                continue
            try:
                out.append(self.verdict(entry, int(position)))
            except LeaseUnknown as error:
                out.append(error)
        return out


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
    epoch = getattr(buffer, "shard_epoch_at", None)
    return IncarnationGate(store, entries, owner_epoch=epoch if callable(epoch) else None)
