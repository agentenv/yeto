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

A read failure never fences (fail open on the sampler; the gate is the second
line). Nothing here talks to a cloud.

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
trainer is older) the lease JSON and the gate work as described above.
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
    the run's trainer is older (no trainer lease) - the lease record alone fences then.

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
    hint = (read_lease(store, shard) or {}).get("owner")
    if isinstance(hint, Mapping) and hint.get("pos") is not None:
        try:
            raw = objects.read(ObjectIndex(objects, prefix=shard_prefix(shard)).key(int(hint["pos"])))
            record = None if raw is None else json.loads(raw)
            if isinstance(record, dict) and is_owner_record(record) and int(record["pos"]) == int(hint["pos"]):
                tail = ShardTail(next=int(hint["pos"]) + 1, owner=_owner_of(record))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a hint only: the whole shard is listed
    return claim_shard_owner(objects, shard, incarnation, by=by, clock=clock, tail=tail)


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
    """One incarnation's side of the lease (module doc).

    ``buffer``: the sampler's shard writer. When it can hold ownership records
    (:func:`owner_records_enabled` at :meth:`claim`), the shard index is the fence
    (``owned``) and the lease JSON is written for observation only."""

    def __init__(self, store: Any, sampler: str, incarnation: str, *,
                 write_check_s: float = DEFAULT_WRITE_CHECK_S,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 buffer: Any = None, by: str = "sampler",
                 max_read_failures: int = DEFAULT_MAX_READ_FAILURES) -> None:
        self.store = store
        self.sampler = str(sampler)
        self.incarnation = str(incarnation)
        self.write_check_s = float(write_check_s)
        self.clock = clock
        self.monotonic = monotonic
        self.buffer = buffer
        self.by = str(by)
        self.max_read_failures = int(max_read_failures)
        self.owned = False  # the shard's ownership records decide, not the lease JSON
        self.owner: Any = None  # this incarnation's ownership record (owned)
        self.holder: str | None = None  # who the last read named
        self.fenced_by: str | None = None
        self.read_failures = 0
        self.failures_in_a_row = 0
        self.reads = 0
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

    def claim(self, entries: Iterable[Any] = (), earlier: Iterable[str] = (), *,
              owner: Any = None) -> dict[str, Any]:
        """Take the shard for this incarnation: its ownership record where the shard
        can hold one (module doc), and the lease record; returns the lease record
        (owned: with ``owner``, the ownership record's epoch and position).
        ``owner``: the ownership record a fence already wrote (:func:`fence_shard_owner`)."""
        if owner is None and owner_records_enabled(self.buffer):
            owner = self.buffer.claim_owner(self.incarnation, by=self.by)
            self.owned, self.owner = True, owner
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
        if owner is not None:
            record["owner"] = {"epoch": owner.epoch, "pos": owner.pos, "by": owner.by}
        try:
            _put(self.store, lease_key(self.sampler), record)
        except Exception:  # noqa: BLE001 - owned: the lease record is observation only
            if owner is None:
                raise
        self.holder = self.incarnation
        return record

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
                    return False
                self.failures_in_a_row = 0
                self._sync()
                return self.fenced_by is not None
            lease = read_lease(self.store, self.sampler)
            if lease is None:
                self.read_failures += 1
                return False
            self.holder = str(lease.get("incarnation"))
            if self.holder != self.incarnation:
                self.fenced_by = self.holder
            return self.fenced_by is not None

    @property
    def unreadable(self) -> bool:
        """Owned: the shard index could not be read ``max_read_failures`` times in a
        row; no group is written until a read succeeds (not a fence: nothing is sticky)."""
        return self.owned and self.failures_in_a_row >= self.max_read_failures

    def check_before_write(self) -> bool:
        """Before a group write: True when it must not be written. Owned: the shard
        writer's verdict, no request here - the write reads the index itself and
        raises ``ShardFenced`` (so there is no window between this check and the
        create) - or the index was unreadable too often. Else the cached lease
        verdict when read in the last ``write_check_s`` seconds, or a fresh :meth:`poll`."""
        if self.owned:
            self._sync()
            return self.fenced_by is not None or self.unreadable
        if self.fenced_by is not None:
            return True
        if self.monotonic() - self._checked_at < self.write_check_s:
            return False
        return self.poll()

    def detail(self) -> dict[str, Any]:
        out = {"fenced": "superseded", "superseded_by": self.fenced_by, "lease_reads": self.reads,
               "lease_read_failures": self.read_failures}
        if self.owned:
            owner = self.buffer.owner
            out["shard_owner"] = None if owner is None else owner.to_dict()
        return out


class IncarnationGate:
    """The buffer reader's stale-incarnation verdict (module doc).

    ``entries()``: the reader's entries in buffer order (``FileSampleBuffer.entries``);
    ``refresh_s``: how often the leases are read again (an unknown incarnation
    forces a read at most that often too)."""

    def __init__(self, store: Any, entries: Callable[[], list[Any]], *, refresh_s: float = 30.0,
                 clock_margin_s: float = DEFAULT_CLOCK_MARGIN_S,
                 monotonic: Callable[[], float] = time.monotonic,
                 owner_epoch: Callable[[int], int] | None = None) -> None:
        """``owner_epoch(position)``: the shard owner epoch of a merged position
        (``FileSampleBuffer.shard_epoch_at``); a position above 0 was merged under the
        shard's ownership records and is never judged here."""
        self.store = store
        self.entries = entries
        self.owner_epoch = owner_epoch
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
        if self.owner_epoch is not None and self.owner_epoch(int(position)) > 0:
            return None  # valid by its shard's ownership records: the merge decided
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
    epoch = getattr(buffer, "shard_epoch_at", None)
    return IncarnationGate(store, entries, owner_epoch=epoch if callable(epoch) else None)
