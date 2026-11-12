"""Durable sample buffer: an ordered index of samples next to their payloads.

rl-trainer-start-stop. The buffer keeps samples across
trainer stops and head restarts. It lives in the existing cut storage (a
:class:`~yeto.rl.engine.resume.CheckpointStore`) under ``sample-buffer/``:

* ``payloads/<name>`` holds one sample payload; it is written before its index
  record, so the index never names a missing payload;
* the index holds one JSON record per sample (``pos``: the sample's position) and
  a last record ``end`` when the samplers append nothing more. Two layouts:

  - ``index.jsonl`` (:class:`JsonlIndex`): one line per record, appended under
    ``flock``. For a store that is a directory every island shares (local disk,
    Modal Volume). A torn last line (crash during a write) is ignored by readers
    and cut off by the next writer;
  - ``index/<pos>.json`` (:class:`ObjectIndex`): one object per record, taken
    with a create-if-absent write. For a store whose mount cannot append or
    rewrite (an S3 bucket): it uses the store's object API
    (``CheckpointStore.objects``), never the mount. A reader lists the keys after
    the last one it read.

The consumed watermark is the position before which every sample is consumed.
The buffer does not persist it: the trainer writes it into the cut record
(``ConsistentCut.buffer_watermark``), and the next start calls :meth:`reset_to`
with the watermark of the cut it restores. So a sample is consumed at most once
per cut lineage, and samples consumed after the last cut are consumed again.

Arrival notification is by polling the index (``poll`` / ``start_polling``).

Limits of this first implementation: one consumer (or one per lane: synced
trainer islands read disjoint lanes, :mod:`.merged_release`); with ``index.jsonl``,
concurrent appenders must share one host (``flock``); the version gate on
selection is the consumer's job (``max_policy_age``).

Appenders on several hosts (convertible nodes, rl-trainer-start-stop): each
appender writes its own shard index ``shards/<shard>.jsonl`` (same line format,
one writer per file) next to the shared payloads, and one merger -- the head --
appends the new shard lines to ``index.jsonl`` (:func:`merge_shards`). The
merged line keeps ``shard`` and ``shard_pos``, so a restarted merger continues
where the index ends.

The merge is idempotent (:meth:`FileSampleBuffer._merge_shards`): a merged
record is identified by ``(shard, shard_pos)``, the merger appends only the
shard records above the highest merged ``shard_pos`` of their shard, and it
decides that again after every lost create. So two mergers at once (a trainer
not dead yet and the one that replaces it), or a create whose answer was lost,
leave every shard record in the index once. An index written before this rule
can hold a shard record twice: the reader passes over the second one
(``duplicate_merged_records``), it is never selected. The reader decides that
by the exact set of ``(shard, shard_pos)`` it has read, so an index that holds a
shard's records out of their order (the older :func:`merge_shards` sorted a
batch by ``appended_at`` alone; a sampler clock that stepped back then put a
later record first) is read with every record once.

Not covered here: a *shard* that holds one group twice. A sampler whose create
of a shard record was applied but answered as lost writes the group again at
the next position: same ``sample_id``, two ``shard_pos``. The merge takes them
for two records. The object store's create must not report such a create as
lost (``S3ObjectStore.create`` reads the object back after a retried request
was refused); this merge relies on that.

Shard ownership (the fence of a sampler shard on an object store). A shard index
can hold *ownership records* between its group records, each taken like any
record with a create-if-absent write of the next position::

    {"schema": "yeto.rl.sample-buffer/v2", "pos": p,
     "owner": {"incarnation": "<incarnation, or pool-fence:<ticket>>", "epoch": e,
               "by": "sampler" | "pool" | "operator", "claimed_at": <information>}}

``epoch`` counts the ownership records of the shard up to this one. One rule,
the same for every reader and independent of any clock: a group record (and an
``end`` record) is valid if no ownership record precedes it (the legacy part of
the shard), or if the last ownership record before it names the incarnation
that wrote it (``sampler_round.incarnation``). The position decides: every group
created before the fence record stays valid, every write of the old incarnation
after it is void. A sampler claims its shard at its join
(:meth:`FileSampleBuffer.claim_owner`); the pool fences a shard by appending a
``pool-fence:<ticket>`` record (:func:`claim_shard_owner`). A shard writer reads
the index before every create, so it sees the fence and raises
:class:`ShardFenced` instead of writing; when it loses the create to the fence
record it reads again and sees it. The merger (:meth:`FileSampleBuffer._merge_shards`)
applies the rule again - a writer that bypassed the check gets nothing merged -
and writes ``shard_epoch`` into the merged records of an owned part, from which
a restarted merger restores the owner (and ``shard_passed``, the shard positions
it passed over right before the record, so a reader tells them from a hole).
``epoch`` is information: the rule reads only the order of the records. The
merged index holds no ownership record of a shard, but one *format marker*
(``{"schema": "yeto.rl.sample-buffer/v2", "marker": "shard-ownership", "note"}``),
written by the merger before the first record it merges from a shard with an
ownership record. A reader passes over the marker's position (never selected,
like a repeated record). It is there for code older than ownership records:
that code resumes a shard by its number of merged records and would merge the
shard from a wrong position; with the marker it stops at its first read of the
index (``record out of order``, the marker's ``note`` in the message) instead
of merging anything. See "Older code" below. An older trainer fails on a v2 record (schema error), so a sampler
writes them only when the trainer of the run is new enough
(``trainer_lease.TrainerLease.current(objects) >= 1``).

Older code (before ownership records) on a store that holds them fails, it does
not misread: the trainer and the head stop at the format marker of the merged
index (``SampleBufferError: ... record out of order``, before any write); a
sampler stops the same way at the ownership record of its shard; the head's
``UnmergedShards`` raises ``KeyError: 'sample_id'`` on an ownership record when
the trainer has merged nothing since the first one (no marker yet). A merger of
older code that got past all of this would copy an ownership record into the
merged index; a reader of this code passes over such a record too
(``foreign_merged_records``). The fields ``shard_epoch`` / ``shard_passed`` of a
merged record are ignored by older code. A run that wrote ownership records
cannot go back to older trainer code.

Damaged shard records. The merger checks every shard record before it writes it
into the merged index: one that is not JSON, has another schema or position, or
lacks a field of an entry is passed over (``bad_shard_records``; never merged, so
the merged index holds only records the merger can read back), and the groups
after it and of every other shard are merged as usual. The shard's own writer
passes over it as well (``bad_records_seen``) instead of failing at its start. Nothing is written for a
passed-over record (a fenced write, a damaged record): a restarted merger reads
each shard again from its highest merged position and passes over the same
records. A shard position that is missing while later ones exist (an object
deleted from outside) is not skipped - the record may be a group - but reported
(``blocked_shards``): the shard's later groups wait until it is back.
``FileSampleBuffer.merge_events`` (``callable(event, **fields)``) receives
:data:`MERGE_EVENT` for every such finding.

Sampler islands on several sites (rl-cross-site-sample-store): an object store
without reliable create-if-absent between hosts (``objects.buffer_shards``, OSS)
gives each sampler island its own shard, ``shards/<shard>/<pos>.json`` (one
writer per shard, same record format), and names the shard once under
``shard-names/``. The consumer -- the trainer, the one reader of the buffer --
is the only writer of ``index/`` (``FileSampleBuffer(merge_shards=True)``): each
refresh first appends the new records of every named shard (by ``appended_at``,
then shard name), keeping ``shard`` and ``shard_pos``, and an ``end`` record
once every shard has ended. The consumed watermark stays one position in the
merged index; :meth:`FileSampleBuffer.shard_positions` turns it into the
per-shard positions (the merged index is durable, so a restore from a cut
resumes every shard where that cut left it).
"""

from __future__ import annotations

import bisect
import dataclasses
import fcntl
import heapq
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from .merged_release import BufferLane, merges_shards
from .object_store import ObjectStore
from .resume import CheckpointStore
from .trainer_lifecycle import BacklogStats, SampleIndexEntry, Selected

SCHEMA = "yeto.rl.sample-buffer/v1"
OWNER_SCHEMA = "yeto.rl.sample-buffer/v2"  # a shard ownership record (module doc)
OWNER_EVENT = "rl_shard_owner"  # a claim or a fence of a shard: shard, incarnation, epoch, by, pos
# the merger passed over shard records or found a damaged index: ``reason`` (stale_incarnation,
# bad_record, duplicate_merged_record, unmerged_shard_gap, missing_shard_position), shard, ...
MERGE_EVENT = "rl_buffer_merge_skipped"
FORMAT_MARKER = "shard-ownership"  # the merged index's format marker (module doc)
_MARKER_NOTE = "written by yeto with shard ownership records: code older than them cannot read this buffer"
BUFFER_DIR = "sample-buffer"
INDEX = "index.jsonl"
INDEX_OBJECTS = "index"
PAYLOADS = "payloads"
SHARDS = "shards"
SHARD_NAMES = "shard-names"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_POS_DIGITS = 12
# create-if-absent races lost in a row before an append gives up (another writer
# takes every position first: that is a fault, not contention)
_MAX_CREATE_RACES = 1000
# concurrent record reads of one ObjectIndex.read_new (an object store GET each)
_READ_WORKERS = 16


def _read_all(objects: ObjectStore, keys: list[str]) -> list[bytes | None]:
    """``objects.read`` of every key, in key order (concurrent for more than one)."""
    if len(keys) <= 1:
        return [objects.read(k) for k in keys]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(_READ_WORKERS, len(keys))) as pool:
        return list(pool.map(objects.read, keys))


class SampleBufferError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ShardOwner:
    """The newest ownership record of a shard (module doc). ``pos``: its position
    in the shard (-1: restored from a merged record's ``shard_epoch``)."""

    incarnation: str
    epoch: int
    by: str = ""
    pos: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {"incarnation": self.incarnation, "epoch": self.epoch, "by": self.by, "pos": self.pos}


class ShardFenced(SampleBufferError):
    """This writer does not own its shard: another incarnation (or a pool fence)
    holds the newest ownership record. Nothing was written."""

    def __init__(self, shard: str, incarnation: str | None, owner: ShardOwner) -> None:
        self.shard, self.incarnation, self.owner = shard, incarnation, owner
        super().__init__(f"shard {shard!r} is owned by {owner.incarnation!r} (epoch {owner.epoch}, by "
                         f"{owner.by or 'unknown'}), not by this writer ({incarnation!r}): nothing written")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def bad_shard_record(record: Mapping[str, Any]) -> str | None:
    """Why the merger cannot take this shard record (None: it can): the record
    would be written into the merged index and read back by every later reader."""
    if "bad" in record:
        return str(record["bad"])
    if record.get("schema") == OWNER_SCHEMA:
        owner = record.get("owner")
        if not (isinstance(owner, Mapping) and isinstance(owner.get("incarnation"), str)
                and _is_int(owner.get("epoch")) and owner["epoch"] >= 1):
            return "ownership record without an owner incarnation and epoch"
        return None
    if record.get("schema") != SCHEMA:
        return f"schema {record.get('schema')!r}"
    for name in ("sampler_round", "provenance"):
        if record.get(name) is not None and not isinstance(record[name], Mapping):
            return f"{name} is not an object"
    if record.get("end"):
        return None
    try:
        _entry(dict(record))
    except (KeyError, TypeError, ValueError) as error:
        return f"not an entry ({type(error).__name__}: {error})"
    return None


def print_merge_event(event: str, **fields: Any) -> None:
    """One ``YETO_RL_EVENT <json>`` stdout record (the merger's default emitter)."""
    from yeto.rl.event_echo import format_record  # noqa: PLC0415

    print(format_record({"island_id": None, "time_unix": time.time(), "event": str(event), **fields}),
          flush=True)


def is_owner_record(record: Mapping[str, Any]) -> bool:
    return record.get("schema") == OWNER_SCHEMA and isinstance(record.get("owner"), Mapping)


def _owner_of(record: Mapping[str, Any]) -> ShardOwner:
    owner = record["owner"]
    return ShardOwner(incarnation=str(owner["incarnation"]), epoch=int(owner["epoch"]),
                      by=str(owner.get("by") or ""), pos=int(record["pos"]))


def record_incarnation(record: Mapping[str, Any]) -> str | None:
    """The incarnation that wrote a shard record: a group's ``sampler_round.incarnation``,
    an ``end`` record's ``incarnation``."""
    inc = record.get("incarnation") if record.get("end") else (record.get("sampler_round") or {}).get("incarnation")
    return None if inc is None else str(inc)


def owner_accepts(owner: ShardOwner | None, record: Mapping[str, Any]) -> bool:
    """The validity rule of a shard record under the last ownership record before it
    (module doc); no clock, no lease."""
    return owner is None or record_incarnation(record) == owner.incarnation
class _SeenPositions:
    """The positions of one shard a reader found in the merged index: every position
    below ``prefix``, plus ``above`` (a merged index in shard order keeps ``above``
    empty). ``high``: the highest one + 1."""

    __slots__ = ("prefix", "above", "high")

    def __init__(self) -> None:
        self.prefix = 0
        self.above: set[int] = set()
        self.high = 0

    def __contains__(self, position: int) -> bool:
        return position < self.prefix or position in self.above

    def add(self, position: int) -> None:
        if position == self.prefix:
            self.prefix += 1
            while self.prefix in self.above:
                self.above.discard(self.prefix)
                self.prefix += 1
        elif position > self.prefix:
            self.above.add(position)
        self.high = max(self.high, position + 1)

    @property
    def missing(self) -> int:
        """Positions below ``high`` that were not found."""
        return self.high - self.prefix - len(self.above)


def payload_name(sample_id: str) -> str:
    return _SAFE_NAME.sub("_", sample_id) + ".bin"


def _hold_reader(store: Any) -> Any:
    """``store.hold_reader()``; a store object without it holds nothing."""
    hold = getattr(store, "hold_reader", None)
    if callable(hold):
        return hold()
    from .resume import StoreWriterGuard

    return StoreWriterGuard()


class JsonlIndex:
    """The index as ``index.jsonl`` in a shared directory (see the module doc).

    Reads of the index and of a payload hold the store's reader guard
    (``CheckpointStore.hold_reader``) while the file is open: a reload by
    another process of the container (the driver reading its policy releases)
    is then skipped instead of failing on the open file."""

    def __init__(self, store: CheckpointStore, *, index: str = INDEX, create_index: bool = True,
                 reload: bool = True) -> None:
        """``index``: the index file under ``sample-buffer/`` (a shard writer
        passes :func:`shard_index`). ``create_index=False``: a reader that must not
        write the index, not even an empty one (on a store mount whose commit would
        replace the merger's index). ``reload=False``: :meth:`read_new` does not
        reload the store (a merger's shard reader: the merger reloaded it once)."""
        self.store = store
        self._reload = bool(reload)
        self.root = Path(store.root) / BUFFER_DIR
        (self.root / PAYLOADS).mkdir(parents=True, exist_ok=True)
        self.path = self.root / index
        if create_index:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch()
        self._offset = 0  # bytes of complete lines read
        self._held = 0  # open one_commit() blocks
        self._uncommitted = False  # appends made inside them, not committed yet

    def write_payload(self, name: str, data: bytes) -> str:
        path = self.root / PAYLOADS / name
        tmp = path.with_suffix(".bin.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return f"{PAYLOADS}/{name}"

    def read_payload(self, ref: str) -> bytes:
        guard = _hold_reader(self.store)  # no reload of the store while the file is open
        try:
            return (self.root / ref).read_bytes()
        finally:
            guard.release()

    def has_payload(self, ref: str) -> bool:
        return (self.root / ref).is_file()

    def read_new(self) -> list[dict[str, Any]]:
        if self._reload:
            self.store.reload()
        if not self.path.exists():  # a reader before the first merge (or a shard not created yet)
            return []
        guard = _hold_reader(self.store)
        try:
            with open(self.path, "rb") as fh:
                return list(self._lines(fh))
        finally:
            guard.release()

    def append(self, absorb: Callable[[list[dict[str, Any]]], None],
               make: Callable[[], dict[str, Any]]) -> None:
        self.append_many(absorb, lambda: [make()])

    def append_many(self, absorb: Callable[[list[dict[str, Any]]], None],
                    make: Callable[[], list[dict[str, Any]]]) -> None:
        """Append ``make()``'s records with one write, one fsync and one store
        commit (a merger appends a whole refresh at once: one Modal Volume commit
        per record would cost seconds each)."""
        with open(self.path, "r+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                absorb(list(self._lines(fh)))
                records = make()
                fh.truncate(self._offset)  # drop a torn last line
                fh.seek(self._offset)
                fh.write(b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in records))
                fh.flush()
                os.fsync(fh.fileno())
                absorb(list(self._lines(fh)))
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        if self._held:
            self._uncommitted = True
        else:
            self.store.commit()

    @contextmanager
    def one_commit(self) -> Iterator[None]:
        """Appends inside the block are committed once, when it ends (also on an
        error: what was appended stays)."""
        self._held += 1
        try:
            yield
        finally:
            self._held -= 1
            if not self._held and self._uncommitted:
                self._uncommitted = False
                self.store.commit()

    # ------------------------------------------------------------ shards (merger side)
    def shard_names(self) -> list[str]:
        """The shard index files under ``sample-buffer/shards/``, sorted."""
        return sorted(p.name[:-len(".jsonl")] for p in (self.root / SHARDS).glob("*.jsonl"))

    def shard_reader(self, shard: str, start: int) -> "JsonlIndex":
        """A reader of one shard's index file that skips its first ``start``
        records (merged before)."""
        reader = JsonlIndex(self.store, index=shard_index(shard), create_index=False, reload=False)
        if start:
            skipped = 0
            if reader.path.exists():
                with open(reader.path, "rb") as fh:
                    for _ in reader._lines(fh):
                        skipped += 1
                        if skipped == start:
                            break
            if skipped < start:
                raise SampleBufferError(f"shard {shard}: {skipped} records, but {start} are merged")
        return reader

    def _lines(self, fh) -> Iterator[dict[str, Any]]:
        fh.seek(self._offset)
        data = fh.read()
        while (end := data.find(b"\n")) >= 0:  # a torn last line has no newline
            raw, data = data[:end], data[end + 1:]
            try:
                record = json.loads(raw)
            except ValueError as error:
                raise SampleBufferError(f"{self.path}: bad line at byte {self._offset}: {raw[:200]!r}") from error
            self._offset += end + 1
            yield record


class ObjectIndex:
    """The index as one object per record, ``index/<pos>.json`` (see the module doc)."""

    def __init__(self, objects: ObjectStore, *, prefix: str = f"{BUFFER_DIR}/{INDEX_OBJECTS}/",
                 start: int = 0, tolerant: bool = False) -> None:
        """``prefix``: the record keys (a shard passes its own); ``start``: the
        position of the first record to read (a merger skips merged records).
        ``tolerant`` (a merger's shard reader): a record that is not a JSON object
        with its own position is returned as ``{"pos": pos, "bad": reason}`` instead
        of raising, so one damaged record does not stop the reader."""
        self.objects = objects
        self.prefix = prefix
        self.tolerant = bool(tolerant)
        self.blocked_at: int | None = None  # a missing position with later ones listed (last read)
        self._next = int(start)  # the position of the next record to read
        self._last_key = self.key(self._next - 1) if self._next else ""  # the newest record key read

    def key(self, pos: int) -> str:
        return f"{self.prefix}{int(pos):0{_POS_DIGITS}d}.json"

    def write_payload(self, name: str, data: bytes) -> str:
        ref = f"{PAYLOADS}/{name}"
        self.objects.write(f"{BUFFER_DIR}/{ref}", data)
        return ref

    def read_payload(self, ref: str) -> bytes:
        data = self.objects.read(f"{BUFFER_DIR}/{ref}")
        if data is None:
            raise SampleBufferError(f"payload {ref} is missing")
        return data

    def has_payload(self, ref: str) -> bool:
        return self.objects.exists(f"{BUFFER_DIR}/{ref}")

    def read_new(self) -> list[dict[str, Any]]:
        """The records after the last one read, in order. Stops before a missing
        position (no writer leaves one: a position is created only after the one
        before it). One LIST; the records it names are fetched concurrently
        (``_READ_WORKERS``), then taken in position order up to the first gap."""
        keys = []
        blocked = None
        for i, key in enumerate(self.objects.list(self.prefix, start_after=self._last_key)):
            if key != self.key(self._next + i):
                blocked = self._next + i  # a later key is listed, this position is not
                break
            keys.append(key)
        out = []
        for key, raw in zip(keys, _read_all(self.objects, keys)):
            if raw is None:
                blocked = self._next  # listed, then gone
                break
            try:
                record = json.loads(raw)
            except ValueError as error:
                if not self.tolerant:
                    raise SampleBufferError(f"{key}: bad record: {raw[:200]!r}") from error
                record = {"pos": self._next, "bad": f"not JSON: {raw[:80]!r}"}
            if self.tolerant and not (isinstance(record, dict) and "bad" not in record
                                      and _is_int(record.get("pos")) and int(record["pos"]) == self._next):
                reason = record["bad"] if isinstance(record, dict) and "bad" in record else \
                    f"not a record of position {self._next}: {raw[:80]!r}"
                record = {"pos": self._next, "bad": reason}
            out.append(record)
            self._last_key = key
            self._next += 1
        self.blocked_at = blocked
        return out

    def append(self, absorb: Callable[[list[dict[str, Any]]], None],
               make: Callable[[], dict[str, Any]]) -> None:
        for _ in range(_MAX_CREATE_RACES):
            absorb(self.read_new())
            record = make()
            data = (json.dumps(record, sort_keys=True) + "\n").encode()
            if self.objects.create(self.key(record["pos"]), data):
                absorb(self.read_new())
                return
        raise SampleBufferError(f"{self.prefix}: lost {_MAX_CREATE_RACES} appends in a row to other writers")

    def append_batch(self, absorb: Callable[[list[dict[str, Any]]], None],
                     make: Callable[[int], list[dict[str, Any]]]) -> None:
        """Append a batch: one ``read_new`` (one LIST), then the records' create-if-absent
        writes in position order, without a LIST or a read-back between them.
        ``make(done)``: the records not created yet (``done`` of the batch are), with
        their positions from the index length; it is called again after every lost
        create, when ``absorb`` has taken the other writer's records (a merger leaves
        out the ones that writer appended too). Each created record is the next
        position, so no reader sees a gap; a lost create (another writer took that
        position) stops the run, the other writer's records are read and the rest is
        made again at the new positions - the same outcome as one ``append`` per
        record, with one LIST per batch instead of two per record."""
        done = 0
        for _ in range(_MAX_CREATE_RACES):
            absorb(self.read_new())
            records = make(done)
            if not records:
                return
            created = []
            for record in records:
                if int(record["pos"]) != self._next + len(created):
                    raise SampleBufferError(f"{self.prefix}: batch record at {record['pos']}, "
                                            f"expected {self._next + len(created)}")
                data = (json.dumps(record, sort_keys=True) + "\n").encode()
                if not self.objects.create(self.key(record["pos"]), data):
                    break
                created.append(record)
            if created:  # ours, at the next positions: absorbed as written, not read back
                self._next += len(created)
                self._last_key = self.key(self._next - 1)
                absorb(created)
                done += len(created)
            if len(created) == len(records):
                return
        raise SampleBufferError(f"{self.prefix}: lost {_MAX_CREATE_RACES} appends in a row to other writers")

    # ------------------------------------------------------------ shards (merger side)
    def shard_names(self) -> list[str]:
        return shard_names(self.objects)

    def shard_reader(self, shard: str, start: int) -> "ObjectIndex":
        return ObjectIndex(self.objects, prefix=shard_prefix(shard), start=start, tolerant=True)


@dataclasses.dataclass
class ShardTail:
    """What a process that only appends ownership records knows of a shard's end
    (:func:`claim_shard_owner`): the next position and the newest owner seen. Kept
    between calls, the next call lists only the keys after it."""

    next: int = 0
    owner: ShardOwner | None = None


def _newest_owner(objects: ObjectStore, keys: list[str], known: ShardOwner | None = None) -> ShardOwner | None:
    """The newest ownership record among the shard records ``keys`` (in position
    order), read from the end backwards: no further than the first one found, or
    than a record written by the ``known`` owner (it still owned the shard there,
    so no newer ownership record precedes it; None is returned then). First the
    last record alone, then ``_READ_WORKERS`` at a time."""
    end = len(keys)
    for size in (1, *([_READ_WORKERS] * (len(keys) // _READ_WORKERS + 1))):
        if end <= 0:
            break
        chunk, end = keys[max(0, end - size):end], max(0, end - size)
        for key, raw in reversed(list(zip(chunk, _read_all(objects, chunk)))):
            try:
                record = json.loads(raw) if raw is not None else None
            except ValueError as error:
                raise SampleBufferError(f"{key}: bad record: {raw[:200]!r}") from error
            if record is None:
                raise SampleBufferError(f"{key}: listed but not readable")
            if isinstance(record, dict) and is_owner_record(record):
                return _owner_of(record)
            if known is not None and isinstance(record, dict) and record_incarnation(record) == known.incarnation:
                return None
    return None


def claim_shard_owner(objects: ObjectStore, shard: str, incarnation: str, *, by: str,
                      clock: Callable[[], float] = time.time,
                      tail: ShardTail | None = None) -> tuple[ShardOwner, bool]:
    """Append an ownership record for ``incarnation`` to ``shard`` without reading the
    shard's groups (the pool's fence: ``pool-fence:<ticket>``, ``by="pool"``).
    Returns ``(owner, created)``; ``created`` is False when ``incarnation`` already
    is the shard's newest owner (a retried fence writes nothing).

    Cost: one LIST of the shard's keys after ``tail`` (1000 keys per request), the
    GETs from the end back to the newest ownership record among them or to a group
    of the owner ``tail`` names (usually the last record: one GET), one create; a
    lost create (the sampler wrote a group meanwhile) lists and reads only what
    was added. Without a ``tail`` owner, a shard with no ownership record yet is
    read whole, once (its first fence)."""
    tail = ShardTail() if tail is None else tail
    index = ObjectIndex(objects, prefix=shard_prefix(shard))
    for _ in range(_MAX_CREATE_RACES):
        keys = []
        after = index.key(tail.next - 1) if tail.next else ""
        for i, key in enumerate(objects.list(index.prefix, start_after=after)):
            if key != index.key(tail.next + i):
                break
            keys.append(key)
        tail.owner = _newest_owner(objects, keys, tail.owner) or tail.owner
        tail.next += len(keys)
        if tail.owner is not None and tail.owner.incarnation == str(incarnation):
            return tail.owner, False
        epoch = (0 if tail.owner is None else tail.owner.epoch) + 1
        record = {"schema": OWNER_SCHEMA, "pos": tail.next,
                  "owner": {"incarnation": str(incarnation), "epoch": epoch, "by": str(by),
                            "claimed_at": float(clock())}}
        if objects.create(index.key(tail.next), (json.dumps(record, sort_keys=True) + "\n").encode()):
            tail.owner, tail.next = _owner_of(record), tail.next + 1
            return tail.owner, True
    raise SampleBufferError(f"{index.prefix}: lost {_MAX_CREATE_RACES} ownership claims in a row to other writers")


def _objects_of(store: CheckpointStore | ObjectStore) -> ObjectStore | None:
    return store.objects if isinstance(store, CheckpointStore) else store


def needs_shards(store: CheckpointStore | ObjectStore) -> bool:
    """Whether every sampler island must write its own shard of this store's buffer."""
    objects = _objects_of(store)
    return bool(getattr(objects, "buffer_shards", False))


def shard_prefix(shard: str) -> str:
    """The record keys of one sampler island's shard on an object store."""
    return f"{BUFFER_DIR}/{SHARDS}/{_shard_name(shard)}/"


def shard_names(objects: ObjectStore) -> list[str]:
    """The shards named on an object store, sorted."""
    prefix = f"{BUFFER_DIR}/{SHARD_NAMES}/"
    return sorted(key[len(prefix):] for key in objects.list(prefix))


def _entry(record: dict[str, Any]) -> SampleIndexEntry:
    return SampleIndexEntry(
        sample_id=record["sample_id"], behavior_version=int(record["behavior_version"]),
        payload_ref=record["payload_ref"], tokens=int(record["tokens"]),
        appended_at=float(record["appended_at"]), provenance=record.get("provenance"),
        sampler_round=record.get("sampler_round"))


class UnmergedShards:
    """The head's read-only view of the sampler shard records that the consumer
    has not merged into the index yet (rl-cross-site-sample-store).

    The head reads the merged index only, and only the trainer merges. With the
    trainer scheduler on, a stopped trainer merges nothing, so without this view
    the head would never see a backlog and never start the trainer again.
    :meth:`pending` reads every shard by object key (never writes) and returns
    the records at or above the merged position of their shard."""

    def __init__(self, objects: ObjectStore) -> None:
        self.objects = objects
        self._readers: dict[str, ObjectIndex] = {}
        self._records: dict[str, list[tuple[int, SampleIndexEntry]]] = {}  # (shard position, entry)
        self._owners: dict[str, ShardOwner | None] = {}
        self._ended: set[str] = set()

    def pending(self, merged: dict[str, int]) -> tuple[list[SampleIndexEntry], bool]:
        """(records not merged yet, every named shard has ended). ``merged``: the
        merged position per shard (:meth:`FileSampleBuffer.shard_positions` of
        the index length): the shard records below it are merged."""
        names = shard_names(self.objects)
        for shard in names:
            if shard in self._ended:  # its end record is its last: no LIST per poll for it (a spot
                continue  # pool leaves one ended shard per reclaim, the run lists only live ones)
            reader = self._readers.get(shard)
            if reader is None:
                reader = self._readers[shard] = ObjectIndex(self.objects, prefix=shard_prefix(shard),
                                                            tolerant=True)
                self._records[shard] = []
            for record in reader.read_new():
                if bad_shard_record(record) is not None:
                    continue  # the trainer passes it over
                if is_owner_record(record):
                    self._owners[shard] = _owner_of(record)
                elif not owner_accepts(self._owners.get(shard), record):
                    continue  # a fenced incarnation's write: the trainer never merges it
                elif record.get("end"):
                    self._ended.add(shard)
                else:
                    self._records[shard].append((int(record["pos"]), _entry(record)))
        out: list[SampleIndexEntry] = []
        for shard, records in self._records.items():
            below = int(merged.get(shard, 0))
            done = next((i for i, (pos, _) in enumerate(records) if pos >= below), len(records))
            del records[:done]  # merged: forgotten here
            out.extend(entry for _, entry in records)
        return out, bool(names) and self._ended >= set(names)


def index_for(store: CheckpointStore | ObjectStore, *, index: str = INDEX,
              create_index: bool = True) -> JsonlIndex | ObjectIndex:
    """:class:`ObjectIndex` for an object store or a store with an object API
    (``CheckpointStore.objects``); :class:`JsonlIndex` for a shared directory.
    ``index``/``create_index``: see :class:`JsonlIndex`. A shard index
    (:func:`shard_index`) on an object store is that shard's own keys
    (:func:`shard_prefix`); the shard is named under ``shard-names/``."""
    objects = _objects_of(store)
    if objects is None:
        return JsonlIndex(store, index=index, create_index=create_index)
    if index == INDEX:
        return ObjectIndex(objects)
    shard = _shard_of_index(index)
    name_key = f"{BUFFER_DIR}/{SHARD_NAMES}/{shard}"
    if create_index and not objects.exists(name_key):
        objects.write(name_key, b"")
    # tolerant: a damaged record of the shard does not stop its writer (module doc)
    return ObjectIndex(objects, prefix=shard_prefix(shard), tolerant=True)


def open_trainer_buffer(store: CheckpointStore | ObjectStore, *, batch_size: int,
                        lane: BufferLane | None = None, create_index: bool = True,
                        shards: Iterable[str] | None = None, **kwargs: Any) -> "FileSampleBuffer":
    """The buffer of a trainer (the reader): it merges the sampler shards of a
    store that shards (:func:`needs_shards`), or the named ``shards`` of several
    sampler islands on one shared directory (the data ends once every named
    shard ended), unless it is a synced trainer other than the publisher
    (``merged_release.merges_shards``); ``lane``: its lane. A merging trainer
    is the only writer of the index, so it creates the index even with
    ``create_index=False`` (no sampler writes ``index.jsonl`` then)."""
    named = tuple(shards or ())
    merge = (needs_shards(store) or bool(named)) and merges_shards(lane)
    return FileSampleBuffer(store, batch_size=batch_size, create_index=create_index or merge, lane=lane,
                            merge_shards=merge, expected_shards=named if merge and named else None, **kwargs)


class FileSampleBuffer:
    """:class:`~yeto.rl.engine.trainer_lifecycle.DurableSampleBuffer` over a store
    (a :class:`CheckpointStore`, or an :class:`ObjectStore` at the store root).

    ``batch_size``: samples per training batch (``BacklogStats.batches``);
    ``arrival_window_s``: the window of ``BacklogStats.arrival_rate``."""

    def __init__(self, store: CheckpointStore | ObjectStore, *, batch_size: int,
                 arrival_window_s: float = 60.0, clock: Callable[[], float] = time.time,
                 index: str = INDEX, create_index: bool = True, merge_shards: bool = False,
                 expected_shards: Iterable[str] | None = None,
                 lane: BufferLane | None = None) -> None:
        """``index``/``create_index``: see :class:`JsonlIndex`. ``merge_shards``:
        this buffer (the consumer) merges the sampler shards into the index on
        every refresh (see the module doc); it must be the only one that does.
        On a shared directory the shards are ``shards/<shard>.jsonl`` files (one
        sampler island writes each; several sampler islands on one
        Modal Volume, where ``flock`` does not hold across containers).
        ``expected_shards``: the shards that must all end before the merged index
        ends (None: the shards seen so far), so an early-ending sampler cannot
        end the data before a slower one has written its first record.
        ``lane``: this consumer is one of several synced trainers and selects
        only the positions of its lane (:mod:`.merged_release`); the read cursor
        and the watermark stay positions of the whole index. Of the lane readers
        only the publisher's lane merges shards (``merged_release.merges_shards``)."""
        if lane is not None and merge_shards and not merges_shards(lane):
            raise SampleBufferError(f"buffer lane {lane.env()}: only the publisher's lane merges the "
                                    "sampler shards (yeto.rl.engine.merged_release)")
        self.lane = lane
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.store = store
        self.index = index_for(store, index=index, create_index=create_index)
        self.shard = _shard_of_index(index) if index != INDEX else None  # the shard this buffer writes
        if merge_shards and self.shard is not None:
            raise SampleBufferError("a shard writer does not merge shards (only the consumer's index does)")
        if expected_shards is not None and not merge_shards:
            raise SampleBufferError("expected_shards needs merge_shards")
        self._merge = merge_shards
        self._expected_shards = (None if expected_shards is None
                                 else frozenset(_shard_name(s) for s in expected_shards))
        self._shard_of: list[str | None] = []  # the shard of each position (merged records)
        self._shard_pos_of: list[int | None] = []  # its position in that shard
        self._shard_next: dict[str, int] = {}  # per shard: the highest merged shard position + 1
        self._shard_seen: dict[str, _SeenPositions] = {}  # per shard: the merged shard positions
        # per shard: the index positions of its records and its ``_shard_next`` after each
        # (shard_positions() looks a cursor up in them)
        self._shard_marks: dict[str, tuple[list[int], list[int]]] = {}
        self._shard_pending: dict[str, list[dict[str, Any]]] = {}  # shard records read, not merged yet
        # positions never selected: a repeated merged shard record, the format marker
        self._skipped: set[int] = set()
        self._duplicates = 0  # of them: repeated merged shard records
        self._format_marked = False  # the merged index holds its format marker (module doc)
        self.foreign_merged_records = 0  # ownership records an older merger copied into the index
        self._gap_warned = False
        # shard ownership (module doc). A shard writer: the index positions read (group,
        # end and ownership records), the shard's newest owner, the incarnation this
        # writer claimed as, the shard position of every entry.
        self._positions = 0
        self._owner: ShardOwner | None = None
        self._writer: str | None = None
        self._entry_pos: list[int] = []
        self.fenced_writes_seen = 0  # shard records of a fenced incarnation read here (never entries)
        self.bad_records_seen = 0  # damaged records of this writer's shard read here (never entries)
        # the merger: every shard's newest owner, the owner epoch of each merged position,
        # (incarnation, epoch) of each shard's last merged record, the fenced writes dropped
        self._shard_owner: dict[str, ShardOwner | None] = {}
        self._shard_epoch_of: list[int] = []
        self._shard_last: dict[str, tuple[str | None, int]] = {}
        self._shard_passed: dict[str, int] = {}  # shard records passed over since the last one taken
        self._fenced_dropped: dict[tuple[str, int, str, str | None], int] = {}  # not reported yet
        self.fenced_groups_dropped = 0
        self.bad_shard_records = 0  # damaged shard records passed over (module doc)
        self._bad_shards: set[str] = set()
        self._blocked: dict[str, int] = {}  # shard -> its missing position (later ones exist)
        # ``callable(event, **fields)``: MERGE_EVENT for what the merge passes over (the
        # merger's default: one ``YETO_RL_EVENT`` stdout record, as the trainer's other events)
        self.merge_events: Callable[..., None] | None = print_merge_event if merge_shards else None
        self._shard_readers: dict[str, JsonlIndex | ObjectIndex] = {}
        self._shards_ended: set[str] = set()
        self._foreign_shards: set[str] = set()  # shards seen but not expected (skipped)
        self.batch_size = int(batch_size)
        self.arrival_window_s = float(arrival_window_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: list[SampleIndexEntry] = []
        self._ended = False
        self._consumer: str | None = None
        self._read = 0  # next position select() returns
        self._watermark = 0  # positions below are consumed (committed)
        self._callbacks: list[Callable[[], None]] = []
        self._polled = 0  # entries seen by the last poll
        self._poller: threading.Thread | None = None
        self._stop_polling = threading.Event()
        self.refresh()
        self._polled = len(self._entries)

    # ---------------------------------------------------------------- writing
    def write_payload(self, sample_id: str, data: bytes) -> str:
        """Store one payload; returns the ``payload_ref`` for its index entry.
        A payload is written once: a name already in the store is refused (a
        replacement sampler must never overwrite a group an index entry names)."""
        name = payload_name(sample_id)
        if self.index.has_payload(f"{PAYLOADS}/{name}"):
            raise SampleBufferError(f"payload {name} is already in the sample buffer; "
                                    "a payload is never overwritten")
        return self.index.write_payload(name, data)

    def read_payload(self, payload_ref: str) -> bytes:
        return self.index.read_payload(payload_ref)

    def append(self, entry: SampleIndexEntry) -> None:
        if entry.payload_ref and not self.index.has_payload(entry.payload_ref):
            raise SampleBufferError(f"sample {entry.sample_id}: payload {entry.payload_ref} is missing")
        if entry.appended_at == 0.0:
            entry = dataclasses.replace(entry, appended_at=self.clock())
        # ``provenance`` is written only when set, so a line without it reads as before.
        source = {} if entry.provenance is None else {"provenance": dict(entry.provenance)}
        if entry.sampler_round is not None:
            source["sampler_round"] = dict(entry.sampler_round)
        if self._writer is not None and (entry.sampler_round or {}).get("incarnation") != self._writer:
            # the trainer would drop it: its validity is its round record's incarnation
            raise SampleBufferError(f"sample {entry.sample_id}: this shard writer claimed the shard as "
                                    f"{self._writer!r}, the group's round record names "
                                    f"{(entry.sampler_round or {}).get('incarnation')!r}")
        self._write_record(lambda pos: {"schema": SCHEMA, "pos": pos, "sample_id": entry.sample_id,
                                        "behavior_version": int(entry.behavior_version),
                                        "payload_ref": entry.payload_ref, "tokens": int(entry.tokens),
                                        "appended_at": float(entry.appended_at), **source})

    @contextmanager
    def one_commit(self) -> Iterator[None]:
        """Appends inside the block reach the store with one commit when it ends.
        A Modal Volume commit costs seconds, so a sampler appends the groups of a
        round inside one block; other islands see them after it. An object-store
        index has no commit: its appends are visible one by one, as before."""
        hold = getattr(self.index, "one_commit", None)
        with hold() if callable(hold) else nullcontext():
            yield

    def mark_data_end(self) -> None:
        """The samplers will append nothing more. A shard writer that lost its shard
        (:class:`ShardFenced`) cannot end it: the owner may still write."""
        who = {} if self._writer is None else {"incarnation": self._writer}
        self._write_record(lambda pos: {"schema": SCHEMA, "pos": pos, "end": True, **who})

    def _write_record(self, make: Callable[[int], dict]) -> None:
        def record() -> dict:
            if self._ended:
                raise SampleBufferError("append after the data end")
            self._check_owner()  # after the index was read: the fence record is in, or the create loses
            return make(self._positions)

        with self._lock:
            self.index.append(self._absorb, record)

    # ---------------------------------------------------------------- shard ownership
    @property
    def can_own(self) -> bool:
        """Whether this buffer writes a shard that can hold ownership records (a
        shard on an object store: create-if-absent decides the positions)."""
        return self.shard is not None and isinstance(self.index, ObjectIndex)

    @property
    def owner(self) -> ShardOwner | None:
        """The newest ownership record of this writer's shard, as far as read."""
        with self._lock:
            return self._owner

    @property
    def writer_incarnation(self) -> str | None:
        """The incarnation this writer claimed its shard as (None: it wrote no claim)."""
        return self._writer

    @property
    def fenced(self) -> bool:
        """This writer's shard is owned by another incarnation or a pool fence (as of
        the last read; :meth:`refresh` reads again). Every write raises :class:`ShardFenced`."""
        with self._lock:
            return self.shard is not None and self._owner is not None and self._owner.incarnation != self._writer

    def claim_owner(self, incarnation: str, *, by: str = "sampler", first_only: bool = False) -> ShardOwner:
        """Append this shard's ownership record for ``incarnation`` (module doc) and
        write as it from now on: every earlier incarnation's later write is void.
        Returns the record's owner. ``first_only``: only when the shard has no
        ownership record yet (a running writer that starts to use them must not take
        the shard back from a fence): else :class:`ShardFenced`, nothing written."""
        if not self.can_own:
            raise SampleBufferError("shard ownership records need a shard index on an object store")
        made: list[dict[str, Any]] = []

        def record() -> dict:
            if first_only and self._owner is not None:
                raise ShardFenced(self.shard, str(incarnation), self._owner)
            epoch = (0 if self._owner is None else self._owner.epoch) + 1
            made.append({"schema": OWNER_SCHEMA, "pos": self._positions,
                         "owner": {"incarnation": str(incarnation), "epoch": epoch, "by": str(by),
                                   "claimed_at": float(self.clock())}})
            return made[-1]

        with self._lock:
            self.index.append(self._absorb, record)
            self._writer = str(incarnation)
            return _owner_of(made[-1])

    def _check_owner(self) -> None:
        if self.shard is not None and self._owner is not None and self._owner.incarnation != self._writer:
            raise ShardFenced(self.shard, self._writer, self._owner)

    def groups_from(self, position: int) -> int:
        """A shard writer: its groups at or after shard position ``position`` (the
        trainer's read position in this shard); ownership records are not groups."""
        with self._lock:
            lo, hi = 0, len(self._entry_pos)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._entry_pos[mid] < int(position):
                    lo = mid + 1
                else:
                    hi = mid
            return len(self._entry_pos) - lo

    # ---------------------------------------------------------------- shard ownership (the merger)
    def shard_owners(self) -> dict[str, dict[str, Any]]:
        """{shard: its newest owner as far as merged} of the shards with an ownership record."""
        with self._lock:
            return {shard: owner.to_dict() for shard, owner in sorted(self._shard_owner.items())
                    if owner is not None}

    def shard_epoch_at(self, position: int) -> int:
        """The owner epoch of merged position ``position``: 0 for a record of a
        shard's legacy part (no ownership record before it)."""
        with self._lock:
            return self._shard_epoch_of[int(position)] if 0 <= int(position) < len(self._shard_epoch_of) else 0

    def take_fenced_dropped(self) -> list[dict[str, Any]]:
        """The fenced incarnations' writes the merge dropped since the last call:
        ``{"shard", "epoch", "owner", "incarnation", "groups"}`` per shard, owner
        epoch and writer."""
        with self._lock:
            out = [{"shard": shard, "epoch": epoch, "owner": owner, "incarnation": inc, "groups": n}
                   for (shard, epoch, owner, inc), n in sorted(self._fenced_dropped.items(), key=str)]
            self._fenced_dropped.clear()
            return out

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        """Append ``records`` (without ``pos``) at the next positions: one write on
        a shared directory, one object per record on an object store. Idempotent
        for merged shard records (with ``shard`` and ``shard_pos``) and the ``end``
        record: the records to write are decided after the index was read (again
        after every lost create), and one the index already holds is left out -
        another merger appended it, or this one did and lost the answer."""
        written: set[int] = set()  # of ``records``: created by this call
        offered: list[int] = []  # the ones the last ``rest`` returned, ``counted`` created before it
        counted = 0

        def rest(done: int = 0) -> list[dict]:
            nonlocal offered, counted
            written.update(offered[:done - counted])
            counted = done
            offered = [i for i, record in enumerate(records)
                       if i not in written and not self._is_merged(record)]
            if self._ended:
                if offered:
                    raise SampleBufferError("append after the data end")
                return []
            start = self._positions
            return [{**records[i], "pos": start + n} for n, i in enumerate(offered)]

        with self._lock:
            if isinstance(self.index, JsonlIndex):
                self.index.append_many(self._absorb, rest)
            else:  # an object store: create-if-absent per record
                self.index.append_batch(self._absorb, rest)

    def _is_merged(self, record: Mapping[str, Any]) -> bool:
        """Whether the index already holds this merged shard record (or, for the
        ``end`` record, has ended). A record of no shard is never taken for merged."""
        if record.get("end"):
            return self._ended
        if record.get("marker"):
            return self._format_marked
        if record.get("shard") is None:
            return False
        seen = self._shard_seen.get(record["shard"])
        return seen is not None and int(record["shard_pos"]) in seen

    def _absorb(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            # a shard writer on an object store passes over a damaged record of its own shard,
            # as the trainer does: it holds a position, it is no entry
            bad = (bad_shard_record(record) if self.shard is not None and getattr(self.index, "tolerant", False)
                   else None)
            owner = bad is None and self.shard is not None and is_owner_record(record)
            other = self.shard is None and record.get("schema") == OWNER_SCHEMA and (
                record.get("marker") == FORMAT_MARKER or (is_owner_record(record) and record.get("shard") is not None))
            if ((record.get("schema") != SCHEMA and not owner and not other and bad is None)
                    or int(record["pos"]) != self._positions):
                raise SampleBufferError(f"sample buffer: record out of order: {json.dumps(record)[:200]}")
            self._positions += 1
            if owner:
                self._owner = _owner_of(record)
                continue
            if other:
                self._absorb_non_group(record)
                continue
            if bad is not None:
                if not self.bad_records_seen:
                    print(f"[sample-buffer] WARNING: shard {self.shard!r} record {record.get('pos')} is "
                          f"damaged ({bad}): passed over", flush=True)
                self.bad_records_seen += 1
                continue
            if self.shard is not None and not owner_accepts(self._owner, record):
                self.fenced_writes_seen += 1  # a fenced incarnation's write: void for every reader
                continue
            if record.get("end"):
                self._ended = True
                continue
            shard, shard_pos = record.get("shard"), None
            if shard is not None:
                seen = self._shard_seen.get(shard)
                if seen is None:
                    seen = self._shard_seen[shard] = _SeenPositions()
                    self._shard_marks[shard] = ([], [])
                # a merged record written without ``shard_pos`` is the next of its shard
                shard_pos = seen.high if record.get("shard_pos") is None else int(record["shard_pos"])
                if shard_pos in seen:
                    self._skip_duplicate(record, shard, shard_pos)
                else:
                    # positions the merger passed over on purpose (ownership records, fenced
                    # or damaged records) are no gap: the merged record says how many
                    for passed in range(max(seen.high, shard_pos - int(record.get("shard_passed") or 0)),
                                        shard_pos):
                        seen.add(passed)
                    if shard_pos > seen.high:
                        self._note_gap(shard, seen.high, shard_pos)
                    seen.add(shard_pos)
                    self._shard_next[shard] = seen.high
                    if record.get("shard_epoch") or not self._shard_last.get(shard, (None, 0))[1]:
                        self._shard_last[shard] = (record_incarnation(record), int(record.get("shard_epoch") or 0))
                    # else: merged by older code after an ownership record (no ``shard_epoch``,
                    # no validity check): the shard's owner is still the one known before it
                positions, highs = self._shard_marks[shard]
                positions.append(len(self._entries))
                highs.append(seen.high)
            self._entry_pos.append(self._positions - 1)
            self._shard_epoch_of.append(int(record.get("shard_epoch") or 0))
            self._shard_of.append(shard)
            self._shard_pos_of.append(shard_pos)
            self._entries.append(_entry(record))

    def _absorb_non_group(self, record: Mapping[str, Any]) -> None:
        """A v2 record of the merged index: the format marker, or a shard's ownership
        record that a merger of older code copied (module doc). It keeps its position
        and is never selected."""
        if record.get("marker") == FORMAT_MARKER:
            self._format_marked = True
        else:
            shard, shard_pos, owner = str(record["shard"]), record.get("shard_pos"), _owner_of(record)
            if not self.foreign_merged_records:
                print(f"[sample-buffer] WARNING: merged index position {record['pos']} is an ownership "
                      f"record of shard {shard!r} (copied by a merger of older code): passed over",
                      flush=True)
            self.foreign_merged_records += 1
            if _is_int(shard_pos):  # the shard continues after it, under this owner
                seen = self._shard_seen.setdefault(shard, _SeenPositions())
                self._shard_marks.setdefault(shard, ([], []))
                if shard_pos not in seen:
                    seen.add(int(shard_pos))
                    self._shard_next[shard] = seen.high
                    self._shard_last[shard] = (owner.incarnation, owner.epoch)
        self._skipped.add(len(self._entries))
        self._entry_pos.append(self._positions - 1)
        self._shard_epoch_of.append(0)
        self._shard_of.append(None)
        self._shard_pos_of.append(None)
        self._entries.append(SampleIndexEntry(sample_id="", behavior_version=0, payload_ref="", tokens=0,
                                              appended_at=0.0))

    def _skip_duplicate(self, record: Mapping[str, Any], shard: str, shard_pos: int) -> None:
        """The index holds this shard record a second time (two mergers, or a lost
        create answer, before the merge was idempotent): its position stays in the
        index, but it is never selected."""
        if not self._duplicates:
            print(f"[sample-buffer] WARNING: merged index position {record['pos']} repeats shard "
                  f"{shard!r} record {shard_pos} ({record.get('sample_id')!r}), merged before: "
                  f"repeated records are passed over, not trained twice", flush=True)
        self._skipped.add(len(self._entries))
        self._duplicates += 1
        self._merge_event("duplicate_merged_record", shard=shard, shard_pos=shard_pos,
                          position=int(record["pos"]), count=self._duplicates)

    def _note_gap(self, shard: str, merged: int, shard_pos: int) -> None:
        if not self._gap_warned:
            self._gap_warned = True
            print(f"[sample-buffer] WARNING: the merged index goes from shard {shard!r} record "
                  f"{merged - 1} to {shard_pos}: {shard_pos - merged} record(s) of the shard were passed "
                  f"over (an earlier merger counted repeated records, or merged the shard out of "
                  f"its order); unless the index holds them further on they stay unmerged", flush=True)
        self._merge_event("unmerged_shard_gap", shard=shard, shard_pos=merged, records=shard_pos - merged)

    @property
    def duplicate_merged_records(self) -> int:
        """The merged records read so far that repeat an earlier one (passed over)."""
        with self._lock:
            return self._duplicates

    @property
    def unmerged_shard_gaps(self) -> int:
        """The shard records an earlier merger passed over, as far as read: the shard
        positions below a shard's highest merged one that the index does not hold (a
        record merged out of its shard's order counts until it is read)."""
        with self._lock:
            return sum(seen.missing for seen in self._shard_seen.values())

    # ---------------------------------------------------------------- reading
    def refresh(self) -> int:
        """Read the records other writers added (after merging the shards, when this
        buffer merges); returns the number of samples."""
        with self._lock:
            self._absorb(self.index.read_new())
            if self._merge:
                self._merge_shards()
            return len(self._entries)

    def _merge_shards(self) -> None:
        new: list[list[dict[str, Any]]] = []  # per shard, in shard order
        for shard in self.index.shard_names():
            if self._expected_shards is not None and shard not in self._expected_shards:
                # a shard this trainer was not launched with (another run's or a stray
                # sampler's under the same prefix): never train on it
                if shard not in self._foreign_shards:
                    self._foreign_shards.add(shard)
                    print(f"[sample-buffer] WARNING: shard {shard!r} has groups but is not one of the "
                          f"expected shards {sorted(self._expected_shards)} (--rl-buffer-shards): its "
                          f"groups are NOT trained on; a sampler under that name is wasted", flush=True)
                continue
            reader = self._shard_readers.get(shard)
            if reader is None:  # first seen here: skip what an earlier merger merged
                reader = self._shard_readers[shard] = self.index.shard_reader(
                    shard, self._shard_next.get(shard, 0))
                # the owner there: the last merged record was valid, so its incarnation
                # owned the shard (epoch 0: the legacy part, no owner yet)
                last, epoch = self._shard_last.get(shard, (None, 0))
                self._shard_owner[shard] = ShardOwner(str(last), epoch) if epoch else None
            # What was read stays pending until the index holds it: a merge that fails
            # half way (a store error) is finished by the next refresh, and a record
            # another merger appended meanwhile is dropped here, not merged again.
            merged = self._shard_next.get(shard, 0)
            records = [r for r in self._shard_pending.get(shard, ()) if r["shard_pos"] >= merged]
            new_records = reader.read_new()
            self._note_blocked(shard, getattr(reader, "blocked_at", None))
            for record in new_records:
                owner = self._shard_owner.get(shard)
                taken = False
                if (bad := bad_shard_record(record)) is not None:
                    self._drop_bad(shard, record, bad)
                elif is_owner_record(record):
                    self._shard_owner[shard] = _owner_of(record)
                elif not owner_accepts(owner, record):
                    self._drop_fenced(shard, owner, record)
                elif record.get("end"):
                    self._shards_ended.add(shard)
                elif int(record["pos"]) >= merged:
                    taken = True
                    more = {} if owner is None else {"shard_epoch": owner.epoch}
                    if self._shard_passed.get(shard):
                        more["shard_passed"] = self._shard_passed[shard]
                    records.append({**record, "shard": shard, "shard_pos": int(record["pos"]), **more})
                # the reader starts right after the last merged record, so this counts
                # exactly the positions between two merged records of the shard
                self._shard_passed[shard] = 0 if taken else self._shard_passed.get(shard, 0) + 1
            self._shard_pending[shard] = records
            new.append(records)
        if self._ended:
            if first := next((r for records in new for r in records), None):
                raise SampleBufferError(f"shard records after the data end: {json.dumps(first)[:200]}")
            return
        # Interleave by appended_at but keep every shard in its own order: the merged
        # records of a shard must stay a prefix of it, because a restarted merger
        # resumes each shard after its highest merged position (a clock step back
        # would otherwise merge a later record first, then skip the earlier one).
        merged = list(heapq.merge(*new, key=lambda r: (float(r.get("appended_at", 0.0)), r["shard"])))
        expected = set(self._shard_readers) if self._expected_shards is None else self._expected_shards
        if expected and self._shards_ended >= expected:
            merged.append({"schema": SCHEMA, "end": True})
        if not self._format_marked and any(owner is not None for owner in self._shard_owner.values()):
            # before the first record merged from a shard with ownership records (module doc)
            merged.insert(0, {"schema": OWNER_SCHEMA, "marker": FORMAT_MARKER, "note": _MARKER_NOTE})
        if merged:
            self._write_records(merged)
            for shard, records in self._shard_pending.items():
                done = self._shard_next.get(shard, 0)
                self._shard_pending[shard] = [r for r in records if r["shard_pos"] >= done]

    def _merge_event(self, reason: str, **fields: Any) -> None:
        if self.merge_events is not None:
            try:
                self.merge_events(MERGE_EVENT, reason=reason, **fields)
            except Exception as error:  # noqa: BLE001 - observation only
                print(f"[sample-buffer] merge event not delivered: {type(error).__name__}: {error}", flush=True)

    def _drop_bad(self, shard: str, record: Mapping[str, Any], why: str) -> None:
        """A damaged shard record: passed over, never written into the merged index."""
        self.bad_shard_records += 1
        if shard not in self._bad_shards:
            self._bad_shards.add(shard)
            print(f"[sample-buffer] WARNING: shard {shard!r} record {record.get('pos')} is damaged "
                  f"({why}): passed over, not merged; the shard's other groups are", flush=True)
        self._merge_event("bad_record", shard=shard, shard_pos=record.get("pos"), detail=str(why)[:200],
                          count=self.bad_shard_records)

    def _note_blocked(self, shard: str, position: int | None) -> None:
        """A shard position is missing while later ones exist: reported, never skipped."""
        if position is None:
            self._blocked.pop(shard, None)
        elif self._blocked.get(shard) != position:
            self._blocked[shard] = position
            print(f"[sample-buffer] WARNING: shard {shard!r} has no record at position {position} but "
                  f"later ones: its groups from there on are NOT merged until that object is back "
                  f"(deleted from outside?)", flush=True)
            self._merge_event("missing_shard_position", shard=shard, shard_pos=position)

    @property
    def blocked_shards(self) -> dict[str, int]:
        """{shard: the missing position its merge waits at} (module doc)."""
        with self._lock:
            return dict(sorted(self._blocked.items()))

    def merge_counters(self) -> dict[str, int]:
        """Cumulative counts of what this reader's merge found and passed over."""
        with self._lock:
            return {"duplicate_merged_records": self._duplicates, "unmerged_shard_gaps": sum(seen.missing for seen in self._shard_seen.values()),
                    "stale_incarnation_groups": self.fenced_groups_dropped,
                    "bad_shard_records": self.bad_shard_records, "blocked_shards": len(self._blocked)}

    def _drop_fenced(self, shard: str, owner: ShardOwner, record: Mapping[str, Any]) -> None:
        """A shard record after an ownership record that names another incarnation
        (a fenced writer that bypassed its own check): never merged."""
        if record.get("end"):
            print(f"[sample-buffer] WARNING: shard {shard!r}: the data end at {record.get('pos')} was "
                  f"written by {record_incarnation(record)!r}, the shard is owned by {owner.incarnation!r} "
                  f"(epoch {owner.epoch}): ignored", flush=True)
            return
        if not self.fenced_groups_dropped:
            print(f"[sample-buffer] WARNING: shard {shard!r} record {record.get('pos')} "
                  f"({record.get('sample_id')!r}) was written by {record_incarnation(record)!r} after "
                  f"{owner.incarnation!r} took the shard (epoch {owner.epoch}): such groups are not "
                  f"merged, not trained", flush=True)
        self.fenced_groups_dropped += 1
        key = (shard, owner.epoch, owner.incarnation, record_incarnation(record))
        self._fenced_dropped[key] = self._fenced_dropped.get(key, 0) + 1
        self._merge_event("stale_incarnation", shard=shard, shard_pos=record.get("pos"), epoch=owner.epoch,
                          owner=owner.incarnation, incarnation=record_incarnation(record),
                          sample_id=record.get("sample_id"), count=self.fenced_groups_dropped)

    def shard_at(self, position: int) -> str | None:
        """The sampler shard of merged position ``position`` (None: no merged shards)."""
        with self._lock:
            return self._shard_of[int(position)] if 0 <= int(position) < len(self._shard_of) else None

    def shard_positions(self, cursor: int) -> dict[str, int]:
        """{shard: that shard's position below merged position ``cursor``}: the
        per-shard form of a consumed watermark or read cursor ({} for a buffer
        without merged shards). A shard's position is its highest ``shard_pos``
        below ``cursor`` plus one - the number of its records there, unless the
        index repeats one (not counted twice)."""
        with self._lock:
            if not 0 <= int(cursor) <= len(self._entries):
                raise SampleBufferError(f"cursor {cursor} outside the index (0..{len(self._entries)})")
            out: dict[str, int] = {}
            for shard, (positions, highs) in self._shard_marks.items():
                below = bisect.bisect_left(positions, int(cursor))  # its records below the cursor
                if below:
                    out[shard] = highs[below - 1]
            return dict(sorted(out.items()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def entries(self) -> list[SampleIndexEntry]:
        """The entries read so far, in buffer order (a copy)."""
        with self._lock:
            return list(self._entries)

    @property
    def data_ended(self) -> bool:
        with self._lock:
            return self._ended

    # ---------------------------------------------------------------- consuming
    @property
    def watermark(self) -> int:
        with self._lock:
            return self._watermark

    @property
    def read_cursor(self) -> int:
        with self._lock:
            return self._read

    def reset_to(self, watermark: int) -> None:
        """Continue at ``watermark`` (from the restored cut; 0 for a first start):
        samples from there are selected again."""
        with self._lock:
            self.refresh()
            if not 0 <= int(watermark) <= len(self._entries):
                raise SampleBufferError(f"watermark {watermark} outside the index (0..{len(self._entries)})")
            self._read = self._watermark = int(watermark)

    def select(self, consumer: str, limit: int) -> list[Selected]:
        with self._lock:
            if self._consumer is None:
                self._consumer = consumer
            elif consumer != self._consumer:
                raise SampleBufferError(f"buffer is read by {self._consumer!r}, not {consumer!r}")
            self.refresh()
            out, self._read = self._next_entries(limit)
            return out

    def _next_entries(self, limit: int) -> tuple[list[Selected], int]:
        """(the next ``limit`` entries from the read cursor, the cursor after them).
        A position of another lane, or one that repeats a merged shard record, is
        passed over: the cursor moves past it, it is not returned."""
        out: list[Selected] = []
        pos = self._read
        while len(out) < max(0, int(limit)) and pos < len(self._entries):
            if (self.lane is None or self.lane.owns(pos)) and pos not in self._skipped:
                out.append(Selected(pos, self._entries[pos]))
            pos += 1
        return out, pos

    def peek(self, limit: int) -> list[Selected]:
        """The entries :meth:`select` would return next (after a refresh), without
        selecting them: the read cursor does not move (a reader prefetches their
        payloads)."""
        with self._lock:
            self.refresh()
            return self._next_entries(limit)[0]

    def commit_consumed(self, cursor: int) -> None:
        """Every sample below ``cursor`` is trained; the next cut records it."""
        with self._lock:
            if not self._watermark <= int(cursor) <= self._read:
                raise SampleBufferError(
                    f"consumed cursor {cursor} outside {self._watermark}..{self._read} (watermark..selected)")
            self._watermark = int(cursor)

    def backlog(self, current_version: int, *, start: int | None = None,
                unmerged: UnmergedShards | None = None) -> BacklogStats:
        """Samples from ``start`` on; default: the samples not yet selected here.
        A head that does not read the buffer passes the trainer's committed
        watermark (:class:`CommittedBacklog`). ``unmerged``: also count the
        shard records not merged yet (their end counts as the data end)."""
        with self._lock:
            self.refresh()
            first = self._read if start is None else max(0, int(start))
            pending = self._entries[first:]
            if self._skipped:  # a repeated merged record is no backlog: nobody trains on it
                pending = [e for pos, e in enumerate(pending, start=first) if pos not in self._skipped]
            ended = self._ended
            if unmerged is not None:
                extra, shards_ended = unmerged.pending(self.shard_positions(len(self._entries)))
                pending = pending + extra
                ended = ended or shards_ended
            versions = Counter(e.behavior_version for e in pending)
            oldest = min(versions) if versions else None
            since = self.clock() - self.arrival_window_s
            recent = sum(1 for e in self._entries if e.appended_at >= since)
            if unmerged is not None:
                recent += sum(1 for e in extra if e.appended_at >= since)
            return BacklogStats(
                batches=len(pending) / self.batch_size, samples=len(pending), oldest_version=oldest,
                version_histogram=dict(sorted(versions.items())),
                arrival_rate=recent / self.arrival_window_s, data_ended=ended,
                max_version_gap=None if oldest is None else int(current_version) - oldest)

    # ---------------------------------------------------------------- arrival
    def on_arrival(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def poll(self) -> int:
        """Read the index once; call every arrival callback when samples arrived
        since the last poll. Returns the number of new samples."""
        with self._lock:
            new = self.refresh() - self._polled
            self._polled += max(new, 0)
            callbacks = list(self._callbacks) if new > 0 else []
        for callback in callbacks:
            callback()
        return max(new, 0)

    def start_polling(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if self._poller is not None:
            raise SampleBufferError("already polling")
        self._stop_polling.clear()

        def loop() -> None:
            while not self._stop_polling.wait(interval_s):
                self.poll()

        self._poller = threading.Thread(target=loop, name="sample-buffer-poll", daemon=True)
        self._poller.start()

    def stop_polling(self) -> None:
        self._stop_polling.set()
        if self._poller is not None:
            self._poller.join()
            self._poller = None


def _shard_name(shard: str) -> str:
    name = _SAFE_NAME.sub("_", str(shard))
    if not name:
        raise ValueError("shard name is empty")
    return name


def shard_index(shard: str) -> str:
    """The index of one appender's shard (relative to ``sample-buffer/``): the
    file of a shared-directory store; on an object store, :func:`index_for` maps
    it to :func:`shard_prefix`."""
    return f"{SHARDS}/{_shard_name(shard)}.jsonl"


def _shard_of_index(index: str) -> str:
    """The shard name of a :func:`shard_index` value."""
    head, _, name = index.partition("/")
    if head != SHARDS or not name.endswith(".jsonl") or "/" in name:
        raise SampleBufferError(f"index {index!r} is not a shard index")
    return name[:-len(".jsonl")]


def _complete_lines(data: bytes) -> list[dict]:
    """The JSON records of the complete lines (a torn last line is skipped)."""
    end = data.rfind(b"\n")
    return [json.loads(raw) for raw in data[:end + 1].splitlines() if raw.strip()] if end >= 0 else []


def shard_lengths(files, shards: Iterable[str]) -> dict[str, int]:
    """The complete sample records in each shard index (``files`` as in
    :func:`merge_shards`)."""
    return {str(shard): sum(1 for r in _complete_lines(
                files.read(f"{BUFFER_DIR}/{shard_index(shard)}") or b"") if not r.get("end"))
            for shard in shards}


def merge_shards(files, shards: Iterable[str]) -> int:
    """Append the new lines of each shard index to ``index.jsonl`` (the caller is the
    only writer of the index). ``files``: store access with ``read(rel)`` /
    ``write(rel, data)`` (``trainer_remote.StoreFiles``). New lines of all shards
    are merged in their ``appended_at`` order, every shard in its own order (a
    shard whose clock stepped back is not reordered: a reader takes a merged
    record below its shard's highest merged position for a repeat); returns the
    number merged."""
    index_rel = f"{BUFFER_DIR}/{INDEX}"
    index = files.read(index_rel) or b""
    end = index.rfind(b"\n")
    index = index[:end + 1]  # the merger writes whole lines; a torn tail is not ours
    merged = Counter()
    lines = _complete_lines(index)
    if any(r.get("end") for r in lines):
        raise SampleBufferError("merge after the data end")
    for record in lines:  # each shard continues after its highest merged position
        if record.get("shard") is not None:
            shard_pos = record.get("shard_pos")
            done = merged[record["shard"]]
            merged[record["shard"]] = max(done, (done if shard_pos is None else int(shard_pos)) + 1)
    per_shard: list[list[dict]] = []
    for shard in shards:
        records = [r for r in _complete_lines(files.read(f"{BUFFER_DIR}/{shard_index(shard)}") or b"")
                   if not r.get("end")]
        per_shard.append([{**record, "shard": shard, "shard_pos": shard_pos}
                          for shard_pos, record in enumerate(records[merged[shard]:], start=merged[shard])])
    # as FileSampleBuffer._merge_shards: by appended_at, never against a shard's own order
    new = list(heapq.merge(*per_shard, key=lambda r: (float(r.get("appended_at", 0.0)), r["shard"])))
    if not new:
        return 0
    out = [index]
    for pos, record in enumerate(new, start=len(lines)):
        out.append((json.dumps({**record, "pos": pos}, sort_keys=True) + "\n").encode())
    files.write(index_rel, b"".join(out))
    return len(new)


class CommittedBacklog:
    """The head's view of a buffer the trainer reads elsewhere (rl-trainer-start-stop
    rl-trainer-start-stop). The backlog starts at the consumed watermark the trainer committed
    into its newest cut (``watermark()``, e.g. ``StoreCutLog.watermark``), not at a
    cursor of this process. While a trainer runs, the samples it selected after its
    newest cut still count as backlog (at most the rounds between two cuts)."""

    def __init__(self, buffer: FileSampleBuffer, watermark: Callable[[], int | None],
                 *, unmerged: UnmergedShards | None = None) -> None:
        """``unmerged``: on a sharded object store, the shard records the trainer
        has not merged yet (they count as backlog)."""
        self.buffer = buffer
        self._watermark = watermark
        self.unmerged = unmerged

    def backlog(self, current_version: int) -> BacklogStats:
        return self.buffer.backlog(current_version, start=self._watermark() or 0, unmerged=self.unmerged)

    def append(self, entry: SampleIndexEntry) -> None:
        self.buffer.append(entry)

    def select(self, consumer: str, limit: int) -> list[Selected]:
        raise SampleBufferError("the head does not select: the trainer island reads the buffer")

    def commit_consumed(self, cursor: int) -> None:
        raise SampleBufferError("the head does not commit: the trainer island writes the watermark")

    def on_arrival(self, callback: Callable[[], None]) -> None:
        self.buffer.on_arrival(callback)


READ_POSITION = "read-position.json"  # under sample-buffer/: the trainer's read heartbeat
_NO_RELEASE_FIELD = object()  # write_read_position: an old-format heartbeat (no ``release``)


def buffer_dir(buffer: Any) -> Path | None:
    """The local ``sample-buffer/`` directory of a buffer (its store's), or None
    for a buffer without a local store (a relay, an object store)."""
    root = getattr(getattr(buffer, "store", None), "root", None)
    if root is None:
        return None
    try:
        return Path(root) / BUFFER_DIR
    except TypeError:
        return None


def write_read_position(buffer_root: str | Path, *, buffer_selected: int,
                        buffer_shards: dict[str, int] | None, train_step: int | None,
                        release: Mapping[str, Any] | None | object = _NO_RELEASE_FIELD,
                        clock: Callable[[], float] = time.time) -> Path:
    """Write the trainer's read position to ``<buffer_root>/read-position.json``
    (tmp + rename), after every group it consumes, released or not.
    ``train_step``: the trainer's last published policy version when it read.
    ``release``: ``{"seq", "policy_hash"}`` of the newest release this trainer knows
    (None: none yet). It ties the heartbeat to a release lineage, so a sampler can
    ignore a heartbeat left by another run or written before a newer release
    (``SamplerLoop.backlog``): sparse or slow releases no longer hold the cap shut."""
    root = Path(buffer_root)
    path = root / READ_POSITION
    payload = {"buffer_selected": int(buffer_selected),
               "buffer_shards": {str(k): int(v) for k, v in (buffer_shards or {}).items()},
               "train_step": None if train_step is None else int(train_step),
               "time_unix": round(float(clock()), 3)}
    if release is not _NO_RELEASE_FIELD:
        payload["release"] = None if release is None else {
            "seq": int(release["seq"]), "policy_hash": str(release["policy_hash"])}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_read_position(buffer_root: str | Path) -> dict[str, Any] | None:
    """The trainer's read heartbeat (:func:`write_read_position`), or None when it
    is missing or unreadable (the caller keeps the release position)."""
    try:
        data = json.loads((Path(buffer_root) / READ_POSITION).read_text(encoding="utf-8"))
        shards = data.get("buffer_shards") or {}
        beat = {"buffer_selected": int(data.get("buffer_selected") or 0),
                "buffer_shards": {str(k): int(v) for k, v in dict(shards).items()},
                "train_step": data.get("train_step"), "time_unix": data.get("time_unix")}
        if "release" in data:  # absent: an old-format heartbeat
            release = data["release"]
            beat["release"] = None if release is None else {
                "seq": int(release["seq"]), "policy_hash": str(release["policy_hash"])}
        return beat
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None
