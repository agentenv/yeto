"""Durable sample buffer: an ordered index of samples next to their payloads.

rl-trainer-start-stop L4.1/L4.3, design D4. The buffer keeps samples across
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
selection is the consumer's job (P0-S ``max_policy_age``).

Appenders on several hosts (convertible nodes, rl-trainer-start-stop L7.5c): each
appender writes its own shard index ``shards/<shard>.jsonl`` (same line format,
one writer per file) next to the shared payloads, and one merger -- the head --
appends the new shard lines to ``index.jsonl`` (:func:`merge_shards`). The
merged line keeps ``shard`` and ``shard_pos``, so a restarted merger continues
where the index ends.

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

import dataclasses
import fcntl
import heapq
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from .merged_release import BufferLane, merges_shards
from .object_store import ObjectStore
from .resume import CheckpointStore
from .trainer_lifecycle import BacklogStats, SampleIndexEntry, Selected

SCHEMA = "yeto.rl.sample-buffer/v1"
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


class SampleBufferError(RuntimeError):
    pass


def payload_name(sample_id: str) -> str:
    return _SAFE_NAME.sub("_", sample_id) + ".bin"


class JsonlIndex:
    """The index as ``index.jsonl`` in a shared directory (see the module doc)."""

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
        return (self.root / ref).read_bytes()

    def has_payload(self, ref: str) -> bool:
        return (self.root / ref).is_file()

    def read_new(self) -> list[dict[str, Any]]:
        if self._reload:
            self.store.reload()
        if not self.path.exists():  # a reader before the first merge (or a shard not created yet)
            return []
        with open(self.path, "rb") as fh:
            return list(self._lines(fh))

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
                 start: int = 0) -> None:
        """``prefix``: the record keys (a shard passes its own); ``start``: the
        position of the first record to read (a merger skips merged records)."""
        self.objects = objects
        self.prefix = prefix
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
        before it)."""
        out = []
        for key in self.objects.list(self.prefix, start_after=self._last_key):
            if key != self.key(self._next):
                break
            raw = self.objects.read(key)
            if raw is None:
                break
            try:
                out.append(json.loads(raw))
            except ValueError as error:
                raise SampleBufferError(f"{key}: bad record: {raw[:200]!r}") from error
            self._last_key = key
            self._next += 1
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

    # ------------------------------------------------------------ shards (merger side)
    def shard_names(self) -> list[str]:
        return shard_names(self.objects)

    def shard_reader(self, shard: str, start: int) -> "ObjectIndex":
        return ObjectIndex(self.objects, prefix=shard_prefix(shard), start=start)


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
    has not merged into the index yet (rl-cross-site-sample-store N3.6).

    The head reads the merged index only, and only the trainer merges. With the
    trainer scheduler on, a stopped trainer merges nothing, so without this view
    the head would never see a backlog and never start the trainer again.
    :meth:`pending` reads every shard by object key (never writes) and returns
    the records at or above the merged count of their shard."""

    def __init__(self, objects: ObjectStore) -> None:
        self.objects = objects
        self._readers: dict[str, ObjectIndex] = {}
        self._records: dict[str, list[SampleIndexEntry]] = {}
        self._dropped: dict[str, int] = {}  # records already merged and forgotten here
        self._ended: set[str] = set()

    def pending(self, merged: dict[str, int]) -> tuple[list[SampleIndexEntry], bool]:
        """(records not merged yet, every named shard has ended). ``merged``: the
        merged records per shard (:meth:`FileSampleBuffer.shard_positions` of
        the index length)."""
        names = shard_names(self.objects)
        for shard in names:
            reader = self._readers.get(shard)
            if reader is None:
                reader = self._readers[shard] = ObjectIndex(self.objects, prefix=shard_prefix(shard))
                self._records[shard], self._dropped[shard] = [], 0
            for record in reader.read_new():
                if record.get("end"):
                    self._ended.add(shard)
                else:
                    self._records[shard].append(_entry(record))
        out: list[SampleIndexEntry] = []
        for shard, records in self._records.items():
            done = max(0, int(merged.get(shard, 0)) - self._dropped[shard])
            del records[:done]
            self._dropped[shard] += done
            out.extend(records)
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
    return ObjectIndex(objects, prefix=shard_prefix(shard))


def open_trainer_buffer(store: CheckpointStore | ObjectStore, *, batch_size: int,
                        lane: BufferLane | None = None, create_index: bool = True,
                        **kwargs: Any) -> "FileSampleBuffer":
    """The buffer of a trainer (the reader): it merges the sampler shards of a
    store that shards (:func:`needs_shards`) unless it is a synced trainer other
    than the publisher (``merged_release.merges_shards``); ``lane``: its lane."""
    return FileSampleBuffer(store, batch_size=batch_size, create_index=create_index, lane=lane,
                            merge_shards=needs_shards(store) and merges_shards(lane), **kwargs)


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
        sampler island writes each; S25 C1: several sampler islands on one
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
        self._shard_readers: dict[str, JsonlIndex | ObjectIndex] = {}
        self._shards_ended: set[str] = set()
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
        """The samplers will append nothing more."""
        self._write_record(lambda pos: {"schema": SCHEMA, "pos": pos, "end": True})

    def _write_record(self, make: Callable[[int], dict]) -> None:
        def record() -> dict:
            if self._ended:
                raise SampleBufferError("append after the data end")
            return make(len(self._entries))

        with self._lock:
            self.index.append(self._absorb, record)

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        """Append ``records`` (without ``pos``) at the next positions: one write on
        a shared directory, one object per record on an object store."""
        def make() -> list[dict]:
            if self._ended:
                raise SampleBufferError("append after the data end")
            start = len(self._entries)
            return [{**record, "pos": start + i} for i, record in enumerate(records)]

        with self._lock:
            if isinstance(self.index, JsonlIndex):
                self.index.append_many(self._absorb, make)
                return
            for record in records:  # an object store: one create-if-absent per record
                self._write_record(lambda pos, record=record: {**record, "pos": pos})

    def _absorb(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if record.get("schema") != SCHEMA or int(record["pos"]) != len(self._entries):
                raise SampleBufferError(f"sample buffer: record out of order: {json.dumps(record)[:200]}")
            if record.get("end"):
                self._ended = True
                continue
            self._shard_of.append(record.get("shard"))
            self._entries.append(_entry(record))

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
            reader = self._shard_readers.get(shard)
            if reader is None:  # first seen here: skip what an earlier merger merged
                done = sum(1 for s in self._shard_of if s == shard)
                reader = self._shard_readers[shard] = self.index.shard_reader(shard, done)
            records = []
            for record in reader.read_new():
                if record.get("end"):
                    self._shards_ended.add(shard)
                else:
                    records.append({**record, "shard": shard, "shard_pos": int(record["pos"])})
            new.append(records)
        if self._ended:
            if first := next((r for records in new for r in records), None):
                raise SampleBufferError(f"shard records after the data end: {json.dumps(first)[:200]}")
            return
        # Interleave by appended_at but keep every shard in its own order: the merged
        # records of a shard must stay a prefix of it, because a restarted merger
        # resumes each shard after its merged count (a clock step back would
        # otherwise merge a later record first, then skip the earlier one).
        merged = list(heapq.merge(*new, key=lambda r: (float(r.get("appended_at", 0.0)), r["shard"])))
        expected = set(self._shard_readers) if self._expected_shards is None else self._expected_shards
        if expected and self._shards_ended >= expected:
            merged.append({"schema": SCHEMA, "end": True})
        if merged:
            self._write_records(merged)

    def shard_positions(self, cursor: int) -> dict[str, int]:
        """{shard: records of that shard below merged position ``cursor``}: the
        per-shard form of a consumed watermark or read cursor ({} for a buffer
        without merged shards)."""
        with self._lock:
            if not 0 <= int(cursor) <= len(self._entries):
                raise SampleBufferError(f"cursor {cursor} outside the index (0..{len(self._entries)})")
            return dict(sorted(Counter(s for s in self._shard_of[:int(cursor)] if s is not None).items()))

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
            out: list[Selected] = []
            pos = self._read
            while len(out) < max(0, int(limit)) and pos < len(self._entries):
                if self.lane is None or self.lane.owns(pos):
                    out.append(Selected(pos, self._entries[pos]))
                pos += 1
            self._read = pos
            return out

    def peek(self, limit: int) -> list[Selected]:
        """The entries :meth:`select` would return next (after a refresh), without
        selecting them: the read cursor does not move (a reader prefetches their
        payloads)."""
        with self._lock:
            self.refresh()
            out: list[Selected] = []
            pos = self._read
            while len(out) < max(0, int(limit)) and pos < len(self._entries):
                if self.lane is None or self.lane.owns(pos):
                    out.append(Selected(pos, self._entries[pos]))
                pos += 1
            return out

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
            pending = self._entries[self._read if start is None else max(0, int(start)):]
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
    are merged in their ``appended_at`` order; returns the number merged."""
    index_rel = f"{BUFFER_DIR}/{INDEX}"
    index = files.read(index_rel) or b""
    end = index.rfind(b"\n")
    index = index[:end + 1]  # the merger writes whole lines; a torn tail is not ours
    merged = Counter()
    lines = _complete_lines(index)
    if any(r.get("end") for r in lines):
        raise SampleBufferError("merge after the data end")
    for record in lines:
        if record.get("shard") is not None:
            merged[record["shard"]] += 1
    new: list[dict] = []
    for shard in shards:
        records = [r for r in _complete_lines(files.read(f"{BUFFER_DIR}/{shard_index(shard)}") or b"")
                   if not r.get("end")]
        for shard_pos, record in enumerate(records[merged[shard]:], start=merged[shard]):
            new.append({**record, "shard": shard, "shard_pos": shard_pos})
    if not new:
        return 0
    new.sort(key=lambda r: (float(r.get("appended_at", 0.0)), r["shard"], r["shard_pos"]))
    out = [index]
    for pos, record in enumerate(new, start=len(lines)):
        out.append((json.dumps({**record, "pos": pos}, sort_keys=True) + "\n").encode())
    files.write(index_rel, b"".join(out))
    return len(new)


class CommittedBacklog:
    """The head's view of a buffer the trainer reads elsewhere (rl-trainer-start-stop
    L5.3c, D4). The backlog starts at the consumed watermark the trainer committed
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
