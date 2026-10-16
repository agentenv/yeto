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

Limits of this first implementation: one consumer; with ``index.jsonl``,
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

import fcntl
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

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

    def __init__(self, store: CheckpointStore, *, index: str = INDEX, create_index: bool = True) -> None:
        """``index``: the index file under ``sample-buffer/`` (a shard writer
        passes :func:`shard_index`). ``create_index=False``: a reader that must not
        write the index, not even an empty one (on a store mount whose commit would
        replace the merger's index)."""
        self.store = store
        self.root = Path(store.root) / BUFFER_DIR
        (self.root / PAYLOADS).mkdir(parents=True, exist_ok=True)
        self.path = self.root / index
        if create_index:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch()
        self._offset = 0  # bytes of complete lines read

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
        self.store.reload()
        if not self.path.exists():  # a reader before the first merge
            return []
        with open(self.path, "rb") as fh:
            return list(self._lines(fh))

    def append(self, absorb: Callable[[list[dict[str, Any]]], None],
               make: Callable[[], dict[str, Any]]) -> None:
        with open(self.path, "r+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                absorb(list(self._lines(fh)))
                record = make()
                fh.truncate(self._offset)  # drop a torn last line
                fh.seek(self._offset)
                fh.write((json.dumps(record, sort_keys=True) + "\n").encode())
                fh.flush()
                os.fsync(fh.fileno())
                absorb(list(self._lines(fh)))
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        self.store.commit()

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


class FileSampleBuffer:
    """:class:`~yeto.rl.engine.trainer_lifecycle.DurableSampleBuffer` over a store
    (a :class:`CheckpointStore`, or an :class:`ObjectStore` at the store root).

    ``batch_size``: samples per training batch (``BacklogStats.batches``);
    ``arrival_window_s``: the window of ``BacklogStats.arrival_rate``."""

    def __init__(self, store: CheckpointStore | ObjectStore, *, batch_size: int,
                 arrival_window_s: float = 60.0, clock: Callable[[], float] = time.time,
                 index: str = INDEX, create_index: bool = True, merge_shards: bool = False) -> None:
        """``index``/``create_index``: see :class:`JsonlIndex`. ``merge_shards``:
        this buffer (the consumer) merges the sampler shards of an object store
        into the index on every refresh (see the module doc); it must be the
        only one that does."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.store = store
        self.index = index_for(store, index=index, create_index=create_index)
        self.shard = _shard_of_index(index) if index != INDEX else None  # the shard this buffer writes
        if merge_shards and (self.shard is not None or not isinstance(self.index, ObjectIndex)):
            raise SampleBufferError("only the index of an object store merges shards")
        self._merge = merge_shards
        self._shard_of: list[str | None] = []  # the shard of each position (merged records)
        self._shard_readers: dict[str, ObjectIndex] = {}
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
        """Store one payload; returns the ``payload_ref`` for its index entry."""
        return self.index.write_payload(payload_name(sample_id), data)

    def read_payload(self, payload_ref: str) -> bytes:
        return self.index.read_payload(payload_ref)

    def append(self, entry: SampleIndexEntry) -> None:
        if entry.payload_ref and not self.index.has_payload(entry.payload_ref):
            raise SampleBufferError(f"sample {entry.sample_id}: payload {entry.payload_ref} is missing")
        if entry.appended_at == 0.0:
            entry = SampleIndexEntry(entry.sample_id, entry.behavior_version, entry.payload_ref,
                                     entry.tokens, self.clock())
        self._write_record(lambda pos: {"schema": SCHEMA, "pos": pos, "sample_id": entry.sample_id,
                                        "behavior_version": int(entry.behavior_version),
                                        "payload_ref": entry.payload_ref, "tokens": int(entry.tokens),
                                        "appended_at": float(entry.appended_at)})

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

    def _absorb(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if record.get("schema") != SCHEMA or int(record["pos"]) != len(self._entries):
                raise SampleBufferError(f"sample buffer: record out of order: {json.dumps(record)[:200]}")
            if record.get("end"):
                self._ended = True
                continue
            self._shard_of.append(record.get("shard"))
            self._entries.append(SampleIndexEntry(
                sample_id=record["sample_id"], behavior_version=int(record["behavior_version"]),
                payload_ref=record["payload_ref"], tokens=int(record["tokens"]),
                appended_at=float(record["appended_at"])))

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
        objects = self.index.objects
        new: list[dict[str, Any]] = []
        for shard in shard_names(objects):
            reader = self._shard_readers.get(shard)
            if reader is None:  # first seen here: skip what an earlier merger merged
                done = sum(1 for s in self._shard_of if s == shard)
                reader = self._shard_readers[shard] = ObjectIndex(objects, prefix=shard_prefix(shard),
                                                                  start=done)
            for record in reader.read_new():
                if record.get("end"):
                    self._shards_ended.add(shard)
                else:
                    new.append({**record, "shard": shard, "shard_pos": int(record["pos"])})
        if self._ended:
            if new:
                raise SampleBufferError(f"shard records after the data end: {json.dumps(new[0])[:200]}")
            return
        new.sort(key=lambda r: (float(r.get("appended_at", 0.0)), r["shard"], r["shard_pos"]))
        for record in new:
            self._write_record(lambda pos, record=record: {**record, "pos": pos})
        if self._shard_readers and self._shards_ended >= set(self._shard_readers):
            self._write_record(lambda pos: {"schema": SCHEMA, "pos": pos, "end": True})

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
            stop = min(len(self._entries), self._read + max(0, int(limit)))
            out = [Selected(pos, self._entries[pos]) for pos in range(self._read, stop)]
            self._read = stop
            return out

    def commit_consumed(self, cursor: int) -> None:
        """Every sample below ``cursor`` is trained; the next cut records it."""
        with self._lock:
            if not self._watermark <= int(cursor) <= self._read:
                raise SampleBufferError(
                    f"consumed cursor {cursor} outside {self._watermark}..{self._read} (watermark..selected)")
            self._watermark = int(cursor)

    def backlog(self, current_version: int, *, start: int | None = None) -> BacklogStats:
        """Samples from ``start`` on; default: the samples not yet selected here.
        A head that does not read the buffer passes the trainer's committed
        watermark (:class:`CommittedBacklog`)."""
        with self._lock:
            self.refresh()
            pending = self._entries[self._read if start is None else max(0, int(start)):]
            versions = Counter(e.behavior_version for e in pending)
            oldest = min(versions) if versions else None
            since = self.clock() - self.arrival_window_s
            recent = sum(1 for e in self._entries if e.appended_at >= since)
            return BacklogStats(
                batches=len(pending) / self.batch_size, samples=len(pending), oldest_version=oldest,
                version_histogram=dict(sorted(versions.items())),
                arrival_rate=recent / self.arrival_window_s, data_ended=self._ended,
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

    def __init__(self, buffer: FileSampleBuffer, watermark: Callable[[], int | None]) -> None:
        self.buffer = buffer
        self._watermark = watermark

    def backlog(self, current_version: int) -> BacklogStats:
        return self.buffer.backlog(current_version, start=self._watermark() or 0)

    def append(self, entry: SampleIndexEntry) -> None:
        self.buffer.append(entry)

    def select(self, consumer: str, limit: int) -> list[Selected]:
        raise SampleBufferError("the head does not select: the trainer island reads the buffer")

    def commit_consumed(self, cursor: int) -> None:
        raise SampleBufferError("the head does not commit: the trainer island writes the watermark")

    def on_arrival(self, callback: Callable[[], None]) -> None:
        self.buffer.on_arrival(callback)
