"""Durable sample buffer: an append-only index next to the sample payloads.

rl-trainer-start-stop L4.1/L4.3, design D4. The buffer keeps samples across
trainer stops and head restarts. It lives in the existing cut storage (a
:class:`~yeto.rl.engine.resume.CheckpointStore`: local directory or bucket mount)
under ``sample-buffer/``:

* ``payloads/<name>`` holds one sample payload; it is written (tmp + rename)
  before its index line, so the index never names a missing payload;
* ``index.jsonl`` holds one JSON line per sample. The line number is the sample's
  position. A torn last line (crash during a write) is ignored by readers and cut
  off by the next writer.

The consumed watermark is the position before which every sample is consumed.
The buffer does not persist it: the trainer writes it into the cut record
(``ConsistentCut.buffer_watermark``), and the next start calls :meth:`reset_to`
with the watermark of the cut it restores. So a sample is consumed at most once
per cut lineage, and samples consumed after the last cut are consumed again.

Arrival notification is by polling the index (``poll`` / ``start_polling``).

Limits of this first implementation: one consumer; concurrent appenders must
share one host (``flock`` on the index); the version gate on selection is the
consumer's job (P0-S ``max_policy_age``).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .resume import CheckpointStore
from .trainer_lifecycle import BacklogStats, SampleIndexEntry, Selected

SCHEMA = "yeto.rl.sample-buffer/v1"
BUFFER_DIR = "sample-buffer"
INDEX = "index.jsonl"
PAYLOADS = "payloads"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class SampleBufferError(RuntimeError):
    pass


class FileSampleBuffer:
    """:class:`~yeto.rl.engine.trainer_lifecycle.DurableSampleBuffer` over a store.

    ``batch_size``: samples per training batch (``BacklogStats.batches``);
    ``arrival_window_s``: the window of ``BacklogStats.arrival_rate``."""

    def __init__(self, store: CheckpointStore, *, batch_size: int, arrival_window_s: float = 60.0,
                 clock: Callable[[], float] = time.time) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.store = store
        self.root = Path(store.root) / BUFFER_DIR
        (self.root / PAYLOADS).mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / INDEX
        self.index_path.touch(exist_ok=True)
        self.batch_size = int(batch_size)
        self.arrival_window_s = float(arrival_window_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: list[SampleIndexEntry] = []
        self._ended = False
        self._offset = 0  # bytes of complete index lines read
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
        name = _SAFE_NAME.sub("_", sample_id) + ".bin"
        path = self.root / PAYLOADS / name
        tmp = path.with_suffix(".bin.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return f"{PAYLOADS}/{name}"

    def read_payload(self, payload_ref: str) -> bytes:
        return (self.root / payload_ref).read_bytes()

    def append(self, entry: SampleIndexEntry) -> None:
        if entry.payload_ref and not (self.root / entry.payload_ref).is_file():
            raise SampleBufferError(f"sample {entry.sample_id}: payload {entry.payload_ref} is missing")
        if entry.appended_at == 0.0:
            entry = SampleIndexEntry(entry.sample_id, entry.behavior_version, entry.payload_ref,
                                     entry.tokens, self.clock())
        self._write_line(lambda pos: {"schema": SCHEMA, "pos": pos, "sample_id": entry.sample_id,
                                      "behavior_version": int(entry.behavior_version),
                                      "payload_ref": entry.payload_ref, "tokens": int(entry.tokens),
                                      "appended_at": float(entry.appended_at)})

    def mark_data_end(self) -> None:
        """The samplers will append nothing more."""
        self._write_line(lambda pos: {"schema": SCHEMA, "pos": pos, "end": True})

    def _write_line(self, make: Callable[[int], dict]) -> None:
        with self._lock, open(self.index_path, "r+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                self._read_new(fh)
                if self._ended:
                    raise SampleBufferError("append after the data end")
                fh.truncate(self._offset)  # drop a torn last line
                fh.seek(self._offset)
                line = (json.dumps(make(len(self._entries)), sort_keys=True) + "\n").encode()
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
                self._read_new(fh)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        self.store.commit()

    # ---------------------------------------------------------------- reading
    def refresh(self) -> int:
        """Read index lines other writers appended; returns the number of samples."""
        with self._lock:
            self.store.reload()
            with open(self.index_path, "rb") as fh:
                self._read_new(fh)
            return len(self._entries)

    def _read_new(self, fh) -> None:
        fh.seek(self._offset)
        data = fh.read()
        while True:
            end = data.find(b"\n")
            if end < 0:
                return  # nothing more, or a torn last line
            raw, data = data[:end], data[end + 1:]
            record = json.loads(raw)
            if record.get("schema") != SCHEMA or int(record["pos"]) != len(self._entries):
                raise SampleBufferError(f"{self.index_path}: bad line at byte {self._offset}: {raw[:200]!r}")
            self._offset += end + 1
            if record.get("end"):
                self._ended = True
                continue
            self._entries.append(SampleIndexEntry(
                sample_id=record["sample_id"], behavior_version=int(record["behavior_version"]),
                payload_ref=record["payload_ref"], tokens=int(record["tokens"]),
                appended_at=float(record["appended_at"])))

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
