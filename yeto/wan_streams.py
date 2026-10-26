"""Per-stream accounting, stall detection and rate-aware striping for the
island side of a striped syncer connection (S26 E3b).

A ``SyncerClient`` stripes one logical message over ``num_streams`` TCP data
sockets as 4 MiB CHUNK envelopes. The syncer reassembles a message by
``(msg_id, offset)``, so any chunk may travel on any data stream of the same
connection group. This module holds the parts of that transport that do not
need a socket:

* :class:`StreamMeter` counts bytes, busy time and the last byte progress of
  one stream. The send loop feeds it; diagnostics and the stall watchdog read
  it.
* :func:`pick_stream` chooses the data stream for the next chunk. ``"rr"``
  is the historical round robin. ``"adaptive"`` sends each chunk to the
  stream with the earliest estimated finish time (backlog / measured rate),
  so a slow stream gets fewer chunks and a stalled stream gets none. The
  number of open streams is fixed by HELLO; the number of streams that carry
  data follows the measured rates.
* :class:`StreamStalledError` is the failure a watchdog reports when a
  stream holds unsent bytes and makes no byte progress for the stall timeout.
  The client handles it like any socket error: the whole group is torn down
  and redialed, and the syncer replays its PULLs, so the island pushes the
  interrupted fragment again at the same (global_step, round_attempt).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

STREAM_SCHEDULES = ("rr", "adaptive")

# EWMA weight of the newest completed send in the per-stream rate estimate.
RATE_EWMA_ALPHA = 0.3
# Sends shorter than this carry no useful rate signal (kernel buffer absorbs
# them), so they do not update the estimate.
MIN_RATE_SAMPLE_SECONDS = 0.01


class StreamStalledError(ConnectionError):
    """One stream held unsent (or partially received) bytes with no progress
    for longer than the stall timeout."""


@dataclass(frozen=True)
class StreamSnapshot:
    """Cumulative counters of one stream at one instant."""

    bytes_sent: int
    busy_seconds: float
    backlog_bytes: int
    rate_bps: float | None


class StreamMeter:
    """Counters of one outbound stream. Thread-safe; cheap enough to update
    on every ``send`` call."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.bytes_sent = 0
        self.busy_seconds = 0.0
        self.queued_bytes = 0  # enqueued, not yet handed to the kernel
        self.rate_bps: float | None = None
        self._item_started: float | None = None
        self._item_bytes = 0
        self._last_progress = clock()

    def enqueued(self, nbytes: int) -> None:
        with self._lock:
            self.queued_bytes += nbytes

    def begin(self) -> None:
        with self._lock:
            now = self._clock()
            self._item_started = now
            self._item_bytes = 0
            self._last_progress = now

    def progress(self, nbytes: int) -> None:
        with self._lock:
            self.bytes_sent += nbytes
            self.queued_bytes -= nbytes
            self._item_bytes += nbytes
            self._last_progress = self._clock()

    def end(self) -> None:
        with self._lock:
            if self._item_started is None:
                return
            elapsed = self._clock() - self._item_started
            self.busy_seconds += elapsed
            if elapsed >= MIN_RATE_SAMPLE_SECONDS and self._item_bytes:
                sample = self._item_bytes / elapsed
                self.rate_bps = (
                    sample
                    if self.rate_bps is None
                    else (1 - RATE_EWMA_ALPHA) * self.rate_bps + RATE_EWMA_ALPHA * sample
                )
            self._item_started = None

    def drop_pending(self) -> None:
        """The connection group died: whatever was queued is gone."""
        with self._lock:
            self.queued_bytes = 0
            self._item_started = None
            self._item_bytes = 0

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._item_started is None and self.queued_bytes <= 0

    def stalled_for(self, now: float | None = None) -> float:
        """Seconds without byte progress while bytes are waiting; 0 when idle."""
        with self._lock:
            if self._item_started is None and self.queued_bytes <= 0:
                return 0.0
            return max(0.0, (self._clock() if now is None else now) - self._last_progress)

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            busy = self.busy_seconds
            if self._item_started is not None:
                busy += self._clock() - self._item_started
            return StreamSnapshot(self.bytes_sent, busy, max(0, self.queued_bytes), self.rate_bps)


def pick_stream(
    schedule: str,
    meters: Sequence[StreamMeter],
    rr_ticket: int,
) -> int:
    """Index into ``meters`` (the data streams) for the next chunk.

    ``adaptive`` minimizes ``backlog / rate``. A stream without a rate sample
    yet uses the mean measured rate (or 1 B/s when nothing is measured, which
    reduces the rule to least backlog). Ties go to the round-robin ticket so
    an idle start still spreads chunks over every stream."""
    count = len(meters)
    if count == 0:
        raise ValueError("no data streams")
    if schedule == "rr":
        return rr_ticket % count
    if schedule != "adaptive":
        raise ValueError(f"unknown stream schedule {schedule!r}")
    snaps = [meter.snapshot() for meter in meters]
    known = [snap.rate_bps for snap in snaps if snap.rate_bps]
    default_rate = sum(known) / len(known) if known else 1.0
    best = None
    for step in range(count):
        index = (rr_ticket + step) % count
        snap = snaps[index]
        finish = snap.backlog_bytes / (snap.rate_bps or default_rate)
        if best is None or finish < best[0]:
            best = (finish, index)
    return best[1]


@dataclass(frozen=True)
class UploadWindow:
    """Data-stream traffic between two snapshots (one sync boundary)."""

    bytes_per_stream: tuple[int, ...]
    busy_seconds_per_stream: tuple[float, ...]

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_per_stream)

    def stream_mbps(self) -> list[float | None]:
        """MB/s (1e6 B) of each stream over the time it was busy."""
        return [
            (b / s / 1e6) if s > 0 and b > 0 else None
            for b, s in zip(self.bytes_per_stream, self.busy_seconds_per_stream, strict=True)
        ]


def upload_window(
    before: Sequence[StreamSnapshot], after: Sequence[StreamSnapshot]
) -> UploadWindow:
    if len(before) != len(after):  # the group was redialed with another width
        before = [StreamSnapshot(0, 0.0, 0, None)] * len(after)
    return UploadWindow(
        tuple(max(0, a.bytes_sent - b.bytes_sent) for b, a in zip(before, after, strict=True)),
        tuple(max(0.0, a.busy_seconds - b.busy_seconds) for b, a in zip(before, after, strict=True)),
    )
