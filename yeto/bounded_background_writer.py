"""Bounded, fail-closed FIFO execution for immutable capture payloads.

This module deliberately knows nothing about artifact formats, paths, CAS, or
``fsync``.  It is a small ownership and concurrency primitive for a later
capture-writer integration:

* producers hand off an exact immutable :class:`bytes` payload;
* one non-daemon worker invokes one fixed sink in FIFO admission order;
* item and byte capacity remain reserved through completion of the sink call;
* full capacity blocks producers rather than dropping work;
* close atomically stops admission and drains every accepted item; and
* the first sink failure stops the worker, abandons queued work, wakes all
  waiters, and is re-raised from every subsequent operation.

The reservation accounts for payload bytes owned by this writer from
admission through sink return.  The sink must not retain an item or its payload
after returning; retaining it would extend memory lifetime outside the
reservation contract.  Exact ``bytes`` inputs are required so callers cannot
mutate an accepted payload while it is queued or in flight.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class WriterState(str, Enum):
    """Externally visible lifecycle states."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class BackgroundWriterError(RuntimeError):
    """Base class for deterministic writer-contract failures."""


class BackgroundWriterClosed(BackgroundWriterError):
    """An item was submitted after close had stopped admission."""


class BackgroundWriterFailed(BackgroundWriterError):
    """The worker's first sink failure made the writer unusable."""

    def __init__(self, sequence: int | None, cause: BaseException) -> None:
        location = "internal worker state" if sequence is None else f"item {sequence}"
        super().__init__(
            f"background writer failed at {location}: {type(cause).__name__}: {cause}"
        )
        self.sequence = sequence
        self.cause = cause


class ReservationError(ValueError):
    """A submission did not satisfy the explicit byte-reservation contract."""


@dataclass(frozen=True, slots=True)
class WriteItem:
    """One immutable payload admitted to the FIFO.

    ``sequence`` is allocated while holding the admission lock, so it is also
    the total FIFO order across concurrent producers.
    """

    sequence: int
    payload: bytes
    reserved_bytes: int
    admitted_ns: int


@dataclass(frozen=True, slots=True)
class WriterStats:
    """One internally consistent snapshot of writer accounting."""

    state: WriterState
    max_items: int
    max_bytes: int
    accepted_items: int
    accepted_bytes: int
    completed_items: int
    completed_bytes: int
    abandoned_items: int
    abandoned_bytes: int
    reserved_items: int
    reserved_bytes: int
    queued_items: int
    queued_bytes: int
    in_flight_items: int
    in_flight_bytes: int
    reservation_high_water_items: int
    reservation_high_water_bytes: int
    queue_high_water_items: int
    queue_high_water_bytes: int
    producer_block_events: int
    producer_block_ns_total: int
    producer_block_ns_max: int
    queue_wait_ns_total: int
    queue_wait_ns_max: int
    worker_calls: int
    worker_ns_total: int
    worker_ns_max: int
    worker_idle_ns_total: int
    close_calls: int
    close_wait_ns_total: int
    failure_sequence: int | None
    failure_type: str | None
    failure_message: str | None
    worker_alive: bool

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        value = asdict(self)
        value["state"] = self.state.value
        return value


_THREAD_SERIAL = itertools.count(1)


class BoundedBackgroundWriter:
    """Run a fixed sink on immutable payloads with bounded FIFO admission.

    Args:
        sink: Called exactly once for each item that completes successfully.
            The sink executes only on the writer thread and must not retain the
            item or payload after it returns.
        max_items: Maximum accepted items, including the item in flight.
        max_bytes: Maximum reserved payload bytes, including the item in flight.
        thread_name: Optional diagnostic name for the single worker.

    The object is also a context manager.  Normal context exit drains the
    queue.  Exceptional context exit still drains; if the sink also failed,
    the sink failure is raised by :meth:`close` and becomes the context-exit
    exception.
    """

    def __init__(
        self,
        sink: Callable[[WriteItem], None],
        *,
        max_items: int,
        max_bytes: int,
        thread_name: str | None = None,
    ) -> None:
        if not callable(sink):
            raise TypeError("sink must be callable")
        if isinstance(max_items, bool) or not isinstance(max_items, int):
            raise TypeError("max_items must be an integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")

        self._sink = sink
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._condition = threading.Condition()
        self._queue: deque[WriteItem] = deque()
        self._state = WriterState.OPEN
        self._next_sequence = 1
        self._failure: tuple[int | None, BaseException] | None = None

        self._accepted_items = 0
        self._accepted_bytes = 0
        self._completed_items = 0
        self._completed_bytes = 0
        self._abandoned_items = 0
        self._abandoned_bytes = 0
        self._reserved_items = 0
        self._reserved_bytes = 0
        self._queued_bytes = 0
        self._in_flight: WriteItem | None = None
        self._reservation_high_water_items = 0
        self._reservation_high_water_bytes = 0
        self._queue_high_water_items = 0
        self._queue_high_water_bytes = 0
        self._producer_block_events = 0
        self._producer_block_ns_total = 0
        self._producer_block_ns_max = 0
        self._queue_wait_ns_total = 0
        self._queue_wait_ns_max = 0
        self._worker_calls = 0
        self._worker_ns_total = 0
        self._worker_ns_max = 0
        self._worker_idle_ns_total = 0
        self._close_calls = 0
        self._close_wait_ns_total = 0
        self._worker_running = True

        name = thread_name or f"yeto-capture-writer-{next(_THREAD_SERIAL)}"
        self._thread = threading.Thread(
            target=self._worker_main,
            name=name,
            daemon=False,
        )
        try:
            self._thread.start()
        except BaseException:
            self._worker_running = False
            raise

    @property
    def thread_name(self) -> str:
        return self._thread.name

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def __enter__(self) -> BoundedBackgroundWriter:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def submit(self, payload: bytes, *, reservation_bytes: int) -> int:
        """Block until capacity is reserved, then admit one immutable payload.

        ``reservation_bytes`` is deliberately explicit and must equal the
        payload length.  This prevents a future caller from quietly
        under-accounting a descriptor or mutable tensor view.  Capacity remains
        reserved until the sink returns, not merely until the worker dequeues
        the item.
        """

        if threading.current_thread() is self._thread:
            raise BackgroundWriterError("the writer sink may not submit recursively")
        block_started_ns: int | None = None
        with self._condition:
            # Terminal state dominates argument validation: after a sink
            # failure, every submit deterministically propagates that first
            # error; after close, every submit is a closed-admission error.
            self._raise_if_failed_locked()
            if self._state is not WriterState.OPEN:
                raise BackgroundWriterClosed(
                    f"background writer is {self._state.value}; admission is closed"
                )
            if type(payload) is not bytes:
                raise TypeError("payload must be exact immutable bytes")
            if isinstance(reservation_bytes, bool) or not isinstance(
                reservation_bytes, int
            ):
                raise TypeError("reservation_bytes must be an integer")
            if reservation_bytes < 1:
                raise ReservationError("reservation_bytes must be positive")
            if reservation_bytes != len(payload):
                raise ReservationError(
                    "reservation_bytes must equal the immutable payload length"
                )
            if reservation_bytes > self._max_bytes:
                raise ReservationError(
                    f"payload reservation {reservation_bytes} exceeds writer byte "
                    f"capacity {self._max_bytes}"
                )

            while True:
                if self._failure is not None:
                    self._finish_producer_block_locked(block_started_ns)
                    self._raise_if_failed_locked()
                if self._state is not WriterState.OPEN:
                    self._finish_producer_block_locked(block_started_ns)
                    raise BackgroundWriterClosed(
                        f"background writer is {self._state.value}; admission is closed"
                    )
                if (
                    self._reserved_items < self._max_items
                    and self._reserved_bytes + reservation_bytes <= self._max_bytes
                ):
                    break
                if block_started_ns is None:
                    block_started_ns = time.monotonic_ns()
                    self._producer_block_events += 1
                self._condition.wait()

            self._finish_producer_block_locked(block_started_ns)
            admitted_ns = time.monotonic_ns()
            sequence = self._next_sequence
            self._next_sequence += 1
            item = WriteItem(
                sequence=sequence,
                payload=payload,
                reserved_bytes=reservation_bytes,
                admitted_ns=admitted_ns,
            )
            self._queue.append(item)
            self._queued_bytes += reservation_bytes
            self._accepted_items += 1
            self._accepted_bytes += reservation_bytes
            self._reserved_items += 1
            self._reserved_bytes += reservation_bytes
            self._reservation_high_water_items = max(
                self._reservation_high_water_items, self._reserved_items
            )
            self._reservation_high_water_bytes = max(
                self._reservation_high_water_bytes, self._reserved_bytes
            )
            self._queue_high_water_items = max(
                self._queue_high_water_items, len(self._queue)
            )
            self._queue_high_water_bytes = max(
                self._queue_high_water_bytes, self._queued_bytes
            )
            self._condition.notify_all()
            return sequence

    def close(self) -> WriterStats:
        """Atomically stop admission, drain accepted work, and join the worker.

        Multiple callers may close concurrently.  They all observe the same
        terminal state.  On failure each call raises
        :class:`BackgroundWriterFailed` chained from the first sink exception.
        """

        if threading.current_thread() is self._thread:
            raise BackgroundWriterError("the writer sink may not close its worker")
        close_started_ns = time.monotonic_ns()
        with self._condition:
            self._close_calls += 1
            if self._state is WriterState.OPEN:
                self._state = WriterState.CLOSING
                self._condition.notify_all()
            while self._state is WriterState.CLOSING:
                self._condition.wait()
            failure = self._failure
        self._thread.join()
        close_elapsed_ns = time.monotonic_ns() - close_started_ns
        with self._condition:
            self._close_wait_ns_total += close_elapsed_ns
            stats = self._snapshot_locked()
        if failure is not None:
            sequence, cause = failure
            raise BackgroundWriterFailed(sequence, cause) from cause
        return stats

    def snapshot(self) -> WriterStats:
        """Return counters from one condition-lock critical section."""

        with self._condition:
            return self._snapshot_locked()

    def check(self) -> WriterStats:
        """Return a snapshot or propagate the worker's first failure.

        Unlike :meth:`close`, this does not change an open writer's lifecycle.
        Capture loops use it at deterministic hook boundaries so asynchronous
        publication errors cannot remain latent until process teardown.
        """

        with self._condition:
            self._raise_if_failed_locked()
            return self._snapshot_locked()

    def _finish_producer_block_locked(self, started_ns: int | None) -> None:
        if started_ns is None:
            return
        elapsed = time.monotonic_ns() - started_ns
        self._producer_block_ns_total += elapsed
        self._producer_block_ns_max = max(self._producer_block_ns_max, elapsed)

    def _raise_if_failed_locked(self) -> None:
        if self._failure is None:
            return
        sequence, cause = self._failure
        raise BackgroundWriterFailed(sequence, cause) from cause

    def _snapshot_locked(self) -> WriterStats:
        failure_sequence: int | None = None
        failure_type: str | None = None
        failure_message: str | None = None
        if self._failure is not None:
            failure_sequence, cause = self._failure
            failure_type = type(cause).__name__
            failure_message = str(cause)
        in_flight_bytes = (
            0 if self._in_flight is None else self._in_flight.reserved_bytes
        )
        return WriterStats(
            state=self._state,
            max_items=self._max_items,
            max_bytes=self._max_bytes,
            accepted_items=self._accepted_items,
            accepted_bytes=self._accepted_bytes,
            completed_items=self._completed_items,
            completed_bytes=self._completed_bytes,
            abandoned_items=self._abandoned_items,
            abandoned_bytes=self._abandoned_bytes,
            reserved_items=self._reserved_items,
            reserved_bytes=self._reserved_bytes,
            queued_items=len(self._queue),
            queued_bytes=self._queued_bytes,
            in_flight_items=0 if self._in_flight is None else 1,
            in_flight_bytes=in_flight_bytes,
            reservation_high_water_items=self._reservation_high_water_items,
            reservation_high_water_bytes=self._reservation_high_water_bytes,
            queue_high_water_items=self._queue_high_water_items,
            queue_high_water_bytes=self._queue_high_water_bytes,
            producer_block_events=self._producer_block_events,
            producer_block_ns_total=self._producer_block_ns_total,
            producer_block_ns_max=self._producer_block_ns_max,
            queue_wait_ns_total=self._queue_wait_ns_total,
            queue_wait_ns_max=self._queue_wait_ns_max,
            worker_calls=self._worker_calls,
            worker_ns_total=self._worker_ns_total,
            worker_ns_max=self._worker_ns_max,
            worker_idle_ns_total=self._worker_idle_ns_total,
            close_calls=self._close_calls,
            close_wait_ns_total=self._close_wait_ns_total,
            failure_sequence=failure_sequence,
            failure_type=failure_type,
            failure_message=failure_message,
            worker_alive=self._worker_running,
        )

    def _worker_main(self) -> None:
        active_item: WriteItem | None = None
        try:
            while True:
                with self._condition:
                    idle_started_ns: int | None = None
                    while not self._queue and self._state is WriterState.OPEN:
                        if idle_started_ns is None:
                            idle_started_ns = time.monotonic_ns()
                        self._condition.wait()
                    if idle_started_ns is not None:
                        self._worker_idle_ns_total += (
                            time.monotonic_ns() - idle_started_ns
                        )
                    if not self._queue:
                        if self._state is WriterState.CLOSING:
                            self._state = WriterState.CLOSED
                            self._condition.notify_all()
                        return
                    active_item = self._queue.popleft()
                    self._queued_bytes -= active_item.reserved_bytes
                    self._in_flight = active_item

                worker_started_ns = time.monotonic_ns()
                queue_wait_ns = worker_started_ns - active_item.admitted_ns
                try:
                    self._sink(active_item)
                except BaseException as exc:
                    worker_elapsed_ns = time.monotonic_ns() - worker_started_ns
                    with self._condition:
                        self._record_worker_timing_locked(
                            worker_elapsed_ns, queue_wait_ns
                        )
                        self._fail_locked(active_item, exc)
                    return

                worker_elapsed_ns = time.monotonic_ns() - worker_started_ns
                with self._condition:
                    self._record_worker_timing_locked(worker_elapsed_ns, queue_wait_ns)
                    self._completed_items += 1
                    self._completed_bytes += active_item.reserved_bytes
                    self._reserved_items -= 1
                    self._reserved_bytes -= active_item.reserved_bytes
                    self._in_flight = None
                    active_item = None
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._fail_locked(active_item, exc)
        finally:
            with self._condition:
                self._worker_running = False
                self._condition.notify_all()

    def _record_worker_timing_locked(
        self, worker_elapsed_ns: int, queue_wait_ns: int
    ) -> None:
        self._worker_calls += 1
        self._worker_ns_total += worker_elapsed_ns
        self._worker_ns_max = max(self._worker_ns_max, worker_elapsed_ns)
        self._queue_wait_ns_total += queue_wait_ns
        self._queue_wait_ns_max = max(self._queue_wait_ns_max, queue_wait_ns)

    def _fail_locked(self, active_item: WriteItem | None, cause: BaseException) -> None:
        if self._failure is not None:
            return
        failure_sequence = None if active_item is None else active_item.sequence
        self._failure = (failure_sequence, cause)
        abandoned_items = len(self._queue) + (0 if active_item is None else 1)
        abandoned_bytes = self._queued_bytes + (
            0 if active_item is None else active_item.reserved_bytes
        )
        self._abandoned_items += abandoned_items
        self._abandoned_bytes += abandoned_bytes
        self._queue.clear()
        self._queued_bytes = 0
        self._in_flight = None
        self._reserved_items = 0
        self._reserved_bytes = 0
        self._state = WriterState.FAILED
        self._condition.notify_all()
