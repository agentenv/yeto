from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from yeto.bounded_background_writer import (
    BackgroundWriterClosed,
    BackgroundWriterFailed,
    BoundedBackgroundWriter,
    ReservationError,
    WriteItem,
    WriterState,
)


TIMEOUT = 3.0


def _wait_until(predicate: Callable[[], bool], *, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for concurrent writer state")
        time.sleep(0.001)


def _join(thread: threading.Thread) -> None:
    thread.join(TIMEOUT)
    assert not thread.is_alive(), f"thread leaked: {thread.name}"


def test_fifo_order_reservations_and_timing_counters() -> None:
    entered = threading.Event()
    release = threading.Event()
    observed: list[tuple[int, bytes]] = []

    def sink(item: WriteItem) -> None:
        if item.sequence == 1:
            entered.set()
            assert release.wait(TIMEOUT)
        observed.append((item.sequence, item.payload))

    writer = BoundedBackgroundWriter(sink, max_items=4, max_bytes=32)
    assert writer.submit(b"one", reservation_bytes=3) == 1
    assert entered.wait(TIMEOUT)
    assert writer.submit(b"two", reservation_bytes=3) == 2
    assert writer.submit(b"three", reservation_bytes=5) == 3

    while_blocked = writer.snapshot()
    assert writer.check().state is WriterState.OPEN
    assert while_blocked.state is WriterState.OPEN
    assert while_blocked.reserved_items == 3
    assert while_blocked.reserved_bytes == 11
    assert while_blocked.in_flight_items == 1
    assert while_blocked.queued_items == 2
    assert while_blocked.queued_bytes == 8
    assert while_blocked.reservation_high_water_items == 3
    assert while_blocked.reservation_high_water_bytes == 11
    assert while_blocked.queue_high_water_items == 2
    assert while_blocked.queue_high_water_bytes == 8

    release.set()
    stats = writer.close()

    assert observed == [(1, b"one"), (2, b"two"), (3, b"three")]
    assert stats.state is WriterState.CLOSED
    assert stats.accepted_items == stats.completed_items == 3
    assert stats.accepted_bytes == stats.completed_bytes == 11
    assert stats.abandoned_items == stats.abandoned_bytes == 0
    assert stats.reserved_items == stats.reserved_bytes == 0
    assert stats.queued_items == stats.in_flight_items == 0
    assert stats.worker_calls == 3
    assert stats.worker_ns_total >= stats.worker_ns_max > 0
    assert stats.queue_wait_ns_total >= stats.queue_wait_ns_max > 0
    assert stats.close_calls == 1
    assert stats.close_wait_ns_total > 0
    assert stats.worker_alive is False
    assert writer.thread_alive is False
    assert stats.as_json()["state"] == "closed"


def test_payload_must_be_exact_immutable_bytes_with_exact_reservation() -> None:
    writer = BoundedBackgroundWriter(lambda _item: None, max_items=1, max_bytes=4)
    try:
        with pytest.raises(TypeError, match="exact immutable bytes"):
            writer.submit(bytearray(b"a"), reservation_bytes=1)  # type: ignore[arg-type]
        with pytest.raises(ReservationError, match="equal.*payload length"):
            writer.submit(b"ab", reservation_bytes=1)
        with pytest.raises(ReservationError, match="must be positive"):
            writer.submit(b"", reservation_bytes=0)
        with pytest.raises(ReservationError, match="exceeds writer byte capacity"):
            writer.submit(b"abcde", reservation_bytes=5)
        assert writer.snapshot().accepted_items == 0
    finally:
        stats = writer.close()
    assert stats.worker_alive is False
    assert writer.thread_alive is False


def test_byte_capacity_blocks_without_dropping_or_releasing_inflight_reservation() -> (
    None
):
    entered = threading.Event()
    release = threading.Event()
    observed: list[bytes] = []

    def sink(item: WriteItem) -> None:
        if item.sequence == 1:
            entered.set()
            assert release.wait(TIMEOUT)
        observed.append(item.payload)

    writer = BoundedBackgroundWriter(sink, max_items=4, max_bytes=4)
    writer.submit(b"aaaa", reservation_bytes=4)
    assert entered.wait(TIMEOUT)

    result: list[int] = []
    producer = threading.Thread(
        target=lambda: result.append(writer.submit(b"b", reservation_bytes=1)),
        name="blocked-byte-producer",
    )
    producer.start()
    _wait_until(lambda: writer.snapshot().producer_block_events == 1)
    blocked = writer.snapshot()
    assert blocked.reserved_bytes == 4
    assert blocked.in_flight_bytes == 4
    assert result == []

    release.set()
    _join(producer)
    stats = writer.close()

    assert result == [2]
    assert observed == [b"aaaa", b"b"]
    assert stats.completed_items == 2
    assert stats.abandoned_items == 0
    assert stats.producer_block_events == 1
    assert stats.producer_block_ns_total >= stats.producer_block_ns_max > 0
    assert stats.reservation_high_water_bytes == 4
    assert stats.worker_alive is False


@pytest.mark.parametrize("failure_site", ["write", "fsync"])
def test_first_sink_failure_abandons_fifo_wakes_waiters_and_propagates(
    failure_site: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    injected = OSError(f"injected {failure_site} failure")
    observed: list[int] = []

    def sink(item: WriteItem) -> None:
        observed.append(item.sequence)
        entered.set()
        assert release.wait(TIMEOUT)
        raise injected

    writer = BoundedBackgroundWriter(sink, max_items=3, max_bytes=16)
    writer.submit(b"one", reservation_bytes=3)
    assert entered.wait(TIMEOUT)
    writer.submit(b"two", reservation_bytes=3)
    writer.submit(b"three", reservation_bytes=5)

    blocked_errors: list[BaseException] = []

    def blocked_submit() -> None:
        try:
            writer.submit(b"four", reservation_bytes=4)
        except BaseException as exc:
            blocked_errors.append(exc)

    blocked_producer = threading.Thread(
        target=blocked_submit, name=f"blocked-{failure_site}-producer"
    )
    blocked_producer.start()
    _wait_until(lambda: writer.snapshot().producer_block_events == 1)
    release.set()
    _join(blocked_producer)
    _wait_until(lambda: writer.snapshot().state is WriterState.FAILED)

    assert len(blocked_errors) == 1
    assert isinstance(blocked_errors[0], BackgroundWriterFailed)
    assert blocked_errors[0].cause is injected  # type: ignore[attr-defined]
    assert blocked_errors[0].__cause__ is injected

    with pytest.raises(BackgroundWriterFailed) as enqueue_error:
        writer.submit(b"late", reservation_bytes=4)
    assert enqueue_error.value.sequence == 1
    assert enqueue_error.value.cause is injected
    assert enqueue_error.value.__cause__ is injected
    with pytest.raises(BackgroundWriterFailed) as health_error:
        writer.check()
    assert health_error.value.cause is injected
    with pytest.raises(BackgroundWriterFailed) as invalid_enqueue_error:
        writer.submit(bytearray(b"x"), reservation_bytes=1)  # type: ignore[arg-type]
    assert invalid_enqueue_error.value.cause is injected

    with pytest.raises(BackgroundWriterFailed) as close_error:
        writer.close()
    assert close_error.value.sequence == 1
    assert close_error.value.cause is injected
    assert close_error.value.__cause__ is injected

    stats = writer.snapshot()
    assert observed == [1]
    assert stats.state is WriterState.FAILED
    assert stats.accepted_items == stats.abandoned_items == 3
    assert stats.accepted_bytes == stats.abandoned_bytes == 11
    assert stats.completed_items == stats.completed_bytes == 0
    assert stats.reserved_items == stats.reserved_bytes == 0
    assert stats.queued_items == stats.in_flight_items == 0
    assert stats.worker_calls == 1
    assert stats.failure_sequence == 1
    assert stats.failure_type == "OSError"
    assert stats.failure_message == f"injected {failure_site} failure"
    assert stats.producer_block_events == 1
    assert stats.producer_block_ns_total > 0
    assert stats.worker_alive is False
    assert writer.thread_alive is False


def test_concurrent_producers_preserve_the_single_admission_sequence() -> None:
    producer_count = 6
    items_per_producer = 30
    start = threading.Barrier(producer_count)
    accepted: list[tuple[int, bytes]] = []
    accepted_lock = threading.Lock()
    observed: list[tuple[int, bytes]] = []
    errors: list[BaseException] = []

    def sink(item: WriteItem) -> None:
        observed.append((item.sequence, item.payload))

    writer = BoundedBackgroundWriter(sink, max_items=7, max_bytes=49)

    def produce(producer_id: int) -> None:
        try:
            start.wait(TIMEOUT)
            for item_id in range(items_per_producer):
                payload = f"{producer_id:02}:{item_id:02}".encode("ascii")
                sequence = writer.submit(payload, reservation_bytes=len(payload))
                with accepted_lock:
                    accepted.append((sequence, payload))
        except BaseException as exc:
            with accepted_lock:
                errors.append(exc)

    producers = [
        threading.Thread(target=produce, args=(index,), name=f"producer-{index}")
        for index in range(producer_count)
    ]
    for producer in producers:
        producer.start()
    for producer in producers:
        _join(producer)
    stats = writer.close()

    assert errors == []
    expected_count = producer_count * items_per_producer
    assert len(accepted) == len(observed) == expected_count
    assert observed == sorted(accepted)
    assert [sequence for sequence, _payload in observed] == list(
        range(1, expected_count + 1)
    )
    assert stats.accepted_items == stats.completed_items == expected_count
    assert stats.accepted_bytes == stats.completed_bytes
    assert stats.abandoned_items == 0
    assert stats.reservation_high_water_items <= 7
    assert stats.reservation_high_water_bytes <= 49
    assert stats.worker_alive is False
    assert writer.thread_alive is False


def test_close_race_rejects_blocked_and_late_producers_then_drains() -> None:
    entered = threading.Event()
    release = threading.Event()

    def sink(item: WriteItem) -> None:
        assert item.sequence == 1
        entered.set()
        assert release.wait(TIMEOUT)

    writer = BoundedBackgroundWriter(sink, max_items=1, max_bytes=8)
    writer.submit(b"first", reservation_bytes=5)
    assert entered.wait(TIMEOUT)

    blocked_errors: list[BaseException] = []

    def blocked_submit() -> None:
        try:
            writer.submit(b"next", reservation_bytes=4)
        except BaseException as exc:
            blocked_errors.append(exc)

    producer = threading.Thread(target=blocked_submit, name="close-race-producer")
    producer.start()
    _wait_until(lambda: writer.snapshot().producer_block_events == 1)

    close_results = []
    close_errors: list[BaseException] = []

    def close_writer() -> None:
        try:
            close_results.append(writer.close())
        except BaseException as exc:
            close_errors.append(exc)

    closers = [
        threading.Thread(target=close_writer, name=f"closer-{index}")
        for index in range(2)
    ]
    for closer in closers:
        closer.start()
    _wait_until(lambda: writer.snapshot().state is WriterState.CLOSING)
    _join(producer)
    assert len(blocked_errors) == 1
    assert isinstance(blocked_errors[0], BackgroundWriterClosed)
    with pytest.raises(BackgroundWriterClosed):
        writer.submit(b"late", reservation_bytes=4)

    release.set()
    for closer in closers:
        _join(closer)

    assert close_errors == []
    assert len(close_results) == 2
    assert all(result.state is WriterState.CLOSED for result in close_results)
    assert all(result.completed_items == 1 for result in close_results)
    final = writer.close()
    assert final.state is WriterState.CLOSED
    assert final.accepted_items == final.completed_items == 1
    assert final.producer_block_events == 1
    assert final.producer_block_ns_total > 0
    assert final.close_calls == 3
    assert final.worker_alive is False
    assert writer.thread_alive is False
    with pytest.raises(BackgroundWriterClosed):
        writer.submit(bytearray(b"x"), reservation_bytes=1)  # type: ignore[arg-type]


def test_empty_context_close_leaves_no_worker() -> None:
    writer: BoundedBackgroundWriter | None = None
    with BoundedBackgroundWriter(
        lambda _item: None,
        max_items=2,
        max_bytes=8,
        thread_name="empty-capture-writer-test",
    ) as opened:
        writer = opened
        assert opened.thread_alive is True
        assert opened.snapshot().state is WriterState.OPEN
    assert writer is not None
    assert writer.snapshot().state is WriterState.CLOSED
    assert writer.snapshot().worker_alive is False
    assert writer.thread_alive is False
    assert not any(
        thread.name == "empty-capture-writer-test" and thread.is_alive()
        for thread in threading.enumerate()
    )
