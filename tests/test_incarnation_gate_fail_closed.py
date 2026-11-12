"""The incarnation gate fails closed on a lease it cannot read: an absent lease
(no such key) and an unreadable one (a timeout, a 5xx, a truncated body) are told
apart; the reader defers the group (neither trained nor counted stale), the failed
read is not cached, and the claim does not overwrite a lease it could not read."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.test_incarnation_lease_fence import _entry, _write_group
from yeto.rl.engine.incarnation_lease import (
    IncarnationGate,
    IncarnationLease,
    LeaseReadError,
    LeaseUnknown,
    lease_key,
    read_lease,
    read_record,
    read_record_strict,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


class FlakyObjects:
    """An object store over a directory whose reads of ``leases`` keys fail on demand
    (``fail``: raise; ``body``: answer these bytes instead), like a timeout or a 5xx."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.fail: BaseException | None = None
        self.body: bytes | None = None
        self.reads = 0
        self.only: str | None = None  # fail only this sampler's lease

    def read(self, key: str) -> bytes | None:
        if "leases/" in key and (self.only is None or key.endswith(f"/{self.only}.json")):
            self.reads += 1
            if self.fail is not None:
                raise self.fail
            if self.body is not None:
                return self.body
        path = self.root / key
        return path.read_bytes() if path.exists() else None

    def write(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class FlakyStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects = FlakyObjects(root)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _superseded(root: Path) -> None:
    """Lease of s: aaaa superseded by bbbb at time 100, none of aaaa sealed."""
    store = CheckpointStore(root)
    IncarnationLease(store, "s", "aaaa", clock=lambda: 0.0).claim()
    IncarnationLease(store, "s", "bbbb", clock=lambda: 100.0).claim()


# --------------------------------------------------------------------------- read_record_strict

def test_strict_read_tells_absent_from_unreadable(tmp_path):
    store = FlakyStore(tmp_path)
    assert read_record_strict(store, lease_key("s")) is None  # no such key: absent
    _superseded(tmp_path)
    assert read_record_strict(store, lease_key("s"))["incarnation"] == "bbbb"
    for fail, body in [(TimeoutError("read timeout"), None), (OSError("503 SlowDown"), None),
                       (None, b'{"incarnation": "bb'), (None, b"[1, 2]")]:
        store.objects.fail, store.objects.body = fail, body
        with pytest.raises(LeaseReadError):
            read_record_strict(store, lease_key("s"))
        with pytest.raises(LeaseReadError):
            read_lease(store, "s", strict=True)
        assert read_record(store, lease_key("s")) is None  # the lenient form is unchanged
    # a directory store (no objects): absent and garbled
    plain = CheckpointStore(tmp_path / "plain")
    assert read_record_strict(plain, lease_key("s")) is None
    (tmp_path / "plain" / lease_key("s")).parent.mkdir(parents=True)
    (tmp_path / "plain" / lease_key("s")).write_text("{trunc")
    with pytest.raises(LeaseReadError):
        read_record_strict(plain, lease_key("s"))


# --------------------------------------------------------------------------- gate

def test_gate_fails_closed_and_does_not_cache_the_failure(tmp_path):
    _superseded(tmp_path)
    store, clock = FlakyStore(tmp_path), Clock()
    late = _entry("s", "aaaa", appended_at=500.0)  # written after bbbb's claim: stale
    gate = IncarnationGate(store, lambda: [late], clock_margin_s=5.0, retry_s=1.0, monotonic=clock)
    store.objects.fail = TimeoutError("read timeout")
    with pytest.raises(LeaseUnknown):
        gate.verdict(late, 0)
    reads = store.objects.reads
    with pytest.raises(LeaseUnknown):  # within retry_s: not read again, still no verdict
        gate.verdict(late, 0)
    assert store.objects.reads == reads and gate.deferred == 2 and gate.stale == 0
    store.objects.fail = None
    clock.now = 1.5  # the failure was not cached as "no lease" for refresh_s (30 s)
    assert "superseded by bbbb" in gate.verdict(late, 0)
    assert gate.stale == 1 and gate.read_failures == 1


def test_gate_failed_refresh_does_not_fall_back_to_no_lease(tmp_path):
    """A lease read once and then unreadable at its refresh: still no verdict."""
    _superseded(tmp_path)
    store, clock = FlakyStore(tmp_path), Clock()
    late = _entry("s", "aaaa", appended_at=500.0)
    gate = IncarnationGate(store, lambda: [late], clock_margin_s=5.0, refresh_s=30.0, retry_s=0.0,
                           monotonic=clock)
    assert gate.verdict(late, 0)
    clock.now = 31.0
    store.objects.body = b'{"incarn'  # truncated body
    with pytest.raises(LeaseUnknown):
        gate.verdict(late, 0)


def test_gate_absent_lease_still_accepts(tmp_path):
    store = FlakyStore(tmp_path)
    entry = _entry("s", "aaaa")
    assert IncarnationGate(store, lambda: [entry]).verdict(entry, 0) is None


def test_gate_warns_at_most_every_warn_s(tmp_path, caplog):
    _superseded(tmp_path)
    store, clock = FlakyStore(tmp_path), Clock()
    late = _entry("s", "aaaa", appended_at=500.0)
    gate = IncarnationGate(store, lambda: [late], retry_s=0.0, warn_s=60.0, monotonic=clock)
    store.objects.fail = OSError("500 InternalError")
    with caplog.at_level(logging.WARNING, logger="yeto.rl.engine.incarnation_lease"):
        for step in range(600):  # ten minutes of failed reads, one a second
            clock.now = float(step)
            with pytest.raises(LeaseUnknown):
                gate.verdict(late, 0)
        store.objects.fail = None
        clock.now = 600.0
        assert gate.verdict(late, 0)
    unreadable = [r for r in caplog.records if "unreadable" in r.getMessage()]
    assert len(unreadable) == 10 and gate.read_failures == 600
    assert any("read again" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- claim

def test_claim_does_not_overwrite_a_lease_it_cannot_read(tmp_path):
    _superseded(tmp_path)
    store = FlakyStore(tmp_path)
    store.objects.fail = TimeoutError("read timeout")
    with pytest.raises(LeaseReadError):
        IncarnationLease(store, "s", "cccc").claim()
    assert read_lease(CheckpointStore(tmp_path), "s")["incarnation"] == "bbbb"  # not overwritten

    class OnceFlaky(FlakyObjects):
        def read(self, key):
            if "leases/" in key and self.reads == 0:
                self.reads += 1
                raise TimeoutError("read timeout")
            return super().read(key)

    store.objects = OnceFlaky(tmp_path)
    record = IncarnationLease(store, "s", "cccc").claim()  # retried: remembers whom it supersedes
    assert record["superseded"] == ["aaaa", "bbbb"] and record["generation"] == 3


def test_poll_marks_an_unreadable_lease_unknown_and_does_not_fence(tmp_path):
    store = FlakyStore(tmp_path)
    lease = IncarnationLease(store, "s", "aaaa")
    lease.claim()
    store.objects.fail = TimeoutError("read timeout")
    assert lease.poll() is False and lease.unknown and lease.read_failures == 1
    store.objects.fail = None
    assert lease.poll() is False and not lease.unknown


# --------------------------------------------------------------------------- Miles reader

def test_miles_reader_defers_groups_whose_lease_is_unreadable(tmp_path):
    """Lease unreadable: the stale group of the fenced incarnation is neither trained
    nor counted stale; the watermark stops before it; once the lease reads again it
    is judged (stale) by the next take."""
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)  # 0: before the claim, sealed
    IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim(writer.entries())
    _write_group(writer, "s0", "aaaa", 1)  # 1: the fenced process still writes
    _write_group(writer, "s1", "xxxx", 0)  # 2: another island, no lease
    _write_group(writer, "s0", "bbbb", 2)  # 3: the holder
    _write_group(writer, "s1", "xxxx", 1)  # 4
    flaky = FlakyStore(root)
    buffer = FileSampleBuffer(store, batch_size=1)
    gate = IncarnationGate(flaky, buffer.entries, retry_s=0.0, clock_margin_s=0.0)
    reader = BatchReader(buffer, max_policy_age=2, prefetch=False, read_workers=1, wait_s=0.0, poll_s=0.0,
                         incarnation_gate=gate)
    first = gate.verdict(buffer.entries()[0], 0)  # the lease is read (and cached) once
    assert first is None
    gate._leases.clear()  # noqa: SLF001 - as if the cache expired
    flaky.objects.fail, flaky.objects.only = TimeoutError("read timeout"), "s0"
    out, report = reader.take(0, 2, current_version=None)
    # s0's groups wait for their lease; s1's (no lease record) are trained
    assert [position for position, _ in out] == [2, 4]
    assert report.stale_incarnation_groups == 0 and report.stale_groups == 0
    assert report.lease_deferred_groups == 3
    assert report.metrics()["sample_buffer/lease_deferred_groups"] == 3.0
    assert report.end == 0  # the round's end never passes a deferred group
    assert reader.watermark_before(1) == 0
    flaky.objects.fail = None
    out, report = reader.take(1, 2, current_version=None)
    assert [position for position, _ in out] == [0, 3]  # judged again: sealed, holder
    assert report.lease_deferred_groups == 0 and report.stale_incarnation_groups == 1  # position 1
    assert reader.watermark_before(2) == 5 and not reader._deferred  # noqa: SLF001


def test_miles_reader_waits_for_a_deferred_group_at_the_data_end(tmp_path):
    """The only group left waits for its lease (no exhaustion, no stale count) and
    is trained once the lease reads again."""
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    writer.mark_data_end()
    flaky = FlakyStore(root)
    flaky.objects.fail = TimeoutError("read timeout")
    buffer = FileSampleBuffer(store, batch_size=1)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            flaky.objects.fail = None  # the store answers again

    reader = BatchReader(buffer, max_policy_age=2, prefetch=False, read_workers=1, wait_s=0.0, poll_s=0.5,
                         sleep=sleep, incarnation_gate=IncarnationGate(flaky, buffer.entries, retry_s=0.0))
    out, report = reader.take(0, 1, current_version=None)
    assert [position for position, _ in out] == [0] and len(sleeps) == 3
    assert report.stale_incarnation_groups == 0


def test_reset_forgets_deferred_groups(tmp_path):
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s0", "aaaa").claim()
    _write_group(writer, "s0", "aaaa", 0)
    _write_group(writer, "s1", "xxxx", 0)
    flaky = FlakyStore(root)
    flaky.objects.fail, flaky.objects.only = TimeoutError("read timeout"), "s0"
    buffer = FileSampleBuffer(store, batch_size=1)
    reader = BatchReader(buffer, max_policy_age=2, prefetch=False, read_workers=1, wait_s=0.0, poll_s=0.0,
                         incarnation_gate=IncarnationGate(flaky, buffer.entries, retry_s=0.0))
    out, _ = reader.take(0, 1, current_version=None)
    assert [position for position, _ in out] == [1] and reader._deferred  # noqa: SLF001
    reader.reset(0)
    assert not reader._deferred  # noqa: SLF001 - selected again from the watermark


# --------------------------------------------------------------------------- verl feed

def test_buffer_feed_waits_for_an_unreadable_lease(tmp_path):
    from yeto.rl.adapters.verl.buffer_feed import BufferFeed, RowShape

    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s", "aaaa", clock=lambda: 0.0).claim()
    IncarnationLease(store, "s", "bbbb", clock=lambda: 0.0).claim([])
    _write_group(writer, "s", "aaaa", 1)  # the fenced process's late write
    _write_group(writer, "s", "bbbb", 2)
    flaky = FlakyStore(root)
    flaky.objects.fail = TimeoutError("read timeout")
    reader = FileSampleBuffer(store, batch_size=1)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4:
            flaky.objects.fail = None

    records = []
    feed = BufferFeed(reader, vmap=lambda: None, shape=RowShape(16, 16, 0), add_group=lambda rows: None,
                      record=records.append, emit=lambda name, **f: None, wait_s=0.0, poll_s=0.25, sleep=sleep,
                      incarnation_gate=IncarnationGate(flaky, reader.entries, retry_s=0.0))
    feed._local = lambda version: 0  # noqa: SLF001 - versions are not under test
    pulled = feed.pull()
    assert pulled["uid"] == "buf1" and feed.lease_waits == 4  # the late write was dropped, not trained
    assert [r["position"] for r in records if r.get("dropped")] == [0]
