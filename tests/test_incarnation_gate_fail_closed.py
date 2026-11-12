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


@pytest.fixture(autouse=True)
def _no_claim_pause(monkeypatch):
    import yeto.rl.engine.incarnation_lease as incarnation_lease

    monkeypatch.setattr(incarnation_lease, "_claim_pause", lambda attempt, sleep=None: None)


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

def _fenced_buffer(tmp_path):
    """s0: aaaa superseded by bbbb (aaaa's position 0 sealed, 1 written after the claim);
    s1 has no lease. Positions: 0 s0/aaaa, 1 s0/aaaa (stale), 2 s1, 3 s0/bbbb, 4 s1."""
    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim(writer.entries())
    _write_group(writer, "s0", "aaaa", 1)
    _write_group(writer, "s1", "xxxx", 0)
    _write_group(writer, "s0", "bbbb", 2)
    _write_group(writer, "s1", "xxxx", 1)
    return root, store


def _outage_reader(root, store, *, recover_after: int, seen: list | None = None):
    """A reader whose gate cannot read s0's lease until ``recover_after`` pauses."""
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    flaky = FlakyStore(root)
    buffer = FileSampleBuffer(store, batch_size=1)
    gate = IncarnationGate(flaky, buffer.entries, retry_s=0.0, clock_margin_s=0.0)
    sleeps: list[float] = []
    reader: list = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if seen is not None:
            seen.append([position for position, _ in reader[0]._deferred])  # noqa: SLF001
        if len(sleeps) >= recover_after:
            flaky.objects.fail = None

    reader.append(BatchReader(buffer, max_policy_age=2, prefetch=False, read_workers=1, wait_s=0.0,
                              poll_s=0.5, sleep=sleep, incarnation_gate=gate))
    flaky.objects.fail, flaky.objects.only = TimeoutError("read timeout"), "s0"
    return reader[0], flaky, sleeps


def test_miles_reader_pauses_in_strict_order_while_a_lease_is_unreadable(tmp_path):
    """s0's lease unreadable: the first s0 group (position 0) is deferred and every group
    behind it waits too, s1's included (strict order, nothing judged); once the lease
    reads again they are judged in order: 0 trained (sealed), 1 stale, 2 trained."""
    root, store = _fenced_buffer(tmp_path)
    seen: list = []
    reader, _, sleeps = _outage_reader(root, store, recover_after=3, seen=seen)
    out, report = reader.take(0, 2, current_version=None)
    assert seen and all(waiting == [0, 1] for waiting in seen)  # nothing behind 0 was trained
    assert [position for position, _ in out] == [0, 2] and len(sleeps) == 3
    assert report.stale_incarnation_groups == 1 and report.stale_groups == 0
    assert report.lease_deferred_groups == 2  # counted once, by the take that deferred them
    assert report.metrics()["sample_buffer/lease_deferred_groups"] == 2.0
    assert report.end == 3 and reader.watermark_before(1) == 3 and not reader._deferred  # noqa: SLF001


def test_a_restore_after_a_lease_outage_trains_no_group_twice(tmp_path):
    """Regression: before strict order, round 0 trained s1's [2, 4] past the deferred
    position 0, committed watermark 0, and a restore trained 2 and 4 again."""
    root, store = _fenced_buffer(tmp_path)
    reader, _, _ = _outage_reader(root, store, recover_after=2)
    out, report = reader.take(0, 2, current_version=None)
    first = [position for position, _ in out]
    watermark = reader.watermark_before(1)
    assert all(position < watermark for position in first) and watermark == report.end
    restored, _, _ = _outage_reader(root, store, recover_after=0)  # a new trainer process
    restored.reset(watermark)
    out, _ = restored.take(1, 2, current_version=None)
    after = [position for position, _ in out]
    assert not set(first) & set(after)
    assert sorted(first + after) == [0, 2, 3, 4]  # every trainable group exactly once (1 is stale)


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
    """A take that ends (here: the trainer stops it) while groups wait: a restore
    selects them again from its watermark; nothing stays deferred."""
    root, store = _fenced_buffer(tmp_path)
    reader, _, _ = _outage_reader(root, store, recover_after=10**9)

    class Stop(Exception):
        pass

    def stop(seconds: float) -> None:
        raise Stop()

    reader.sleep = stop
    with pytest.raises(Stop):
        reader.take(0, 2, current_version=None)
    reader.reset(0)
    assert not reader._deferred and not reader._retry  # noqa: SLF001


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


# --------------------------------------------------------------------------- an S3 / R2 store

class _Body:
    def __init__(self, data: bytes | None = None, error: BaseException | None = None) -> None:
        self.data, self.error = data, error

    def read(self) -> bytes:
        if self.error is not None:
            raise self.error
        return self.data or b""


class _StubS3:
    """A boto3 S3 client double: ``get_object`` raises ``error`` (a request-level
    failure) or answers ``body`` (whose read may raise ``body_error``)."""

    def __init__(self) -> None:
        from botocore.exceptions import ClientError

        class NoSuchKey(ClientError):
            pass

        self.exceptions = type("Exceptions", (), {"NoSuchKey": NoSuchKey})
        self.error: BaseException | None = None
        self.body: bytes | None = None
        self.body_error: BaseException | None = None
        self.calls = 0

    def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803 - boto3's names
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.body is None and self.body_error is None:
            raise self.exceptions.NoSuchKey({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.body, self.body_error)}


def _client_error(code: str, status: int = 500) -> BaseException:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject")


def _s3() -> tuple[_StubS3, object]:
    from yeto.cloud.s3_store import S3ObjectStore

    client = _StubS3()
    return client, S3ObjectStore("s3://bucket/run", client=client)


def _request_errors() -> list[BaseException]:
    from botocore.exceptions import ConnectionClosedError, EndpointConnectionError, ReadTimeoutError

    return [_client_error("InternalError", 500), _client_error("SlowDown", 503),
            _client_error("AccessDenied", 403), EndpointConnectionError(endpoint_url="https://r2"),
            ReadTimeoutError(endpoint_url="https://r2"), ConnectionClosedError(endpoint_url="https://r2")]


@pytest.mark.parametrize("index", range(6))
def test_strict_read_over_s3_turns_request_errors_into_lease_read_errors(index, monkeypatch):
    """R2 / boto raise their own exception types (ClientError, BotoCoreError), most
    of them not OSError: each is an unreadable lease, never an escaping error."""
    client, store = _s3()
    client.error = _request_errors()[index]
    with pytest.raises(LeaseReadError):
        read_record_strict(store, lease_key("s"))
    assert read_record(store, lease_key("s")) is None


def test_strict_read_over_s3_body_errors_and_truncation(monkeypatch):
    import http.client

    from botocore.exceptions import IncompleteReadError

    import yeto.cloud.s3_store as s3_store

    monkeypatch.setattr(s3_store, "_read_retry_pause", lambda attempt: None)
    client, store = _s3()
    for error in (IncompleteReadError(actual_bytes=3, expected_bytes=9), http.client.IncompleteRead(b"{")):
        client.body_error = error
        with pytest.raises(LeaseReadError):
            read_record_strict(store, lease_key("s"))
    client.body_error, client.body = None, b'{"incarnation": "aa'  # cut short
    with pytest.raises(LeaseReadError):
        read_record_strict(store, lease_key("s"))


def test_strict_read_over_s3_absent_answers(monkeypatch):
    """NoSuchKey (the modeled class) and a bare ClientError naming 404 / NoSuchKey /
    NotFound are an absent lease (None), not a read error."""
    client, store = _s3()
    assert read_record_strict(store, lease_key("s")) is None  # modeled NoSuchKey
    for code in ("NoSuchKey", "404", "NotFound"):
        client.error = _client_error(code, 404)
        assert read_record_strict(store, lease_key("s")) is None
    client.error = _client_error("NoSuchBucket", 404)  # a wrong bucket is no verdict
    with pytest.raises(LeaseReadError):
        read_record_strict(store, lease_key("s"))


def test_gate_over_s3_defers_on_a_5xx_instead_of_raising_it(tmp_path):
    client, store = _s3()
    client.error = _client_error("InternalError", 500)
    entries = [_entry("s", "aaaa", 0)]
    gate = IncarnationGate(store, lambda: entries, retry_s=0.0)
    with pytest.raises(LeaseUnknown):
        gate.verdict(entries[0], 0)


def test_poll_over_s3_does_not_raise_on_a_5xx():
    client, store = _s3()
    lease = IncarnationLease(store, "s", "aaaa")
    client.error = _client_error("InternalError", 500)
    assert lease.poll() is False and lease.unknown


def test_claim_pauses_between_its_reads_and_then_raises(monkeypatch):
    """The claim's retries are spaced (a transient 5xx outlasts three reads in a row),
    and a lease that stays unreadable is not overwritten."""
    import yeto.rl.engine.incarnation_lease as incarnation_lease

    pauses: list[float] = []
    monkeypatch.setattr(incarnation_lease, "_claim_pause", lambda attempt, sleep=None: pauses.append(attempt))
    client, store = _s3()
    client.error = _client_error("InternalError", 500)
    with pytest.raises(LeaseReadError):
        IncarnationLease(store, "s", "bbbb").claim()
    assert client.calls == incarnation_lease.CLAIM_READ_ATTEMPTS
    assert len(pauses) == incarnation_lease.CLAIM_READ_ATTEMPTS - 1


def test_claim_patiently_waits_out_an_outage_then_claims(tmp_path):
    from yeto.rl.engine.incarnation_lease import claim_patiently

    store = FlakyStore(tmp_path)
    store.objects.fail = _client_error("SlowDown", 503)
    clock, sleeps = Clock(), []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds
        if clock.now > 20:
            store.objects.fail = None  # the store answers again: the lease is absent

    claimed = claim_patiently(IncarnationLease(store, "s", "bbbb"), lambda: [], patience_s=600,
                              monotonic=clock, sleep=sleep)
    assert claimed is not None and claimed["incarnation"] == "bbbb" and sleeps
    assert read_lease(store, "s")["incarnation"] == "bbbb"


def test_claim_patiently_stops_or_gives_up():
    from yeto.rl.engine.incarnation_lease import claim_patiently

    client, store = _s3()
    client.error = _client_error("InternalError", 500)
    clock = Clock()

    def sleep(seconds: float) -> None:
        clock.now += seconds

    assert claim_patiently(IncarnationLease(store, "s", "b"), lambda: [], should_stop=lambda: True,
                           monotonic=clock, sleep=sleep) is None
    with pytest.raises(LeaseReadError):
        claim_patiently(IncarnationLease(store, "s", "b"), lambda: [], patience_s=60, monotonic=clock, sleep=sleep)


def test_a_take_stopped_while_groups_wait_loses_none_of_them(tmp_path):
    """A take that raises while groups wait for their lease returned nothing: the next
    take on the reader judges them again, in order (none lost, none counted stale)."""
    root, store = _fenced_buffer(tmp_path)
    reader, flaky, _ = _outage_reader(root, store, recover_after=10**9)

    class Stop(Exception):
        pass

    def stop(seconds: float) -> None:
        raise Stop()

    reader.sleep = stop
    with pytest.raises(Stop):
        reader.take(0, 2, current_version=None)
    flaky.objects.fail = None
    out, report = reader.take(0, 2, current_version=None)
    assert [position for position, _ in out] == [0, 2] and report.stale_incarnation_groups == 1
