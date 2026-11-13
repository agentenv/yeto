"""A lease read the store refuses for good (refused credentials, no such bucket) is
not waited out: the gate raises :class:`LeaseStoreFatal` (shard, store code, what to
do) and the trainer exits through its failure path instead of pausing forever. An
expired session token and an outage (5xx, timeout) stay fail-closed pauses."""

from __future__ import annotations

import logging

import pytest

from tests.test_incarnation_gate_fail_closed import (
    FlakyStore,
    _client_error,
    _fenced_buffer,
    _outage_reader,
    _s3,
)
from tests.test_incarnation_lease_fence import _entry, _write_group
from yeto.rl.engine.incarnation_lease import (
    FATAL_MARK,
    IncarnationGate,
    IncarnationLease,
    LeaseStoreFatal,
    LeaseUnknown,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

FATAL = [("AccessDenied", 403), ("InvalidAccessKeyId", 403), ("SignatureDoesNotMatch", 403),
         ("NoSuchBucket", 404)]
RETRYABLE = [("ExpiredToken", 400), ("InternalError", 500), ("SlowDown", 503)]


@pytest.fixture(autouse=True)
def _no_claim_pause(monkeypatch):
    import yeto.rl.engine.incarnation_lease as incarnation_lease

    monkeypatch.setattr(incarnation_lease, "_claim_pause", lambda attempt, sleep=None: None)


@pytest.mark.parametrize("code,status", FATAL)
def test_gate_raises_a_fatal_store_error_with_shard_code_and_advice(code, status, caplog):
    client, store = _s3()
    client.error = _client_error(code, status)
    entries = [_entry("s7", "aaaa", 0)]
    gate = IncarnationGate(store, lambda: entries, retry_s=0.0)
    with caplog.at_level(logging.ERROR), pytest.raises(LeaseStoreFatal) as raised:
        gate.verdict(entries[0], 0)
    message = str(raised.value)
    assert FATAL_MARK in message and "shard s7" in message and code in message and "credentials" in message
    assert not isinstance(raised.value, LeaseUnknown) and gate.deferred == 0
    assert any(FATAL_MARK in r.getMessage() for r in caplog.records)
    with pytest.raises(LeaseStoreFatal):  # not cached as a pause: read and refused again
        gate.verdict(entries[0], 0)


@pytest.mark.parametrize("code,status", RETRYABLE)
def test_gate_pauses_on_an_expired_token_or_an_outage(code, status):
    client, store = _s3()
    client.error = _client_error(code, status)
    entries = [_entry("s", "aaaa", 0)]
    gate = IncarnationGate(store, lambda: entries, retry_s=0.0)
    with pytest.raises(LeaseUnknown):
        gate.verdict(entries[0], 0)
    assert gate.deferred == 1


def test_gate_recheck_raises_on_a_fatal_store_error():
    client, store = _s3()
    client.error = _client_error("AccessDenied", 403)
    entries = [_entry("s", "aaaa", 0)]
    gate = IncarnationGate(store, lambda: entries, retry_s=0.0)
    with pytest.raises(LeaseStoreFatal):
        gate.recheck([(entries[0], 0)], {})


# --------------------------------------------------------------------------- Miles BatchReader

@pytest.mark.parametrize("code,status", FATAL)
def test_miles_take_exits_on_a_fatal_store_error_without_waiting(tmp_path, code, status):
    root, store = _fenced_buffer(tmp_path)
    reader, flaky, sleeps = _outage_reader(root, store, recover_after=10_000)
    flaky.objects.fail = _client_error(code, status)
    with pytest.raises(LeaseStoreFatal, match=code):
        reader.take(0, 2, current_version=None)
    assert sleeps == []  # no fail-closed pause


@pytest.mark.parametrize("code,status", RETRYABLE)
def test_miles_take_waits_out_an_expired_token_or_an_outage(tmp_path, code, status):
    root, store = _fenced_buffer(tmp_path)
    reader, flaky, sleeps = _outage_reader(root, store, recover_after=2)
    flaky.objects.fail = _client_error(code, status)
    out, _ = reader.take(0, 2, current_version=None)
    assert [position for position, _ in out] == [0, 2] and len(sleeps) == 2


def test_miles_recheck_exits_on_a_fatal_store_error(tmp_path):
    """The lease turns unreadable for good while the take waits for its second group:
    the recheck before training raises instead of deferring."""
    from tests.test_incarnation_gate_recheck import _reader, _store

    root, store, writer = _store(tmp_path)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    counts: list[int] = []

    def on_sleep(count, flaky):
        counts.append(count)
        if count == 1:
            _write_group(writer, "s1", "xxxx", 0)
            flaky.objects.fail, flaky.objects.only = _client_error("NoSuchBucket", 404), "s0"

    reader, _, _ = _reader(root, store, on_sleep)
    with pytest.raises(LeaseStoreFatal, match="NoSuchBucket"):
        reader.take(0, 2, current_version=None)
    assert counts == [1]


# --------------------------------------------------------------------------- verl buffer_feed

def _feed(tmp_path, error, on_sleep=None):
    from yeto.rl.adapters.verl.buffer_feed import BufferFeed, RowShape

    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s", "aaaa", 1)
    flaky = FlakyStore(root)
    flaky.objects.fail = error
    reader = FileSampleBuffer(store, batch_size=1)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        (on_sleep or (lambda n, f: None))(len(sleeps), flaky)

    feed = BufferFeed(reader, vmap=lambda: None, shape=RowShape(16, 16, 0), add_group=lambda rows: None,
                      record=lambda r: None, emit=lambda name, **f: None, wait_s=0.0, poll_s=0.25, sleep=sleep,
                      incarnation_gate=IncarnationGate(flaky, reader.entries, retry_s=0.0))
    feed._local = lambda version: 0  # noqa: SLF001 - versions are not under test
    return feed, flaky, sleeps


@pytest.mark.parametrize("code,status", FATAL)
def test_verl_pull_exits_on_a_fatal_store_error(tmp_path, code, status):
    feed, _, sleeps = _feed(tmp_path, _client_error(code, status))
    with pytest.raises(LeaseStoreFatal, match=code):
        feed.pull()
    assert sleeps == [] and feed.lease_waits == 0


@pytest.mark.parametrize("code,status", RETRYABLE)
def test_verl_pull_waits_out_an_expired_token_or_an_outage(tmp_path, code, status):
    def recover(count, flaky):
        if count == 2:
            flaky.objects.fail = None

    feed, _, sleeps = _feed(tmp_path, _client_error(code, status), recover)
    assert feed.pull()["uid"] == "buf0" and feed.lease_waits == 2


def test_verl_recheck_exits_on_a_fatal_store_error(tmp_path):
    feed, flaky, sleeps = _feed(tmp_path, None)
    assert feed.pull()["uid"] == "buf0"
    feed.keep()
    flaky.objects.fail = _client_error("InvalidAccessKeyId", 403)
    feed._lease_reads.clear()  # noqa: SLF001 - the recheck reads now
    with pytest.raises(LeaseStoreFatal, match="InvalidAccessKeyId"):
        feed.recheck_kept()
    assert feed.lease_waits == 0
