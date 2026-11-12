"""The verl buffer feed judges a round's kept groups again right before the round is
handed to training (the verl twin of the BatchReader recheck): a round collects
``required`` groups one pull at a time, and a group judged on a lease cached up to
``refresh_s`` while the round waited for later groups must not be trained once its
incarnation was fenced meanwhile. An unreadable lease waits (fail closed)."""

from __future__ import annotations

import pytest

from tests.test_incarnation_gate_fail_closed import FlakyStore
from tests.test_incarnation_lease_fence import _write_group
from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl.buffer_feed import BufferCall, BufferFeed, RowShape
from yeto.rl.engine.incarnation_lease import STALE_REASON, IncarnationGate, IncarnationLease
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


@pytest.fixture(autouse=True)
def _no_claim_pause(monkeypatch):
    import yeto.rl.engine.incarnation_lease as incarnation_lease

    monkeypatch.setattr(incarnation_lease, "_claim_pause", lambda attempt, sleep=None: None)


def _setup(tmp_path, on_sleep):
    """A feed whose gate caches leases for 30 s (no time passes) and whose ``sleep``
    (the wait for the next group, the wait on an unreadable lease) runs ``on_sleep``."""
    root = tmp_path / "store"
    store = CheckpointStore(root)
    writer = FileSampleBuffer(store, batch_size=1)
    flaky = FlakyStore(root)
    reader = FileSampleBuffer(store, batch_size=1)
    gate = IncarnationGate(flaky, reader.entries, clock_margin_s=0.0, retry_s=0.0, monotonic=lambda: 0.0)
    added, records, sleeps = [], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        on_sleep(len(sleeps), flaky)

    feed = BufferFeed(reader, vmap=lambda: None, shape=RowShape(16, 16, 0), add_group=added.append,
                      record=records.append, emit=lambda name, **f: None, sleep=sleep, poll_s=0.5,
                      wait_s=600.0, clock=lambda: 0.0, incarnation_gate=gate)
    feed._local = lambda version: 0  # noqa: SLF001 - versions are not under test
    return store, writer, flaky, feed, added, records


def _round(feed, required, *, recheck=True):
    """The buffer pool's collection loop (``FullyAsyncRolloutPool.generate``) over the feed."""
    collector = far.RoundCollector(far.VersionMap(), 0, 2, required)
    collector.vmap.record(0, 0)
    while True:
        while not collector.full:
            meta = feed.pull()
            keep = collector.offer(far.SampleMeta(**meta))
            feed("yeto_keep_last" if keep else "yeto_drop_last")
        fenced = set(feed.recheck_kept()) if recheck else set()
        if not fenced:
            break
        collector.kept = [v for v in collector.kept if v.meta.uid not in fenced]
    feed.close_round(0)
    return [v.meta.uid for v in collector.kept]


def _uids(added):
    return [rows["non_tensor"]["uid"][0] for rows in added]


def _fence_while_waiting(store, writer):
    def on_sleep(count, _flaky):
        if count == 1:  # the round waits for its second group; meanwhile bbbb fences aaaa
            IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim()  # seals nothing
            _write_group(writer, "s1", "xxxx", 0)
            _write_group(writer, "s1", "xxxx", 1)
    return on_sleep


def test_the_window_exists_without_the_recheck(tmp_path):
    """The reproduction: position 0 passes on the cached lease, the round waits, aaaa is
    fenced; without the recheck 0 reaches training."""
    holder = {}
    store, writer, flaky, feed, added, _ = _setup(tmp_path, lambda c, f: holder["on"](c, f))
    holder["on"] = _fence_while_waiting(store, writer)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    assert _round(feed, 2, recheck=False) == ["buf0", "buf1"]
    assert _uids(added) == ["buf0", "buf1"]  # a fenced incarnation's group trained


def test_a_fence_while_the_round_waits_keeps_the_earlier_group_out(tmp_path):
    holder = {}
    store, writer, flaky, feed, added, records = _setup(tmp_path, lambda c, f: holder["on"](c, f))
    holder["on"] = _fence_while_waiting(store, writer)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    assert _round(feed, 2) == ["buf1", "buf2"]
    assert _uids(added) == ["buf1", "buf2"]
    assert feed.dropped == {STALE_REASON: 1} and feed.accepted == 2
    stale = [r for r in records if r.get("dropped") == STALE_REASON]
    assert [r["position"] for r in stale] == [0]
    assert [r["position"] for r in records if r.get("kept")] == [1, 2]
    assert sorted(r["position"] for r in records) == [0, 1, 2]  # one record per position
    assert feed.commit_round()[1] == [1, 2]  # the cursor record holds the trained positions


def test_no_fence_nothing_dropped_one_read_per_shard(tmp_path):
    def on_sleep(count, _flaky):
        raise AssertionError("the buffer holds the round: no wait")

    store, writer, flaky, feed, added, _ = _setup(tmp_path, on_sleep)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    for rid in range(3):
        _write_group(writer, "s0", "aaaa", rid)
    flaky.objects.only = "s0"
    assert _round(feed, 3) == ["buf0", "buf1", "buf2"]
    assert _uids(added) == ["buf0", "buf1", "buf2"] and feed.dropped == {}
    assert flaky.objects.reads == 2  # the selection's (cached), then the recheck's


def test_an_unreadable_lease_at_the_recheck_waits_then_judges(tmp_path):
    """The recheck cannot read s0's lease: nothing is handed over or dropped; it waits
    (fail closed) and, the lease readable again, finds 0 fenced."""
    root_store = {}

    def on_sleep(count, flaky):
        assert added == []  # nothing trained while the lease is unknown
        if count == 1:
            IncarnationLease(root_store["store"], "s0", "bbbb", clock=lambda: 0.0).claim()
            _write_group(root_store["writer"], "s0", "bbbb", 0)
            flaky.objects.fail = None

    store, writer, flaky, feed, added, records = _setup(tmp_path, on_sleep)
    root_store.update(store=store, writer=writer)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    assert feed.pull()["uid"] == "buf0"
    feed.keep()
    flaky.objects.fail = TimeoutError("read timeout")
    feed._lease_reads.clear()  # noqa: SLF001 - the recheck reads now
    original = feed.gate.monotonic
    ticks = iter(range(0, 1000, 5))
    feed.gate.monotonic = lambda: float(next(ticks))  # the retry reads again
    assert feed.recheck_kept() == ["buf0"]
    feed.gate.monotonic = original
    assert feed.lease_waits >= 1 and feed.dropped == {STALE_REASON: 1}
    assert feed.pull()["uid"] == "buf1"
    feed.keep()
    assert feed.recheck_kept() == []
    feed.close_round(0)
    assert _uids(added) == ["buf1"]


def test_buffer_call_hands_the_round_over_at_assembly(tmp_path):
    def on_sleep(count, _flaky):
        raise AssertionError("no wait")

    store, writer, _, feed, added, _ = _setup(tmp_path, on_sleep)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    calls = []
    call = BufferCall(feed, lambda name, *a, **k: calls.append((name, len(added))), round_id=lambda: 0)
    call("yeto_pull_sample")
    call("yeto_keep_last")
    assert added == []  # not handed over before the round is judged again
    call("yeto_assemble")
    assert calls == [("yeto_assemble", 1)]  # handed over before the trainer assembles
