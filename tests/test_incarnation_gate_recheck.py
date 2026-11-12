"""The incarnation gate judges the groups of a round once more right before the take
hands them to training: a fence that came while the take waited for later groups
keeps an earlier group out (store simulation finding ``stale_incarnation_group_trained``,
preset fence-zombie), one lease read per shard; an unreadable lease defers the group
(strict order, #373), and a dropped group holds no task (#368)."""

from __future__ import annotations

import pytest

from tests.test_incarnation_gate_fail_closed import FlakyStore
from tests.test_incarnation_lease_fence import _entry, _write_group
from yeto.rl.engine.incarnation_lease import IncarnationGate, IncarnationLease, LeaseUnknown
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


@pytest.fixture(autouse=True)
def _no_claim_pause(monkeypatch):
    import yeto.rl.engine.incarnation_lease as incarnation_lease

    monkeypatch.setattr(incarnation_lease, "_claim_pause", lambda attempt, sleep=None: None)


def _write_task(buffer, sampler: str, inc: str, rid: int, slot: int, attempt: int) -> None:
    from yeto.rl.engine.buffer_payload import GroupWriter
    from yeto.rl.engine.policy_versions import engine_weight_version

    sample = {"tokens": [1, 2, 3], "response_length": 1, "rollout_log_probs": [-0.1], "loss_mask": [1],
              "reward": 1.0, "metadata": {"weight_version": engine_weight_version(0)}}
    GroupWriter(buffer, sampler).write([dict(sample, index=0), dict(sample, index=1)],
                                       group_key=f"{inc}-r{rid}-p",
                                       sampler_round={"sampler": sampler, "incarnation": inc, "rollout_id": rid,
                                                      "last": True, "dispense_position": slot,
                                                      "dispense_attempt": attempt, "dispense_plan": "plan"})


def _reader(root, store, on_sleep, **gate_kwargs):
    """A reader whose gate caches leases for 30 s (no time passes: the cache never
    expires on its own) and whose ``sleep`` runs ``on_sleep(count)``."""
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    flaky = FlakyStore(root)
    buffer = FileSampleBuffer(store, batch_size=1)
    gate = IncarnationGate(flaky, buffer.entries, clock_margin_s=0.0, retry_s=0.0,
                           monotonic=lambda: 0.0, **gate_kwargs)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        on_sleep(len(sleeps), flaky)

    reader = BatchReader(buffer, max_policy_age=2, prefetch=False, read_workers=1, wait_s=600.0,
                         poll_s=0.5, sleep=sleep, incarnation_gate=gate)
    return reader, flaky, gate


def _store(tmp_path):
    root = tmp_path / "store"
    store = CheckpointStore(root)
    return root, store, FileSampleBuffer(store, batch_size=1)


def test_a_fence_while_the_take_waits_keeps_the_earlier_group_out(tmp_path):
    """Position 0 (s0/aaaa) passes the gate at selection; the take waits for a second
    group; meanwhile bbbb fences aaaa (0 not sealed). Before the fix 0 was trained
    against the cached lease; now it is dropped and the take trains s1's 1 and 2."""
    root, store, writer = _store(tmp_path)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)

    def on_sleep(count, _flaky):
        if count == 1:
            IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim()  # seals nothing
            _write_group(writer, "s1", "xxxx", 0)
            _write_group(writer, "s1", "xxxx", 1)

    reader, flaky, gate = _reader(root, store, on_sleep)
    out, report = reader.take(0, 2, current_version=None)
    assert [position for position, _ in out] == [1, 2]
    assert report.stale_incarnation_groups == 1 and report.lease_deferred_groups == 0
    assert report.end == 3 and gate.stale == 1


def test_the_recheck_reads_each_shard_once_per_take(tmp_path):
    """No fence: the round is judged again with one lease read per shard (s0 here;
    s1 has no lease and is read once too), nothing is dropped."""
    root, store, writer = _store(tmp_path)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    for rid in range(3):
        _write_group(writer, "s0", "aaaa", rid)

    def on_sleep(count, _flaky):
        raise AssertionError("the buffer holds the round: no wait")

    reader, flaky, _ = _reader(root, store, on_sleep)
    flaky.objects.only = "s0"
    out, report = reader.take(0, 3, current_version=None)
    assert [position for position, _ in out] == [0, 1, 2]
    assert flaky.objects.reads == 2  # the selection's (cached), then the recheck's
    assert report.stale_incarnation_groups == 0


def test_an_unreadable_lease_at_the_recheck_defers_in_strict_order(tmp_path):
    """The recheck cannot read s0's lease: 0 and every group behind it wait (not
    trained, not counted stale); when the lease reads again 0 is found fenced."""
    root, store, writer = _store(tmp_path)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_group(writer, "s0", "aaaa", 0)
    seen: list = []
    reader_box: list = []

    def on_sleep(count, flaky):
        seen.append(sorted(p for p, _ in reader_box[0]._deferred))  # noqa: SLF001
        if count == 1:
            IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim()
            _write_group(writer, "s1", "xxxx", 0)
            flaky.objects.fail, flaky.objects.only = TimeoutError("read timeout"), "s0"
        if count == 3:
            flaky.objects.fail = None
            _write_group(writer, "s1", "xxxx", 1)

    reader, flaky, _ = _reader(root, store, on_sleep)
    reader_box.append(reader)
    out, report = reader.take(0, 2, current_version=None)
    assert seen[1:3] == [[0, 1], [0, 1]]  # the outage: 0 and 1 wait, nothing trained
    assert [position for position, _ in out] == [1, 2]
    assert report.stale_incarnation_groups == 1 and report.lease_deferred_groups >= 2
    assert not reader._deferred  # noqa: SLF001


def test_a_group_dropped_at_the_recheck_holds_no_task(tmp_path):
    """Slot 7 is written by s0/aaaa (position 0) and, dispensed again, by s1 (1):
    the take passes 1 over as a duplicate of 0. aaaa is fenced while the take
    waits; 0 is dropped, so 1 is judged again and trains slot 7 (#368: one group
    per task, and a fenced group is none)."""
    root, store, writer = _store(tmp_path)
    IncarnationLease(store, "s0", "aaaa", clock=lambda: 0.0).claim()
    _write_task(writer, "s0", "aaaa", 0, slot=7, attempt=0)
    _write_task(writer, "s1", "xxxx", 0, slot=7, attempt=1)

    def on_sleep(count, _flaky):
        if count == 1:
            IncarnationLease(store, "s0", "bbbb", clock=lambda: 0.0).claim()
            _write_task(writer, "s1", "xxxx", 1, slot=8, attempt=0)

    reader, _, _ = _reader(root, store, on_sleep)
    out, report = reader.take(0, 2, current_version=None)
    assert [position for position, _ in out] == [1, 2]
    assert report.stale_incarnation_groups == 1 and report.duplicate_task_groups == 0
    assert reader.tasks.owner == {("plan", 7): 1, ("plan", 8): 2}


def test_gate_recheck_reads_once_per_shard_and_defers_an_unreadable_one(tmp_path):
    root = tmp_path / "store"
    store = CheckpointStore(root)
    IncarnationLease(store, "s", "aaaa", clock=lambda: 0.0).claim()
    flaky = FlakyStore(root)
    now = [0.0]
    entries = [_entry("s", "aaaa", appended_at=10.0) for _ in range(4)]
    gate = IncarnationGate(flaky, lambda: entries, clock_margin_s=0.0, retry_s=0.0, monotonic=lambda: now[0])
    assert all(gate.verdict(entry, p) is None for p, entry in enumerate(entries))
    reads, read = flaky.objects.reads, {}
    IncarnationLease(store, "s", "bbbb", clock=lambda: 5.0).claim()
    verdicts = gate.recheck([(entry, p) for p, entry in enumerate(entries)], read)
    assert all("superseded by bbbb" in v for v in verdicts) and flaky.objects.reads == reads + 1
    assert gate.recheck([(entries[0], 0)], read) and flaky.objects.reads == reads + 1  # read just now
    now[0] = 2.0
    flaky.objects.fail = TimeoutError("read timeout")
    [unknown] = gate.recheck([(entries[0], 0)], read)
    assert isinstance(unknown, LeaseUnknown)
    with pytest.raises(LeaseUnknown):  # an older cached read does not answer while the reads fail
        gate.verdict(entries[0], 0)
