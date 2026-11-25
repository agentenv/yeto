"""The core (Miles) buffer reader's ``trainer.sample_wait``: the wait for a group it can
train, the same rule as the verl buffer feed (``adapters.verl.buffer_feed``).

* The fault path: the samplers keep writing groups the take passes over (all over the
  policy-age limit). Every selection used to start the wait again, and a selection that
  passed everything over went straight to the next one, so the take waited for good. It
  now stops by ``trainer.sample_wait`` once ``wait_s`` passed without a trainable group.
* The normal path (groups it trains keep arriving, slowly): the take's result, its
  events and its every clock read and sleep are those of the reader before this change,
  item for item (the expected values below were recorded on it).
"""

from __future__ import annotations

import pytest

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


def _buffer(root):
    return FileSampleBuffer(CheckpointStore(root), batch_size=2)


def _group(g, version, n=4):
    return [sample(n * g + j, g, version) for j in range(n)]


class _Run:
    """A fake clock and sleep that log every call, and the reader's events."""

    def __init__(self, limit=100_000):
        self.now, self.limit = 0.0, limit
        self.log: list[tuple] = []
        self.events: list[tuple[str, dict]] = []
        self.on_sleep = lambda: None

    def clock(self):
        self.log.append(("clock", self.now))
        assert len(self.log) < self.limit, "the take never stopped"
        return self.now

    def sleep(self, seconds):
        self.log.append(("sleep", seconds))
        self.now += seconds
        self.on_sleep()

    def emit(self, event, **fields):
        self.events.append((event, fields))


def _reader(buffer, run, **kwargs):
    return br.BatchReader(buffer, max_policy_age=1, wait_s=50.0, poll_s=10.0, group_size=4, clock=run.clock,
                          sleep=run.sleep, prefetch=False, read_workers=1, incarnation_gate=None,
                          heartbeat_s=20.0, max_inflight_bytes=None, **kwargs)


# ---------------------------------------------------------------------- the fault path
def test_over_age_groups_that_keep_coming_stop_the_take_by_sample_wait(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    run = _Run()
    written = []

    def a_stale_group_every_poll():
        written.append(len(written))
        writer.write(_group(len(written), 1), group_key=f"old{len(written)}")

    run.on_sleep = a_stale_group_every_poll
    writer.write(_group(0, 1), group_key="old0")
    reader = _reader(buffer, run)
    reader.emit = run.emit
    with pytest.raises(br.SampleBufferRolloutError, match="without a new group it can train") as raised:
        reader.take(0, 2, current_version=5)  # age floor 4: every group of v1 is passed over
    assert getattr(raised.value, "yeto_breaker", None) == "trainer.sample_wait"
    assert len(written) >= 5 and raised.value.passed_over["stale"] >= 5  # they kept coming
    [trip] = [f for e, f in run.events if e == "rl_breaker" and f.get("state") == "trip"]
    assert trip["name"] == "trainer.sample_wait" and 50 < trip["value"] <= 60


def test_an_empty_buffer_after_passed_over_groups_waits_from_the_last_trainable_one(tmp_path):
    """Over-age groups, then nothing: the wait started before them (the take's start), not
    at the last group passed over."""
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    run = _Run()
    late = iter(range(1, 4))

    def three_stale_groups_then_nothing():
        g = next(late, None)
        if g is not None:
            writer.write(_group(g, 1), group_key=f"old{g}")

    run.on_sleep = three_stale_groups_then_nothing
    writer.write(_group(0, 1), group_key="old0")
    reader = _reader(buffer, run)
    reader.emit = run.emit
    with pytest.raises(br.SampleBufferRolloutError, match="without a new group it can train"):
        reader.take(0, 2, current_version=5)
    # the last stale group arrived at t=30; the take stopped 50 s after its start, not after t=30
    assert run.now == 60.0


def test_a_trainable_group_after_passed_over_ones_starts_the_wait_again(tmp_path):
    """Over-age groups, a trainable one at t=20, then nothing: the take (two groups) stops
    50 s after the trainable group, with the message of a wait with no new group."""
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    run = _Run()

    def stale_then_one_trainable():
        if run.now == 10:
            writer.write(_group(1, 1), group_key="old1")
        elif run.now == 20:
            writer.write(_group(2, 5), group_key="new2")

    run.on_sleep = stale_then_one_trainable
    writer.write(_group(0, 1), group_key="old0")
    reader = _reader(buffer, run)
    reader.emit = run.emit
    with pytest.raises(br.SampleBufferRolloutError) as raised:
        reader.take(0, 2, current_version=5)
    assert str(raised.value).startswith("rollout 0: 1/2 groups after 50 s without a new group (buffer position 3)")
    assert run.now == 80.0  # t=20 + 50 s, at the next poll


def test_a_group_waiting_for_its_lease_starts_the_wait_as_before(tmp_path):
    """A selected group whose lease cannot be read is not passed over: it starts the wait
    (as before), the lease pause is not counted, and when it is then found fenced the take
    stops 50 s after that selection (t=40), not at once."""
    from yeto.rl.engine.incarnation_lease import LeaseUnknown

    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    run = _Run()

    def one_group_at_40():
        if run.now == 40:
            writer.write(_group(0, 5), group_key="late")

    class Gate:  # the lease of the group's shard is unreadable until t=80, then fenced
        def verdict(self, entry, position):
            if run.now < 80:
                raise LeaseUnknown("lease unreadable")
            return "superseded incarnation"

    run.on_sleep = one_group_at_40
    reader = br.BatchReader(buffer, max_policy_age=1, wait_s=50.0, poll_s=10.0, group_size=4, clock=run.clock,
                            sleep=run.sleep, prefetch=False, read_workers=1, incarnation_gate=Gate(),
                            heartbeat_s=0, max_inflight_bytes=None)
    reader.emit = run.emit
    with pytest.raises(br.SampleBufferRolloutError, match="without a new group"):
        reader.take(0, 1, current_version=5)
    assert run.now == 100.0


# ---------------------------------------------------------------------- the normal path
def _compact(log):
    """``[("clock", 0.0), ("clock", 0.0), ("sleep", 10.0), ...]`` as ``"c0x2 s10 ..."``."""
    tokens = [("c" if kind == "clock" else "s") + f"{value:g}" for kind, value in log]
    runs: list[list] = []
    for token in tokens:
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    return " ".join(token + (f"x{n}" if n > 1 else "") for token, n in runs)


# Recorded on the reader before this change (main 385384b6c): every clock read and sleep of
# the two takes, every event, and the first take's report.
FIRST_TAKE = ("c0x10 s10 c10x4 s10 c20x5 s10 c30x4 s10 c40x9 s10 c50x4 s10 c60x5 s10 c70x4 s10 c80x9 s10 "
              "c90x4 s10 c100x5 s10 c110x4 s10 c120x6")
SECOND_TAKE = "c120x5 s10 c130x4 s10 c140x5 s10 c150x4 s10 c160x5 s10 c170x4 s10 c180x4"
HEARTBEAT = {"heartbeat_s": 20.0, "limit_s": 50.0}
EVENTS = [
    ("rl_sample_wait", {"groups": 1, "needed": 4, "rollout_id": 0, "waited_s": 20.0, **HEARTBEAT}),
    ("rl_sample_wait", {"groups": 2, "needed": 4, "rollout_id": 0, "waited_s": 20.0, **HEARTBEAT}),
    ("rl_sample_wait", {"groups": 3, "needed": 4, "rollout_id": 0, "waited_s": 20.0, **HEARTBEAT}),
    ("rl_sample_wait", {"groups": 0, "needed": 1, "rollout_id": 1, "waited_s": 20.0, **HEARTBEAT}),
    ("rl_sample_wait", {"groups": 0, "needed": 1, "rollout_id": 1, "waited_s": 40.0, **HEARTBEAT}),
    ("rl_breaker", {"action": "stop", "count": 1,
                    "detail": "rollout 1: 0/1 groups after 50 s without a new group (buffer position 4)",
                    "direction": "max", "kind": "breaker", "layer": "trainer",
                    "legacy_event": "SampleBufferRolloutError", "limit": 50.0, "name": "trainer.sample_wait",
                    "schema": "yeto-breaker-v1", "scope": {}, "state": "trip",
                    "stop_reason": "breaker:trainer.sample_wait", "unit": "s", "value": 60.0}),
]
NONZERO_METRICS = {"sample_buffer/end": 4.0, "sample_buffer/groups": 4.0, "sample_buffer/wait_s": 120.0}


def test_a_slow_round_of_trainable_groups_reads_and_reports_exactly_as_before(tmp_path):
    """One trainable group every 40 s (the round needs 120 s > wait_s 50), then a take with
    nothing to read: the same result, clock reads, sleeps, heartbeats and stop as before."""
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    run = _Run()
    arrivals = iter(range(1, 4))

    def a_trainable_group_every_40_s():
        if run.now % 40 == 0:
            g = next(arrivals, None)
            if g is not None:
                writer.write(_group(g, 1), group_key=f"g{g}")

    run.on_sleep = a_trainable_group_every_40_s
    writer.write(_group(0, 1), group_key="g0")
    reader = _reader(buffer, run)
    reader.emit = run.emit
    groups, report = reader.take(0, 4, current_version=1)
    assert [position for position, _ in groups] == [0, 1, 2, 3]
    first = list(run.log)
    with pytest.raises(br.SampleBufferRolloutError) as raised:
        reader.take(1, 1, current_version=1)
    assert str(raised.value) == "rollout 1: 0/1 groups after 50 s without a new group (buffer position 4)"
    assert raised.value.yeto_breaker == "trainer.sample_wait"
    assert _compact(first) == FIRST_TAKE and _compact(run.log[len(first):]) == SECOND_TAKE
    assert [(e, {k: v for k, v in f.items() if k != "time_unix"}) for e, f in run.events] == EVENTS
    metrics = report.metrics()
    assert {k: v for k, v in metrics.items() if v} == NONZERO_METRICS and len(metrics) == 27
