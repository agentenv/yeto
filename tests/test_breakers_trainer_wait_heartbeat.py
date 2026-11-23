"""The buffer reader's wait heartbeat (``rl_sample_wait``) against the launcher's stall clock.

While a take waits for new groups (inside ``trainer.sample_wait``) the reader emits one island
event per ``max(60 s, stall / 4)``; ``launch.stall`` counts island events, so a legitimate wait no
longer ends the run as a hang, and the sample wait decides. A trainer that is truly hung (no
take waiting) or waiting on unreadable leases (no ``wait_s`` bound) emits nothing, and the stall
clock trips exactly when it did before.

The reader and the fleet share one fake clock; the reader's records go through the real echo
path (``event_echo.format_record`` -> ``TapeCollector``) the launcher's progress probe counts.
"""

from __future__ import annotations

import argparse

import pytest

from tests.buffer_rollout_procs import sample
from yeto.launch.fleet import FleetController, RunStalled
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import sample_roles, trainer_breakers
from yeto.rl.engine.incarnation_lease import LeaseUnknown
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.event_echo import TapeCollector, format_record

STALL_S = 900.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _buffer(root):
    return FileSampleBuffer(CheckpointStore(root), batch_size=2)


def _island(tmp_path, clock, *, stall_s=STALL_S):
    """A collector fed by the reader's emit, and a fleet whose stall clock counts it."""
    collector = TapeCollector(tmp_path / "tape" / "island-0.jsonl")

    class Ops:
        def now(self):
            return clock()

    fleet = FleetController({}, None, Ops(), poll_interval=1, recover_timeout=0,
                            progress_probe=lambda: collector.count, stall_timeout=stall_s)

    def emit(event, **fields):
        collector.feed(format_record({"island_id": 0, "time_unix": clock(), "event": event, **fields}) + "\n")
    return collector, fleet, emit


def _waiting_reader(tmp_path, clock, fleet, emit, **kwargs):
    """A reader whose every poll advances the clock and runs one fleet stall check."""

    def sleep(dt):
        clock.now += dt
        fleet._check_stall()

    reader = br.BatchReader(_buffer(tmp_path / "buf"), clock=clock, sleep=sleep, **kwargs)
    reader.emit = emit
    return reader


def _events(collector, name):
    import json

    return [json.loads(line) for line in collector.path.read_text().splitlines()
            if json.loads(line)["event"] == name]


# ------------------------------------------------------------------ cadence
@pytest.mark.parametrize("stall_s, beat_s", [(900.0, 225.0), (7200.0, 1800.0), (120.0, 60.0), (60.0, 60.0)])
def test_the_heartbeat_is_a_quarter_of_the_stall_clock_and_at_least_a_minute(stall_s, beat_s):
    assert trainer_breakers.wait_heartbeat_s(stall_s) == beat_s
    assert trainer_breakers.wait_heartbeat_env(stall_s) == {br.HEARTBEAT_ENV: f"{beat_s:g}"}


@pytest.mark.parametrize("stall_s", [60.0, 120.0, 900.0, 3600.0, 7200.0, 86400.0])
def test_rule_d5_uses_the_same_heartbeat_formula(stall_s):
    """The rules stay import-light: D5 keeps the formula as literals (rules_timing)."""
    from yeto.breakers import rules_timing

    assert rules_timing.wait_heartbeat_s(stall_s) == trainer_breakers.wait_heartbeat_s(stall_s)
    assert rules_timing.WAIT_HEARTBEAT_MIN_S == trainer_breakers.MIN_WAIT_HEARTBEAT_S


BASE = ["--training-mode", "rl", "--cluster-prefix", "hb", "--model", "Qwen/Qwen3-1.7B",
        "--data", "HuggingFaceH4/MATH-500", "--rollout-batch-size", "8", "--n-samples-per-prompt", "8"]


@pytest.mark.parametrize("argv, expected", [
    (["--gpu", "modal:8xh100", "--rl-sample-role", "trainer", "--rl-max-policy-age", "2"], True),
    (["--gpu", "modal:8xh100,modal:8xh100", "--rl-island-sample-role", "0:trainer",
      "--rl-island-sample-role", "1:sampler", "--rl-max-policy-age", "2"], True),
    (["--gpu", "modal:8xh100"], False),
    (["--gpu", "modal:8xh100", "--rl-sample-role", "trainer", "--rl-max-policy-age", "2",
      "--rl-backend", "verl"], False),  # the verl buffer feed does not heartbeat
])
def test_the_rule_facts_know_a_buffer_trainer_as_the_launcher_does(argv, expected):
    from yeto.breakers.rules_facts import buffer_trainer
    from yeto.rl import up_consistency

    args = up_consistency.trainer_args([*BASE, *argv])
    assert args is not None
    assert buffer_trainer(args) is expected


def test_no_heartbeat_env_without_a_stall_clock():
    assert trainer_breakers.wait_heartbeat_env(0.0) == {}


def test_the_launcher_gives_the_interval_to_buffer_trainer_islands_only():
    from yeto.launcher import _wait_heartbeat_env

    launch = argparse.Namespace(rl_stall_timeout=7200)
    role = lambda r: argparse.Namespace(rl_sample_role=r)  # noqa: E731
    assert _wait_heartbeat_env(launch, role(sample_roles.TRAINER)) == {br.HEARTBEAT_ENV: "1800"}
    assert _wait_heartbeat_env(launch, role(sample_roles.CONVERTIBLE)) == {br.HEARTBEAT_ENV: "1800"}
    assert _wait_heartbeat_env(launch, role(sample_roles.SAMPLER)) == {}
    assert _wait_heartbeat_env(launch, role(sample_roles.BOTH)) == {}
    assert _wait_heartbeat_env(argparse.Namespace(rl_stall_timeout=None), role(sample_roles.TRAINER)) == {
        br.HEARTBEAT_ENV: "225"}  # the default stall clock (900 s)
    assert _wait_heartbeat_env(argparse.Namespace(rl_stall_timeout=0), role(sample_roles.TRAINER)) == {}
    assert br.HEARTBEAT_ENV in br.ISLAND_ENVS  # forwarded to the Miles rollout workers


def test_a_waiting_take_beats_once_per_interval_and_starts_over_on_a_new_group(tmp_path):
    clock = Clock()
    writer = br.GroupWriter(_buffer(tmp_path / "buf"), "s0")
    events = []

    def sleep(dt):
        clock.now += dt
        if clock.now == 300.0:  # a sampler writes one group 300 s into the wait
            writer.write([sample(0, 0, 1), sample(1, 0, 1)], group_key="g0")

    reader = br.BatchReader(_buffer(tmp_path / "buf"), max_policy_age=1, wait_s=1000.0, poll_s=10.0,
                            heartbeat_s=200.0, clock=clock, sleep=sleep)
    reader.emit = lambda event, **fields: events.append((event, fields))
    with pytest.raises(br.SampleBufferRolloutError, match="1/2 groups after 1000 s"):
        reader.take(0, 2, current_version=1)
    beats = [fields for event, fields in events if event == br.WAIT_EVENT]
    # at 200 s, then the group at 300 s restarts both clocks: 500, 700, 900, 1100, 1300 s
    assert [round(b["waited_s"]) for b in beats] == [200, 200, 400, 600, 800, 1000]
    assert all(b["limit_s"] == 1000.0 and b["heartbeat_s"] == 200.0 and b["needed"] == 2 for b in beats)
    assert [b["groups"] for b in beats] == [0, 1, 1, 1, 1, 1]


def test_heartbeat_zero_emits_nothing(tmp_path):
    clock = Clock()
    events = []

    def sleep(dt):
        clock.now += dt

    reader = br.BatchReader(_buffer(tmp_path / "buf"), max_policy_age=1, wait_s=500.0, poll_s=10.0,
                            heartbeat_s=0, clock=clock, sleep=sleep)
    reader.emit = lambda event, **fields: events.append(event)
    with pytest.raises(br.SampleBufferRolloutError):
        reader.take(0, 1, current_version=1)
    assert br.WAIT_EVENT not in events


# ------------------------------------------------------------------ against the stall clock
def test_a_legitimate_wait_longer_than_the_stall_clock_ends_on_the_sample_wait(tmp_path):
    """Stall 900 s, sample wait 3600 s: before, the launcher ended the run at 900 s of a wait
    (RunStalled, exit 6); now the heartbeat keeps the stall clock fed and the sample wait
    decides at 3600 s."""
    clock = Clock()
    collector, fleet, emit = _island(tmp_path, clock)
    emit("rl_driver_start")  # the island is up: the stall clock runs
    fleet._check_stall()
    reader = _waiting_reader(tmp_path, clock, fleet, emit, max_policy_age=1, wait_s=3600.0, poll_s=5.0,
                             heartbeat_s=trainer_breakers.wait_heartbeat_s(STALL_S))
    with pytest.raises(br.SampleBufferRolloutError, match="after 3600 s without a new group"):
        reader.take(0, 1, current_version=1)
    assert clock.now > 3600.0
    assert len(_events(collector, br.WAIT_EVENT)) == 16  # every 225 s up to 3600 s
    assert _events(collector, "rl_breaker")[-1]["name"] == "trainer.sample_wait"


def test_a_hung_trainer_still_trips_the_stall_clock_at_the_same_time(tmp_path):
    """No take waiting (the trainer hangs elsewhere): no heartbeat, RunStalled at 900 s."""
    clock = Clock()
    collector, fleet, emit = _island(tmp_path, clock)
    emit("rl_driver_start")
    fleet._check_stall()
    clock.now = STALL_S - 1
    fleet._check_stall()  # not yet
    clock.now = STALL_S
    with pytest.raises(RunStalled) as info:
        fleet._check_stall()
    assert info.value.silent_s == STALL_S and info.value.timeout_s == STALL_S


def test_a_wait_on_unreadable_leases_gets_no_heartbeat_and_still_trips_the_stall_clock(tmp_path):
    """A group whose lease cannot be read waits without the wait_s bound; it is not a
    legitimate wait for samplers, so it gets no heartbeat and the stall clock ends it at
    900 s as before."""

    class UnreadableLeases:
        def verdict(self, entry, position):
            raise LeaseUnknown("lease read timed out")

    clock = Clock()
    br.GroupWriter(_buffer(tmp_path / "buf"), "s0").write([sample(0, 0, 1), sample(1, 0, 1)], group_key="g0")
    collector, fleet, emit = _island(tmp_path, clock)
    emit("rl_driver_start")
    fleet._check_stall()
    reader = _waiting_reader(tmp_path, clock, fleet, emit, max_policy_age=1, wait_s=3600.0, poll_s=5.0,
                             heartbeat_s=trainer_breakers.wait_heartbeat_s(STALL_S),
                             incarnation_gate=UnreadableLeases(), one_group_per_task=False)
    with pytest.raises(RunStalled) as info:
        reader.take(0, 1, current_version=1)
    assert info.value.silent_s == STALL_S
    assert _events(collector, br.WAIT_EVENT) == []
