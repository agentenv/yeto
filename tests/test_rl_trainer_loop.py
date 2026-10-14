"""rl-trainer-start-stop L5.3b (CPU, fakes): the scheduler in the controller loop.

A fake cloud (the island builder: a fresh fake trainer per incarnation, released
on stop) and the real file sample buffer, scheduler S1 and controller journal.
The head restarts in the middle: a new loop over the same journal continues."""

from __future__ import annotations

import argparse
import time

import pytest
import torch
from test_rl_engine_driver import NAME, PROFILE, _driver

from yeto import runs
from yeto.launch import trainer_loop as launch_loop
from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine import recovery
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.fake_lifecycle import (
    BufferFedFakeRollout,
    FakeMembership,
    FakeRoundCuts,
    FakeSampler,
    cuttable_fake_engine,
)
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.trainer_lifecycle import DriverLifecycle, Hold, LifecycleState, Start, Stop
from yeto.rl.engine.trainer_loop import LOOP_KIND, TrainerLoop, TrainerLoopThread, next_incarnation
from yeto.rl.engine.trainer_scheduler import SCHEDULER_STATE_KIND, S1Config, S1Scheduler

BATCH = 4  # fake engine: 2 groups x 2 samples
T_S = 5.0


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FakeCloud:
    """Acquire = build a fresh fake trainer island; release = give the cards back."""

    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path
        self.cuts = FakeRoundCuts(tmp_path / "cuts", watermark=lambda: self.trainer_buffer.watermark)
        self.membership = FakeMembership()
        self.buffer = None  # the head's buffer
        self.trainer_buffer = None  # the buffer of the running trainer
        self.rollouts = []
        self.engine = None
        self.acquired: list[int] = []
        self.released: list[int] = []

    def new_buffer(self) -> FileSampleBuffer:
        # every head opens the buffer in the shared store; this in-process trainer shares it
        self.buffer = FileSampleBuffer(CheckpointStore(self.tmp_path / "store"), batch_size=BATCH)
        return self.buffer

    def build(self, handle):
        self.acquired.append(handle.incarnation)
        engine = cuttable_fake_engine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                                      capability_profile=PROFILE)
        resume = handle.resume_from
        self.trainer_buffer = self.buffer
        self.buffer.reset_to(0 if resume is None else resume.watermark)
        engine.rollout = BufferFedFakeRollout(engine, self.buffer)
        self.rollouts.append(engine.rollout)
        self.engine = engine
        sync = LocalOnlySync(1000)
        sync.round_cuts = self.cuts
        return _driver(engine, sync, self.tmp_path, name=f"events-{handle.incarnation}.jsonl")

    def release(self, handle) -> None:
        self.released.append(handle.incarnation)

    def published(self):
        return None if self.engine is None or self.engine.published is None else self.engine.published[0]


class _Head:
    """One head process: journal, scheduler, lifecycle, loop."""

    def __init__(self, tmp_path, cloud: _FakeCloud, clock: _Clock, *, release_orphan=None,
                 data_ended=lambda: False) -> None:
        self.journal = Journal(tmp_path / "head-journal")
        records = self.journal.records
        cloud.new_buffer()
        self.scheduler = S1Scheduler(S1Config(a_hard=0, start_seconds=T_S),
                                     record=self.journal.append, clock=clock)
        self.lifecycle = DriverLifecycle(cloud.build, membership=cloud.membership, cuts=cloud.cuts,
                                         release=cloud.release,
                                         first_incarnation=next_incarnation(records))
        self.loop = TrainerLoop(scheduler=self.scheduler, lifecycle=self.lifecycle,
                                buffer=cloud.buffer, cuts=cloud.cuts, record=self.journal.append,
                                data_ended=data_ended, release_orphan=release_orphan,
                                ready_timeout_s=30.0)
        self.restored = self.loop.restore(records)
        # after the orphan is released: the backlog continues at the newest cut
        cut = cloud.cuts.latest()
        cloud.buffer.reset_to(0 if cut is None else cut.watermark)

    def crash(self) -> None:
        self.journal.close()  # the process dies: the journal lock goes with it


def _ticks(loop, n):
    return [type(loop.tick()) for _ in range(n)]


def _idle_stop(loop, clock):
    """Backlog empty: the first tick starts the idle time, T_s later the loop stops."""
    assert _ticks(loop, 1) == [Hold] and loop.state is LifecycleState.TRAINING
    clock.now += T_S
    assert _ticks(loop, 1) == [Stop] and loop.state is LifecycleState.STOPPED


def test_start_stop_cycles_across_a_head_restart(tmp_path):
    clock = _Clock()
    cloud = _FakeCloud(tmp_path)
    data_end = {"ended": False}
    head = _Head(tmp_path, cloud, clock, data_ended=lambda: data_end["ended"])
    assert head.restored is False and head.loop.state is LifecycleState.COLD
    sampler = FakeSampler(cloud.buffer, cloud.published)

    # cycle 0: Q_on = A_hard + 1 = 1 batch; 2 batches -> start, train 2 rounds, idle stop
    assert _ticks(head.loop, 1) == [Hold]  # empty backlog: cold trainer holds
    sampler.produce(2 * BATCH)
    assert _ticks(head.loop, 3) == [Start, Hold, Hold]
    _idle_stop(head.loop, clock)
    cut0 = cloud.cuts.latest()
    assert (cut0.trigger, cut0.policy_version, cut0.watermark) == (recovery.SCHEDULED, 2, 2 * BATCH)

    # cycle 1: one more batch; the head dies while the trainer runs and is idle
    sampler.load()
    sampler.produce(BATCH)
    assert _ticks(head.loop, 3) == [Start, Hold, Hold]
    assert head.scheduler.idle_since == clock.now
    old_lifecycle, old_handle = head.lifecycle, head.loop.handle
    head.crash()

    # the cloud reclaims the orphaned trainer (spot): PREEMPT cut, LEAVE PREEMPTED
    orphans = []

    def release_orphan(incarnation):
        orphans.append(incarnation)
        old_lifecycle.stop(old_handle, LeaveReason.PREEMPTED)

    head = _Head(tmp_path, cloud, clock, release_orphan=release_orphan,
                 data_ended=lambda: data_end["ended"])
    assert head.restored and orphans == [1]
    assert head.loop.state is LifecycleState.STOPPED and next_incarnation(head.journal.records) == 2
    cut1 = cloud.cuts.latest()
    assert (cut1.trigger, cut1.policy_version, cut1.watermark) == (recovery.PREEMPT, 3, 3 * BATCH)
    sampler = FakeSampler(cloud.buffer, cloud.published, name="s1")

    # cycle 2 after the restart: continues at v3, idle stop -> SCHEDULED
    assert _ticks(head.loop, 1) == [Hold]
    sampler.load()
    sampler.produce(BATCH)
    assert _ticks(head.loop, 2) == [Start, Hold]
    assert head.loop.handle.start_version == 3 and head.loop.current_version() == 4
    _idle_stop(head.loop, clock)
    assert cloud.cuts.latest().trigger == recovery.SCHEDULED

    # cycle 3: data end -> FINISHED, the loop is done
    sampler.load()
    sampler.produce(BATCH)
    assert _ticks(head.loop, 2) == [Start, Hold]
    data_end["ended"] = True
    assert _ticks(head.loop, 1) == [Stop] and head.loop.finished
    assert _ticks(head.loop, 1) == [Hold]
    head.crash()

    assert cloud.membership.tape == [
        ("JOIN", 0), ("LEAVE", 0, LeaveReason.STOPPED),
        ("JOIN", 1), ("LEAVE", 1, LeaveReason.PREEMPTED),
        ("JOIN", 2), ("LEAVE", 2, LeaveReason.STOPPED),
        ("JOIN", 3), ("LEAVE", 3, LeaveReason.FINISHED),
    ]
    assert cloud.acquired == [0, 1, 2, 3] and cloud.released == [0, 1, 2, 3]
    # versions continue across starts; no sample is trained twice
    trained = [sid for r in cloud.rollouts for batch in r.trained.values() for sid, _ in batch]
    assert len(trained) == 5 * BATCH and len(set(trained)) == len(trained)
    assert sorted(v for r in cloud.rollouts for v in r.trained) == [0, 1, 2, 3, 4]

    # the journal holds the loop and the scheduler records of both heads
    head = Journal(tmp_path / "head-journal")
    loop_events = [(r["event"], r["incarnation"]) for r in head.records if r["kind"] == LOOP_KIND]
    assert loop_events == [
        ("started", 0), ("ready", 0), ("stopped", 0),
        ("started", 1), ("ready", 1), ("orphaned", 1),
        ("started", 2), ("ready", 2), ("stopped", 2),
        ("started", 3), ("ready", 3), ("stopped", 3),
    ]
    stops = [r for r in head.records if r["kind"] == LOOP_KIND and r["event"] == "stopped"]
    assert [(r["reason"], r["trigger"]) for r in stops] == [
        ("STOPPED", recovery.SCHEDULED), ("STOPPED", recovery.SCHEDULED), ("FINISHED", recovery.MANUAL)]
    assert any(r["kind"] == SCHEDULER_STATE_KIND for r in head.records)
    head.close()


def test_restart_continues_the_idle_time(tmp_path):
    """D8: the idle time spent before the head restart counts after it."""
    clock = _Clock()
    cloud = _FakeCloud(tmp_path)
    head = _Head(tmp_path, cloud, clock)
    FakeSampler(cloud.buffer, cloud.published).produce(BATCH)
    assert _ticks(head.loop, 3) == [Start, Hold, Hold]  # train, then idle starts
    idle_since = head.scheduler.idle_since
    handle, lifecycle = head.loop.handle, head.lifecycle
    head.crash()

    clock.now += T_S - 1
    head = _Head(tmp_path, cloud, clock, release_orphan=lambda inc: None)
    assert head.scheduler.idle_since == idle_since
    # the orphan is released by the cloud layer; this fake keeps it, end it here
    lifecycle.stop(handle, LeaveReason.STOPPED)
    head.crash()


def test_thread_ticks_until_finished(tmp_path):
    clock = _Clock()
    cloud = _FakeCloud(tmp_path)
    head = _Head(tmp_path, cloud, clock, data_ended=lambda: True)
    FakeSampler(cloud.buffer, cloud.published).produce(BATCH)
    runner = TrainerLoopThread(head.loop, interval_s=0.01).start()
    deadline = time.monotonic() + 30
    while not head.loop.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()
    assert runner.error is None and head.loop.finished
    assert cloud.membership.tape == [("JOIN", 0), ("LEAVE", 0, LeaveReason.FINISHED)]
    head.crash()


def test_new_journal_starts_at_incarnation_zero(tmp_path):
    assert next_incarnation([]) == 0
    assert next_incarnation([{"kind": LOOP_KIND, "incarnation": 4}, {"kind": "other", "incarnation": 9}]) == 5


# ---------------------------------------------------------------- launcher switch
def _args(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    parser = argparse.ArgumentParser()
    launch_loop.add_arguments(parser)
    args = parser.parse_args([] if mode is None else ["--rl-trainer-scheduler", mode,
                                                      "--rl-trainer-scheduler-interval", "0.01",
                                                      "--rl-trainer-start-seconds", "0"])
    args.cluster_prefix, args.rl_max_policy_age = "run-a", 0
    return args


def test_switch_off_by_default_runs_unchanged(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, None)
    assert args.rl_trainer_scheduler == "off" and not launch_loop.enabled(args)
    launch_loop.check(args)  # no binding needed
    assert launch_loop.run_with_trainer_loop(args, lambda: "codes") == "codes"
    assert not (tmp_path / "runs").exists()  # no journal written


def test_switch_on_without_a_binding_refuses(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, "s1")
    monkeypatch.setattr(launch_loop, "binding_factory", None)
    with pytest.raises(SystemExit, match="no trainer island binding"):
        launch_loop.check(args)


def test_switch_on_runs_the_loop_beside_the_controller(tmp_path, monkeypatch):
    """The fleet controller's run() returns after the samplers finished; the loop
    started and stopped the trainer meanwhile, and stops it at the end."""
    args = _args(tmp_path, monkeypatch, "s1")
    cloud = _FakeCloud(tmp_path)
    cloud.new_buffer()
    FakeSampler(cloud.buffer, cloud.published).produce(2 * BATCH)

    def binding(_args, _island):
        return launch_loop.CloudTrainerBinding(
            lifecycle=lambda first: DriverLifecycle(cloud.build, membership=cloud.membership,
                                                    cuts=cloud.cuts, release=cloud.release,
                                                    first_incarnation=first),
            buffer=cloud.buffer, cuts=cloud.cuts)

    def controller_run():
        deadline = time.monotonic() + 30
        while (cloud.membership.tape[-1:] != [("LEAVE", 0, LeaveReason.STOPPED)]
               and time.monotonic() < deadline):
            time.sleep(0.01)
        return "codes"

    assert launch_loop.run_with_trainer_loop(args, controller_run, factory=binding) == "codes"
    assert cloud.membership.tape == [("JOIN", 0), ("LEAVE", 0, LeaveReason.STOPPED)]
    assert cloud.cuts.latest().policy_version == 2
    journal = Journal(launch_loop.journal_dir(args))
    assert [r["event"] for r in journal.records if r["kind"] == LOOP_KIND] == ["started", "ready", "stopped"]
    journal.close()
