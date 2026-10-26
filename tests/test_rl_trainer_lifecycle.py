"""rl-trainer-start-stop L3 (CPU, fakes): lifecycle states, SCHEDULED cuts,
JOIN/LEAVE per incarnation, and repeated starts over IslandDriver."""

from __future__ import annotations

import time

import pytest
import torch
from test_rl_engine_driver import NAME, PROFILE, _driver

from yeto.rl.pool_client import LeaveReason
from yeto.rl.engine import recovery
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import RoundFailedError
from yeto.rl.engine.fake_lifecycle import FakeMembership, FakeRoundCuts, cuttable_fake_engine
from yeto.rl.engine.trainer_lifecycle import (
    AlwaysOnScheduler,
    BacklogStats,
    CutRef,
    DriverLifecycle,
    Hold,
    LifecycleError,
    LifecycleState,
    Start,
    Stop,
    StopCondition,
    TrainerLifecycle,
    allowed_transition,
)

FULL = [s.value for s in LifecycleState]


class _Island:
    """Builds a fresh fake trainer + LocalOnlySync per incarnation (acquire)."""

    def __init__(self, tmp_path, *, num_rollout=100, **engine_kwargs):
        self.tmp_path = tmp_path
        self.num_rollout = num_rollout
        self.engine_kwargs = engine_kwargs
        self.cuts = FakeRoundCuts(tmp_path / "cuts")
        self.membership = FakeMembership()
        self.engines = []
        self.released = []

    def build(self, handle):
        engine = cuttable_fake_engine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                                      capability_profile=PROFILE, **self.engine_kwargs)
        self.engines.append(engine)
        sync = LocalOnlySync(self.num_rollout)
        sync.round_cuts = self.cuts
        return _driver(engine, sync, self.tmp_path, name=f"events-{handle.incarnation}.jsonl")

    def lifecycle(self):
        return DriverLifecycle(self.build, membership=self.membership, cuts=self.cuts,
                               release=lambda h: self.released.append(h.incarnation))


def _deadline(s=30.0):
    return time.monotonic() + s


def _publishes(engine):
    return [c[1] for c in engine.calls if c[0] == "publish"]


def test_transitions_move_forward_or_to_stopped():
    s = LifecycleState
    assert allowed_transition(s.COLD, s.ACQUIRING) and allowed_transition(s.TRAINING, s.DRAINING)
    assert allowed_transition(s.RESTORING, s.STOPPED)
    assert not allowed_transition(s.COLD, s.TRAINING)
    assert not allowed_transition(s.TRAINING, s.WARMING)
    assert not allowed_transition(s.STOPPED, s.STOPPED)


def test_three_start_stop_cycles_continue_the_policy(tmp_path):
    """L3.1 + L3.4: start -> train -> stop three times on fresh fake trainers; each
    start restores the SCHEDULED cut of the previous stop and goes on at v+1."""
    island = _Island(tmp_path)
    lc = island.lifecycle()
    assert isinstance(lc, TrainerLifecycle)
    cuts = []
    for cycle in range(3):
        resume = island.cuts.latest()
        h = lc.start(lc.plan_start({"cycle": cycle}), resume_from=resume)
        assert lc.ready(h, _deadline())
        receipt = lc.run(h, None, StopCondition(rounds=2))
        assert h.state is LifecycleState.TRAINING
        assert (receipt.rounds, receipt.first_version, receipt.trained_version) == \
            (2, 2 * cycle, 2 * cycle + 2)
        cut = lc.drain_and_cut(h, LeaveReason.STOPPED)
        lc.stop(h, LeaveReason.STOPPED)
        assert h.states == FULL and h.error is None
        assert cut == CutRef(cut_id=cut.cut_id, trigger=recovery.SCHEDULED, policy_version=2 * cycle + 2)
        cuts.append(cut)
        engine = island.engines[cycle]
        # the restored version is published again first, then v+1, v+2
        assert _publishes(engine) == [2 * cycle, 2 * cycle + 1, 2 * cycle + 2]
        restored = [c for c in engine.calls if c[0] == "restore_cut"]
        assert restored == ([] if resume is None else [("restore_cut", resume.cut_id)])
        assert torch.equal(engine.tensors[NAME], torch.full((1, 2), float(2 * cycle + 2)))
    assert len({id(e) for e in island.engines}) == 3  # a fresh trainer per start
    assert island.membership.tape == [
        ("JOIN", 0), ("LEAVE", 0, LeaveReason.STOPPED),
        ("JOIN", 1), ("LEAVE", 1, LeaveReason.STOPPED),
        ("JOIN", 2), ("LEAVE", 2, LeaveReason.STOPPED),
    ]
    assert island.released == [0, 1, 2]
    assert [c.trigger for c in cuts] == [recovery.SCHEDULED] * 3


def test_run_twice_before_the_stop_trains_on(tmp_path):
    island = _Island(tmp_path)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    assert lc.run(h, None, StopCondition(1)).trained_version == 1
    assert lc.run(h, None, StopCondition(2)).trained_version == 3
    lc.stop(h, LeaveReason.STOPPED)  # drains first
    assert h.cut.policy_version == 3 and h.cut.trigger == recovery.SCHEDULED
    assert island.membership.tape[-1] == ("LEAVE", 0, LeaveReason.STOPPED)


def test_data_end_drains_on_its_own_with_finished(tmp_path):
    island = _Island(tmp_path, num_rollout=3)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    receipt = lc.run(h, None, StopCondition())
    assert receipt.data_ended and receipt.trained_version == 3
    assert h.state is LifecycleState.DRAINING
    lc.stop(h, LeaveReason.FINISHED)
    assert h.states == FULL
    assert h.cut.trigger == recovery.MANUAL and h.cut.policy_version == 3
    assert island.membership.tape == [("JOIN", 0), ("LEAVE", 0, LeaveReason.FINISHED)]


def test_reclaim_cuts_with_preempt_and_leaves_preempted(tmp_path):
    island = _Island(tmp_path)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    lc.run(h, None, StopCondition(1))
    cut = lc.drain_and_cut(h, LeaveReason.PREEMPTED)
    lc.stop(h, LeaveReason.PREEMPTED)
    assert cut.trigger == recovery.PREEMPT and cut.policy_version == 1
    assert island.membership.tape[-1] == ("LEAVE", 0, LeaveReason.PREEMPTED)
    # the next start restores the PREEMPT cut like any other
    h2 = lc.start(lc.plan_start({}), resume_from=island.cuts.latest())
    assert lc.run(h2, None, StopCondition(1)).trained_version == 2
    lc.stop(h2, LeaveReason.STOPPED)


def test_stop_before_any_round_keeps_the_previous_cut(tmp_path):
    island = _Island(tmp_path)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    lc.run(h, None, StopCondition(1))
    lc.stop(h, LeaveReason.STOPPED)
    first = island.cuts.latest()
    h2 = lc.start(lc.plan_start({}), resume_from=first)
    assert lc.ready(h2, _deadline())
    lc.stop(h2, LeaveReason.STOPPED)
    assert h2.states == FULL and island.cuts.latest() == first
    assert island.membership.tape[-2:] == [("JOIN", 1), ("LEAVE", 1, LeaveReason.STOPPED)]


def test_training_failure_stops_with_the_p5_code(tmp_path):
    island = _Island(tmp_path, failed_step_rounds={1})
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    with pytest.raises(LifecycleError) as raised:
        lc.run(h, None, StopCondition(3))
    assert isinstance(raised.value.__cause__, RoundFailedError)
    assert h.state is LifecycleState.STOPPED and h.states[-2] == "training"
    assert island.membership.tape == [("JOIN", 0), ("LEAVE", 0, LeaveReason.TRAINING_FAILED)]
    assert island.released == [0]
    lc.stop(h, LeaveReason.STOPPED)  # already stopped: no second LEAVE
    assert len(island.membership.tape) == 2


def test_start_refuses_a_restore_at_another_version(tmp_path):
    island = _Island(tmp_path)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    lc.run(h, None, StopCondition(1))
    lc.stop(h, LeaveReason.STOPPED)
    stale = CutRef(cut_id="other", trigger=recovery.SCHEDULED, policy_version=7)
    h2 = lc.start(lc.plan_start({}), resume_from=stale)
    assert not lc.ready(h2, _deadline())
    assert h2.state is LifecycleState.STOPPED and isinstance(h2.error, LifecycleError)
    assert island.membership.tape[-1] == ("LEAVE", 1, LeaveReason.INFRA_FAILED)


def test_acquire_failure_never_joins(tmp_path):
    membership = FakeMembership()

    def build(handle):
        raise RuntimeError("no capacity")

    lc = DriverLifecycle(build, membership=membership)
    with pytest.raises(RuntimeError, match="no capacity"):
        lc.start(lc.plan_start({}), resume_from=None)
    assert membership.tape == []


def test_always_on_scheduler_is_start_once():
    s = AlwaysOnScheduler()
    busy = BacklogStats(batches=2.0, samples=8)
    assert s.decide(busy, LifecycleState.COLD) == Start()
    assert s.decide(busy, LifecycleState.TRAINING, {"h100": 3.0}) == Hold()
    ended = BacklogStats(batches=0.0, samples=0, data_ended=True)
    assert s.decide(ended, LifecycleState.TRAINING) == Stop(LeaveReason.FINISHED)
