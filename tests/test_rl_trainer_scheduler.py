"""Scheduler S1 (rl-trainer-start-stop L5.1-L5.3): every scheduler scenario of the spec."""

from __future__ import annotations

import pytest

from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.trainer_lifecycle import (
    AlwaysOnScheduler,
    BacklogStats,
    Hold,
    LifecycleState,
    Start,
    Stop,
    TrainerScheduler,
)
from yeto.rl.engine.trainer_scheduler import (
    SCHEDULER_STATE_KIND,
    S1Config,
    S1Scheduler,
    last_start_seconds,
)

COLD, TRAINING = LifecycleState.COLD, LifecycleState.TRAINING
EMPTY = BacklogStats(batches=0.0, samples=0)


def _q(batches: float) -> BacklogStats:
    return BacklogStats(batches=batches, samples=int(batches * 4))


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_s1_implements_the_port():
    assert isinstance(S1Scheduler(S1Config(a_hard=2, start_seconds=500.0)), TrainerScheduler)


def test_cap_by_version_limit():
    # T_s=500, T_b=44, alpha=0.2 -> Q_s1 = ceil(500*0.8/(0.2*44)) = 46 >= 45; cap A_hard+1 = 3
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=500.0, alpha=0.2, batch_seconds=44.0))
    assert s.q_on(500.0) == 3
    assert s.decide(_q(3), COLD) == Start()


def test_alpha_below_cap_starts_at_q_s1():
    # Q_s1 = ceil(10*0.5/(0.5*10)) = 1 < A_hard + 1
    s = S1Scheduler(S1Config(a_hard=4, start_seconds=10.0, alpha=0.5, batch_seconds=10.0))
    assert s.decide(_q(0.5), COLD) == Hold()
    assert s.decide(_q(1), COLD) == Start()


def test_default_threshold_without_alpha():
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=500.0))
    assert s.decide(_q(2), COLD) == Hold()
    assert s.decide(_q(3), COLD) == Start()


def test_alpha_has_no_default_and_needs_batch_seconds():
    assert S1Config(a_hard=2, start_seconds=1.0).alpha is None
    with pytest.raises(ValueError, match="batch_seconds"):
        S1Config(a_hard=2, start_seconds=1.0, alpha=0.2)
    with pytest.raises(ValueError, match="alpha"):
        S1Config(a_hard=2, start_seconds=1.0, alpha=0.0, batch_seconds=1.0)


def test_ski_rental_stop():
    clock = _Clock()
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=300.0), clock=clock)
    assert s.decide(EMPTY, TRAINING) == Hold()  # idle starts here
    clock.t += 299.0
    assert s.decide(EMPTY, TRAINING) == Hold()
    clock.t += 1.0
    assert s.decide(EMPTY, TRAINING) == Stop(LeaveReason.STOPPED)


def test_new_samples_reset_the_idle_time():
    clock = _Clock()
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=300.0), clock=clock)
    s.decide(EMPTY, TRAINING)
    clock.t += 200.0
    assert s.decide(_q(1), TRAINING) == Hold()
    clock.t += 200.0
    assert s.decide(EMPTY, TRAINING) == Hold()  # idle again from here, not from t=0
    clock.t += 299.0
    assert s.decide(EMPTY, TRAINING) == Hold()


def test_data_end_finishes_and_trains_the_remainder():
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=300.0))
    ended = BacklogStats(batches=1.0, samples=4, data_ended=True)
    assert s.decide(ended, COLD) == Start()  # below Q_on, but nothing more will come
    assert s.decide(ended, TRAINING) == Hold()
    done = BacklogStats(batches=0.0, samples=0, data_ended=True)
    assert s.decide(done, TRAINING) == Stop(LeaveReason.FINISHED)
    # nothing left: a stopped (or never started) trainer also finishes the run (L5.3c)
    assert s.decide(done, COLD) == Stop(LeaveReason.FINISHED)


def test_transition_states_hold_and_prices_are_ignored():
    s = S1Scheduler(S1Config(a_hard=0, start_seconds=0.0))
    for state in (LifecycleState.ACQUIRING, LifecycleState.JOINING, LifecycleState.RESTORING,
                  LifecycleState.WARMING, LifecycleState.DRAINING):
        assert s.decide(EMPTY, state, {"h100": 99.0}) == Hold()
    assert s.decide(_q(1), COLD, {"h100": 99.0}) == Start()


def test_start_once():
    s = AlwaysOnScheduler()
    assert s.decide(EMPTY, COLD) == Start()
    assert s.decide(EMPTY, TRAINING) == Hold()  # an empty backlog does not stop it
    done = BacklogStats(batches=0.0, samples=0, data_ended=True)
    assert s.decide(done, TRAINING) == Stop(LeaveReason.FINISHED)


# ---------------------------------------------------------------- L5.2: T_s
def _summary(total_s, island=1):
    return {"event": "trainer_start_summary", "island_id": island, "total_s": total_s}


def test_t_s_from_last_start_summary():
    events = [{"event": "trainer_start_phase", "phase": "submitted"}, _summary(120.0),
              _summary(None), {"event": "trainer_start_phase", "phase": "first_step_done"},
              _summary(80.5)]
    assert last_start_seconds(events) == 80.5
    clock = _Clock()
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=500.0),
                    start_events=lambda: events, clock=clock)
    assert s.start_seconds() == 80.5
    s.decide(EMPTY, TRAINING)
    clock.t += 80.5
    assert s.decide(EMPTY, TRAINING) == Stop(LeaveReason.STOPPED)


def test_t_s_falls_back_to_configuration():
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=500.0),
                    start_events=lambda: [_summary(None)])
    assert s.start_seconds() == 500.0
    assert S1Scheduler(S1Config(a_hard=2, start_seconds=7.0)).start_seconds() == 7.0


def test_measured_t_s_changes_q_s1():
    events: list[dict] = []
    s = S1Scheduler(S1Config(a_hard=10, start_seconds=500.0, alpha=0.5, batch_seconds=10.0),
                    start_events=lambda: events)
    assert s.q_on(s.start_seconds()) == 11  # Q_s1 = 50 -> cap 11
    events.append(_summary(20.0))
    assert s.q_on(s.start_seconds()) == 2


# ---------------------------------------------------------------- L5.3: journal
def test_head_restart_while_idle_continues_the_idle_time(tmp_path):
    clock = _Clock()
    cfg = S1Config(a_hard=2, start_seconds=300.0)
    with Journal(tmp_path) as journal:
        s = S1Scheduler(cfg, record=journal.append, clock=clock)
        assert s.decide(EMPTY, TRAINING) == Hold()
        clock.t += 100.0
        s.decide(EMPTY, TRAINING)
        [rec] = [r for r in journal.records if r["kind"] == SCHEDULER_STATE_KIND]
        assert rec["state"] == {"idle_since": 1000.0}
        assert rec["inputs"]["start_seconds"] == 300.0 and rec["decision"] == {"action": "hold"}

    clock.t += 150.0  # the head is down; idle time t = 250 s at restart
    with Journal(tmp_path) as journal:
        s = S1Scheduler(cfg, record=journal.append, clock=clock)
        assert s.restore(journal.records)
        assert s.decide(EMPTY, TRAINING) == Hold()
        clock.t += 50.0  # 300 s since idle began, not 50 s
        assert s.decide(EMPTY, TRAINING) == Stop(LeaveReason.STOPPED)
        last = [r for r in journal.records if r["kind"] == SCHEDULER_STATE_KIND][-1]
        assert last["decision"] == {"action": "stop", "reason": "STOPPED"}
        assert last["state"] == {"idle_since": None}


def test_restore_without_records_starts_fresh():
    s = S1Scheduler(S1Config(a_hard=2, start_seconds=300.0))
    assert not s.restore([{"kind": "auto_state", "state": {}}])
    assert s.idle_since is None
