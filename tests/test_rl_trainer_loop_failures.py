"""T-FPUB: the trainer loop stops after too many failed incarnations in a row.

GPU run s24-l54-pair-20261014i: every trainer incarnation failed at its first
publish and the scheduler started a new one about 15 s later, with no limit.
Now ``max_consecutive_failures`` (default 3) failed incarnations with no trained
round between them stop the loop: it releases the trainer, journals
``gave_up`` and raises; a trained round resets the count.
"""

from __future__ import annotations

import argparse
import threading
from types import SimpleNamespace

import pytest

from yeto import runs
from yeto.launch import trainer_loop as launch_loop
from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.trainer_lifecycle import (
    BacklogStats,
    Hold,
    LifecycleError,
    LifecycleState,
    RunReceipt,
    Start,
)
from yeto.rl.engine.trainer_loop import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    GAVE_UP,
    LOOP_KIND,
    STARTED,
    TrainerLoop,
    TrainerLoopGaveUp,
    TrainerLoopThread,
)


class _Scheduler:
    """Start whenever no trainer runs; else hold (the loop trains on a hold)."""

    def __init__(self, *_a, **_k) -> None:
        pass

    def decide(self, backlog, state):
        return Hold() if state is LifecycleState.TRAINING else Start()


class _Buffer:
    def backlog(self, _version):
        return BacklogStats(batches=5.0, samples=20)


class _Cuts:
    def latest(self):
        return None


class _Lifecycle:
    """``outcomes``: per incarnation, "start_error" (start raises), "train_error"
    (the first run fails, as the L5.4 publish), or "ok" (each run trains a round)."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.started: list[int] = []
        self.stopped: list[tuple[int, LeaveReason]] = []
        self._next = 0

    def plan_start(self, spec):
        plan = SimpleNamespace(incarnation=self._next)
        self._next += 1
        return plan

    def start(self, plan, resume_from=None):
        self.started.append(plan.incarnation)
        outcome = self.outcomes[plan.incarnation] if plan.incarnation < len(self.outcomes) else "train_error"
        if outcome == "start_error":
            raise RuntimeError("island did not come up")
        return SimpleNamespace(incarnation=plan.incarnation, state=LifecycleState.TRAINING,
                               outcome=outcome, start_version=0, trained_version=None,
                               error=None, leave_reason=None, cut=None, released=None)

    def ready(self, handle, deadline):
        return True

    def run(self, handle, buffer, condition):
        if handle.outcome == "ok":
            handle.outcome = "train_error"  # one good round, then it breaks
            handle.trained_version = (handle.trained_version or 0) + 1
            return RunReceipt(rounds=1, first_version=0, trained_version=handle.trained_version,
                              data_ended=False)
        handle.state = LifecycleState.STOPPED
        handle.error = RuntimeError("Megatron produced an empty or duplicate LoRA mapping")
        handle.leave_reason = LeaveReason.INFRA_FAILED
        raise LifecycleError("trainer failed")

    def stop(self, handle, reason):
        self.stopped.append((handle.incarnation, reason))
        handle.state = LifecycleState.STOPPED


def _loop(outcomes, **kwargs):
    records: list[dict] = []
    lifecycle = _Lifecycle(outcomes)
    loop = TrainerLoop(scheduler=_Scheduler(), lifecycle=lifecycle, buffer=_Buffer(), cuts=_Cuts(),
                       record=lambda kind, **f: records.append({"kind": kind, **f}), **kwargs)
    return loop, lifecycle, records


def _tick_until_gave_up(loop, limit=50):
    for _ in range(limit):
        try:
            loop.tick()
        except TrainerLoopGaveUp as error:
            return error
    raise AssertionError("the loop never gave up")


def test_three_failed_incarnations_stop_the_loop():
    loop, lifecycle, records = _loop([])
    assert loop.max_consecutive_failures == DEFAULT_MAX_CONSECUTIVE_FAILURES == 3
    error = _tick_until_gave_up(loop)
    assert "3 trainer incarnations failed in a row" in str(error)
    assert lifecycle.started == [0, 1, 2]  # no fourth Start
    events = [r["event"] for r in records if r["kind"] == LOOP_KIND]
    assert events.count(STARTED) == 3 and events[-1] == GAVE_UP
    assert records[-1]["failures"] == 3 and records[-1]["limit"] == 3
    assert loop.finished and isinstance(loop.tick(), Hold)  # nothing more happens
    assert lifecycle.started == [0, 1, 2]


def test_failed_starts_count_and_the_limit_is_configurable():
    loop, lifecycle, records = _loop(["start_error"] * 5, max_consecutive_failures=2)
    _tick_until_gave_up(loop)
    assert lifecycle.started == [0, 1]
    with pytest.raises(ValueError, match="max_consecutive_failures"):
        _loop([], max_consecutive_failures=0)


def test_a_trained_round_resets_the_count():
    # 0, 1 fail; 2 trains a round (count back to 0) then fails; 3, 4 fail -> give up
    loop, lifecycle, _records = _loop(["train_error", "start_error", "ok"])
    _tick_until_gave_up(loop)
    assert lifecycle.started == [0, 1, 2, 3, 4]


def test_thread_reports_the_give_up_to_the_launcher():
    loop, _lifecycle, _records = _loop([], max_consecutive_failures=1)
    told: list[BaseException] = []
    runner = TrainerLoopThread(loop, 0.001, on_error=told.append).start()
    assert runner.join(10)
    assert isinstance(runner.error, TrainerLoopGaveUp) and told == [runner.error]


def _args(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    parser = argparse.ArgumentParser()
    launch_loop.add_arguments(parser)
    args = parser.parse_args(["--rl-trainer-scheduler", "s1", "--rl-trainer-scheduler-interval",
                              "0.001", *extra])
    args.cluster_prefix, args.rl_max_policy_age = "run-a", 0
    return args


def _binding(lifecycle):
    def factory(_args, _island):
        return launch_loop.CloudTrainerBinding(lifecycle=lambda first: lifecycle,
                                               buffer=_Buffer(), cuts=_Cuts())
    return factory


def test_launcher_aborts_the_fleet_and_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_loop, "S1Scheduler", _Scheduler)
    args = _args(tmp_path, monkeypatch, "--rl-trainer-max-failures", "2")
    lifecycle = _Lifecycle([])
    aborted = threading.Event()
    reasons: list[BaseException] = []

    def abort(error):
        reasons.append(error)
        aborted.set()

    def controller_run():  # the fleet controller: ends with the abort error
        assert aborted.wait(10)
        raise reasons[0]

    with pytest.raises(TrainerLoopGaveUp):
        launch_loop.run_with_trainer_loop(args, controller_run, factory=_binding(lifecycle),
                                          abort=abort)
    assert lifecycle.started == [0, 1]
    journal = Journal(launch_loop.journal_dir(args))
    assert [r["event"] for r in journal.records if r["kind"] == LOOP_KIND][-1] == GAVE_UP
    journal.close()


def test_launcher_without_abort_raises_after_the_fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_loop, "S1Scheduler", _Scheduler)
    args = _args(tmp_path, monkeypatch, "--rl-trainer-max-failures", "1")
    ended = threading.Event()

    class _Thread(TrainerLoopThread):
        def _run(self) -> None:
            super()._run()
            ended.set()

    monkeypatch.setattr(launch_loop, "TrainerLoopThread", _Thread)

    def controller_run():
        assert ended.wait(10)
        return "codes"

    with pytest.raises(TrainerLoopGaveUp):
        launch_loop.run_with_trainer_loop(args, controller_run, factory=_binding(_Lifecycle([])))


def test_bad_limit_is_refused_before_launch(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, "--rl-trainer-max-failures", "0")
    monkeypatch.setattr(launch_loop, "binding_factory", lambda a, i: None)
    with pytest.raises(SystemExit, match="max-failures"):
        launch_loop.check(args)


def test_fleet_controller_run_ends_with_the_abort_error():
    from yeto.launch.fleet import FleetController

    controller = FleetController.__new__(FleetController)
    controller.learners = {}
    controller._abort_error = None
    controller._fleet = lambda *a, **k: None
    error = TrainerLoopGaveUp("gave up")
    controller.abort(error)
    with pytest.raises(TrainerLoopGaveUp):
        controller.run()
