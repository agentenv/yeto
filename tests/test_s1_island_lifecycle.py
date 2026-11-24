"""The trainer scheduler s1 on the unified island lifecycle (task 2.1 of unified-trainer-islands):
its journal records move the held island's state record, its failures go through the
decision table (a configuration error is never restarted), an s1 journal written before
the lifecycle was on is imported once, read only. Off by default: s1 runs one trainer
island, which takes no new path unless ``--island-lifecycle on``. No cloud."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import yeto.launcher as launcher
from tests.test_rl_trainer_loop_failures import (
    _args,
    _binding,
    _Buffer,
    _Cuts,
    _Lifecycle,
    _Scheduler,
    _tick_until_gave_up,
)
from yeto.config_errors import ConfigError
from yeto.launch import island_lifecycle as lc
from yeto.launch import trainer_loop as launch_loop
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.trainer_lifecycle import LifecycleError, LifecycleState
from yeto.rl.engine.trainer_loop import LOOP_KIND, TrainerLoop
from yeto.rl.pool_client import LeaveReason


def _states(tmp_path):
    return lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))


# -- the journal records and the states they mean ----------------------------------------------------

@pytest.mark.parametrize("record, state, cause", [
    ({"event": "started", "incarnation": 0}, "ordering", None),
    ({"event": "ready", "incarnation": 0}, "running", None),
    ({"event": "stopped", "incarnation": 0, "reason": "STOPPED"}, "done", "stopped"),  # an idle stop
    ({"event": "stopped", "incarnation": 0, "reason": "FINISHED"}, "done", "finished"),
    ({"event": "stopped", "incarnation": 0, "reason": "INFRA_FAILED"}, "recovering", "infra"),
    ({"event": "stopped", "incarnation": 0, "reason": None, "error": "RuntimeError('NCCL')"}, "recovering", "infra"),
    ({"event": "stopped", "incarnation": 0, "reason": "INFRA_FAILED", "error": "x", "cause": "config_error"},
     "failed", "config_error"),
    ({"event": "stopped", "incarnation": 0, "reason": None, "error": "ConfigError('bad fragments')"},
     "failed", "config_error"),  # a journal of a loop without the table
    ({"event": "stopped", "incarnation": 0, "reason": "PREEMPTED"}, "recovering", "preempted"),
    ({"event": "orphaned", "incarnation": 2}, "recovering", "lost"),
    ({"event": "gave_up", "incarnation": 2, "failures": 3, "limit": 3}, "abandoned", "infra"),
    ({"event": "finished", "incarnation": None}, "done", "finished"),
])
def test_each_s1_record_maps_to_one_state(record, state, cause):
    moved = lc.loop_event({"kind": LOOP_KIND, **record})
    assert moved[0] == state
    assert (None if moved[1] is None else moved[1].cause) == cause


def test_records_that_move_no_state():
    assert lc.loop_event({"kind": LOOP_KIND, "event": "first_policy", "incarnation": 0}) is None
    assert lc.loop_event({"kind": "trainer_scheduler_state", "idle_s": 3}) is None


def test_the_table_classifies_a_failed_incarnation():
    assert lc.loop_classifier(None, ConfigError("layout")) == ("config_error", False)
    assert lc.loop_classifier(LeaveReason.INFRA_FAILED, None) == ("infra", True)
    assert lc.loop_classifier(None, RuntimeError("island did not come up")) == ("infra", True)
    assert lc.loop_classifier(LeaveReason.TRAINING_FAILED, None) == ("strict", False)
    assert lc.loop_classifier(LeaveReason.STOPPED, None)[1]  # a failure read as a stop: still relaunched


# -- the loop with the recorder ----------------------------------------------------------------------

def _recorded_loop(tmp_path, outcomes, *, lifecycle=None, **kwargs):
    states = _states(tmp_path)
    journal: list[dict] = []
    recorder = lc.LoopRecorder(lambda kind, **f: journal.append({"kind": kind, **f}) or journal[-1], states, 0,
                               source="modal", name="run-a-l0-modal")
    lifecycle = lifecycle or _Lifecycle(outcomes)
    loop = TrainerLoop(scheduler=_Scheduler(), lifecycle=lifecycle, buffer=_Buffer(), cuts=_Cuts(),
                       record=recorder, classify_stop=lc.loop_classifier, **kwargs)
    return loop, lifecycle, states, journal


def test_the_recorder_keeps_the_journal_and_the_island_record(tmp_path):
    loop, lifecycle, states, journal = _recorded_loop(tmp_path, ["ok", "train_error", "start_error"])
    _tick_until_gave_up(loop)
    record = states.read(0)
    assert record["state"] == "abandoned" and record["mode"] == "s1" and record["source"] == "modal"
    assert record["incarnation"] == lifecycle.started[-1] and record["relaunches"] == len(lifecycle.started) - 1
    assert record["handle"] == {"incarnation": lifecycle.started[-1], "cluster": "run-a-l0-modal"}
    moves = [h["to"] for h in record["history"]]
    assert moves[:3] == ["ordering", "running", "recovering"] and moves[-1] == "abandoned"
    assert [r["event"] for r in journal if r["kind"] == LOOP_KIND][-1] == "gave_up"  # the journal is unchanged


class _ConfigErrorLifecycle(_Lifecycle):
    """Every incarnation refuses its configuration in its first round."""

    def run(self, handle, buffer, condition):
        handle.state = LifecycleState.STOPPED
        handle.error = ConfigError("FullParameterLayoutMismatch: expected_fragments=4, actual_fragments=10")
        handle.leave_reason = LeaveReason.CONFIG_MISMATCH
        raise LifecycleError("trainer refused its configuration")


def test_a_configuration_error_is_never_restarted(tmp_path):
    loop, lifecycle, states, journal = _recorded_loop(tmp_path, [], lifecycle=_ConfigErrorLifecycle([]))
    _tick_until_gave_up(loop)
    assert lifecycle.started == [0]  # the table: config_error -> fail, no second incarnation
    stopped = [r for r in journal if r.get("event") == "stopped"]
    assert stopped[0]["cause"] == "config_error"
    record = states.read(0)
    assert record["state"] == "failed" and record["decision"]["cause"] == "config_error"
    assert "actual_fragments=10" in record["exit"]


def test_without_the_table_the_loop_is_unchanged():
    records: list[dict] = []
    lifecycle = _ConfigErrorLifecycle([])
    loop = TrainerLoop(scheduler=_Scheduler(), lifecycle=lifecycle, buffer=_Buffer(), cuts=_Cuts(),
                       record=lambda kind, **f: records.append({"kind": kind, **f}))
    _tick_until_gave_up(loop)
    assert lifecycle.started == [0, 1, 2]  # the failure count decides, as before
    assert all("cause" not in r for r in records)


# -- the one-time import of an older journal ----------------------------------------------------------

def test_an_older_journal_is_imported_once_read_only(tmp_path):
    jdir = tmp_path / "trainer-loop"
    with Journal(jdir) as journal:
        journal.append(LOOP_KIND, event="started", incarnation=0, resume_cut=None, resume_version=None)
        journal.append(LOOP_KIND, event="ready", incarnation=0, version=0)
        journal.append(LOOP_KIND, event="stopped", incarnation=0, reason="INFRA_FAILED", error=None)
        journal.append(LOOP_KIND, event="started", incarnation=1, resume_cut="c7", resume_version=7)
        journal.append(LOOP_KIND, event="ready", incarnation=1, version=7)
        before = {p.name: p.read_bytes() for p in jdir.iterdir() if p.is_file()}
        states = _states(tmp_path)
        recorder = lc.LoopRecorder(journal.append, states, 0, source="sky", name="run-a-l0-aws")
        record = recorder.import_journal(journal.records)
        assert record["state"] == "running" and record["imported_from_journal"] is True
        assert record["incarnation"] == 1 and record["relaunches"] == 1
        assert record["resume"] == {"cut": "c7", "version": 7}
        assert recorder.import_journal(journal.records) is None  # once: the store has a record now
        after = {p.name: p.read_bytes() for p in jdir.iterdir() if p.is_file()}
    assert after == before  # read only


def test_an_empty_journal_imports_nothing(tmp_path):
    states = _states(tmp_path)
    recorder = lc.LoopRecorder(lambda kind, **f: None, states, 0, source="modal")
    assert recorder.import_journal([]) is None and states.read(0) is None


# -- the switch: off by default, on with --island-lifecycle on ----------------------------------------

def _run_s1(tmp_path, monkeypatch, *, lifecycle_mode):
    monkeypatch.setattr(launch_loop, "S1Scheduler", _Scheduler)
    args = _args(tmp_path, monkeypatch, "--rl-trainer-max-failures", "1")
    args.island_lifecycle, args.gpu, args.dry_run = lifecycle_mode, "modal:1xl40s", False
    opened: list[object] = []

    def store(_args):
        opened.append(_args)
        return _states(tmp_path)

    monkeypatch.setattr(launcher, "_island_state_store", store)
    lifecycle = _Lifecycle([])
    aborted = threading.Event()

    def controller_run():
        assert aborted.wait(10)
        raise RuntimeError("fleet aborted")

    with pytest.raises(Exception):  # noqa: B017 - the gave-up error or the fleet's
        launch_loop.run_with_trainer_loop(args, controller_run, factory=_binding(lifecycle),
                                          island=SimpleNamespace(name="run-a-l0-modal", cloud="modal"),
                                          abort=lambda error: aborted.set())
    return opened, _states(tmp_path).read(0)


def test_s1_keeps_no_record_by_default(tmp_path, monkeypatch):
    opened, record = _run_s1(tmp_path, monkeypatch, lifecycle_mode="auto")
    assert opened == [] and record is None  # a single trainer island: no new path


def test_s1_opted_in_records_its_island(tmp_path, monkeypatch):
    opened, record = _run_s1(tmp_path, monkeypatch, lifecycle_mode="on")
    assert len(opened) == 1
    assert record["state"] == "abandoned" and record["mode"] == "s1" and record["source"] == "modal"
    assert record["name"] == "run-a-l0-modal"
