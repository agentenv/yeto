"""0.26/0.27 launcher side: no relaunch into a finished elastic run; an island that
left the pool (re-JOIN exhausted, exit 7) is not the fixed-roster abandon (exit 4)."""

from __future__ import annotations

import json

from yeto.launcher import FleetController, FixedRosterIslandAbandoned
from yeto.rl.event_echo import PREFIX, TapeCollector

from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread


def _ctl(ops, **kw):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(learners={"l0": ("task-l0", 1), "l1": ("task-l1", 2)},
                           syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30,
                           recover_timeout=600, thread_cls=ImmediateThread, fixed_roster=True, **kw)


def test_late_island_failure_after_syncer_finished_is_not_relaunched():
    ops = FakeOps()
    ops.status_seq[SYNCER] = [RUNNING, SUCCEEDED]
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING, RUNNING, FAILED]  # BrokenPipe after the syncer exited
    ctl = _ctl(ops)
    ctl.run()
    assert ops.relaunch_calls == []
    assert "syncer finished" in ctl.learners["l1"]["exit"]


def test_head_mode_syncer_finished_probe_stops_recovery():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    finished = {"v": True}
    ctl = FleetController(learners={"l0": ("task-l0", 1), "l1": ("task-l1", 2)}, syncer=None,
                          sky_ops=ops, poll_interval=30, recover_timeout=600,
                          thread_cls=ImmediateThread, fixed_roster=True,
                          syncer_probe=lambda: None, syncer_restart=lambda: None,
                          syncer_finished_probe=lambda: finished["v"])
    ctl.run()
    assert ops.relaunch_calls == []


def test_left_pool_island_relaunched_not_abandoned_with_exit_4():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, RUNNING, RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    ops.relaunch_results["l1"] = [None] * 200  # the relaunch never comes back
    ctl = _ctl(ops, elastic=True, left_pool_probe=lambda name: name == "l1")
    ctl.run()  # no FixedRosterIslandAbandoned
    assert ops.relaunch_calls and ctl.learners["l1"]["state"] == "abandoned"


def test_left_pool_with_no_island_relaunch_ends_without_relaunch():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    ctl = _ctl(ops, elastic=True, no_island_relaunch=True, left_pool_probe=lambda name: name == "l1")
    ctl.run()
    assert ops.relaunch_calls == [] and ctl.learners["l1"]["exit"].startswith("LEFT_POOL")


def test_legacy_failure_still_abandons_with_exit_4_path():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.status_seq["l1"] = [FAILED]
    ops.up["l1"] = False
    ctl = _ctl(ops, left_pool_probe=lambda name: True)  # elastic=False: probe ignored
    ctl.recover_timeout = 0
    try:
        ctl.run()
    except FixedRosterIslandAbandoned:
        return
    raise AssertionError("legacy fixed-roster failure must still abandon")


def test_tape_collector_sees_left_pool(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    c.feed(PREFIX + json.dumps({"event": "elastic_left_pool", "island_id": 1, "time_unix": 1.0}) + "\n")
    assert c.left_pool and not c.finalized


def test_final_grace_flag_and_status_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_rl_launcher import _args
    from yeto import launcher

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic", "--rl-soft-deadline-s", "240")), 2)
    assert " --final-grace-s 240" in cmd
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic", "--rl-final-grace-s", "60")), 2)
    assert " --final-grace-s 60" in cmd
    assert "--final-grace-s" not in launcher.syncer_command(_args(), 2)

    # head mode: status.json state decides (process still in its grace window)
    ls = launcher.LocalSyncer.__new__(launcher.LocalSyncer)
    ls.proc = SimpleNamespace(poll=lambda: None)
    ls.event_tape = str(tmp_path / "yeto-tape.jsonl")
    base = {"schema": "yeto.syncer.elastic-status/v1", "syncer_epoch": 0, "islands": {}}
    (tmp_path / "status.json").write_text(json.dumps({**base, "state": "running"}))
    assert not ls.finished()
    (tmp_path / "status.json").write_text(json.dumps({**base, "state": "finished"}))
    assert ls.finished()


# -- 10-09 ruling: elastic is not a fixed roster -----------------------------------
def test_controller_fixed_roster_only_for_non_elastic_rl():
    from types import SimpleNamespace

    from yeto.launcher import controller_fixed_roster

    assert controller_fixed_roster(SimpleNamespace(training_mode="rl", rl_island_scheduling="legacy"))
    assert not controller_fixed_roster(SimpleNamespace(training_mode="rl", rl_island_scheduling="elastic"))
    assert not controller_fixed_roster(SimpleNamespace(training_mode="sft", rl_island_scheduling="legacy"))


def _refusing_ops(text):
    ops = FakeOps()
    ops.rl_strict_failure = lambda name, job: text if name == "l1" else None
    return ops


def test_elastic_refused_island_dropped_rest_continue():
    ops = _refusing_ops("ElasticProtocolError: JOIN refused or timed out: backend identity mismatch, "
                        "JOIN refused: island 1 declares aa but this elastic session is pinned to bb")
    ops.status_seq["l0"] = [RUNNING, RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    fleet = []
    ctl = FleetController(learners={"l0": ("task-l0", 1), "l1": ("task-l1", 2)},
                          syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30,
                          recover_timeout=600, thread_cls=ImmediateThread, fixed_roster=False,
                          elastic=True)
    ctl._fleet = lambda kind, name=None, **kw: fleet.append((kind, name, kw.get("reason", "")))
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    codes = ctl.run()  # no exception: the pool goes on
    assert ops.relaunch_calls == [] and "l1" in ops.down_calls
    assert ctl.learners["l1"]["state"] == "abandoned" and codes["l0"] == "JobStatus.SUCCEEDED"
    assert any(k == "island_lost" and n == "l1" and "strict failure" in r for k, n, r in fleet)


def test_elastic_strict_failure_drops_only_that_island():
    ops = _refusing_ops("[yeto-rl-strict-failure] metric=x")
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    ctl = FleetController(learners={"l0": ("task-l0", 1), "l1": ("task-l1", 2)},
                          syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30,
                          recover_timeout=600, thread_cls=ImmediateThread, fixed_roster=False,
                          elastic=True)
    ctl.run()
    assert ops.relaunch_calls == [] and ops.down_calls.count("l1") >= 1 and "l0" not in ops.down_calls


def test_legacy_refused_island_still_stops_the_run():
    import pytest

    ops = _refusing_ops("RuntimeError: session mismatch (HELLO refused, session keeps running): x")
    ops.status_seq["l0"] = [RUNNING] * 50
    ops.status_seq["l1"] = [FAILED]
    ops.relaunch_results["l1"] = [None] * 200
    with pytest.raises(FixedRosterIslandAbandoned):
        _ctl(ops).run()  # fixed roster (legacy): unchanged
    ops2 = _refusing_ops("[yeto-rl-strict-failure] metric=x")
    ops2.status_seq["l0"] = [RUNNING] * 5
    ops2.status_seq["l1"] = [FAILED]
    with pytest.raises(RuntimeError, match="strict RL job l1 failed"):
        _ctl(ops2).run()  # legacy strict failure: unchanged
