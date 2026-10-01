"""Unit tests for FleetController: fake sky ops + fake clock, no network.

Relaunch attempts run through an inline "thread" stub so every poll is
deterministic and the tests never sleep for real.
"""

import pytest

from yeto.launcher import FleetController

SYNCER = "yeto-syncer"


class FakeStatus:
    def __init__(self, name: str, terminal: bool):
        self._name = name
        self._terminal = terminal

    def is_terminal(self) -> bool:
        return self._terminal

    def __str__(self) -> str:
        return f"JobStatus.{self._name}"


RUNNING = FakeStatus("RUNNING", False)
SUCCEEDED = FakeStatus("SUCCEEDED", True)
FAILED = FakeStatus("FAILED", True)


class ImmediateThread:
    """threading.Thread stand-in that runs the target inline on start()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class FakeOps:
    """Scripted sky_ops: statuses are consumed per poll (last one repeats)."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = 0
        self.status_seq = {}  # cluster -> [FakeStatus, ...]; last repeats
        self.up = {}  # cluster -> bool (default True)
        self.relaunch_results = {}  # cluster -> [job_id or None]; empty -> None
        self.after_relaunch = {}  # cluster -> status seq installed on success
        self.relaunch_calls = []  # cluster names, in order
        self.relaunch_tasks = []  # tasks passed to relaunch, in order
        self.down_calls = []  # cluster names, in order
        self.alive = {}  # cluster -> bool answered by job_alive (default False)
        self.alive_calls = []  # (cluster, job_id) queries, in order

    def job_status(self, cluster, job_id):
        seq = self.status_seq[cluster]
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item

    def cluster_up(self, cluster):
        return self.up.get(cluster, True)

    def job_alive(self, cluster, job_id):
        self.alive_calls.append((cluster, job_id))
        return self.alive.get(cluster, False)

    def relaunch(self, task, cluster):
        self.relaunch_calls.append(cluster)
        self.relaunch_tasks.append(task)
        queue = self.relaunch_results.get(cluster)
        job_id = queue.pop(0) if queue else None
        if job_id is not None:
            self.status_seq[cluster] = list(self.after_relaunch.get(cluster, [RUNNING]))
            self.up[cluster] = True
        return job_id

    def down(self, cluster):
        self.down_calls.append(cluster)

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps += 1
        assert self.sleeps < 500, "controller poll loop did not terminate"


def make_controller(ops, learners, recover_timeout=100, poll=30, on_relaunch=None,
                    stop_flag=None):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
        stop_flag=stop_flag,
        learners={name: (f"task-{name}", job_id) for name, job_id in learners.items()},
        syncer=(SYNCER, "task-syncer", 1),
        sky_ops=ops,
        poll_interval=poll,
        recover_timeout=recover_timeout,
        on_relaunch=on_relaunch,
        thread_cls=ImmediateThread,
    )


def test_learner_recovers_within_timeout():
    # (a) cluster preempted (not UP), relaunch succeeds -> running -> done.
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False
    ops.relaunch_results["l0"] = [101]
    ops.after_relaunch["l0"] = [RUNNING, SUCCEEDED]
    relaunched = []
    ctl = make_controller(
        ops, {"l0": 1}, on_relaunch=lambda name, job: relaunched.append((name, job))
    )

    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert ctl.learners["l0"]["state"] == "done"
    assert ctl.learners["l0"]["job_id"] == 101
    assert relaunched == [("l0", 101)]
    # Relaunched with the original task spec, and never torn down.
    assert ops.relaunch_calls == ["l0"]
    assert ops.relaunch_tasks == ["task-l0"]
    assert ops.down_calls == []


INIT_REFUSED = RuntimeError(
    "Getting job status: skipped for cluster 'l0' (status: INIT). "
    "It is only allowed for UP and AUTOSTOPPING clusters."
)


def test_status_refusal_with_live_job_is_not_a_failure():
    # sky refuses job_status while the cluster reads INIT, but the job is
    # still running on it: keep waiting, never relaunch (live-run-failures #37).
    ops = FakeOps()
    ops.status_seq["l0"] = [INIT_REFUSED, INIT_REFUSED, SUCCEEDED]
    ops.alive["l0"] = True
    ctl = make_controller(ops, {"l0": 1})

    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert ops.relaunch_calls == []
    assert ops.down_calls == []
    assert ops.alive_calls == [("l0", 1), ("l0", 1)]
    assert ctl.learners["l0"]["job_id"] == 1


def test_status_refusal_with_dead_job_still_recovers():
    ops = FakeOps()
    ops.status_seq["l0"] = [INIT_REFUSED]
    ops.alive["l0"] = False
    ops.relaunch_results["l0"] = [101]
    ops.after_relaunch["l0"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1})

    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert ops.relaunch_calls == ["l0"]
    assert ctl.learners["l0"]["job_id"] == 101


def test_cluster_not_up_with_live_job_is_not_a_failure():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, RUNNING, SUCCEEDED]
    ops.up["l0"] = False
    ops.alive["l0"] = True
    ctl = make_controller(ops, {"l0": 1})

    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert ops.relaunch_calls == []
    assert ops.alive_calls == [("l0", 1), ("l0", 1)]


def test_learner_abandoned_after_timeout_run_continues(capsys):
    # (b) job FAILED, every relaunch fails, timeout passes -> down() once,
    # abandoned; the run completes with the surviving learner.
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    ops.status_seq["l1"] = [RUNNING] * 5 + [SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1, "l1": 2}, recover_timeout=100, poll=30)

    exit_codes = ctl.run()

    assert exit_codes["l1"] == "JobStatus.SUCCEEDED"
    assert exit_codes["l0"].startswith("ABANDONED after ")
    assert ctl.learners["l0"]["state"] == "abandoned"
    assert ops.down_calls == ["l0"]  # exactly once; never the syncer
    assert set(ops.relaunch_calls) == {"l0"}  # retried until timeout
    err = capsys.readouterr().err
    assert "ABANDONED" in err and "fleet continues with 1 learner" in err


def test_zero_recover_timeout_tears_down_immediately():
    # (c) recover_timeout=0 disables recovery: first failure -> teardown,
    # no relaunch attempt at all.
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    ops.status_seq["l1"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1, "l1": 2}, recover_timeout=0)

    exit_codes = ctl.run()

    assert exit_codes["l0"].startswith("ABANDONED")
    assert exit_codes["l1"] == "JobStatus.SUCCEEDED"
    assert ops.down_calls == ["l0"]
    assert ops.relaunch_calls == []


def test_all_learners_abandoned_raises_and_downs_syncer():
    # (d) nothing survives -> syncer torn down too, RuntimeError.
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    ops.status_seq["l1"] = [FAILED]
    ctl = make_controller(ops, {"l0": 1, "l1": 2}, recover_timeout=0)

    with pytest.raises(RuntimeError):
        ctl.run()

    assert sorted(ops.down_calls) == sorted(["l0", "l1", SYNCER])
    assert SYNCER in ctl.downed_clusters


def test_syncer_never_abandoned(capsys):
    # (e) syncer job fails; relaunches keep failing well past the timeout,
    # but the controller never downs it and eventually recovers it.
    ops = FakeOps()
    ops.status_seq[SYNCER] = [FAILED]
    ops.relaunch_results[SYNCER] = [None, None, 201]
    ops.after_relaunch[SYNCER] = [RUNNING]
    ops.status_seq["l0"] = [RUNNING] * 6 + [SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1}, recover_timeout=50, poll=30)

    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert SYNCER not in ops.down_calls
    assert ctl.syncer["state"] == "running"
    assert ctl.syncer["job_id"] == 201
    assert ops.relaunch_calls.count(SYNCER) == 3
    err = capsys.readouterr().err
    assert "syncer unrecovered" in err and "still retrying" in err


# -- fixed-roster RL recovery windows (P0 batch review) ------------------------

from yeto.launcher import (  # noqa: E402
    FIXED_ROSTER_MAX_RELAUNCHES,
    RECOVERY_STABLE_S,
    FixedRosterIslandAbandoned,
)


def _fixed(ops, recover_timeout=3600):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
        learners={"l0": ("task-l0", 1)}, syncer=(SYNCER, "task-syncer", 1), sky_ops=ops,
        poll_interval=30, recover_timeout=recover_timeout, thread_cls=ImmediateThread,
        fixed_roster=True,
    )


def test_fixed_roster_recovered_island_that_fails_hours_later_is_relaunched_again():
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    healthy_for_hours = [RUNNING] * int(6 * 3600 / 30)  # >> RECOVERY_STABLE_S
    ops.relaunch_results["l0"] = [101, 102, 103]
    ops.after_relaunch["l0"] = healthy_for_hours + [FAILED]
    original = ops.relaunch

    def relaunch(task, cluster):
        job = original(task, cluster)
        # each later relaunch is again healthy for hours, then the last one finishes
        ops.after_relaunch["l0"] = (healthy_for_hours + [FAILED] if job < 102
                                    else [RUNNING, SUCCEEDED])
        return job

    ops.relaunch = relaunch
    ops.sleeps = -100000  # the long healthy phases poll many times
    # 3 failures separated by hours: each opens a fresh window (count/budget
    # reset), so even more relaunches than the per-window cap are allowed
    assert _fixed(ops, recover_timeout=600).run() == {"l0": "JobStatus.SUCCEEDED"}
    assert len(ops.relaunch_calls) == 3 > FIXED_ROSTER_MAX_RELAUNCHES
    assert RECOVERY_STABLE_S < 6 * 3600


def test_fixed_roster_persistently_failing_island_abandoned_after_the_cap():
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    ops.relaunch_results["l0"] = list(range(200, 220))
    ops.after_relaunch["l0"] = [FAILED]  # dies right after every relaunch
    with pytest.raises(FixedRosterIslandAbandoned):
        _fixed(ops).run()
    assert len(ops.relaunch_calls) == FIXED_ROSTER_MAX_RELAUNCHES
    assert "l0" in ops.down_calls


def test_non_fixed_roster_abandon_keeps_the_fleet_running():
    ops = FakeOps()
    ops.status_seq["l0"] = [FAILED]
    ops.status_seq["l1"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1, "l1": 2}, recover_timeout=0)
    codes = ctl.run()  # SFT: no exception, l0 abandoned, l1 finishes
    assert ctl.learners["l0"]["state"] == "abandoned" and codes["l1"] == "JobStatus.SUCCEEDED"


# ---------------------------------------------------------------- no launcher relaunch
def _parsed(extra):
    from yeto.cli import parse_args as parse_cli

    return parse_cli(["--gpu", "modal:1xh100", "--model", "org/model", "--data", "org/data",
                      "--training-mode", "rl", "--total-steps", "3", "--rollout-batch-size", "4",
                      "--n-samples-per-prompt", "2", "--rollout-max-response-len", "128",
                      "--local-rl-rounds-per-sync", "1", "--reward-function", "pkg.reward:score",
                      *extra])


def test_modal_retries_zero_or_no_island_relaunch_disables_the_launcher_relaunch():
    from yeto.launcher import effective_recover_timeout

    assert effective_recover_timeout(_parsed(())) == 1200  # default unchanged
    assert effective_recover_timeout(_parsed(("--modal-retries", "3"))) == 1200
    assert effective_recover_timeout(_parsed(("--modal-retries", "0"))) == 0
    assert effective_recover_timeout(_parsed(("--no-island-relaunch",))) == 0
    assert effective_recover_timeout(_parsed(("--recover-timeout", "60"))) == 60
    # the controller built with that budget never relaunches a failed island
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, FAILED]
    ops.relaunch_results["l0"] = [101]
    ctl = make_controller(ops, {"l0": 1},
                          recover_timeout=effective_recover_timeout(_parsed(("--modal-retries", "0"))))
    try:
        ctl.run()
    except RuntimeError:
        pass  # all learners gone
    assert ops.relaunch_calls == [] and "l0" in ops.down_calls


def test_launch_uses_the_effective_budget():
    import inspect

    from yeto import launcher

    assert "recover_timeout=effective_recover_timeout(args)" in inspect.getsource(launcher)


def test_stop_flag_prevents_the_relaunch(tmp_path):
    """STOP is set, then the cluster disappears: the island is not relaunched."""
    flag = tmp_path / "STOP"
    flag.write_text("stop\n")
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False  # preempted / gone
    ops.relaunch_results["l0"] = [101]  # would succeed if it were tried
    ctl = make_controller(ops, {"l0": 1}, stop_flag=flag)
    with pytest.raises(RuntimeError):
        ctl.run()  # all learners gone
    assert ops.relaunch_calls == []
    assert "l0" in ops.down_calls
    assert "STOP flag" in ctl.learners["l0"]["exit"]


def test_stop_flag_written_after_the_poll_check_still_prevents_the_relaunch(tmp_path):
    """The flag appears between the recovery decision and the relaunch call itself."""
    flag = tmp_path / "STOP"
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False
    ops.relaunch_results["l0"] = [101]
    ctl = make_controller(ops, {"l0": 1}, stop_flag=flag)
    real = ctl._stop_requested
    calls = []

    def stop_requested(rec, where):
        calls.append(where)
        if where == "recovery":  # first check passes; the flag lands before the relaunch runs
            flag.write_text("stop\n")
            return False
        return real(rec, where)

    ctl._stop_requested = stop_requested
    with pytest.raises(RuntimeError):
        ctl.run()
    assert "relaunch" in calls and ops.relaunch_calls == []


def test_without_the_flag_the_island_is_relaunched(tmp_path):
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False
    ops.relaunch_results["l0"] = [101]
    ops.after_relaunch["l0"] = [RUNNING, SUCCEEDED]
    ctl = make_controller(ops, {"l0": 1}, stop_flag=tmp_path / "STOP")
    ctl.run()
    assert ops.relaunch_calls == ["l0"]


def test_stop_run_cli_only_writes_the_flag(tmp_path, monkeypatch, capsys):
    from yeto import cli, runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    assert cli.main(["stop-run", "myrun"]) == 0
    assert runs.stop_flag_path("myrun") == tmp_path / "myrun" / "STOP"
    assert (tmp_path / "myrun" / "STOP").exists()
    assert not (tmp_path / "myrun" / "meta.json").exists()  # nothing else touched
    # the launcher wires the same path into the fleet controller
    import inspect

    from yeto import launcher

    assert "stop_flag=runs.stop_flag_path(args.cluster_prefix)" in inspect.getsource(launcher)
