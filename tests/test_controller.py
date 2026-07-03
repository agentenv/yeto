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

    def job_status(self, cluster, job_id):
        seq = self.status_seq[cluster]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def cluster_up(self, cluster):
        return self.up.get(cluster, True)

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


def make_controller(ops, learners, recover_timeout=100, poll=30, on_relaunch=None):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
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
