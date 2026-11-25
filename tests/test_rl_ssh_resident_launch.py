"""launcher.run() with a resident ssh: trainer island (yeto.launch.ssh_resident): the
fleet starts it over SSH (fake cluster), the trainer's end ends the run, the teardown
releases every node and verifies it, and an unconfirmed release exits 5."""

from __future__ import annotations

import time

import pytest

import yeto.launcher as launcher
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)
from test_rl_launcher_island_failure import Clock, _setup
from test_rl_ssh_resident import NAMES, FakeCluster, _cards  # noqa: F401  (fixture)
from yeto.rl import event_echo

FINALIZED = event_echo.format_record({"island_id": 0, "time_unix": 1.0, "event": "rl_learner_finalized"}) + "\n"


def _inventory(tmp_path, n=3):
    path = tmp_path / "inventory.yaml"
    path.write_text("nodes:\n" + "".join(
        f"  - {{name: b300-{i}, host: ops@192.0.2.{10 + i}, address: 198.51.100.{10 + i}, interface: bond0, "
        "device_family: cuda, cpu_arch: x86_64, card: B300, cards_per_node: 8, hbm_gib: 288, "
        "host_memory_gib: 2048, fabric: ib, runtime: docker, docker_args: [--gpus, all, --device, "
        "/dev/infiniband, --ulimit, 'memlock=-1:-1']}\n" for i in range(n)))
    return path


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster(NAMES)
    monkeypatch.setattr("yeto.launch.ssh_resident.SshOps", lambda: fake)
    return fake


def _args(tmp_path, *extra):
    inv = _inventory(tmp_path)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync",
                                    "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
                                    "--rl-node-inventory", str(inv), "--rl-ssh-trainer-workdir", "/opt/yeto",
                                    "--tensor-parallel", "8", "--pipeline-parallel", "3",
                                    *extra), gpu="ssh:3x8xb300@bm")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    return args


def _fleet(args) -> list[tuple[str, str, str | None]]:
    """(event, island, reason) of the run's fleet.jsonl lifecycle records."""
    import json

    from yeto import runs

    path = runs.run_dir(args.cluster_prefix) / "fleet.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [(r["event"], r["island"], r.get("reason")) for r in recs if r["event"].startswith("island_")]


class _ImmediateThread:
    """The fleet's ``thread_cls``: a relaunch runs to its end inside the poll that starts it,
    so the fake clock never moves while one is in flight (a loaded runner cannot make a
    relaunch outlast the recovery window in fake time)."""

    def __init__(self, target, daemon=None, **kw):
        self._target = target

    def start(self):
        self._target()

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class DrivenClock(Clock):
    """Fake time with the test's script on it: before each fake sleep of the launcher, every
    step whose condition holds runs, in order, on the launcher's own thread. No wall-clock
    thread races the fake clock."""

    def __init__(self):
        super().__init__()
        self.steps: list[tuple] = []
        self.controller = None

    def then(self, ready, act):
        self.steps.append((ready, act))

    def sleep(self, seconds):
        while self.steps and self.steps[0][0]():
            self.steps.pop(0)[1]()
        super().sleep(seconds)

    def finalized(self) -> bool:
        c = self.controller
        return c is not None and bool(c.learners) and all(c._finalized(r) for r in c.learners.values())  # noqa: SLF001


def _drive(monkeypatch) -> DrivenClock:
    """The launcher's FleetController runs relaunches inline and is known to the clock (its
    finalized probe tells when the island's end record went through the log stream)."""
    clock = DrivenClock()

    class Controller(launcher.FleetController):
        def __init__(self, *a, **kw):
            super().__init__(*a, **{**kw, "thread_cls": _ImmediateThread})
            clock.controller = self

    monkeypatch.setattr(launcher, "FleetController", Controller)
    return clock


def _started(cluster, k):
    return lambda: all(k in cluster.runs[n] for n in NAMES)


def _finish_when_started(clock, cluster, k=0):
    """Once incarnation ``k`` runs on every node: node 0 reports the trainer's end, and, once
    the launcher has read it (event sync, not a sleep), every rank exits 0."""

    def finish():
        cluster.say("b300-0", k, FINALIZED)
        deadline = time.monotonic() + 60
        while not clock.finalized():  # the log pump is a thread: wait for it, fake time stands still
            if time.monotonic() > deadline:
                raise AssertionError("the island's rl_learner_finalized record was never read")
            time.sleep(0.001)
        for name in reversed(NAMES):
            cluster.exit(name, k, 0)

    clock.then(_started(cluster, k), finish)


def _crash_then_finish(clock, cluster):
    """Incarnation 0: rank 1 dies once every node runs; incarnation 1 then runs to its end."""
    clock.then(_started(cluster, 0), lambda: cluster.exit("b300-1", 0, 1))
    _finish_when_started(clock, cluster, k=1)


def test_a_resident_ssh_trainer_runs_to_its_end_and_is_released(monkeypatch, tmp_path, cluster):
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    _finish_when_started(clock, cluster)
    code = launcher.run(args)
    assert code == 0, code
    assert cluster.count("start", 0) == 3 and cluster.count("kill", 0) == 3  # teardown verified each node
    assert not [r for r in record if r[0] in ("terminate", "relaunch")]  # no sky call for it


def test_a_failed_rank_is_relaunched_as_the_next_incarnation(monkeypatch, tmp_path, cluster):
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    _crash_then_finish(clock, cluster)
    code = launcher.run(args)
    assert code == 0, code
    assert cluster.count("start", 0) == 3 and cluster.count("start", 1) == 3
    assert cluster.count("kill", 0) == 3 and cluster.count("kill", 1) == 3


def test_an_unconfirmed_release_exits_5_and_keeps_the_head(monkeypatch, tmp_path, cluster):
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    cluster.kill_rc["b300-2"] = 4  # its container survives docker rm -f
    _finish_when_started(clock, cluster)
    assert launcher.run(args) == launcher.TEARDOWN_UNVERIFIED_EXIT == 5
    assert args._teardown_incomplete is True
    stops = [e for e in _fleet(args) if e[0] == "island_stop"]
    assert [e[2] for e in stops] == ["SUCCEEDED"]  # the normal end's record only: the unconfirmed release adds none


def test_the_stop_flag_stops_the_island_and_exits_9(monkeypatch, tmp_path, cluster):
    from yeto import breakers, runs

    breakers.clear_stop_reason()  # a worker is one process per run: no earlier test's trip

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    runs.request_stop(args.cluster_prefix)
    assert launcher.run(args) == launcher.RUN_STOPPED_EXIT == 9
    assert cluster.count("start", 0) == 3 and cluster.count("kill", 0) == 3
    assert all(cluster.runs[n][0]["state"] == "exited" for n in NAMES)
    # the stop's teardown closes the island in fleet.jsonl (the page stops billing it)
    (ready,), (stop,) = ([e for e in _fleet(args) if e[0] == k] for k in ("island_ready", "island_stop"))
    assert stop == ("island_stop", ready[1], "teardown")


def test_a_second_launch_of_the_run_reattaches_to_the_kept_island(monkeypatch, tmp_path, cluster):
    from yeto import runs

    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    args.keep = True
    runs.request_stop(args.cluster_prefix)  # the first launcher ends at once, the island is kept
    assert launcher.run(args) == 9 and cluster.count("kill") == 0
    assert not [e for e in _fleet(args) if e[0] == "island_stop"]  # kept: still running and billed
    runs.stop_flag_path(args.cluster_prefix).unlink()
    again = _args(tmp_path)
    _finish_when_started(clock, cluster)
    assert launcher.run(again) == 0
    assert cluster.count("start") == 3 and cluster.count("follow", 0) == 3  # reattached, no second start
    assert cluster.count("kill", 0) == 3


def test_a_launch_is_refused_when_the_inventory_lacks_a_node(monkeypatch, tmp_path, cluster):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    path = _inventory(tmp_path, n=2)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync",
                                    "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
                                    "--rl-node-inventory", str(path), "--rl-ssh-trainer-workdir", "/w",
                                    "--tensor-parallel", "8", "--pipeline-parallel", "3"), gpu="ssh:3x8xb300@bm")
    with pytest.raises(SystemExit, match="has 3 node"):
        launcher.run(args)
    assert not cluster.calls


def test_a_resident_island_copies_the_weights_onto_every_node_before_its_first_start(
        monkeypatch, tmp_path, cluster):
    """--model-store-source on a resident ssh: trainer (trainer-only, no scheduler): the
    island's preflight runs the weights copy (ssh_weights.sync_weights) on every node,
    once, before any node starts; a relaunch does not copy again."""
    import yeto.launch.ssh_weights as ssh_weights

    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    for key in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.setenv("YETO_WEIGHTS_S3_" + key, "test")
    synced = []

    def sync(plan, nodes, addresses, ops, *, images):
        synced.append((plan.store, plan.source, [n.name for n in nodes], images,
                       sum(cluster.count("start", k) for k in range(3))))

    monkeypatch.setattr(ssh_weights, "sync_weights", sync)
    args = _args(tmp_path, "--model-store", "local:///data/models",
                 "--model-store-source", "s3://weights/models/k31")
    inv = tmp_path / "inventory.yaml"
    inv.write_text(inv.read_text().replace("docker_args:", "mounts: {/mnt/nvme/models: /data/models}, docker_args:"))
    _crash_then_finish(clock, cluster)
    assert launcher.run(args) == 0
    image = "ghcr.io/x/y@sha256:" + "a" * 64
    assert synced == [("/data/models", "s3://weights/models/k31", NAMES, [image] * 3, 0)]
    assert cluster.count("start", 0) == 3 and cluster.count("start", 1) == 3
