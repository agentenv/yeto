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


def _finish_when_started(cluster, k=0):
    def finish():
        for _ in range(500):
            if all(k in cluster.runs[n] for n in NAMES):
                break
            time.sleep(0.01)
        cluster.say("b300-0", k, FINALIZED)
        for name in reversed(NAMES):
            cluster.exit(name, k, 0)
    import threading

    threading.Thread(target=finish, daemon=True).start()


def test_a_resident_ssh_trainer_runs_to_its_end_and_is_released(monkeypatch, tmp_path, cluster):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    _finish_when_started(cluster)
    code = launcher.run(args)
    assert code == 0, code
    assert cluster.count("start", 0) == 3 and cluster.count("kill", 0) == 3  # teardown verified each node
    assert not [r for r in record if r[0] in ("terminate", "relaunch")]  # no sky call for it


def test_a_failed_rank_is_relaunched_as_the_next_incarnation(monkeypatch, tmp_path, cluster):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    import threading

    def crash_then_finish():
        for _ in range(500):
            if all(0 in cluster.runs[n] for n in NAMES):
                break
            time.sleep(0.01)
        cluster.exit("b300-1", 0, 1)  # incarnation 0: rank 1 dies
        _finish_when_started(cluster, k=1)

    threading.Thread(target=crash_then_finish, daemon=True).start()
    code = launcher.run(args)
    assert code == 0, code
    assert cluster.count("start", 0) == 3 and cluster.count("start", 1) == 3
    assert cluster.count("kill", 0) == 3 and cluster.count("kill", 1) == 3


def test_an_unconfirmed_release_exits_5_and_keeps_the_head(monkeypatch, tmp_path, cluster):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    cluster.kill_rc["b300-2"] = 4  # its container survives docker rm -f
    _finish_when_started(cluster)
    assert launcher.run(args) == launcher.TEARDOWN_UNVERIFIED_EXIT == 5
    assert args._teardown_incomplete is True


def test_the_stop_flag_stops_the_island_and_exits_9(monkeypatch, tmp_path, cluster):
    from yeto import runs

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    runs.request_stop(args.cluster_prefix)
    assert launcher.run(args) == launcher.RUN_STOPPED_EXIT == 9
    assert cluster.count("start", 0) == 3 and cluster.count("kill", 0) == 3
    assert all(cluster.runs[n][0]["state"] == "exited" for n in NAMES)


def test_a_second_launch_of_the_run_reattaches_to_the_kept_island(monkeypatch, tmp_path, cluster):
    from yeto import runs

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path)
    args.keep = True
    runs.request_stop(args.cluster_prefix)  # the first launcher ends at once, the island is kept
    assert launcher.run(args) == 9 and cluster.count("kill") == 0
    runs.stop_flag_path(args.cluster_prefix).unlink()
    again = _args(tmp_path)
    _finish_when_started(cluster)
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
