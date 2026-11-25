"""Resident ``ssh:`` islands on the unified decision table (task 2.2 of unified-trainer-islands):
``SshIslandOps.island_decision`` reads what the source observed (a rank's exit code, node 0's
markers, a node that stops answering, a process gone without an exit code) through
``island_lifecycle.classify``; the fleet controller records it, and with the island lifecycle
on a rank's exit 10 is a configuration error (never relaunched). The reattach rule of a
restarted launcher is unchanged. No SSH, no cloud."""

from __future__ import annotations

import pytest

from tests.test_rl_ssh_resident import NAMES, Clock, FakeCluster, make_ops, spec, wait_for
from yeto.config_errors import CONFIG_ERROR_EXIT, ConfigError, config_error_line
from yeto.launch import island_lifecycle as lc
from yeto.launch import ssh_resident


@pytest.fixture(autouse=True)
def _cards(monkeypatch, tmp_path):
    from yeto.hw import card_tools

    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))


def _island(tmp_path, **kw):
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster, **kw)
    ops.add("isl", spec())
    ops.start("isl")
    return cluster, ops


def test_one_marker_list_for_every_source():
    assert ssh_resident.STRICT_MARKERS[:-2] == lc.STRICT_MARKERS
    assert ssh_resident.STRICT_MARKERS[-2:] == ("[yeto-config-error]", "[yeto-island-stop]")


def test_a_running_island_has_no_decision(tmp_path):
    _cluster, ops = _island(tmp_path)
    assert ops.island_decision("isl", 0) is None
    assert ops.island_decision("nope", 0) is None


def test_rank0_exit_0_is_finished(tmp_path):
    cluster, ops = _island(tmp_path)
    for name in reversed(NAMES):
        cluster.exit(name, 0, 0)
    assert wait_for(lambda: all(p.poll() is not None for p in ops.islands["isl"].job.procs))
    assert str(ops.job_status("isl", 0)) == "SUCCEEDED"
    d = ops.island_decision("isl", 0)
    assert (d.cause, d.state) == ("finished", "done")


def test_a_rank_exit_code_is_an_infrastructure_failure(tmp_path):
    cluster, ops = _island(tmp_path)
    cluster.exit("b300-1", 0, 3)
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    d = ops.island_decision("isl", 0)
    assert (d.cause, d.action) == ("infra", "relaunch") and "exited with code 3" in d.detail
    assert ops.island_decision("isl", 1) is None  # another incarnation


def test_a_rank_exit_10_is_a_configuration_error(tmp_path):
    cluster, ops = _island(tmp_path)
    cluster.exit("b300-2", 0, CONFIG_ERROR_EXIT)
    assert wait_for(lambda: ops.islands["isl"].job.procs[2].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    d = ops.island_decision("isl", 0)
    assert (d.cause, d.action, d.state) == ("config_error", "fail", "failed")


def test_a_rank_exit_11_is_a_deterministic_stop(tmp_path):
    """A trainer breaker, a strict check or a refusal stopped the island: never relaunched."""
    from yeto.config_errors import ISLAND_STOP_EXIT

    cluster, ops = _island(tmp_path)
    cluster.exit("b300-0", 0, ISLAND_STOP_EXIT)
    assert wait_for(lambda: ops.islands["isl"].job.procs[0].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    d = ops.island_decision("isl", 0)
    assert (d.cause, d.action, d.state) == ("breaker", "fail", "failed")


def test_node0_markers_decide(tmp_path):
    cluster, ops = _island(tmp_path)
    cluster.say("b300-0", 0, config_error_line(ConfigError("FullParameterLayoutMismatch")) + "\n")
    cluster.exit("b300-0", 0, 1)
    assert wait_for(lambda: ops.islands["isl"].job.procs[0].poll() is not None)
    assert wait_for(lambda: str(ops.job_status("isl", 0)) == "FAILED")
    # node 0's log lines arrive on the stream thread: wait for the marker, not only for the exit
    assert wait_for(lambda: ops.island_decision("isl", 0).cause == "config_error")
    cluster2, ops2 = _island(tmp_path / "b")
    cluster2.say("b300-0", 0, "[yeto-rl-strict-failure] mixed_version_group_count\n")
    cluster2.exit("b300-0", 0, 1)
    assert wait_for(lambda: str(ops2.job_status("isl", 0)) == "FAILED")
    assert wait_for(lambda: ops2.island_decision("isl", 0).cause == "strict")


def test_a_node_that_stops_answering_is_lost_and_a_vanished_process_is_infra(tmp_path):
    clock = Clock()
    cluster, ops = _island(tmp_path, clock=clock, health_every_s=60.0, unreachable_polls=3)
    cluster.unreachable.add("b300-1")
    for _ in range(3):
        clock.t += 61
        ops.job_status("isl", 0)
    d = ops.island_decision("isl", 0)
    assert (d.cause, d.action) == ("lost", "relaunch")
    cluster2, ops2 = _island(tmp_path / "b")
    cluster2.vanish("b300-2", 0)
    assert wait_for(lambda: ops2.islands["isl"].job.procs[2].poll() is not None)
    assert str(ops2.job_status("isl", 0)) == "FAILED"
    assert ops2.island_decision("isl", 0).cause == "infra"


def test_routing_ops_passes_the_decision_through(tmp_path):
    from yeto.modal_runner import RoutingOps

    cluster, ops = _island(tmp_path)
    routing = RoutingOps(object(), None, ops)
    assert routing.island_decision("isl", 0) is None
    cluster.exit("b300-1", 0, CONFIG_ERROR_EXIT)
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    ops.job_status("isl", 0)
    assert routing.island_decision("isl", 0).cause == "config_error"
    assert RoutingOps(object(), None, None).island_decision("r-l0-aws", 1) is None  # sky: none


# -- the fleet controller ------------------------------------------------------------------------------

def _ctl(ops, *, recorder):
    from test_controller import ImmediateThread

    from yeto.launch.fleet import FleetController

    class Ops:  # the resident island's ops plus a controller clock
        def __init__(self, inner, clock):
            self.inner, self.clock = inner, clock

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def now(self):
            return self.clock.t

        def sleep(self, seconds):
            self.clock.t += seconds

    return FleetController(learners={"isl": ("task", 0)}, syncer=None, sky_ops=Ops(ops, Clock()), poll_interval=30,
                           recover_timeout=0, thread_cls=ImmediateThread, elastic=True, recorder=recorder)


def test_the_controller_records_the_source_decision(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    cluster, ops = _island(tmp_path)
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    recorder = lc.FleetRecorder(states, {"isl": lc.IslandRef(0, "ssh")})
    cluster.exit("b300-1", 0, 3)
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    ctl = _ctl(ops, recorder=recorder)
    with pytest.raises(RuntimeError, match="all learners abandoned"):  # its only island, no recovery time
        ctl.run()
    record = states.read(0)
    assert record["decision"]["cause"] == "infra" and "exited with code 3" in record["decision"]["detail"]


def test_with_the_lifecycle_a_rank_exit_10_is_never_relaunched(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    cluster, ops = _island(tmp_path)
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    recorder = lc.FleetRecorder(states, {"isl": lc.IslandRef(0, "ssh")})
    cluster.exit("b300-2", 0, CONFIG_ERROR_EXIT)  # no marker line: the exit code alone
    assert wait_for(lambda: ops.islands["isl"].job.procs[2].poll() is not None)
    ctl = _ctl(ops, recorder=recorder)
    ctl.recover_timeout = 600
    with pytest.raises(RuntimeError, match="all learners abandoned"):  # dropped, not relaunched
        ctl.run()
    assert cluster.count("start", 1) == 0  # no incarnation 1
    record = states.read(0)
    assert record["state"] == "failed" and record["decision"]["cause"] == "config_error"
