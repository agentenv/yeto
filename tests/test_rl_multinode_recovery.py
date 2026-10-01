"""rl-multinode-island task 1.7: node loss -> whole-island RECOVERY_REQUIRED; restart precondition."""

from __future__ import annotations

import pytest
from test_rl_reconfig_recovery import _commit_four, _events, _island

from yeto.rl.engine.controller import Rejected
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.journal import read_journal


class _Nodes:
    def __init__(self, alive):
        self.alive = alive

    def __call__(self):
        return dict(self.alive)


def _journal(tmp, kind):
    return [r for r in read_journal(tmp / "state/reconfig") if r["kind"] == kind]


def test_node_loss_mid_run_is_recovery_required(tmp_path):
    nodes = _Nodes({"n0": 4, "n1": 4})
    driver, ctl, fork, pool, publisher, engine = _island(
        tmp_path, controller_kw={"topology": (2, 4), "node_probe": nodes})
    assert ctl.topology == (2, 4) and ctl.check_nodes() is None
    topo = _journal(tmp_path, "topology")
    assert topo and topo[0]["nodes"] == 2 and sorted(topo[0]["alive"]) == ["n0", "n1"]
    original = driver._generate

    def lose_node_then_generate(rollout_id):
        if rollout_id == 1:
            nodes.alive.pop("n1")
        return original(rollout_id)

    driver._generate = lose_node_then_generate
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED: node_lost: 1 of 2 island nodes alive"):
        driver.run()
    assert ctl.inspect().health == "RECOVERY_REQUIRED" and not ctl.admission_open
    lost = _journal(tmp_path, "node_lost")
    assert lost and lost[0]["alive"] == ["n0"] and lost[0]["topology"] == [2, 4]
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == "RECOVERY_REQUIRED" and "node_lost" in ev[-1]["error"]
    # no generation/training happened on the surviving node after the loss
    assert ("generate", 2) not in engine.calls and ("train", 2) not in engine.calls
    assert ctl.check_nodes() == ctl.recovery_required  # idempotent terminal
    with pytest.raises(Rejected, match="RECOVERY_REQUIRED"):
        ctl.request("late", "T4R4S0", 0, 60)
    ctl.close()


def test_wrong_gpu_count_on_a_node_is_node_loss(tmp_path):
    nodes = _Nodes({"n0": 4, "n1": 2})
    driver, ctl, *_ = _island(tmp_path, controller_kw={"topology": (2, 4), "node_probe": nodes})
    assert "do not expose 4 GPUs" in ctl.check_nodes()
    ctl.close()


def test_restart_with_missing_node_refuses_recovery(tmp_path):
    clock = _commit_four(tmp_path)
    nodes = _Nodes({"n0": 4})
    driver, ctl, fork, pool, publisher, engine = _island(
        tmp_path, clock=clock, controller_kw={"topology": (2, 4), "node_probe": nodes})
    assert ctl.inspect().health == "RECOVERY_REQUIRED"
    assert "island topology: 1 of 2 island nodes alive" in ctl.recovery_required
    assert not [c for c in fork.calls if c[0] in ("restore", "start", "stop")]  # no differential recovery
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        driver.run()
    ctl.close()


def test_restart_same_shape_other_hostnames_recovers(tmp_path):
    clock = _commit_four(tmp_path)
    nodes = _Nodes({"host-x": 4, "host-y": 4})  # same shape, different node ids
    driver, ctl, fork, pool, publisher, engine = _island(
        tmp_path, clock=clock, controller_kw={"topology": (2, 4), "node_probe": nodes})
    assert ctl.inspect().health == "RECOVERING"
    driver.run()
    assert ctl.inspect().health == "RUNNING" and ctl.admission_open
    verified = [r for r in read_journal(tmp_path / "state/reconfig") if r.get("status") == "verified"][0]
    assert verified["checks"]["nodes"] == "ok"
    ctl.close()


def test_probe_failure_is_fail_closed(tmp_path):
    def boom():
        raise RuntimeError("ray down")
    driver, ctl, *_ = _island(tmp_path, controller_kw={"topology": (2, 4), "node_probe": boom})
    assert "node probe failed" in ctl.check_nodes()
    ctl.close()


def test_single_node_island_has_no_node_checks(tmp_path):
    driver, ctl, *_ = _island(tmp_path)
    assert ctl.topology is None and ctl.check_nodes() is None and ctl.topology_rejection() is None
    assert not _journal(tmp_path, "topology")
    with pytest.raises(Rejected):
        ctl.set_topology((0, 4))
    ctl.close()
