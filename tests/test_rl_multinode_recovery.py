"""rl-multinode-island task 1.7: node loss -> whole-island RECOVERY_REQUIRED; restart precondition."""

from __future__ import annotations

import pytest
from test_rl_reconfig_recovery import _commit_four, _ctl, _events, _island

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


def test_refuse_partial_island_records_topology_and_recovery(tmp_path):
    """tasks 3.3: the startup precondition runs before any placement group exists."""
    ctl = _ctl(tmp_path / "state", {"t": 1000.0}, topology=(2, 4), node_probe=_Nodes({"n0": 4}))
    why = ctl.refuse_partial_island()
    assert why == "1 of 2 island nodes alive (['n0'])"
    assert ctl.inspect().health == "RECOVERY_REQUIRED" and not ctl.admission_open
    assert "island topology: 1 of 2 island nodes alive" in ctl.recovery_required
    assert "recovery refused on fewer nodes" in ctl.recovery_required
    topo = _journal(tmp_path, "topology")
    assert len(topo) == 1 and topo[0]["nodes"] == 2 and topo[0]["gpus_per_node"] == 4
    assert topo[0]["alive"] == ["n0"] and topo[0]["incarnation"] == ctl.incarnation["id"]
    phases = _journal(tmp_path, "phase")
    assert phases[-1]["phase"] == "RECOVERY_REQUIRED" and "island topology" in phases[-1]["error"]
    assert ctl.refuse_partial_island() == why  # idempotent terminal (second record, same why)
    ctl.close()


def test_refuse_partial_island_full_island_then_open_writes_topology_once(tmp_path):
    nodes = _Nodes({"n0": 4, "n1": 4})
    ctl = _ctl(tmp_path / "state", {"t": 1000.0}, topology=(2, 4), node_probe=nodes)
    assert ctl.refuse_partial_island() is None
    assert ctl.recovery_required is None and len(_journal(tmp_path, "topology")) == 1
    from test_rl_reconfig_recovery import NAME, FakeEngine, Fork, Pool
    import torch

    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, placement_kind="fixed-partition")
    pool = Pool(engine, Fork(engine, running=("engine:c0", "engine:c1")))
    assert ctl.open(pool).health == "RUNNING"
    assert len(_journal(tmp_path, "topology")) == 1  # open() does not repeat the preflight record
    ctl.close()
    # no topology -> no-op
    ctl2 = _ctl(tmp_path / "two" / "state", {"t": 1000.0})
    assert ctl2.refuse_partial_island() is None and not _journal(tmp_path / "two", "topology")
    ctl2.close()


def test_entry_preflight_refuses_partial_island_before_any_placement_group(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.engine.miles_adapter import entry

    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    elastic = SimpleNamespace(controller=ctl)
    topology = SimpleNamespace(nodes=2, gpus_per_node=1)
    miles_args = SimpleNamespace(yeto_rl_event_tape=str(tmp_path / "events.jsonl"), yeto_rl_learner_id=0)
    monkeypatch.setattr(entry, "_ray_alive_nodes", lambda: {"head": 1})  # worker DEAD in the GCS
    pinned = []
    monkeypatch.setattr(entry, "pin_placement_group_to_head", lambda *a, **k: pinned.append(a))
    with pytest.raises(RuntimeError, match="island is RECOVERY_REQUIRED: island topology: 1 of 2"):
        entry.refuse_partial_island_preflight(elastic, topology, miles_args)
    assert not pinned and ctl.inspect().health == "RECOVERY_REQUIRED"
    topo = _journal(tmp_path, "topology")
    assert len(topo) == 1 and topo[0]["alive"] == ["head"]
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == "RECOVERY_REQUIRED" and "recovery refused on fewer nodes" in ev[-1]["error"]
    ctl.close()
    # full island: passes, no event, single-node / elastic None: no-op
    ok = tmp_path / "ok"
    ctl = _ctl(ok / "state", {"t": 1000.0})
    monkeypatch.setattr(entry, "_ray_alive_nodes", lambda: {"head": 1, "w1": 1})
    assert entry.refuse_partial_island_preflight(SimpleNamespace(controller=ctl), topology, miles_args) is None
    assert ctl.recovery_required is None and len(_journal(ok, "topology")) == 1
    assert len([e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]) == 1  # no new event
    ctl.close()
    assert entry.refuse_partial_island_preflight(None, topology, miles_args) is None
    assert entry.refuse_partial_island_preflight(elastic, SimpleNamespace(nodes=1, gpus_per_node=1), miles_args) is None


# ------------------------------------------------- Q4 (C5): parallel layout baseline
def _layout(**kw):
    base = {"tp": 1, "pp": 1, "cp": 1, "ep": 1, "trainer": 4, "nodes": 2, "gpus_per_node": 4,
            "bundle_map": None}
    base.update(kw)
    return base


def test_topology_record_carries_the_layout_and_same_shape_new_node_ids_pass(tmp_path):
    kw = {"topology": (2, 4), "node_probe": _Nodes({"n0": 4, "n1": 4}),
          "layout": {"pp": 2, "bundle_map": {"trainer": (0, 4), "rollout": (1, 5)}}}
    _d, ctl, *_ = _island(tmp_path, controller_kw=kw)
    topo = _journal(tmp_path, "topology")
    assert topo[0]["layout"] == _layout(pp=2, bundle_map={"rollout": [1, 5], "trainer": [0, 4]})
    assert topo[0]["layout_accepted"] is True and topo[0]["layout_error"] is None
    assert ctl.layout_baseline() is None  # first incarnation: nothing to compare against
    ctl.close()
    # machine replaced: other node ids, same layout -> accepted, baseline is the first record
    kw2 = dict(kw, node_probe=_Nodes({"m0": 4, "m1": 4}))
    _d, ctl2, fork2, *_ = _island(tmp_path, controller_kw=kw2)
    assert ctl2.inspect().health == "RUNNING" and ctl2.layout_rejection() is None
    assert ctl2.layout_baseline() == topo[0]["layout"]
    assert _journal(tmp_path, "topology")[1]["layout_accepted"] is True
    ctl2.close()


def test_changed_layout_is_refused_before_any_membership_call_and_is_not_a_journal_terminal(tmp_path):
    kw = {"topology": (2, 4), "node_probe": _Nodes({"n0": 4, "n1": 4}), "layout": {"pp": 2}}
    _d, ctl, *_ = _island(tmp_path, controller_kw=kw)
    ctl.close()
    # relaunch with PP1 TP2 (another cut layout) on the same journal
    bad = dict(kw, layout={"tp": 2, "pp": 1})
    _d, ctl2, fork2, *_ = _island(tmp_path, controller_kw=bad)
    assert ctl2.inspect().health == "RECOVERY_REQUIRED" and not ctl2.admission_open
    assert "layout changed: tp 1 -> 2, pp 2 -> 1" in ctl2.recovery_required
    assert not [c for c in fork2.calls if c[0] in ("restore", "start", "stop")], fork2.calls
    rec = _journal(tmp_path, "topology")[-1]
    assert rec["layout_accepted"] is False and rec["layout_error"].startswith("layout changed")
    assert not [r for r in _journal(tmp_path, "phase") if r["phase"] == "RECOVERY_REQUIRED"]
    ctl2.close()
    # the same shape again (operator fixed the recipe): the island opens
    _d, ctl3, *_ = _island(tmp_path, controller_kw=kw)
    assert ctl3.inspect().health == "RUNNING" and ctl3.layout_rejection() is None
    ctl3.close()
    # a bundle-map change (mixed placement moved) is a layout change too
    moved = dict(kw, layout={"pp": 2, "bundle_map": {"trainer": (0, 1), "rollout": (4, 5)}})
    _d, ctl4, *_ = _island(tmp_path, controller_kw=moved)
    assert "bundle_map None -> {'rollout': [4, 5], 'trainer': [0, 1]}" in ctl4.recovery_required
    ctl4.close()


def test_restart_recovery_precondition_and_confirm_check_the_layout(tmp_path):
    from yeto.rl.engine.controller import RecoveryRequired

    nodes = _Nodes({"n0": 4, "n1": 4})
    kw = {"topology": (2, 4), "node_probe": nodes, "layout": {"pp": 2}}
    driver, ctl, *_ = _island(tmp_path, controller_kw=kw)
    ctl.close()
    # the restart precondition (recovery-design §2) refuses another layout
    _d, ctl2, *_ = _island(tmp_path, controller_kw=dict(kw, layout={"pp": 1}))
    assert ctl2._recovery_precondition(_d.rollout, frozenset(), frozenset()).startswith("layout changed")
    ctl2.close()
    # confirm_recovery: same layout -> checks.layout ok; a layout drift between the
    # restart precondition and the first publication is a failed recovery
    _d3, ctl3, *_ = _island(tmp_path, controller_kw=kw)
    ctl3.recovery_pending = {"recovery_id": "r1", "members": frozenset(_d3.rollout.members()),
                             "deadline_wall": 1e12}
    ctl3.layout = dict(ctl3.layout, pp=4)
    with pytest.raises(RecoveryRequired, match="layout changed: pp 2 -> 4"):
        ctl3.confirm_recovery(_d3)
    rec = [r for r in _journal(tmp_path, "recovery") if r["status"] == "failed"][-1]
    assert rec["checks"]["layout"].startswith("layout changed")
    ctl3.close()


def test_entry_layout_from_megatron_args_and_placement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.engine.miles_adapter import entry

    topology = SimpleNamespace(nodes=2, gpus_per_node=2)
    miles_args = SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=2,
                                 context_parallel_size=1, expert_model_parallel_size=1,
                                 actor_num_nodes=2, actor_num_gpus_per_node=1,
                                 yeto_rl_event_tape=str(tmp_path / "events.jsonl"), yeto_rl_learner_id=0)
    placement = SimpleNamespace(bundle_map={"trainer": (0, 2), "rollout": (1,), "standby": (3,)})
    assert entry.island_layout_of(miles_args, topology, placement) == {
        "tp": 1, "pp": 2, "cp": 1, "ep": 1, "nodes": 2, "gpus_per_node": 2, "trainer": 2,
        "bundle_map": {"trainer": [0, 2], "rollout": [1], "standby": [3]}}
    assert entry.island_layout_of(SimpleNamespace(), topology)["bundle_map"] is None
    # preflight journals it; the next incarnation with PP1 is refused with the tape event
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    monkeypatch.setattr(entry, "_ray_alive_nodes", lambda: {"head": 2, "w1": 2})
    assert entry.refuse_partial_island_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                                 placement=placement) is None
    assert _journal(tmp_path, "topology")[0]["layout"]["pp"] == 2
    ctl.close()
    ctl = _ctl(tmp_path / "state", {"t": 1001.0})
    miles_args.pipeline_model_parallel_size = 1
    miles_args.actor_num_gpus_per_node = 2
    miles_args.actor_num_nodes = 1
    with pytest.raises(RuntimeError, match="RECOVERY_REQUIRED: layout changed: pp 2 -> 1"):
        entry.refuse_partial_island_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                              placement=SimpleNamespace(bundle_map=None))
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == "RECOVERY_REQUIRED" and "bundle_map" in ev[-1]["error"]
    ctl.close()
