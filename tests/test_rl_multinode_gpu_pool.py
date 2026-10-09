"""rl-multinode-island Q6 (ruling 2026-10-04): runtime GPU uuid reconciliation.

Pure rule (``multinode.reconcile_gpu_pool``), the controller's ``gpu_pool`` journal
record, the entry preflight with a fake per-node nvidia-smi probe, and the
``--rl-elastic-accept-rebind`` switch end to end (CLI -> launcher -> learner)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_rl_multinode_recovery import _journal
from test_rl_reconfig_recovery import _ctl, _events

from yeto.rl.engine import multinode as mn

A, B, C, D = "GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"
TOPO = mn.Topology(2, 2)


def _obs(*nodes):
    return tuple(tuple(n) for n in nodes)


# ------------------------------------------------------------------ pure rule
def test_shape_mismatch_is_refused_even_with_accept_rebind():
    r = mn.reconcile_gpu_pool([[A, B], [C, D]], _obs([A, B], [C]), accept_rebind=True)
    assert not r.ok and not r.rebind and "shape [2, 1]" in r.error and "[2, 2]" in r.error
    r = mn.reconcile_gpu_pool([[A, B]], _obs([A, B], [C, D]), accept_rebind=True)
    assert not r.ok and "shape" in r.error
    # the observed side: a probe of the wrong shape is refused before any comparison
    with pytest.raises(mn.TopologyError, match="covers 1 nodes"):
        mn.observed_gpu_pool([[(0, A), (1, B)]], TOPO)
    with pytest.raises(mn.TopologyError, match="exposes 1 GPUs"):
        mn.observed_gpu_pool([[(0, A), (1, B)], [(0, C)]], TOPO)
    with pytest.raises(mn.TopologyError, match="indices"):
        mn.observed_gpu_pool([[(0, A), (1, B)], [(0, C), (2, D)]], TOPO)
    with pytest.raises(mn.TopologyError, match="repeated uuid"):
        mn.observed_gpu_pool([[(0, A), (1, B)], [(0, A), (1, D)]], TOPO)


def test_matching_uuids_pass_without_rebind():
    observed = mn.observed_gpu_pool([[(1, B), (0, A)], [(0, C), (1, D)]], TOPO)
    assert observed == _obs([A, B], [C, D])  # local index order, not probe order
    r = mn.reconcile_gpu_pool([[A, B], [C, D]], observed, accept_rebind=False)
    assert r.ok and not r.rebind and r.mapping == {} and r.diffs == () and r.error is None
    assert r.flat == (A, B, C, D)
    # partial cfg spelling: only the named slots are compared
    r = mn.reconcile_gpu_pool([[A, None], [None, D]], _obs([A, "x"], ["y", D]), accept_rebind=False)
    assert r.ok and not r.rebind


def test_changed_uuid_is_refused_without_accept_rebind():
    r = mn.reconcile_gpu_pool([[A, B], [C, D]], _obs([A, B], ["GPU-new", D]), accept_rebind=False)
    assert not r.ok and not r.rebind
    assert r.diffs == (f"n1:0 declared {C} observed GPU-new",)
    assert "--rl-elastic-accept-rebind" in r.error and "n1:0" in r.error


def test_accept_rebind_yields_mapping_and_journals_rebind(tmp_path):
    r = mn.reconcile_gpu_pool([[A, B], [C, D]], _obs([A, "GPU-b2"], ["GPU-c2", D]), accept_rebind=True)
    assert r.ok and r.rebind and r.mapping == {B: "GPU-b2", C: "GPU-c2"} and len(r.diffs) == 2
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    assert ctl.record_gpu_pool(r, source="cfg", accept_rebind=True) is None
    assert ctl.recovery_required is None
    (rec,) = _journal(tmp_path, "gpu_pool")
    assert rec["incarnation"] == ctl.incarnation["id"] and rec["accepted"] and rec["rebind"]
    assert rec["uuids"] == [[A, "GPU-b2"], ["GPU-c2", D]] and rec["mapping"] == {B: "GPU-b2", C: "GPU-c2"}
    assert rec["source"] == "cfg" and rec["accept_rebind"] is True and len(rec["diffs"]) == 2
    # the new pool is now the binding baseline
    assert ctl.gpu_pool_baseline() == _obs([A, "GPU-b2"], ["GPU-c2", D])
    ctl.close()
    # a refused result is journaled too and is the island's RECOVERY_REQUIRED cause
    ctl = _ctl(tmp_path / "two" / "state", {"t": 1000.0})
    bad = mn.reconcile_gpu_pool([[A, B], [C, D]], _obs([A, B], [C, "GPU-x"]), accept_rebind=False)
    why = ctl.record_gpu_pool(bad, source="cfg", accept_rebind=False)
    assert why == bad.error and ctl.recovery_required.startswith("gpu_pool: 1 GPU uuid(s) differ")
    assert ctl.inspect().health == "RECOVERY_REQUIRED" and not ctl.admission_open
    (rec,) = [r for r in _journal(tmp_path / "two", "gpu_pool")]
    assert not rec["accepted"] and rec["error"] == bad.error and not rec["rebind"]
    # not a journal terminal: the refusal is restartable with --rl-elastic-accept-rebind
    assert not [r for r in _journal(tmp_path / "two", "phase") if r["phase"] == "RECOVERY_REQUIRED"]
    assert ctl.gpu_pool_baseline() is None  # a refused pool never becomes the baseline
    ctl.close()
    again = _ctl(tmp_path / "two" / "state", {"t": 1001.0})
    assert again.recovery_required is None and again.inspect().health != "RECOVERY_REQUIRED"
    again.close()


def test_first_run_without_uuids_binds_baseline_and_the_next_incarnation_compares(tmp_path):
    cfg = mn.declared_gpu_pool({"gpus": []}, TOPO)
    assert cfg == ((None, None), (None, None))
    assert mn.merge_declared_pool(cfg, None) is None
    r = mn.reconcile_gpu_pool(None, _obs([A, B], [C, D]), accept_rebind=False)
    assert r.ok and not r.rebind
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    assert ctl.gpu_pool_baseline() is None
    assert ctl.record_gpu_pool(r, source="none", accept_rebind=False) is None
    ctl.close()
    # second incarnation: the journal baseline is the declared side
    ctl2 = _ctl(tmp_path / "state", {"t": 1001.0})
    base = ctl2.gpu_pool_baseline()
    assert base == _obs([A, B], [C, D])
    declared = mn.merge_declared_pool(cfg, base)
    assert declared == _obs([A, B], [C, D])
    same = mn.reconcile_gpu_pool(declared, _obs([A, B], [C, D]), accept_rebind=False)
    assert same.ok and not same.rebind
    swapped = mn.reconcile_gpu_pool(declared, _obs([A, B], ["GPU-c2", D]), accept_rebind=False)
    assert not swapped.ok and swapped.diffs == (f"n1:0 declared {C} observed GPU-c2",)
    assert ctl2.record_gpu_pool(swapped, source="journal", accept_rebind=False) == swapped.error
    assert ctl2.recovery_required.startswith("gpu_pool:")
    ctl2.close()
    # cfg uuid wins over the baseline where it names a slot
    merged = mn.merge_declared_pool(((A, "GPU-cfg"), (None, None)), base)
    assert merged == _obs([A, "GPU-cfg"], [C, D])
    # the cfg pool with node/index/uuid entries
    cfg2 = mn.declared_gpu_pool({"gpus": [{"node": 1, "index": 0, "uuid": C}, {"node": 0, "index": 1, "uuid": B}]}, TOPO)
    assert cfg2 == ((None, B), (C, None))
    with pytest.raises(mn.TopologyError, match="outside"):
        mn.declared_gpu_pool({"gpus": [{"node": 2, "index": 0, "uuid": C}]}, TOPO)


# --------------------------------------------------------------- entry preflight
def _probe(rows):
    def probe(_topology):
        return rows
    return probe


def test_entry_preflight_refuses_changed_pool_and_accepts_with_flag(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import entry

    topology = SimpleNamespace(nodes=2, gpus_per_node=2)
    resources = {"nodes": 2, "gpus_per_node": 2, "gpus": [
        {"node": 0, "index": 0, "uuid": A}, {"node": 0, "index": 1, "uuid": B},
        {"node": 1, "index": 0, "uuid": C}, {"node": 1, "index": 1, "uuid": D}]}
    miles_args = SimpleNamespace(yeto_rl_event_tape=str(tmp_path / "events.jsonl"), yeto_rl_learner_id=0,
                                 yeto_rl_elastic={"resources": resources})
    live = [[(0, A), (1, B)], [(0, C), (1, D)]]
    monkeypatch.setattr(entry, "_ray_gpu_uuids", _probe(live))
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    res = entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    assert res.ok and not res.rebind and ctl.recovery_required is None
    (rec,) = _journal(tmp_path, "gpu_pool")
    assert rec["uuids"] == [[A, B], [C, D]] and rec["source"] == "cfg"
    ctl.close()
    # the worker was replaced: refused (fail closed), tape event, no placement group
    swapped = [[(0, A), (1, B)], [(0, "GPU-c2"), (1, "GPU-d2")]]
    monkeypatch.setattr(entry, "_ray_gpu_uuids", _probe(swapped))
    ctl = _ctl(tmp_path / "state", {"t": 1001.0})
    with pytest.raises(RuntimeError, match="island is RECOVERY_REQUIRED: gpu_pool: 2 GPU uuid"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    assert ctl.inspect().health == "RECOVERY_REQUIRED"
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == "RECOVERY_REQUIRED" and "--rl-elastic-accept-rebind" in ev[-1]["error"]
    recs = _journal(tmp_path, "gpu_pool")
    assert len(recs) == 2 and not recs[-1]["accepted"] and recs[-1]["source"] == "cfg+journal"
    ctl.close()
    # explicit restart with --rl-elastic-accept-rebind: accepted, rebind journaled, baseline moves
    miles_args.yeto_rl_elastic["accept_rebind"] = True
    ctl = _ctl(tmp_path / "state", {"t": 1002.0})
    res = entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    assert res.ok and res.rebind and res.mapping == {C: "GPU-c2", D: "GPU-d2"}
    assert ctl.recovery_required is None
    recs = _journal(tmp_path, "gpu_pool")
    assert recs[-1]["rebind"] and recs[-1]["accept_rebind"] and recs[-1]["uuids"] == [[A, B], ["GPU-c2", "GPU-d2"]]
    assert ctl.gpu_pool_baseline() == _obs([A, B], ["GPU-c2", "GPU-d2"])
    assert len([e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]) == 1
    ctl.close()


def test_entry_preflight_probe_failure_is_fail_closed_and_single_node_is_a_no_op(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import entry

    topology = SimpleNamespace(nodes=2, gpus_per_node=1)
    miles_args = SimpleNamespace(yeto_rl_event_tape=str(tmp_path / "events.jsonl"), yeto_rl_learner_id=0,
                                 yeto_rl_elastic={"resources": {"nodes": 2, "gpus_per_node": 1}})

    def boom(_topology):
        raise RuntimeError("nvidia-smi not found on node w1")

    monkeypatch.setattr(entry, "_ray_gpu_uuids", boom)
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    with pytest.raises(RuntimeError, match="gpu_pool: nvidia-smi not found on node w1"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    assert ctl.inspect().health == "RECOVERY_REQUIRED"
    (rec,) = _journal(tmp_path, "gpu_pool")
    assert not rec["accepted"] and rec["source"] == "probe" and rec["uuids"] == []
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == "RECOVERY_REQUIRED"
    ctl.close()
    # a probe of the wrong shape (one node only answered) is refused the same way
    monkeypatch.setattr(entry, "_ray_gpu_uuids", _probe([[(0, A)]]))
    ctl = _ctl(tmp_path / "shape" / "state", {"t": 1000.0})
    with pytest.raises(RuntimeError, match="covers 1 nodes"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    ctl.close()
    # no cfg uuids, first run: observed pool becomes the baseline, nothing refused
    monkeypatch.setattr(entry, "_ray_gpu_uuids", _probe([[(0, A)], [(0, C)]]))
    ctl = _ctl(tmp_path / "first" / "state", {"t": 1000.0})
    res = entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args)
    assert res.ok and _journal(tmp_path / "first", "gpu_pool")[0]["source"] == "none"
    ctl.close()
    # single node / no elastic: untouched (no probe, no record)
    called = []
    monkeypatch.setattr(entry, "_ray_gpu_uuids", lambda t: called.append(t))
    ctl = _ctl(tmp_path / "single" / "state", {"t": 1000.0})
    one = SimpleNamespace(nodes=1, gpus_per_node=4)
    assert entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), one, miles_args) is None
    assert entry.reconcile_gpu_pool_preflight(None, topology, miles_args) is None
    assert not called and not _journal(tmp_path / "single", "gpu_pool")
    ctl.close()


def test_main_calls_the_gpu_pool_preflight_right_after_the_partial_island_check():
    import inspect

    from yeto.rl.adapters.miles import entry

    src = inspect.getsource(entry)
    i = src.index("        refuse_partial_island_preflight(elastic, topology, miles_args, placement=launch.placement)\n")
    j = src.index("        reconcile_gpu_pool_preflight(elastic, topology, miles_args, placement=launch.placement)\n")
    k = src.index("        pin_placement_group_to_head(topology.gpus_per_node)\n")
    assert i < j < k


# ------------------------------------------------------------- CLI -> learner switch
def test_accept_rebind_flag_flows_from_cli_to_the_learner(tmp_path, monkeypatch):
    from test_rl_engine_selection import _cli, _island_task, _learner_argv
    from test_rl_infra_switches import ELASTIC_LEARNER, _elastic_cli, _elastic_files

    from yeto import launcher
    from yeto.launcher import _prepare_rl_args
    from yeto.rl import learner

    res, _ = _elastic_files(tmp_path)
    gpu = ("--rl-rollout-gpus", "1", "--gpu", "aws:2xa100@us-east-1")
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0", "--rl-placement", "fixed-partition",
                        "--rl-elastic-accept-rebind", *gpu)
    assert args.rl_elastic_accept_rebind is True
    launcher._check_ports_infra_switches(args, "ports")
    _prepare_rl_args(args)
    assert " --rl-elastic-accept-rebind" in _island_task(args, monkeypatch).run
    plain = _elastic_cli(res, "--rl-elastic-initial-config", "c0", "--rl-placement", "fixed-partition", *gpu)
    _prepare_rl_args(plain)
    assert "--rl-elastic-accept-rebind" not in _island_task(plain, monkeypatch).run
    with pytest.raises(ValueError, match="--rl-elastic-accept-rebind need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-elastic-accept-rebind",)), "ports")
    # learner side: parsed into yeto_rl_elastic["accept_rebind"], refused without --rl-elastic
    parsed = learner.parse_args(_learner_argv(ELASTIC_LEARNER + ("--rl-elastic-accept-rebind",)))
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert miles_args.yeto_rl_elastic["accept_rebind"] is True
    parsed = learner.parse_args(_learner_argv(ELASTIC_LEARNER))
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert "accept_rebind" not in miles_args.yeto_rl_elastic
    with pytest.raises(SystemExit):
        learner.parse_args(_learner_argv(("--rl-elastic-accept-rebind",)))
