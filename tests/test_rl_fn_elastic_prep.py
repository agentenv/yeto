"""fn-elastic: Flash-Next 4x8 H200 elastic prep (CPU only).

* tests/multinode_gpu/resources-fn-4x8.json is name/shape-identical to
  ``flash_next_elastic_declaration()`` (the hook intersects certified edges with
  the declaration by config NAME, so a drift would silently drop every candidate);
* the launcher shapes both fixed configs (trainer 2 nodes x 8, TP2 PP8 EP2; 1 or 2
  TP8 SGLang engines; standby = a whole node) without any cloud call;
* a placeholder / foreign / wrong-kind attestation never yields a candidate and
  auto mode only holds until E1 certifies the FN edges.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_rl_elastic_wire import BIG, DECL, FP, SMALL, _ev, cost_table, setup
from test_rl_launcher_multinode import _elastic_task
from yeto import launcher
from yeto.rl.elastic_benchmark.capabilities import (Attestation, attestation_from_dict,
                                                    parse_configs, validate_edges)
from yeto.rl.engine.controller import read_journal
from yeto.rl.engine.miles_adapter.elastic_hook import elastic_hook_for
from yeto.rl.profiles import qwen3_8_next as q

MANIFEST = Path(__file__).resolve().parent / "multinode_gpu" / "resources-fn-4x8.json"
MANIFEST_2X8 = MANIFEST.with_name("resources-fn-2x8.json")
FN_GPU = "nebius:4x8xh200"
FN_GPU_2X8 = "nebius:2x8xh200"
TRAINER_PAR_2X8 = ("--tensor-parallel", "2", "--pipeline-parallel", "4", "--expert-parallel", "2",
                   "--rollout-num-gpus-per-engine", "8")
TRAINER_PAR = ("--tensor-parallel", "2", "--pipeline-parallel", "8", "--expert-parallel", "2",
               "--rollout-num-gpus-per-engine", "8")


def _manifest():
    return json.loads(MANIFEST.read_text())


def test_manifest_matches_the_declaration():
    raw = _manifest()
    configs = parse_configs(raw)
    assert set(configs) == set(DECL["configs"]) == {SMALL, BIG}
    for name, decl in DECL["configs"].items():
        got = configs[name]
        assert (got.trainer, got.rollout, got.standby, got.rollout_engine_gpus) == \
            (decl.trainer, decl.rollout, decl.standby, decl.rollout_engine_gpus)
        assert dict(got.parallel) == dict(decl.parallel) == {"tp": 2, "pp": 8, "cp": 1, "ep": 2}
    assert (raw["nodes"], raw["gpus_per_node"]) == (DECL["nodes"], DECL["gpus_per_node"])
    edges = {(e.source, e.target, e.kind) for e in validate_edges(raw, configs)}
    assert edges == {(a, b, q.ELASTIC_EDGE_KIND) for a, b in DECL["declared_edges"]}
    assert DECL["initial_config"] == SMALL


@pytest.mark.parametrize("config,rollout,standby,engines", [(SMALL, 8, 8, 1), (BIG, 16, 0, 2)])
def test_launcher_dry_shapes_both_fn_configs(monkeypatch, tmp_path, config, rollout, standby, engines):
    args, spec, task = _elastic_task(monkeypatch, tmp_path, FN_GPU, MANIFEST, config,
                                     rollout=rollout, standby=standby, extra=TRAINER_PAR)
    assert spec.num_nodes == 4 and spec.gpus_per_node == 8
    nodes, per_node, bundles = launcher.rl_island_layout(args, spec)
    assert (nodes, per_node) == (2, 8)
    assert bundles["trainer"] == tuple(range(16))
    assert bundles["rollout"] == tuple(range(16, 16 + rollout))
    assert bundles["standby"] == tuple(range(16 + rollout, 32))
    assert task.num_nodes == 4
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 8 --rl-island-gpus-per-node 8" in task.run
    for flag in ("--tensor-parallel 2", "--pipeline-parallel 8", "--expert-parallel 2",
                 "--rollout-num-gpus-per-engine 8", f"--rollout-num-gpus {rollout}"):
        assert flag in task.run
    assert f"--rl-elastic-initial-config {config}" in task.run
    assert rollout // 8 == engines


def test_2x8_manifest_matches_the_declaration_and_shapes_the_launch(monkeypatch, tmp_path):
    """Formal 2x8 shape (user decision 2026-10-07): one config FN-T8R8S0, n0 = trainer
    TP2 PP4 EP2, n1 = one TP8 engine, no standby, no edges (nothing to recommend)."""
    raw = json.loads(MANIFEST_2X8.read_text())
    decl = q.flash_next_elastic_declaration(nodes=2, trainer_gpus=8)
    configs = parse_configs(raw)
    assert set(configs) == set(decl["configs"]) == {"FN-T8R8S0"} and decl["initial_config"] == "FN-T8R8S0"
    got, d = configs["FN-T8R8S0"], decl["configs"]["FN-T8R8S0"]
    assert (got.trainer, got.rollout, got.standby, got.rollout_engine_gpus) == (8, 8, 0, 8) == \
        (d.trainer, d.rollout, d.standby, d.rollout_engine_gpus)
    assert dict(got.parallel) == dict(d.parallel) == {"tp": 2, "pp": 4, "cp": 1, "ep": 2}
    assert validate_edges(raw, configs) == [] and decl["declared_edges"] == frozenset()
    args, spec, task = _elastic_task(monkeypatch, tmp_path, FN_GPU_2X8, MANIFEST_2X8, "FN-T8R8S0",
                                     rollout=8, extra=TRAINER_PAR_2X8)
    assert spec.num_nodes == 2 and spec.gpus_per_node == 8
    nodes, per_node, bundles = launcher.rl_island_layout(args, spec)
    assert (nodes, per_node) == (1, 8)
    assert bundles["trainer"] == tuple(range(8)) and bundles["rollout"] == tuple(range(8, 16))
    assert bundles["standby"] == ()
    assert task.num_nodes == 2
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rl-island-gpus-per-node 8" in task.run
    for flag in ("--tensor-parallel 2", "--pipeline-parallel 4", "--expert-parallel 2",
                 "--rollout-num-gpus-per-engine 8", "--rollout-num-gpus 8",
                 "--rl-elastic-initial-config FN-T8R8S0"):
        assert flag in task.run, flag
    assert "--rl-standby-gpus" not in task.run


def test_launcher_refuses_a_flag_set_that_disagrees_with_the_cfg(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="placement is T16 R8 S8"):
        _elastic_task(monkeypatch, tmp_path, FN_GPU, MANIFEST, SMALL, rollout=16, standby=0,
                      extra=TRAINER_PAR)


# ---------------------------------------------------------------- attestation placeholders
def _hook(att: Attestation, mode="recommend"):
    configs = parse_configs(_manifest())
    controller = SimpleNamespace(configs=configs, attestation=att, recommend_mode=mode,
                                 journal=SimpleNamespace(epochs=SimpleNamespace(config_id=SMALL)))
    controller.set_recommend_mode = lambda m, reason="": None
    miles_args = SimpleNamespace(yeto_rl_recommend_mode=mode, num_rollout=100)
    hook = elastic_hook_for(miles_args, controller=controller, profile=q.flash_next_execution_profile(),
                            observe=True)
    return hook, controller


PLACEHOLDERS = {
    "none": Attestation.none(),
    "auto_only": attestation_from_dict({"execution_modes": ["partitioned-serial"],
                                        "auto_controller": True, "certified_edges": []}),
    "foreign_2x2": attestation_from_dict({"auto_controller": True, "certified_edges": [
        {"source": "T2R1S1", "target": "T2R2S0", "kind": "rollout-only"},
        {"source": "T2R2S0", "target": "T2R1S1", "kind": "rollout-only"}]}),
    "undeclared_fn_names": attestation_from_dict({"auto_controller": True, "certified_edges": [
        {"source": "FN-T16R8S8", "target": "FN-T24R8S0", "kind": "rollout-only"}]}),
    "fn_wrong_kind": attestation_from_dict({"auto_controller": True, "certified_edges": [
        {"source": "FN-T16R8S8", "target": "FN-T16R16S0", "kind": "trainer-dp"}]}),
}


@pytest.mark.parametrize("which", sorted(PLACEHOLDERS))
def test_uncertified_fn_edges_are_never_candidates(which):
    hook, controller = _hook(PLACEHOLDERS[which])
    assert hook is not None and hook.declared_edges is not None
    assert hook.candidates(controller) == []


def test_e1_certified_fn_attestation_yields_exactly_the_declared_edge():
    att = attestation_from_dict({"runtime_fingerprint": FP, "execution_modes": ["partitioned-serial"],
                                 "auto_controller": True, "certified_edges": [
                                     {"source": a, "target": b, "kind": q.ELASTIC_EDGE_KIND}
                                     for a, b in sorted(DECL["declared_edges"])]})
    hook, controller = _hook(att)
    assert [(e.source, e.target, e.source_engines, e.target_engines)
            for e in hook.candidates(controller)] == [(SMALL, BIG, 1, 2)]


def test_auto_without_e1_certification_only_holds(tmp_path):
    driver, ctl, h, _ = setup(tmp_path, certified=False, mode="auto", events=_ev("saturated"),
                              costs=cost_table(tmp_path), rounds=6)
    driver.run()
    assert h.decisions and all(d["action"] == "hold" and d["candidates"] == [] for d in h.decisions)
    assert ctl.journal.epochs.config_epoch == 0
    assert "request" not in [r["kind"] for r in read_journal(tmp_path / "state/reconfig")]


# ---------------------------------------------------------------- fnrun.sh argv (print-only)
@pytest.mark.parametrize("case,config,rollout", [("fn32s", SMALL, 8), ("fn32b", BIG, 16)])
def test_fnrun_argv_parses_and_shapes(monkeypatch, tmp_path, case, config, rollout):
    import shlex
    import subprocess
    import sys
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args

    out = subprocess.run(["bash", str(MANIFEST.parent / "fnrun.sh"), case], check=True,
                         capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "COSTS": str(tmp_path / "c.json")}).stdout
    argv = shlex.split(out.strip())
    assert argv[0] == "launch" and "--rl-elastic-attestation" not in argv  # nothing certified by default
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = parse_args(argv[1:] + ["--rl-elastic-state-dir", str(tmp_path / "state")])
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    spec = parse_gpu_spec(args.gpu)[0]
    task = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert launcher.rl_island_layout(args, spec)[:2] == (2, 8)
    assert "--rl-recommend-mode recommend" in task.run
    assert f"--rl-elastic-initial-config {config}" in task.run and f"--rollout-num-gpus {rollout}" in task.run
