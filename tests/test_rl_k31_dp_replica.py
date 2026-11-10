"""The full-parameter sync at DP>1 on the k31 pin (Miles k31/b2 on), on CPU.

B2 ported the ports-v1 full-parameter transport together with the commits after
it: DP=1 distributed-optimizer masters (fork 56190a5a8), data-parallel replicas
(921cde8c7), sharded FP32 masters (d50fb04dd), the verified apply (102672760)
and the shape-free layout (8d03f386a). The k31 lineage registers them on B2, so:

* the pins features hold from B2 on (not on F1) and the ports-v1 entries are unchanged;
* ``run_config_rules.full_parameter_dp_{replica,sharded_master}_gap`` are None on the pin;
* a K3.1 trainer island with ``full_parameter_sync=True`` translates at DP 2 and DP 8:
  ``--rl-dp-optimizer auto|distributed`` keep Miles' distributed optimizer (sharded
  masters), ``replicated`` turns it off (whole masters on every replica), offload and
  the precision-aware optimizer stay refused at DP>1, and a pin without the features
  (F1) still refuses DP>1 with the 921cde8c7 message;
* the launcher's synced K3.1 plan (2 x 1x8xb300, EP8 TP1 = DP 8) reaches the island's
  ``build_ports_launch`` without a refusal;
* the DP=1 distributed-optimizer reader (optimizer offload with the sync) is the
  ``distopt_full_parameter`` feature, so the k31 pin takes it;
* the runtime manifest probes one interface per registered feature, needed by no
  capability, and the probes resolve in a k31/b2+ checkout.

The Miles side (the fork's replica / sharded-master tests on k31/b8) is
``tests/fast/backends/megatron_utils/test_full_parameter_dp_*.py`` in Miles;
``tests/test_dp_replicas_miles.py`` and ``tests/test_rl_shape_free_miles.py`` run
yeto's adapter on the real Miles transport at DP 4 (``importorskip``).
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path

import pytest
from test_rl_k31_e2e_cpu import _hf_config, _k31_config
from test_rl_miles_adapter_config import _parsed, sub

from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import pins, run_config_rules
from yeto.rl.adapters.miles import runtime_manifest as rm
from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec

SELECTION = "/run/expert_selection.json"
K31 = Path(__file__).resolve().parents[1] / "examples" / "k31"
DP_FEATURES = ("distopt_full_parameter", "dp_replica_full_parameter", "dp_sharded_master",
               "verified_apply", "shape_free_layout")
PROBES = {
    "fork_b2_distopt_full_parameter": (
        "miles.backends.megatron_utils.full_parameter_state", "_propagate_master_writes"),
    "fork_b2_dp_replicas": ("miles.backends.megatron_utils.full_parameter_replicas", "install_replica_plan"),
    "fork_b2_dp_sharded_masters": ("miles.backends.megatron_utils.full_parameter_sharded_masters", "master_ranges"),
    "fork_b2_verified_apply": ("miles.backends.megatron_utils.full_parameter_state", "verify_stored_fragment_chunks"),
    "fork_b2_shape_free_layout": (
        "miles.ray.train.full_parameter_controller", "FullParameterTransportMixin.install_full_parameter_owner_maps"),
}


def _spec() -> AlgorithmSpec:
    """The synced K3.1 example spec (age 2, tis correction)."""
    return AlgorithmSpec.from_dict(json.loads((K31 / "algorithm-spec.async-age2-tis.json").read_text()))


def _trainer(dp: int, **memory):
    """A K3.1 trainer island (EP8 TP1, samples from the store) at DP ``dp``:
    DP 8 on 8 GPUs; DP 2 as CP 4 on the same 8 GPUs; ``memory``: TrainMemoryConfig fields."""
    cfg = _k31_config(SELECTION, role="trainer", cp=8 // dp)
    assert cfg.parallel.data_parallel == dp and cfg.parallel.expert_parallel == 8
    if memory:
        cfg = dataclasses.replace(cfg, train_memory=dataclasses.replace(cfg.train_memory, **memory))
    return cfg


def _two_gpu_trainer(**memory):
    """DP 2 without context parallelism: 2 GPUs, EP 2."""
    cfg = sub(_k31_config(SELECTION, role="trainer", ep=2), "parallel", actor_num_gpus_per_node=2,
              data_parallel=2, visible_gpus_per_node=2)
    if memory:
        cfg = dataclasses.replace(cfg, train_memory=dataclasses.replace(cfg.train_memory, **memory))
    return cfg


def _synced(cfg):
    return mc.translate_run_config(cfg, _spec(), max_policy_age=2, full_parameter_sync=True)


def _miles_parsed(cfg, *, use_distributed_optimizer: bool):
    """The namespace Miles hands back for a trainer island: no engine, the island's GPUs."""
    return _parsed(use_distributed_optimizer=use_distributed_optimizer, colocate=False, debug_train_only=True,
                   rollout_num_gpus=0, actor_num_gpus_per_node=cfg.parallel.actor_num_gpus_per_node)


@pytest.fixture
def k31_f1_pin(monkeypatch):
    """Pin Miles k31/f1: the k31 lineage before B2 (no full-parameter transport)."""
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_K31_F1_COMMIT)


# ------------------------------------------------------------------ 1. the registration

def test_the_k31_lineage_registers_the_dp_features_on_b2():
    assert pins.CURRENT_LINEAGE == "k31" and pins.MILES_NEXT_COMMIT == pins.MILES_K31_FULL_D_COMMIT
    for feature in DP_FEATURES:
        assert pins.K31_FEATURES[feature] == pins.MILES_K31_B2_COMMIT, feature
        for commit in (pins.MILES_K31_B2_COMMIT, pins.MILES_K31_B3B7_COMMIT, pins.MILES_K31_B4_COMMIT,
                       pins.MILES_K31_B8_COMMIT, pins.MILES_K31_FULL_COMMIT,
                       pins.MILES_K31_FULL_STUB_COMMIT, pins.MILES_K31_FULL_B_COMMIT,
                       pins.MILES_K31_FULL_C_COMMIT, pins.MILES_K31_FULL_D_COMMIT):
            assert pins.pin_has(feature, commit), (feature, commit)
        assert not pins.pin_has(feature, pins.MILES_K31_F1_COMMIT), feature
        assert pins.pin_has(feature, pins.MILES_PORTS_V1_COMMIT), feature  # the frozen lineage keeps its own
        assert pins.commits_with(feature) >= {pins.MILES_K31_B2_COMMIT, pins.MILES_PORTS_V1_COMMIT}
    # the ports-v1 first commits (and the constants naming them) are unchanged
    assert pins.PORTS_V1_FEATURES["dp_replica_full_parameter"] == pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT
    assert pins.PORTS_V1_FEATURES["dp_sharded_master"] == pins.MILES_DP_SHARDED_MASTER_COMMIT
    assert pins.PORTS_V1_FEATURES["distopt_full_parameter"] == pins.MILES_DISTOPT_FULL_PARAMETER_COMMIT
    # the same batch registers the transport the features ride on
    assert pins.K31_FEATURES["full_parameter_transport"] == pins.MILES_K31_B2_COMMIT


def test_the_dp_gaps_are_none_on_the_pin_and_name_the_fork_commit_before_b2(k31_f1_pin, monkeypatch):
    assert "921cde8c7 or later (data-parallel replicas)" in run_config_rules.full_parameter_dp_replica_gap()
    assert "d50fb04dd or later" in run_config_rules.full_parameter_dp_sharded_master_gap()
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_K31_B2_COMMIT)
    assert run_config_rules.full_parameter_dp_replica_gap() is None
    assert run_config_rules.full_parameter_dp_sharded_master_gap() is None
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_K31_B8_COMMIT)
    assert run_config_rules.full_parameter_dp_replica_gap() is None
    assert run_config_rules.full_parameter_dp_sharded_master_gap() is None


# ------------------------------------------------------------------ 2. the translation

@pytest.mark.parametrize("make", [lambda **m: _trainer(8, **m), lambda **m: _trainer(2, **m), _two_gpu_trainer],
                         ids=["dp8-ep8", "dp2-ep8-cp4", "dp2-ep2"])
def test_a_synced_k31_trainer_translates_at_dp_above_one(make):
    for choice in ("auto", "distributed"):
        cfg = make(dp_optimizer=choice)
        launch = _synced(cfg)
        # sharded FP32 masters: Miles' default distributed optimizer for Adam stays on
        assert "use_distributed_optimizer" not in launch.runtime_attrs, choice
        argv = list(launch.argv)
        assert "--debug-train-only" in argv and "--rollout-num-gpus" not in argv
        # what the sharded masters refuse is not in the argv
        # (full_parameter_sharded_masters.refuse_unsupported_args, master_ranges)
        for flag in ("--overlap-param-gather", "--num-distributed-optimizer-instances", "--optimizer-cpu-offload",
                     "--use-precision-aware-optimizer"):
            assert flag not in argv, (choice, flag)
        mc.validate_parsed_args(_miles_parsed(cfg, use_distributed_optimizer=True), launch, num_cells=lambda a: 1)
    # whole masters on every data-parallel replica (the replica path)
    cfg = make(dp_optimizer="replicated")
    launch = _synced(cfg)
    assert launch.runtime_attrs["use_distributed_optimizer"] is False
    assert "--use-distributed-optimizer" not in launch.argv
    with pytest.raises(mc.MilesConfigError, match="re-enabled the distributed optimizer"):
        mc.validate_parsed_args(_miles_parsed(cfg, use_distributed_optimizer=True), launch, num_cells=lambda a: 1)
    mc.validate_parsed_args(_miles_parsed(cfg, use_distributed_optimizer=False), launch, num_cells=lambda a: 1)
    # the same island without the outer sync has no full-parameter reader
    assert "use_distributed_optimizer" not in mc.translate_run_config(
        make(dp_optimizer="replicated"), _spec(), max_policy_age=2, full_parameter_sync=False).runtime_attrs


@pytest.mark.parametrize("dp", [2, 8])
def test_dp_above_one_keeps_refusing_offload_and_the_precision_aware_optimizer(dp):
    for choice in rc.DP_OPTIMIZERS:
        with pytest.raises(mc.MilesConfigError, match=f"at DP {dp} needs its FP32 masters on the device"):
            _synced(_trainer(dp, dp_optimizer=choice, optimizer_offload_fraction=1.0))
    with pytest.raises(mc.MilesConfigError, match="drop --rl-precision-aware-optimizer"):
        _synced(_trainer(dp, precision_aware_optimizer=True))


@pytest.mark.parametrize("dp", [2, 8])
def test_a_k31_pin_without_the_features_still_refuses_dp_above_one(k31_f1_pin, dp):
    with pytest.raises(mc.MilesConfigError, match=f"full-parameter sync at DP {dp} needs Miles 921cde8c7"):
        _synced(_trainer(dp))
    mc.translate_run_config(_trainer(dp), _spec(), max_policy_age=2, full_parameter_sync=False)


def test_without_the_sharded_masters_auto_falls_back_to_whole_masters(monkeypatch):
    k31 = pins.MILES_LINEAGES["k31"]
    features = {k: v for k, v in k31.features.items() if k != "dp_sharded_master"}
    monkeypatch.setattr(pins, "MILES_LINEAGES", {**pins.MILES_LINEAGES, "k31": k31._replace(features=features)})
    assert run_config_rules.full_parameter_dp_replica_gap() is None
    assert _synced(_trainer(8, dp_optimizer="auto")).runtime_attrs["use_distributed_optimizer"] is False
    with pytest.raises(mc.MilesConfigError, match="--rl-dp-optimizer distributed at DP 8 needs Miles d50fb04dd"):
        _synced(_trainer(8, dp_optimizer="distributed"))


def test_the_dp1_distributed_optimizer_reader_is_the_distopt_feature(monkeypatch):
    # optimizer offload with the sync at DP 1 reads the HybridDeviceOptimizer's FP32 copies
    # (fork 56190a5a8, ported by B2): refused before B2, taken from B2 on. The adapter
    # reads the pin it imported (config.MILES_NEXT_COMMIT).
    cfg = _trainer(1, optimizer_offload_fraction=1.0)
    monkeypatch.setattr(mc, "MILES_NEXT_COMMIT", pins.MILES_K31_F1_COMMIT)
    with pytest.raises(mc.MilesConfigError, match="56190a5a8 or later"):
        _synced(cfg)
    monkeypatch.setattr(mc, "MILES_NEXT_COMMIT", pins.MILES_K31_B8_COMMIT)
    launch = _synced(cfg)
    assert "use_distributed_optimizer" not in launch.runtime_attrs
    mc.validate_parsed_args(_miles_parsed(cfg, use_distributed_optimizer=True), launch, num_cells=lambda a: 1)
    assert ("--use-distributed-optimizer --use-precision-aware-optimizer --optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d --optimizer-offload-fraction 1.0") in " ".join(launch.argv)
    # without offload DP 1 keeps the whole-master path
    assert _synced(_trainer(1)).runtime_attrs["use_distributed_optimizer"] is False


def test_engines_on_the_island_at_dp_above_one_pass_the_layout_rule(k31_f1_pin, monkeypatch):
    from yeto.rl.engine.run_config import full_parameter_layout_problem

    assert "DP>1 needs" in full_parameter_layout_problem(data_parallel=4, actor_num_nodes=1, rollout_gpus=4,
                                                         rollout_gpus_per_engine=1)
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_K31_B8_COMMIT)
    assert full_parameter_layout_problem(data_parallel=4, actor_num_nodes=1, rollout_gpus=4,
                                         rollout_gpus_per_engine=1) is None


# ------------------------------------------------------------------ 3. launcher plan -> island argv

@pytest.fixture
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k31-dp-replica-test-key")


@pytest.mark.usefixtures("_island_hmac_key")
@pytest.mark.parametrize("gpu", ["modal:1x8xb300,modal:1x8xb300", "modal:1x8xb300"], ids=["two", "one"])
def test_the_synced_launch_plan_reaches_the_miles_argv_at_dp8(tmp_path, gpu):
    from test_rl_k31_synced_cpu import _plan

    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.adapters.miles.models.kimi_k31 import provider as k31_provider

    _, plan = _plan(tmp_path, gpu)
    assert plan["outer_sync"]
    model = tmp_path / "Kimi-K3-4layer"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_hf_config()))
    provider = k31_provider.kimi_k31_provider(model)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"problem": "q", "answer": "a"}\n')
    local = {"$SYNCER_ADDR": "10.0.0.1:29400",
             "~/yeto-rl-expert-selection.json": str(tmp_path / "expert_selection.json"),
             "~/yeto-rl/algorithm_spec.json": str(tmp_path / "algorithm_spec.json")}
    for k, island in enumerate(plan["island_requests"]):
        tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{**local, "$LEARNER_ID": str(k)}.get(t, t) for t in shlex.split(tail)]
        args = island_entry.parse_args(tokens)
        assert not args.rl_single_island_no_sync and args.parameter_mode == "full"
        config = rc.resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=provider,
                                          target_modules=[], verify_ref_load=False)
        assert config.parallel.data_parallel == 8 and config.parallel.expert_parallel == 8
        launch = island_entry.build_ports_launch(args, config)  # full_parameter_sync=True
        assert "use_distributed_optimizer" not in launch.runtime_attrs  # auto: sharded masters
        argv = list(launch.argv)
        assert argv[argv.index("--expert-model-parallel-size") + 1] == "8"
        assert "--overlap-param-gather" not in argv and "--optimizer-cpu-offload" not in argv


# ------------------------------------------------------------------ 4. the runtime manifest

def test_the_runtime_manifest_probes_one_interface_per_feature():
    for name, target in PROBES.items():
        assert rm.INTERFACES[name] == target, name
        assert not [c for c, need in rm.CAPABILITY_REQUIREMENTS.items() if name in need], name


def test_the_probes_resolve_in_a_k31_b2_checkout():
    pytest.importorskip("torch")
    pytest.importorskip("miles.backends.megatron_utils.full_parameter_replicas")
    state = pytest.importorskip("miles.backends.megatron_utils.full_parameter_state")
    if not hasattr(state, "full_parameter_rank_state"):
        pytest.skip("Miles checkout before k31/b2")
    from yeto.rl.engine.runtime_manifest import _probe

    for name in PROBES:
        assert _probe(*rm.INTERFACES[name]), name
    from miles.ray import full_parameter_transport as transport

    assert {"dp_replicas", "sharded_masters", "verified_apply", "shape_free_layout"} <= set(
        transport.FULL_PARAMETER_TRANSPORT_FEATURES)
