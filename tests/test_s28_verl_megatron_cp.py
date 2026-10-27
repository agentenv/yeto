"""S28 G-CPW: context parallelism on the verl Megatron island, and the seq_idx gate.

* the seq_idx probe of the verl module a process imports (``probe_seq_idx``) and
  the gate on it (``seq_idx_problem``, ``validate``): a Mamba model is refused
  without the patch in effect, at CP=1 as well, and at CP>1 without a CP-aware one;
* the island flag ``--context-parallel``, the plan's shape and overrides, the
  asserted key, the token budget ``max_tokens_per_gpu x CP``;
* the workers' runtime check (parallel sizes, their own seq_idx probe);
* ownership under CP: CP ranks hold the same parameters, so one owns them.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from yeto.rl.adapters.verl import megatron_engine as me
from yeto.rl.adapters.verl import patch_verl

CP_KEY = "actor_rollout_ref.actor.megatron.context_parallel_size"
NANO_PATTERN = "MEMEM*EMEMEM*E"  # Mamba (M), MoE (E), attention (*)
D0 = me.MegatronShape(trainer_nodes=1, trainer_gpus_per_node=8, rollout_nodes=1, rollout_gpus_per_node=8,
                      rollout_tp=2, tp=1, pp=1, ep=8, num_experts=128)


def _util(*, patched: bool = True, supports_total_tokens: bool = True, cp_aware: bool | None = None) -> ModuleType:
    """A stand-in for ``verl/models/mcore/util.py`` over a stubbed Megatron-Core check.

    ``patched``: the real helper text of ``patch_verl`` (fifth patch). ``cp_aware``
    replaces it with a helper of a known CP behaviour: True passes the pack's
    ``total_tokens`` at every CP size (the S28 CP2 helper), False at CP=1 only (the
    S27 helper). So these tests do not depend on which of the two this branch's
    ``patch_verl`` carries."""
    module = ModuleType(me.SEQ_IDX_MODULE)
    module.__file__ = "/verl/verl/models/mcore/util.py"
    module._packed_seq_params_supports = lambda field: supports_total_tokens and field == "total_tokens"
    if cp_aware is not None:
        def helper(cp_size, seqlens_in_batch_padded_cpu):
            if (cp_size != 1 and not cp_aware) or not module._packed_seq_params_supports("total_tokens"):
                return {}
            return {"total_tokens": int(sum(seqlens_in_batch_padded_cpu))}

        module._yeto_seq_idx_args = helper
    elif patched:
        exec(patch_verl.SEQ_IDX_HELPER, module.__dict__)  # noqa: S102 - the patch's own text
    return module


PATCHED = me.probe_seq_idx(1, _util())
CP_AWARE = me.probe_seq_idx(2, _util(cp_aware=True))
CP1_ONLY = me.probe_seq_idx(2, _util(cp_aware=False))
UNPATCHED = me.probe_seq_idx(1, _util(patched=False))


# ----------------------------------------------------------------------------- the probe
def test_probe_calls_the_helper_of_the_module_at_the_islands_cp():
    assert (PATCHED.cp, PATCHED.ready, PATCHED.error) == (1, True, None)
    assert PATCHED.source == "/verl/verl/models/mcore/util.py"
    assert (CP_AWARE.cp, CP_AWARE.ready) == (2, True)
    assert not CP1_ONLY.ready and "CP=1 only" in CP1_ONLY.error
    assert me.probe_seq_idx(1, _util(cp_aware=False)).ready
    assert not UNPATCHED.ready and me.SEQ_IDX_HELPER in UNPATCHED.error


def test_probe_refuses_a_helper_that_sets_no_total_tokens():
    # the helper is there, but this Megatron-Core's PackedSeqParams has no total_tokens
    patch = me.probe_seq_idx(1, _util(supports_total_tokens=False))
    assert not patch.ready and "lack total_tokens" in patch.error


def test_probe_reports_a_helper_whose_signature_changed():
    module = _util(patched=False)
    module._yeto_seq_idx_args = lambda cp_size: {}
    patch = me.probe_seq_idx(1, module)
    assert not patch.ready and "raised TypeError" in patch.error


def test_probe_reports_a_verl_it_cannot_import(monkeypatch):
    monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, None)  # import raises ImportError
    patch = me.probe_seq_idx(2)
    assert (patch.source, patch.cp, patch.ready) == (None, 2, False) and "cannot import" in patch.error


def test_probe_imports_the_module_the_process_runs(monkeypatch):
    monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, _util())
    assert me.probe_seq_idx(1).ready


def test_the_probe_reads_this_branchs_patch_text_as_it_is():
    """The patch text of this branch, applied: ready at CP=1, and at CP=2 exactly when
    its helper passes total_tokens there (the S28 CP2 change does, the S27 helper does
    not). So merging that change opens the gate with no change here."""
    module = _util()
    assert me.probe_seq_idx(1, module).ready
    passes = module._yeto_seq_idx_args(2, [3, 5]) == {"total_tokens": 8}
    assert me.probe_seq_idx(2, module).ready is passes


# ----------------------------------------------------------------------------- the gate
def test_a_mamba_model_needs_the_patch_in_effect_at_cp1():
    me.validate(dataclasses.replace(D0, mamba=True, seq_idx=PATCHED))
    for patch, match in ((UNPATCHED, "CP=1 下需要 seq_idx 补丁.*patch_verl.py --profile megatron"),
                         (None, "没有查过")):
        with pytest.raises(me.MegatronShapeError, match=match):
            me.validate(dataclasses.replace(D0, mamba=True, seq_idx=patch))


def test_cp2_needs_the_cp_aware_patch_on_a_mamba_model():
    shape = dataclasses.replace(D0, cp=2, mamba=True)
    with pytest.raises(me.MegatronShapeError, match="CP=2 下需要 seq_idx 补丁.*CP=1 only"):
        me.validate(dataclasses.replace(shape, seq_idx=CP1_ONLY))
    # a probe made at another CP size proves nothing about this one
    with pytest.raises(me.MegatronShapeError, match="按 CP=1 查的"):
        me.validate(dataclasses.replace(shape, seq_idx=PATCHED))
    me.validate(dataclasses.replace(shape, seq_idx=CP_AWARE))
    assert dataclasses.replace(shape, seq_idx=CP_AWARE).to_dict()["data_parallel"] == 4


def test_a_model_without_mamba_layers_takes_cp_without_the_patch():
    me.validate(dataclasses.replace(D0, cp=2))
    me.validate(dataclasses.replace(D0, cp=2, seq_idx=UNPATCHED))


def test_the_layer_pattern_decides_mamba():
    assert me.has_mamba({"hybrid_override_pattern": NANO_PATTERN})
    assert me.has_mamba({"layers_block_type": ["mamba", "moe", "attention"]})  # the 120B text model
    assert not me.has_mamba({"hybrid_override_pattern": "*E*E"})
    assert not me.has_mamba({"n_routed_experts": 128})  # no hybrid pattern at all


def test_context_parallel_alone_selects_the_megatron_engine():
    from yeto.rl.adapters.verl import run_config_rules

    assert me.engine_for(context_parallel=2) == me.ENGINE_MEGATRON
    assert me.engine_for(context_parallel=1) == me.ENGINE_FSDP2
    assert run_config_rules.megatron_run(SimpleNamespace(context_parallel=2))
    assert not run_config_rules.megatron_run(SimpleNamespace(context_parallel=1))


# ----------------------------------------------------------------------------- the island plan
def _plan(tmp_path, monkeypatch, *, extra=(), config=None, util=None):
    from yeto.rl.adapters.verl import island_entry

    if util is not None:
        monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, util)
    argv = ["--model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--model-revision", "r",
            "--data", "~/yeto-data.jsonl", "--learner-id", "0",
            "--reward-function", "yeto.rl.numeric_answer_reward:score", "--global-rounds", "3",
            "--groups-per-round", "32", "--samples-per-group", "16", "--rollout-max-response-len", "2048",
            "--event-tape", str(tmp_path / "tape.jsonl"), "--inner-lr", "1e-06", "--seq-len", "2560",
            "--seed", "17", "--rl-single-island-no-sync", "--parameter-mode", "full",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8", "--rl-island-gpus-per-node", "8",
            "--tensor-parallel", "1", "--pipeline-parallel", "1", "--expert-parallel", "8",
            "--rollout-num-gpus-per-engine", "2", "--sglang-mem-fraction-static", "0.8",
            "--rl-placement", "fixed-partition", "--rollout-num-gpus", "8",
            "--rl-lr-schedule", "constant", "--rl-max-policy-age", "0", *extra]
    args, ignored = island_entry.parse_args(argv)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(config or {"n_routed_experts": 128}))
    plan = island_entry.megatron_island_plan(
        args, model_path=str(model), train_file="/d/train.parquet", val_file="/d/test.parquet", out_dir="/o",
        algo={"correction": "tis", "tis_upper": 2.0, "tis_lower": 0.0}, chat={},
        ray_info={"after": {"GPU": 16}}, reward_path="/r.py", reward_name="compute_score_numeric_answer")
    return island_entry, plan, ignored


MAMBA = {"n_routed_experts": 128, "hybrid_override_pattern": NANO_PATTERN, "mtp_num_layers": 0}


def test_the_island_takes_the_neutral_context_parallel_flag(tmp_path, monkeypatch):
    island_entry, plan, ignored = _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2"])
    assert "--context-parallel" not in ignored and "2" not in ignored
    assert (plan["shape"]["cp"], plan["shape"]["data_parallel"]) == (2, 4)
    assert island_entry._override_value(plan["overrides"], CP_KEY) == "2"
    assert plan["shape"]["mamba"] is False and plan["shape"]["seq_idx"] is None
    _entry, default, _ = _plan(tmp_path, monkeypatch)
    assert default["shape"]["cp"] == 1 and island_entry._override_value(default["overrides"], CP_KEY) == "1"


def _check_asserted(monkeypatch):
    """``trainer.check_asserted`` without verl: the module imports verl only through
    ``advantage`` (the estimator registration) at module level."""
    import importlib

    name = "yeto.rl.adapters.verl.trainer"
    if name not in sys.modules:
        monkeypatch.setitem(sys.modules, "yeto.rl.adapters.verl.advantage", ModuleType("advantage"))
        module = importlib.import_module(name)
        # not left behind over the stub (monkeypatch.delitem would restore it at teardown)
        del sys.modules[name]
        package = sys.modules["yeto.rl.adapters.verl"]
        if getattr(package, "trainer", None) is module:
            delattr(package, "trainer")
        return module.check_asserted
    return sys.modules[name].check_asserted


def test_the_cp_key_is_a_real_assertion(tmp_path, monkeypatch):
    """The island builds ``asserted`` as ``{key: override value}``; a key written with
    "++" reads back as want None and asserts nothing (S28 G-120). This one has a want,
    so a resolved config that disagrees fails the island."""
    check_asserted = _check_asserted(monkeypatch)
    island_entry, plan, _ = _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2"])
    assert CP_KEY in plan["asserted_keys"]
    asserted = {key: island_entry._override_value(plan["overrides"], key) for key in plan["asserted_keys"]}
    assert asserted[CP_KEY] == "2"
    cfg = {"actor_rollout_ref": {"actor": {"megatron": {"context_parallel_size": 1}}}}
    with pytest.raises(RuntimeError, match="context_parallel_size: want '2' got 1"):
        check_asserted({CP_KEY: asserted[CP_KEY]}, cfg)
    cfg["actor_rollout_ref"]["actor"]["megatron"]["context_parallel_size"] = 2
    check_asserted({CP_KEY: asserted[CP_KEY]}, cfg)


def test_the_island_refuses_a_mamba_model_on_an_unpatched_verl(tmp_path, monkeypatch):
    with pytest.raises(me.MegatronShapeError, match="seq_idx 补丁.*patch_verl.py --profile megatron"):
        _plan(tmp_path, monkeypatch, config=MAMBA, util=_util(patched=False))
    _entry, plan, _ = _plan(tmp_path, monkeypatch, config=MAMBA, util=_util())
    assert plan["shape"]["mamba"] is True
    assert plan["shape"]["seq_idx"] == {"source": "/verl/verl/models/mcore/util.py", "cp": 1, "ready": True,
                                        "error": None}


def test_the_island_takes_cp2_on_a_mamba_model_only_with_the_cp_aware_patch(tmp_path, monkeypatch):
    with pytest.raises(me.MegatronShapeError, match="CP=2 下需要 seq_idx 补丁"):
        _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2"], config=MAMBA, util=_util(cp_aware=False))
    _entry, plan, _ = _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2"], config=MAMBA,
                            util=_util(cp_aware=True))
    assert plan["shape"]["cp"] == 2 and plan["shape"]["seq_idx"]["cp"] == 2 and plan["shape"]["seq_idx"]["ready"]


def test_the_token_budget_counts_context_parallel(tmp_path, monkeypatch):
    """A row is 512 + 2048 = 2560 tokens here. verl packs max_tokens_per_gpu x CP
    tokens per micro-batch and asserts the longest sequence fits."""
    _entry, plan, _ = _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2",
                                                          "--rl-max-tokens-per-gpu", "1280"])
    assert plan["shape"]["max_tokens_per_gpu"] == 1280
    with pytest.raises(me.MegatronShapeError, match="x CP 2 = 2558"):
        _plan(tmp_path, monkeypatch, extra=["--context-parallel", "2", "--rl-max-tokens-per-gpu", "1279"])
    with pytest.raises(me.MegatronShapeError, match="x CP 1 = 1280"):
        _plan(tmp_path, monkeypatch, extra=["--rl-max-tokens-per-gpu", "1280"])


# ----------------------------------------------------------------------------- the workers' runtime check
def _report(rank, *, cp=2, patch=CP_AWARE):
    return {"rank": rank, "sizes": {"tp": 1, "pp": 1, "ep": 8, "cp": cp},
            "seq_idx": dataclasses.asdict(patch)}


def test_runtime_check_compares_every_rank_with_the_plan():
    shape = dataclasses.replace(D0, cp=2, mamba=True, seq_idx=CP_AWARE).to_dict()
    assert me.runtime_problems(shape, [_report(0), _report(1)]) == []
    wrong_cp = me.runtime_problems(shape, [_report(0), _report(1, cp=1)])
    assert wrong_cp == ["rank 1: cp world size 1, the plan has 2"]
    # the driver's verl is patched, a worker's is not (another node's tree)
    unpatched = me.runtime_problems(shape, [_report(0), _report(1, patch=UNPATCHED)])
    assert len(unpatched) == 1 and unpatched[0].startswith("rank 1: 含 Mamba 层")
    not_cp_aware = me.runtime_problems(shape, [_report(0, patch=CP1_ONLY)])
    assert len(not_cp_aware) == 1 and "CP=2 下需要" in not_cp_aware[0]
    # a model without Mamba layers asks the workers for the sizes only
    plain = dataclasses.replace(D0, cp=2).to_dict()
    assert me.runtime_problems(plain, [_report(0, patch=UNPATCHED)]) == []


def _fake_megatron(monkeypatch, *, cp_rank=0, cp=2):
    """``megatron.core.parallel_state`` of one rank of TP1/PP1/EP8/CP2 (DP x CP = 2)."""
    mpu = ModuleType("megatron.core.parallel_state")
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_pipeline_model_parallel_world_size = lambda: 1
    mpu.get_expert_model_parallel_world_size = lambda: 8
    mpu.get_context_parallel_world_size = lambda: cp
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.get_pipeline_model_parallel_rank = lambda: 0
    mpu.get_expert_model_parallel_rank = lambda: cp_rank
    mpu.get_expert_data_parallel_rank = lambda: 0
    mpu.get_data_parallel_rank = lambda with_context_parallel=False: cp_rank if with_context_parallel else 0
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    megatron = ModuleType("megatron")
    megatron.core = core
    for name, module in (("megatron", megatron), ("megatron.core", core), ("megatron.core.parallel_state", mpu)):
        monkeypatch.setitem(sys.modules, name, module)


def test_the_worker_reports_its_own_sizes_and_its_own_verl(monkeypatch):
    from yeto.rl.adapters.verl import megatron_ports as mp

    _fake_megatron(monkeypatch)
    monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, _util(cp_aware=False))
    report = mp.remote_megatron_runtime({})
    assert report["sizes"] == {"tp": 1, "pp": 1, "ep": 8, "cp": 2}
    assert (report["seq_idx"]["cp"], report["seq_idx"]["ready"]) == (2, False)  # probed at its own CP

    tape = []
    shape = dataclasses.replace(D0, cp=2, mamba=True, seq_idx=CP_AWARE).to_dict()
    with pytest.raises(mp.MegatronRuntimeShapeError, match="rank 0: 含 Mamba 层"):
        mp._check_runtime_shape(lambda fn: [fn({})], shape, tape)
    assert tape[0]["event"] == "rl_megatron_runtime_shape"
    monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, _util(cp_aware=True))
    mp._check_runtime_shape(lambda fn: [fn({})], shape, tape)


def test_cp_ranks_hold_the_same_parameters_and_one_of_them_owns_them(monkeypatch):
    """Megatron replicates the parameters over CP (the gradients reduce over DP x CP),
    so a dense parameter is owned on DP-with-CP rank 0 only: the policy digest and
    the layout count it once (combine_digests refuses a key two ranks own)."""
    from yeto.rl.adapters.verl import megatron_ports as mp

    class _Chunk:
        def named_parameters(self):
            return [("decoder.layers.0.mixer.in_proj.weight", SimpleNamespace(shape=(4, 2)))]

    owned = []
    for cp_rank in (0, 1):
        _fake_megatron(monkeypatch, cp_rank=cp_rank)
        owned += [key for key, _ in mp.owned_parameters([_Chunk()], mp._parallel_coordinates())]
    assert owned == ["0:decoder.layers.0.mixer.in_proj.weight@tp0.pp0"]


def test_the_buffer_trainer_checks_the_runtime_shape_before_anything_else(monkeypatch):
    """``island_megatron_checks`` is the buffer trainer's start-up readback (the 120B
    path). A worker whose Megatron built CP=1 for a CP=2 plan stops it there."""
    from yeto.rl.adapters.verl import megatron_ports as mp

    _fake_megatron(monkeypatch, cp=1)
    monkeypatch.setitem(sys.modules, me.SEQ_IDX_MODULE, _util(cp_aware=True))
    island = SimpleNamespace(call_workers=lambda fn: [fn({})])
    plan = {"megatron": dataclasses.replace(D0, cp=2, mamba=True, seq_idx=CP_AWARE).to_dict()}
    tape = []
    with pytest.raises(mp.MegatronRuntimeShapeError, match="cp world size 1, the plan has 2"):
        mp.island_megatron_checks(island, plan, tape)
    assert [row["event"] for row in tape] == ["rl_megatron_runtime_shape"]
