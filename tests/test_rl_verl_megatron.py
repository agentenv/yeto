"""S27 G-DC (V8/V10): the verl Megatron engine island, CPU checks.

* ``megatron_engine``: engine choice, shape rules, Hydra overrides (FSDP2 keys
  replaced, data / sampling / correction keys kept, D0 shape);
* ``island_entry``: the Megatron plan from the launcher's island flags, the
  refusals, local JSONL task files, the resume-store note;
* ``megatron_ports``: parameter ownership, the combined policy digest, the
  resident policy handle and its weights check, the NCCL publisher, and a
  real ``IslandDriver`` run over these ports (fake rollout/trainer);
* image tag variant, patch profile, V10 roles.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import megatron_engine as me
from yeto.rl.engine import backends

D0 = me.MegatronShape(trainer_nodes=1, trainer_gpus_per_node=8, rollout_nodes=1, rollout_gpus_per_node=8,
                      rollout_tp=2, tp=1, pp=1, ep=8, num_experts=128, router_replay="R3")


def _run(**kw) -> vconf.VerlRunConfig:
    base = dict(model_path="/m", train_file="/d/train.parquet", val_file="/d/test.parquet", out_dir="/o",
                parameter_mode="full", lora_rank=0, lora_alpha=0, lr=1e-6, groups_per_round=32,
                samples_per_group=16, max_prompt_len=512, max_response_len=2048, total_rounds=4,
                correction="tis", tis_upper=2.0, n_gpus=8, reward_path="/r.py", reward_name="compute_score")
    return vconf.VerlRunConfig(**{**base, **kw})


def _value(overrides, key):
    hits = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0].lstrip("+") == key]
    assert len(hits) <= 1, (key, hits)
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- engine choice and shape
def test_engine_is_megatron_exactly_when_a_model_parallel_size_is_above_one():
    assert me.engine_for() == me.ENGINE_FSDP2
    assert me.engine_for(tensor_parallel=1, pipeline_parallel=1, expert_parallel=1) == me.ENGINE_FSDP2
    for kw in (dict(expert_parallel=8), dict(tensor_parallel=2), dict(pipeline_parallel=2)):
        assert me.engine_for(**kw) == me.ENGINE_MEGATRON


@pytest.mark.parametrize("change,match", [
    (dict(cp=2, mamba=True), "没有查过"),  # S28 G-CPW: was a blanket CP>1 refusal
    (dict(etp=2), "ETP=1"),
    (dict(tp=3), "TP\\*PP\\*CP=3"),
    (dict(tp=16, trainer_nodes=2), "跨节点"),
    (dict(ep=3), "EP\\*ETP=3"),
    (dict(num_experts=100), "专家数"),
    (dict(rollout_tp=3), "rollout TP=3"),
    (dict(rollout_engine="sglang"), "vLLM"),
    (dict(dispatcher="flex"), "DeepEP"),
    (dict(router_replay="R2"), "router_replay"),
    (dict(recompute_modules=("moe", "everything")), "everything"),
    (dict(gpu_memory_utilization=1.0), "gpu_memory_utilization"),
    (dict(max_tokens_per_gpu=0), "max_tokens_per_gpu"),
])
def test_shape_rules_refuse(change, match):
    shape = dataclasses.replace(D0, **change)
    with pytest.raises(me.MegatronShapeError, match=match):
        me.validate(shape)


def test_d0_shape_is_valid_and_reports_its_sizes():
    me.validate(D0)
    d = D0.to_dict()
    assert (d["trainer_gpus"], d["rollout_gpus"], d["data_parallel"]) == (8, 8, 8)
    assert d["recompute_modules"] == ["moe", "layernorm"]


def test_megatron_island_is_full_parameter_only():
    with pytest.raises(me.MegatronShapeError, match="全参数"):
        me.build_overrides(_run(parameter_mode="lora", lora_rank=32, lora_alpha=32), D0)


# ----------------------------------------------------------------------------- overrides
def test_overrides_replace_the_fsdp_keys_and_keep_the_shared_ones():
    fsdp = vconf.build_overrides(_run())
    out = me.build_overrides(_run(), D0, eps_clip=0.2, eps_clip_high=0.28)
    assert out[0] == "model_engine=megatron"
    keys = [o.split("=", 1)[0].lstrip("+") for o in out]
    assert len(keys) == len(set(keys)), "a key is set twice"
    assert not [o for o in out if o.startswith(("actor_rollout_ref.actor.fsdp_config.",
                                               "actor_rollout_ref.actor.strategy="))]
    kept = [o for o in fsdp if o.startswith(("data.", "algorithm.", "reward.", "actor_rollout_ref.rollout.n=",
                                             "actor_rollout_ref.rollout.temperature=",
                                             "actor_rollout_ref.rollout.top_p=", "actor_rollout_ref.actor.optim.lr=",
                                             "actor_rollout_ref.actor.ppo_mini_batch_size=",
                                             "actor_rollout_ref.rollout.calculate_log_probs="))]
    assert kept and all(o in out for o in kept)
    expected = {
        "trainer.v1.trainer_mode": me.TRAINER_MODE, "actor_rollout_ref.hybrid_engine": "False",
        "trainer.v1.separate_async.parameter_sync_step": "1",
        "trainer.v1.separate_async.num_warmup_batches": "0",
        "trainer.n_gpus_per_node": "8", "trainer.nnodes": "1",
        "actor_rollout_ref.rollout.nnodes": "1", "actor_rollout_ref.rollout.n_gpus_per_node": "8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": "2",
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "nccl",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size": "8",
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": "1",
        "actor_rollout_ref.actor.megatron.use_distributed_optimizer": "True",
        "actor_rollout_ref.actor.megatron.param_offload": "False",
        "actor_rollout_ref.actor.megatron.optimizer_offload": "False",
        "actor_rollout_ref.actor.megatron.router_replay.mode": "R3",
        "actor_rollout_ref.rollout.enable_rollout_routing_replay": "True",
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity": "selective",
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules": "[moe,layernorm]",
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type": "alltoall",
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate": "0.0",
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff": "0.0",
        "actor_rollout_ref.actor.clip_ratio_low": "0.2", "actor_rollout_ref.actor.clip_ratio_high": "0.28",
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": "2.0",
    }
    for key, want in expected.items():
        assert _value(out, key) == want, key


def test_asserted_keys_are_all_set_by_the_overrides():
    out = me.build_overrides(_run(), D0)
    missing = [k for k in me.ASSERTED_KEYS if _value(out, k) is None]
    assert missing == []


def test_new_transformer_config_keys_use_force_add():
    # the fork's megatron.yaml lists only recompute_* and attention_backend: every other
    # override_transformer_config key must be added with "++" (hydra refuses "=" on a new key)
    out = me.build_overrides(_run(), D0)
    tcfg = [o for o in out if "override_transformer_config." in o]
    assert tcfg and all(o.startswith("++") for o in tcfg)


# ----------------------------------------------------------------------------- island plan
def _island_args(tmp_path, **kw):
    from yeto.rl.adapters.verl import island_entry

    argv = ["--model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--model-revision", "r",
            "--data", "~/yeto-data.jsonl", "--learner-id", "0",
            "--reward-function", "yeto.rl.numeric_answer_reward:score", "--global-rounds", "3",
            "--groups-per-round", "32", "--samples-per-group", "16", "--rollout-max-response-len", "2048",
            "--event-tape", str(tmp_path / "tape.jsonl"), "--inner-lr", "1e-06", "--seq-len", "2560",
            "--seed", "17", "--rl-single-island-no-sync", "--parameter-mode", "full",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8", "--rl-island-gpus-per-node", "8",
            "--tensor-parallel", "1", "--pipeline-parallel", "1", "--expert-parallel", "8",
            "--rollout-num-gpus-per-engine", "2", "--sglang-mem-fraction-static", "0.8",
            "--rl-placement", "fixed-partition", "--rollout-num-gpus", "8", "--use-rollout-routing-replay",
            "--rl-lr-schedule", "constant", "--trust-remote-code", "--rl-max-policy-age", "0"]
    args, ignored = island_entry.parse_args(argv + list(kw.pop("extra", [])))
    for key, value in kw.items():
        setattr(args, key, value)
    return island_entry, args, ignored


def _plan(tmp_path, **kw):
    island_entry, args, _ignored = _island_args(tmp_path, **kw)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"n_routed_experts": 128}))
    return island_entry.megatron_island_plan(
        args, model_path=str(model), train_file="/d/train.parquet", val_file="/d/test.parquet", out_dir="/o",
        algo={"correction": "tis", "tis_upper": 2.0, "tis_lower": 0.0, "eps_clip": 0.2, "eps_clip_high": 0.28},
        chat={"enable_thinking": False}, ray_info={"after": {"GPU": 16}}, reward_path="/r.py",
        reward_name="compute_score_numeric_answer")


def test_island_plan_of_the_d0_flags(tmp_path):
    plan = _plan(tmp_path)
    assert plan["shape"]["trainer_gpus"] == 8 and plan["shape"]["rollout_gpus"] == 8
    assert plan["shape"]["router_replay"] == "R3" and plan["shape"]["gpu_memory_utilization"] == 0.8
    assert plan["shape"]["num_experts"] == 128 and plan["hf_exports"] is None
    run = plan["run"]
    assert (run.parameter_mode, run.lora_rank, run.max_prompt_len, run.max_model_len) == ("full", 0, 512, 2560)
    assert _value(plan["overrides"], "actor_rollout_ref.actor.clip_ratio_high") == "0.28"
    assert _value(plan["overrides"], "data.apply_chat_template_kwargs.enable_thinking") == "False"


def test_the_gradient_nan_check_is_on_by_default_and_can_be_turned_off(tmp_path):
    """S27 G-FL: verl's DDP config leaves Megatron's check_for_nan_in_grad at False, and
    with it off a NaN gradient is scaled into every parameter by clip_grads instead of
    raising. The island asks for it through override_ddp_config unless told not to."""
    from yeto.rl import megatron_grad_checks

    assert megatron_grad_checks.CHECK_FOR_NAN_IN_GRAD is True
    assert megatron_grad_checks.ddp_overrides() == {"check_for_nan_in_grad": True}
    assert megatron_grad_checks.ddp_overrides(check_for_nan_in_grad=False) == {}

    key = "++actor_rollout_ref.actor.megatron.override_ddp_config.check_for_nan_in_grad"
    plan = _plan(tmp_path)
    assert plan["shape"]["grad_nan_check"] is True
    assert f"{key}=True" in plan["overrides"]
    off = _plan(tmp_path, extra=["--rl-no-grad-nan-check"])
    assert off["shape"]["grad_nan_check"] is False
    assert not [o for o in off["overrides"] if o.startswith(key)]
    # large-grad checking stays off: its rejection is a warning, not a guard
    assert not [o for o in plan["overrides"] if "check_for_large_grads" in o]


def test_the_island_refuses_a_trainer_without_the_gradient_nan_check():
    """S27 G-FL: the flag is read back off the DDP wrappers, so an override Hydra
    dropped fails at island start instead of running unguarded."""
    from types import SimpleNamespace

    from yeto.rl import megatron_grad_checks as gc

    class _Chunk:  # a Megatron DDP wrapper over an inner module
        def __init__(self, flag):
            self.ddp_config = SimpleNamespace(check_for_nan_in_grad=flag, check_for_large_grads=False)

    on = gc.report([_Chunk(True), _Chunk(True)])
    assert on == {"chunks": 2, "wrapped": 2, "check_for_nan_in_grad": True, "check_for_large_grads": False}
    off = gc.report([_Chunk(True), _Chunk(False)])
    assert off["check_for_nan_in_grad"] is False
    # a chunk that is not DDP-wrapped at all (forward_only engines)
    bare = gc.report([SimpleNamespace(module=SimpleNamespace())])
    assert bare == {"chunks": 1, "wrapped": 0, "check_for_nan_in_grad": False, "check_for_large_grads": False}
    # nested one level down (DDP under another wrapper)
    nested = gc.report([SimpleNamespace(module=_Chunk(True))])
    assert nested["wrapped"] == 1 and nested["check_for_nan_in_grad"] is True

    gc.check_reports([{**on, "rank": 0}], wanted=True)  # does not raise
    gc.check_reports([{**off, "rank": 1}], wanted=False)  # the run asked for no check
    with pytest.raises(gc.MegatronGradCheckError, match="without check_for_nan_in_grad on rank"):
        gc.check_reports([{**on, "rank": 0}, {**off, "rank": 1}], wanted=True)
    with pytest.raises(gc.MegatronGradCheckError, match="no Megatron DDP config"):
        gc.check_reports([{**bare, "rank": 0}], wanted=True)


def test_the_ascend_plugin_stack_is_off_unless_the_flag_asks_for_it(tmp_path):
    """S27 G-FL: no flag, no section; with the flag the plan carries the switches, the
    plugin order and the reasons, and never the clamp rewrite."""
    from yeto.rl import ascend_plugin_stack

    assert _plan(tmp_path)["ascend_plugin_stack"] is None
    section = _plan(tmp_path, extra=["--rl-ascend-plugin-stack", "/data/work/sft-run/plugins"]
                    )["ascend_plugin_stack"]
    assert section["root"] == "/data/work/sft-run/plugins"
    assert section["env"]["STACK_REWRITES"] == "cumsum,squeeze"
    assert [row["plugin"] for row in section["plugins"]] == list(ascend_plugin_stack.PLUGINS)
    assert section["excluded"] and section["unset"]
    assert all(row["present"] is False for row in section["plugins"])  # not this machine's directory


def test_island_plan_with_the_hf_export(tmp_path):
    plan = _plan(tmp_path, extra=["--rl-export-hf-rounds", "final",
                                  "--rl-export-hf-store", "modal-volume://v/run/hf"])
    assert plan["hf_exports"]["rounds"] == "final"
    assert plan["hf_exports"]["store"].startswith("modal-volume://v/run/hf")


@pytest.mark.parametrize("change,match", [
    (dict(parameter_mode="lora"), "--tuning full"),
    # S27 G-FG: a synced Megatron island is allowed; what its outer sync cannot run is refused
    (dict(rl_single_island_no_sync=False, rl_peer_recovery_store="modal-volume://v/p"), "no peer recovery"),
    (dict(rl_placement="colocated"), "fixed-partition"),
    (dict(rl_max_policy_age=1), "--rl-max-policy-age 0"),
    (dict(rl_sample_role="trainer"), "--rl-sample-role"),
    (dict(rollout_num_gpus=4), "!= 岛内 Ray 的 16 卡"),
    (dict(rl_lr_schedule="cosine"), "constant"),
    (dict(rl_export_hf_rounds="final"), "go together"),
])
def test_island_plan_refuses(tmp_path, change, match):
    with pytest.raises(SystemExit, match=match):
        _plan(tmp_path, **change)


def test_resume_store_is_noted_not_used_and_cut_flags_are_refused(tmp_path):
    island_entry, args, _ = _island_args(tmp_path)
    resume, note = island_entry.megatron_resume(args, {"store": "modal-volume://v/run"})
    assert resume is None and "HF exports only" in note
    assert island_entry.megatron_resume(args, None) == (None, None)
    args.rl_cut_every = 4
    with pytest.raises(SystemExit, match="--rl-cut-every"):
        island_entry.megatron_resume(args, {"store": "s"})


def test_fsdp_island_flags_are_unchanged_by_the_new_parser_entries(tmp_path):
    island_entry, args, ignored = _island_args(tmp_path)
    assert "--expert-parallel" not in ignored and "--rl-placement" not in ignored
    assert args.expert_parallel == 8 and args.use_rollout_routing_replay


def test_local_jsonl_task_file_becomes_verl_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from yeto.rl.adapters.verl import island_entry

    labels = ["0", "1", "2", "3", "0.5"]  # one decimal label: pandas would make the column float
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}], "label": labels[i], "metadata": {"task_id": i}}
            for i in range(5)]
    src = tmp_path / "tasks.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    train, test = island_entry.prepare_data(str(src), None, tmp_path / "out", val_rows=2)
    frame = pd.read_parquet(train)
    assert len(frame) == 5 and len(pd.read_parquet(test)) == 2
    assert frame.iloc[3]["reward_model"]["ground_truth"] == "3"  # the label text, not "3.0"
    assert list(frame.iloc[0]["prompt"]) == [{"role": "user", "content": "q0"}]


def test_numeric_answer_reward_is_mapped():
    from yeto.rl.adapters.verl import reward_fn

    name = reward_fn.function_for("yeto.rl.numeric_answer_reward:score")
    fn = getattr(reward_fn, name)
    assert fn("yeto/x", "work...\nAnswer: \\boxed{34}", "34") == 1.0
    assert fn("yeto/x", "Answer: \\boxed{35}", "34") == 0.0


# ----------------------------------------------------------------------------- ports
class _Param:
    def __init__(self, shape, allreduce=None):
        self.shape = shape
        if allreduce is not None:
            self.allreduce = allreduce


class _Chunk:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return list(self.params.items())


def test_ownership_dense_on_dp0_experts_on_edp0():
    from yeto.rl.adapters.verl import megatron_ports as mp

    chunk = _Chunk({"decoder.layers.0.mixer.in_proj.weight": _Param((4, 2)),
                    "decoder.layers.1.mlp.experts.linear_fc1.weight0": _Param((2, 2), allreduce=False),
                    "decoder.layers.1.mlp.router.weight": _Param((8, 2), allreduce=True)})
    rank0 = [k for k, _ in mp.owned_parameters([chunk], dict(tp=0, pp=0, ep=0, dp=0, edp=0))]
    rank3 = [k for k, _ in mp.owned_parameters([chunk], dict(tp=0, pp=0, ep=3, dp=3, edp=0))]
    assert rank0 == ["0:decoder.layers.0.mixer.in_proj.weight@tp0.pp0",
                     "0:decoder.layers.1.mlp.experts.linear_fc1.weight0@tp0.pp0.ep0",
                     "0:decoder.layers.1.mlp.router.weight@tp0.pp0"]
    assert rank3 == ["0:decoder.layers.1.mlp.experts.linear_fc1.weight0@tp0.pp0.ep3"]
    assert mp.is_expert_param("a.experts.b", object()) and not mp.is_expert_param("a.b", object())


def _records(version: int):
    return [{"rank": 0, "tensors": [["0:dense@tp0.pp0", [4, 2], "torch.bfloat16", f"{version:064x}"],
                                    ["0:exp@tp0.pp0.ep0", [2, 2], "torch.bfloat16", "a" * 64]],
             "bytes": 24, "seconds": 0.5},
            {"rank": 1, "tensors": [["0:exp@tp0.pp0.ep1", [2, 2], "torch.bfloat16", "b" * 64]],
             "bytes": 8, "seconds": 0.7}]


def test_combined_digest_is_order_free_and_refuses_duplicates():
    from yeto.rl.adapters.verl import megatron_ports as mp

    a, b = mp.combine_digests(_records(1)), mp.combine_digests(list(reversed(_records(1))))
    assert a == b and len(a.policy_tensor_hash) == 64 and a.payload_bytes == 32 and a.hash_seconds == 0.7
    assert a.policy_tensor_hash != mp.combine_digests(_records(2)).policy_tensor_hash
    assert [k for k, _ in a.specs] == ["0:dense@tp0.pp0", "0:exp@tp0.pp0.ep0", "0:exp@tp0.pp0.ep1"]
    with pytest.raises(ValueError, match="same parameter key"):
        mp.combine_digests(_records(1) + _records(1)[:1])
    layout = mp.layout_of([[r[:3] for r in rec["tensors"]] for rec in _records(1)])
    assert layout == a.specs


class _Manager:
    def __init__(self):
        self.calls = []

    def update_weights(self, version):
        self.calls.append(version)
        return {"sync/bytes": 32.0}


class _FakeIsland:
    """The parts of ``MegatronIsland`` the policy state and the publisher read."""

    def __init__(self, tmp_path):
        from yeto.rl.adapters.verl import megatron_ports as mp

        self.trainer = SimpleNamespace(standalone_checkpoint_manager=_Manager(), global_steps=1)
        self.learner_id, self.out, self.events = 0, Path(tmp_path), []
        self.base_model_revision, self.lora_config_hash = "rev", "c" * 64
        self.layout_hash = mp.layout_hash_of(mp.combine_digests(_records(0)).specs)
        self.trained_rounds, self.process_id, self.published = 0, "p1", None
        self.model_path = str(tmp_path)

    def call_workers(self, fn, *args):
        return _records(self.trained_rounds)

    def append_jsonl(self, name, record):
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def emit(self, event, **fields):
        self.events.append((event, fields))


def test_resident_policy_handle_and_its_weights_check(tmp_path):
    from yeto.rl.adapters.verl import megatron_ports as mp
    from yeto.rl.engine.policy_digest import WeightsChanged

    island = _FakeIsland(tmp_path)
    state = mp.MegatronPolicyState(island).export_digest(policy_version=0)
    assert state.layout == "fragments" and state.policy_version == 0
    state.check_holds()
    island.trained_rounds = 1
    with pytest.raises(WeightsChanged):
        state.check_holds()
    from yeto.rl.engine.trainable_state import UnsupportedLayoutError

    with pytest.raises(UnsupportedLayoutError):  # a fragments handle never materializes
        state.materialize()
    with pytest.raises(mp.MegatronTensorExportError):
        mp.MegatronPolicyState(island).export()
    island.layout_hash = "0" * 64
    with pytest.raises(mp.MegatronTensorExportError, match="layout changed"):
        mp.MegatronPolicyState(island).export_digest(policy_version=1)


def test_publisher_uses_the_nccl_engine_with_the_policy_version(tmp_path):
    from yeto.rl.adapters.verl import megatron_ports as mp

    island = _FakeIsland(tmp_path)
    state = mp.MegatronPolicyState(island).export_digest(policy_version=3)
    result = mp.MegatronPublisher(island).publish(state)
    assert island.trainer.standalone_checkpoint_manager.calls == [3]
    assert result.manifest.target_policy_version == 3
    assert result.manifest.target_policy_hash == state.policy_tensor_hash()
    assert island.published == (3, state.policy_tensor_hash())
    record = json.loads((tmp_path / "verl-publish-0.jsonl").read_text())
    assert record["status"] == mp.READBACK_STATUS and record["sync_metrics"] == {"sync/bytes": 32.0}
    island.trained_rounds = 1  # trained after the hash: the publisher refuses
    with pytest.raises(Exception, match="trained since"):
        mp.MegatronPublisher(island).publish(state)


@pytest.mark.parametrize("spec_file", [None, "tools/eval/efficiency/algo-eff-miles.json"])
def test_driver_runs_rounds_over_the_megatron_ports(tmp_path, spec_file):
    from yeto.rl.adapters.verl import megatron_ports as mp
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.contracts import LocalStepReceipt
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver, TrainStepMetrics, policy_token
    from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

    island = _FakeIsland(tmp_path)

    class Rollout:
        def generate(self, rollout_id, *, expected_policy_version=None):
            version, digest = island.published
            assert version == rollout_id
            token = policy_token(version, digest)
            groups = tuple(GroupMetadata(f"r{rollout_id}-g{g}", (f"r{rollout_id}-g{g}-s0", f"r{rollout_id}-g{g}-s1"),
                                         token, 0.5, 0.5, 6) for g in range(2))
            return RolloutBatchHandle(rollout_id, version, digest, groups, 2, 0, payload=None)

        def abort(self):
            return None

        def members(self):
            return frozenset({mp.MEMBER})

    class Trainer:
        publish_offloaded = False

        def onload(self):
            return None

        def offload(self):
            return None

        def train_step(self, batch):
            island.trained_rounds += 1
            ids = tuple(s for g in batch.groups for s in g.sample_ids)
            return LocalStepReceipt(
                algorithm="grpo", learner_id=0, learner_generation=0, base_policy_version=batch.policy_version,
                base_policy_hash=batch.policy_hash, input_batch_hash="a" * 64, trajectory_ids=ids,
                trained_tokens=12, optimizer_steps=1, optimizer_step_succeeded=True,
                parameter_layout_hash=island.layout_hash)

        def step_metrics(self):
            return TrainStepMetrics(grad_norm=0.5, loss=0.1, pg_loss=0.1, lr=1e-6, train_step=0,
                                    applied_lrs=(1e-6,))

    # the D0 spec (TIS, eps_clip / eps_clip_high) passes the handshake of the Megatron capabilities
    spec = AlgorithmSpec.from_dict(json.loads(Path(spec_file).read_text())) if spec_file else AlgorithmSpec()
    plan = {"groups_per_round": 2, "samples_per_group": 2}
    driver = IslandDriver(
        learner_id=0, rollout=Rollout(), trainer=Trainer(), policy_state=mp.MegatronPolicyState(island),
        publisher=mp.MegatronPublisher(island), placement=mp.MegatronPlacement(D0.to_dict()),
        capabilities=verl_capabilities("sha256:" + "1" * 64, megatron=True), algorithm=spec,
        sync=LocalOnlySync(3), events=EventTape(tmp_path / "tape.jsonl", 0),
        profile=mp._execution_profile(spec, plan), layout="fragments")
    state = driver.run()
    assert driver.rounds_completed == 3 and state.policy_version == 3
    # versions 0..3 published in order, each with the weights of its round
    assert island.trainer.standalone_checkpoint_manager.calls == [0, 1, 2, 3]
    events = [json.loads(line)["event"] for line in (tmp_path / "tape.jsonl").read_text().splitlines()]
    assert events.count("rl_publication") == 4 and "rl_driver_start" in events
    start = next(json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines()
                 if json.loads(line)["event"] == "rl_driver_start")
    assert start["placement"] == "fixed-partition" and start["execution_mode"] == "partitioned-serial"


def test_megatron_capabilities_and_identity():
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.identity import backend_identity

    caps = verl_capabilities("sha256:" + "1" * 64, megatron=True)
    assert caps.parameter_layouts == {"fragments"} and caps.placements == {"fixed-partition"}
    with pytest.raises(ValueError, match="fully_async"):
        verl_capabilities("sha256:" + "1" * 64, megatron=True, fully_async_limit=1)
    fsdp = backend_identity(device_family="nvidia", compat_group="nvidia-h200")
    meg = backend_identity(device_family="nvidia", train_backend="megatron", compat_group="nvidia-h200")
    assert fsdp.sha256() != meg.sha256()
    with pytest.raises(ValueError):
        backend_identity(device_family="nvidia", train_backend="veomni", compat_group="nvidia-h200")


# ----------------------------------------------------------------------------- image, patch, roles
def test_megatron_image_tag_variant():
    from yeto.modal_runner import ENGINE_BUILT_IMAGE_RE, image_ref_from_rl_image
    from yeto.rl.adapters.verl import launch_flags, pins

    assert pins.VERL_MEGATRON_IMAGE == f"verl-megatron-build:{pins.VERL_COMMIT}"
    built = ENGINE_BUILT_IMAGE_RE.fullmatch(pins.VERL_MEGATRON_IMAGE)
    assert (built["backend"], built["variant"], built["commit"]) == ("verl", "megatron", pins.VERL_COMMIT)
    plain = ENGINE_BUILT_IMAGE_RE.fullmatch(pins.VERL_IMAGE)
    assert (plain["backend"], plain["variant"]) == ("verl", None)
    assert image_ref_from_rl_image(pins.VERL_MEGATRON_IMAGE) == pins.VERL_MEGATRON_IMAGE
    assert launch_flags.rl_image_ok(pins.VERL_IMAGE) and launch_flags.rl_image_ok(pins.VERL_MEGATRON_IMAGE)
    assert not launch_flags.rl_image_ok("verl-megatron-build:" + "0" * 40)
    assert not launch_flags.rl_image_ok(f"verl-sglang-build:{pins.VERL_COMMIT}")


def test_megatron_image_layers_and_pins():
    from yeto.rl.adapters.verl import image, pins

    cmds = image.megatron_commands()
    assert "--inexact --extra vllm --extra fsdp --extra megatron" in cmds[0]
    assert any(f"mamba-ssm=={pins.MAMBA_SSM_VERSION}" in c for c in cmds)
    assert any(f"causal-conv1d=={pins.CAUSAL_CONV1D_VERSION}" in c for c in cmds)
    expected = pins.expected_versions_for_engine("megatron")
    assert expected["mamba_ssm"] == pins.MAMBA_SSM_VERSION and expected["vllm"] == pins.EXPECTED_VERSIONS["vllm"]
    assert pins.expected_versions_for_engine("fsdp2") == pins.EXPECTED_VERSIONS
    # S27 G-FN: an Ascend Megatron island asserts MindSpeed's stack, not the CUDA
    # image's; nothing here builds that environment, so a mismatch is a recorded
    # problem (YETO_VERL_ALLOW_UNPINNED waives it) rather than a refusal to start
    npu = pins.expected_versions_for_engine("megatron", "ascend")
    assert npu["megatron_bridge"] == "0.5.0" and npu["mindspeed"] == "0.18.0"
    assert npu["megatron_core"] != expected["megatron_core"]
    assert "vllm" not in npu or npu["vllm"] != expected.get("vllm")
    # a family with no pins at all still fails on the family pins first
    with pytest.raises(ValueError, match="no verl version pins"):
        pins.expected_versions_for_engine("megatron", "amd")
    # and a family with pins but no Megatron stack names that gap instead
    assert set(pins.EXPECTED_VERSIONS_MEGATRON_BY_FAMILY) == set(pins.EXPECTED_VERSIONS_BY_FAMILY)
    with pytest.raises(ValueError):
        image.modal_image(object(), variant="sglang")


VERL_FORK = Path("/home/michael/work/verl-fork-acad9875")


@pytest.mark.skipif(not (VERL_FORK / "verl/models/mcore/util.py").is_file(), reason="no verl fork checkout")
def test_megatron_patch_profile_sets_seq_idx_and_leaves_lora_wire_out(tmp_path):
    import shutil

    from yeto.rl.adapters.verl import patch_verl

    for rel in {t for t, *_ in patch_verl.PATCHES + patch_verl.SEQ_IDX + patch_verl.ZMQ_SUB
                + (patch_verl.AUTO_CLASS,)}:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(VERL_FORK / rel, tmp_path / rel)
    patch_verl.apply(tmp_path, "megatron")
    patch_verl.apply(tmp_path, "megatron")  # idempotent
    util = (tmp_path / patch_verl.SEQ_IDX_TARGET).read_text()
    assert util.count("**_yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu)") == 2
    assert util.count("def _yeto_seq_idx_args(") == 1
    compile(util, "util.py", "exec")
    assert patch_verl.SEND_HOOK not in (tmp_path / patch_verl.SEND_TARGET).read_text()
    engine = (tmp_path / patch_verl.ZMQ_TARGET).read_text()  # one SUB socket per publisher
    assert engine.count("_yeto_sub_key") == 2 and engine.count("time.sleep(1.0)  # yeto") == 1
    compile(engine, "nccl_checkpoint_engine.py", "exec")
    # S27 G-EQ2: the auto-class fix applies on both profiles and stays valid python
    model = (tmp_path / patch_verl.AUTO_CLASS[0]).read_text()
    assert model.count("yeto S27 G-EQ2") == 1 and patch_verl.AUTO_CLASS[1] not in model
    compile(model, "model.py", "exec")
    with pytest.raises(SystemExit, match="unknown profile"):
        patch_verl.apply(tmp_path, "nope")


def test_v10_roles_are_registered_for_verl():
    for role in ("nemotron_h_recipe", "placement", "rollout", "publish"):
        backends.module(role, "verl")
    recipe = backends.module("nemotron_h_recipe", "verl")
    provider = SimpleNamespace(hybrid_layer_pattern="MEM*E", num_moe_experts=128, mtp_num_layers=None)
    recipe.check_provider(provider, parameter_mode="full")
    assert recipe.model_flags(provider) == ()
    with pytest.raises(recipe.NemotronHRecipeError, match="MTP"):
        recipe.check_provider(SimpleNamespace(**{**vars(provider), "mtp_num_layers": 1}), parameter_mode="full")
    with pytest.raises(recipe.NemotronHRecipeError, match="VL"):
        recipe.check_provider(SimpleNamespace(**{**vars(provider), "architectures": ["NemotronH_Omni_Reasoning_V3"]}),
                              parameter_mode="full")
    placement = backends.module("placement", "verl")
    request = placement.PlacementRequest(kind="fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=2,
                                         gpus_per_node=8, expert_parallel=8)
    assert request.trainer_shape() == (1, 8)
    with pytest.raises(ValueError, match="standby"):
        placement.PlacementRequest(kind="fixed-partition", trainer_gpus=8, rollout_gpus=8,
                                   standby_gpus=8).trainer_shape()
    with pytest.raises(ValueError, match="rollout-inject|启动延迟"):
        backends.module("rollout", "verl").INJECT_START_DELAY_ENV
    with pytest.raises(ValueError, match="update-weights-block"):
        backends.module("publish", "verl").INJECT_UPDATE_BLOCK_ENV
    assert backends.module("publish", "verl").publisher_class("megatron").__name__ == "MegatronPublisher"


def test_launcher_allows_the_hf_export_on_the_megatron_island_only():
    from yeto.launch import rl_ports

    base = dict(rl_export_hf_rounds="final", rl_checkpoint_store="modal-volume://v/x", rl_backend="verl",
                rl_elastic=False, total_steps=4, local_rl_rounds_per_sync=1, tensor_parallel=1,
                pipeline_parallel=1, expert_parallel=8)
    rl_ports._check_hf_export_launch_flags(argparse.Namespace(**base))
    with pytest.raises(ValueError, match="verl Megatron"):
        rl_ports._check_hf_export_launch_flags(argparse.Namespace(**{**base, "expert_parallel": 1}))


def test_worker_digest_hashes_the_owned_bf16_bytes(monkeypatch):
    import hashlib

    torch = pytest.importorskip("torch")
    from yeto.rl.adapters.verl import megatron_ports as mp

    layer = torch.nn.Linear(3, 2).to(torch.bfloat16)
    expert = torch.nn.Parameter(torch.arange(4, dtype=torch.bfloat16).reshape(2, 2))
    expert.allreduce = False
    layer.register_parameter("experts_w", expert)
    worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(module=[layer])))
    monkeypatch.setattr(mp, "_parallel_coordinates", lambda: dict(tp=0, pp=0, ep=2, dp=1, edp=0))
    record = mp.remote_megatron_digest(worker)
    keys = [row[0] for row in record["tensors"]]
    assert keys == ["0:experts_w@tp0.pp0.ep2"]  # dp rank 1: dense weights are owned by dp rank 0
    raw = expert.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
    assert record["tensors"][0][3] == hashlib.sha256(raw).hexdigest() and record["bytes"] == 8
    monkeypatch.setattr(mp, "_parallel_coordinates", lambda: dict(tp=0, pp=0, ep=0, dp=0, edp=0))
    rows = mp.remote_megatron_digest(worker)["tensors"]
    assert [r[0] for r in rows] == ["0:weight@tp0.pp0", "0:bias@tp0.pp0", "0:experts_w@tp0.pp0.ep0"]
    assert mp.remote_megatron_layout(worker) == [r[:3] for r in rows]


def test_master_ports_are_drawn_at_random_below_the_ephemeral_range():
    import random
    import socket

    from yeto.rl.adapters.verl import ray_ports

    ports = {ray_ports.pick_free_port(rng=random.Random(seed)) for seed in range(20)}
    assert len(ports) > 15 and all(ray_ports.PORT_RANGE[0] <= p < ray_ports.PORT_RANGE[1] for p in ports)
    with pytest.raises(RuntimeError, match="no free port"):  # the only port in the range is taken
        with socket.socket() as busy:
            busy.bind(("", 0))
            taken = busy.getsockname()[1]
            busy.listen()
            ray_ports.pick_free_port(taken, taken + 1)


def test_hf_export_on_the_trainer_ranks_leaves_the_marker_and_reaches_the_store(tmp_path, monkeypatch):
    """--rl-export-hf-rounds on the Megatron island (s27-dc-train1 round 3 failed with "the writer
    left no .complete"): rank 0 of the real ``remote_hf_export`` writes the round, leaves the
    marker after every rank finished, and ``HfRoundExports`` puts it into the store; another
    rank only joins the collective write."""
    torch = pytest.importorskip("torch")
    import tempfile

    from yeto.rl.adapters.verl import megatron_ports as mp
    from yeto.rl.engine.resume import STORE_IMPL_ENV

    monkeypatch.delenv(STORE_IMPL_ENV, raising=False)
    staging_root = tmp_path / "tmp"
    staging_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging_root))
    model = tmp_path / "base"
    model.mkdir()
    for name in ("config.json", "tokenizer.json"):
        (model / name).write_text("{}")
    (model / "model-00001-of-00013.safetensors").write_bytes(b"base")  # base weights are not copied

    class _Bridge:
        def save_hf_weights(self, module, path):
            (Path(path) / "model-00001-of-00001.safetensors").write_bytes(b"w" * 8)

    worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(module=[object()], bridge=_Bridge())))
    store = tmp_path / "store"
    config = {"rounds": "1,final", "store": str(store)}
    record = mp.remote_hf_export(worker, config, 0, 1, True, str(model))
    assert record["ok"] and record["round"] == 1 and record["files"] == 3, record
    out = store / "round-1"
    assert (out / ".complete").is_file() and (out / "config.json").is_file()
    assert (out / "model-00001-of-00001.safetensors").read_bytes() == b"w" * 8
    assert not (out / "model-00001-of-00013.safetensors").exists()
    assert json.loads((store / "final.json").read_text())["path"] == "round-1"
    assert not (staging_root / mp.HF_STAGING / "round-1").exists()  # the staging copy is deleted

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    assert mp.remote_hf_export(worker, config, 1, 2, False, str(model)) is None
    joined = staging_root / mp.HF_STAGING / "round-2"
    assert (joined / "model-00001-of-00001.safetensors").is_file()
    assert not (joined / ".complete").exists() and not (joined / "config.json").exists()


# ------------------------------------------------- S28 G-120C: fused log-prob kernels
def test_fused_logprob_kernels_are_off_by_default_and_written_from_the_shape():
    off = me.build_overrides(_run(), D0)
    assert _value(off, "actor_rollout_ref.model.use_fused_kernels") == "False"
    on = me.build_overrides(_run(), dataclasses.replace(D0, fused_logprob_kernels=True))
    assert _value(on, "actor_rollout_ref.model.use_fused_kernels") == "True"


def test_fused_logprob_kernels_key_is_asserted():
    # the engine wrote this key for months with nothing checking that verl took it;
    # it decides whether the [tokens, vocab] logits are materialised
    assert "actor_rollout_ref.model.use_fused_kernels" in me.ASSERTED_KEYS


def test_override_value_sees_through_hydras_add_prefix():
    # S28 G-120 handoff section 6 item 12: every asserted key written with "++" read
    # back as want:null, so check_asserted skipped it -- a read-back, not an assertion
    from yeto.rl.adapters.verl.island_entry import _override_value

    key = "actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion"
    assert _override_value([f"++{key}=False"], key) == "False"
    assert _override_value([f"+{key}=False"], key) == "False"
    assert _override_value([f"{key}=False"], key) == "False"
    assert _override_value(["actor_rollout_ref.model.lora_rank=0"], key) is None


def test_every_asserted_key_of_a_megatron_island_gets_a_want(tmp_path):
    # the same failure as above, end to end: the plan's asserted map must carry a value
    # for every key the engine writes, including the "++" ones
    from yeto.rl.adapters.verl.island_entry import _override_value

    out = me.build_overrides(_run(), D0)
    nulls = [k for k in me.ASSERTED_KEYS if _override_value(out, k) is None]
    assert nulls == []
