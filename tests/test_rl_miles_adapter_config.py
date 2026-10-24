"""CPU tests for miles_adapter.config (task 3.1)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yeto.rl.engine import run_config as rc
from yeto.rl.adapters.miles import lr_schedule as lrs
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AlgorithmSpec,
)
from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles.placement import PlacementRewriteError


def make_config(*, colocated: bool = True, with_eval: bool = False, with_moe: bool = False, **top):
    geometry = rc.ModelGeometry(
        num_layers=4, hidden_size=64, num_attention_heads=4, num_query_groups=2,
        kv_channels=16, ffn_hidden_size=128, max_position_embeddings=4096,
        vocab_size=1000, normalization="RMSNorm", norm_epsilon=1e-6,
        position_embedding_type="rope", rope_type=None, rotary_base=10000,
        rotary_percent=1.0, mrope_section=None, gated_linear_unit=True,
        untie_embeddings_and_output_weights=True, disable_bias_linear=True,
        add_qkv_bias=False, qk_layernorm=True, multi_latent_attention=False,
        mla_dims=(),
        moe=rc.MoeGeometry(8, 32, 2, 1, None) if with_moe else None,
    )
    cfg = rc.RLRunConfig(
        hf_checkpoint="/models/qwen", ref_load="/models/qwen",
        model_recipe=rc.ModelRecipe(
            name=rc.RECIPE_GENERIC, provider_class="Qwen3ModelProvider",
            training_attention_backend="flash", gdn=rc.GdnRecipe(gated_delta_net=False),
        ),
        geometry=geometry,
        parallel=rc.ParallelLayout(
            actor_num_nodes=1, actor_num_gpus_per_node=2, tensor_parallel=1,
            pipeline_parallel=1, expert_parallel=1, data_parallel=2,
            rollout_num_gpus_per_engine=1,
            dedicated_rollout_gpus=None if colocated else 2,
            visible_gpus_per_node=4 if not colocated else 2, uneven_pipeline_layers=None,
        ),
        trainable=rc.TrainableConfig(
            parameter_mode="lora", lora_rank=8, lora_targets="attention",
            target_modules=("linear_qkv", "linear_proj"), expert_full_count=0,
        ),
        data=rc.DataConfig(
            prompt_path="/data/p.jsonl",
            columns=rc.DatasetColumns(prompt_column="problem", label_column="answer"),
            apply_chat_template=True, chat_template_kwargs={"enable_thinking": False},
        ),
        batch=rc.BatchConfig(
            seq_len=2048, global_rounds=3, eval_only=False, groups_per_round=4,
            samples_per_group=4, over_sampling_batch_size=4, optimizer_steps=1,
            global_batch=16, rollout_max_response_len=1024,
        ),
        algorithm=rc.AlgorithmConfig(
            advantage_estimator="grpo", reward_function="yeto.rl.rewards:math", lr=1e-5,
            seed=1, rollout_seed=2, lr_schedule=rc.LrSchedule("linear", 3),
        ),
        eval=rc.EvalConfig(interval=2, dataset_name="math", prompt_path="/data/e.jsonl",
                           samples_per_prompt=1, temperature=0.0) if with_eval else None,
        serving=rc.ServingConfig(mem_fraction_static=0.5, deterministic_inference=True,
                                 offload_train=True),
        ports=rc.NetworkPorts(),
        agent=rc.AgentConfig(
            custom_generate_function_path=None, custom_agent_function_path=None,
            agent_max_seq_len=None, dynamic_sampling_filter_path=None,
            use_rollout_routing_replay=False, use_session_server=False,
            session_server_ip=None, session_server_port=None, tito_model=None,
            tito_allowed_append_roles=None,
        ),
        yeto_policy_sync=True, distributed_timeout_minutes=30,
    )
    return dataclasses.replace(cfg, **top) if top else cfg


def sub(cfg, section, **kw):
    return dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **kw)})


def flag_value(argv, flag):
    i = argv.index(flag)
    return argv[i + 1]


def test_colocated_translation_core_flags():
    spec = AlgorithmSpec(kl_coef=0.0)
    launch = mc.translate_run_config(make_config(), spec)
    argv = list(launch.argv)
    assert argv[0] == "train.py"
    assert "--colocate" in argv and "--rollout-num-gpus" not in argv
    assert flag_value(argv, "--advantage-estimator") == "grpo"
    assert flag_value(argv, "--kl-coef") == "0.0"
    assert flag_value(argv, "--rollout-all-samples-process-path") == mc.ROLLOUT_META_HOOK_PATH
    assert flag_value(argv, "--rollout-sample-filter-path") == mc.TRAINED_GROUPS_HOOK_PATH
    assert flag_value(argv, "--lora-rank") == "8"
    assert flag_value(argv, "--target-modules") == "linear_qkv,linear_proj"
    assert flag_value(argv, "--custom-rm-path") == "yeto.rl.rewards.math"
    assert flag_value(argv, "--input-key") == "messages"
    assert flag_value(argv, "--label-key") == "label"
    assert "--external-policy-sync-path" not in argv  # driver owns sync
    assert "--rollout-function-path" not in argv
    assert not set(argv) & mc.FAULT_TOLERANCE_FLAGS
    assert launch.algorithm_sha256 == spec.sha256()
    assert launch.placement.kind == "colocated"
    assert launch.placement.trainer_gpus == launch.placement.rollout_gpus == 2


def test_fixed_partition_translation():
    # rl-infra-spec 2.1: a LoRA fixed partition keeps the trainer resident
    # (NCCL broadcast publish every round); offload_train is refused below.
    cfg = sub(make_config(colocated=False), "serving", offload_train=False)
    launch = mc.translate_run_config(cfg, AlgorithmSpec())
    argv = list(launch.argv)
    assert "--colocate" not in argv
    assert flag_value(argv, "--rollout-num-gpus") == "2"
    assert launch.placement.kind == "fixed-partition"
    assert launch.placement.expected_layout() == (4, 2)
    assert flag_value(argv, "--update-weight-transfer-mode") == "broadcast"
    assert "--lora-base-cpu-backup" not in argv
    assert "--yeto-placement-map" not in argv  # no standby: upstream offset layout


def test_lora_fixed_partition_refuses_trainer_offload_and_emits_standby_map():
    with pytest.raises(mc.MilesConfigError, match="resident"):
        mc.translate_run_config(make_config(colocated=False), AlgorithmSpec())
    cfg = sub(make_config(colocated=False), "serving", offload_train=False)
    cfg = sub(cfg, "parallel", standby_gpus=2, visible_gpus_per_node=6)
    launch = mc.translate_run_config(cfg, AlgorithmSpec())
    argv = list(launch.argv)
    assert json.loads(flag_value(argv, "--yeto-placement-map")) == {
        "trainer": [0, 1], "rollout": [2, 3], "standby": [4, 5]
    }
    assert launch.placement.standby_gpus == 2
    # colocated argv is unchanged by the partition work
    colocated = list(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--lora-base-cpu-backup" in colocated
    assert "--update-weight-transfer-mode" not in colocated


def test_colocated_forces_offload_train():
    # Upstream colocate + LoRA base CPU backup needs torch_memory_saver (GPU
    # acceptance 2026-09-29: update_weights failed on LD_PRELOAD without it).
    colocated = list(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--offload-train" in colocated and "--no-offload-train" not in colocated
    cfg = sub(make_config(colocated=False), "serving", offload_train=False)
    partition = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert "--no-offload-train" in partition


def test_router_port_derives_disjoint_coresident_ports():
    argv = list(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--router-prometheus-port" not in argv and "--worker-dynamic-port-start" not in argv
    island = []
    for router in (21000, 22000):
        cfg = sub(make_config(), "ports", sglang_router_port=router)
        argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
        assert flag_value(argv, "--sglang-router-port") == str(router)
        island.append(
            (flag_value(argv, "--router-prometheus-port"), flag_value(argv, "--worker-dynamic-port-start"))
        )
    assert island == [("21001", "21100"), ("22001", "22100")]


def test_eval_and_moe_and_qwen35_recipe_map():
    cfg = make_config(with_eval=True, with_moe=True)
    cfg = dataclasses.replace(
        cfg,
        model_recipe=dataclasses.replace(
            cfg.model_recipe, name=rc.RECIPE_QWEN3_5, gdn=rc.GdnRecipe(gated_delta_net=True)
        ),
    )
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert flag_value(argv, "--model-name") == "qwen3_5"
    assert argv[argv.index("--spec") + 1 : argv.index("--spec") + 3] == list(mc.QWEN3_5_SPEC)
    assert flag_value(argv, "--qkv-format") == "bshd"
    assert flag_value(argv, "--num-experts") == "8"
    assert flag_value(argv, "--eval-interval") == "2"
    assert "--eval-function-path" not in argv


def test_bounded_filter_and_runtime_attrs():
    spec = AlgorithmSpec(
        dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER, dynamic_sampling_max_replacements=3
    )
    cfg = sub(make_config(), "agent", dynamic_sampling_filter_path=STOCK_NONZERO_STD_FILTER)
    launch = mc.translate_run_config(cfg, spec)
    assert flag_value(list(launch.argv), "--dynamic-sampling-filter-path") == (
        "yeto.rl.filters.bounded_nonzero_reward_std")  # 4.4: Miles spelling of the neutral name
    assert launch.runtime_attrs == {"yeto_rl_dynamic_sampling_max_replacements": 3}
    ns = mc.apply_runtime_attrs(SimpleNamespace(), launch)
    assert ns.yeto_rl_dynamic_sampling_max_replacements == 3


def test_algorithm_disagreement_rejected():
    with pytest.raises(mc.UnmappedConfigError, match="dynamic_sampling_filter_path"):
        mc.translate_run_config(
            sub(make_config(), "agent", dynamic_sampling_filter_path="x.y.z"), AlgorithmSpec()
        )
    cfg = sub(make_config(), "algorithm", advantage_estimator="ppo")
    with pytest.raises(mc.UnmappedConfigError, match="advantage_estimator"):
        mc.translate_run_config(cfg, AlgorithmSpec())


@pytest.mark.parametrize(
    "section, field, value, path",
    [
        ("ports", "train_master_base_port", 20000, "ports.train_master_base_port"),
        ("ports", "rollout_engine_base_port", 15000, "ports.rollout_engine_base_port"),
        ("agent", "custom_agent_function_path", "a.b.c", "agent.custom_agent_function_path"),
        ("agent", "tito_allowed_append_roles", ("tool",), "agent.tito_allowed_append_roles"),
        ("trainable", "parameter_mode", "adapter", "trainable.parameter_mode"),
        ("trainable", "expert_full_count", 4, "trainable.expert_full_count"),
        ("serving", "tp_size", 4, "serving.tp_size"),
    ],
)
def test_unmappable_values_rejected_by_name(section, field, value, path):
    cfg = sub(make_config(), section, **{field: value})
    with pytest.raises(mc.UnmappedConfigError) as err:
        mc.translate_run_config(cfg, AlgorithmSpec())
    assert err.value.path == path
    assert path in str(err.value)


@pytest.mark.usefixtures("dp_replicas")
def test_full_parameter_mode_emits_no_lora_flags():
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    argv = mc.translate_run_config(cfg, AlgorithmSpec()).argv
    assert not [token for token in argv if "lora" in token or token == "--target-modules"]




@pytest.fixture
def dp_replicas(monkeypatch):
    """A Miles pin with data-parallel replicas: ``make_config`` is a DP=2 trainer."""
    monkeypatch.setattr(mc, "full_parameter_dp_replica_gap", lambda: None)


def _pin_before_dp_replicas(monkeypatch):
    """Pin Miles 8edea35a7: in the lineage, but before the DP replica commit."""
    from yeto.rl.adapters.miles import pins

    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", "8edea35a76026d83e0a819038baf09d88bd448d9")


def test_full_parameter_sync_at_dp_above_one_needs_the_replica_pin(monkeypatch):
    # S27 G-AX: a Miles without data-parallel replicas refuses DP>1 at its first manifest.
    _pin_before_dp_replicas(monkeypatch)
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    with pytest.raises(mc.MilesConfigError, match="DP 2 needs Miles 921cde8c7 or later"):
        mc.translate_run_config(cfg, AlgorithmSpec())
    # A no-sync trainer island has no full-parameter reader.
    assert "use_distributed_optimizer" not in mc.translate_run_config(
        cfg, AlgorithmSpec(), full_parameter_sync=False).runtime_attrs


def _pin_without_sharded_masters(monkeypatch):
    """Pin Miles 59aea07f9 (S27 closing): DP replicas, but no sharded masters."""
    from yeto.rl.adapters.miles import pins

    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", "59aea07f9a1e48b6fcb9fed03c70964a00275a40")


def _dp_full_parameter_config(dp_optimizer: str = "auto", offload: float = 0.0):
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    return dataclasses.replace(cfg, train_memory=rc.TrainMemoryConfig(
        dp_optimizer=dp_optimizer, optimizer_offload_fraction=offload))


@pytest.mark.usefixtures("dp_replicas")
@pytest.mark.parametrize("sharded_pin", [True, False])
def test_full_parameter_sync_at_dp_above_one_refuses_optimizer_offload(monkeypatch, sharded_pin):
    if not sharded_pin:
        _pin_without_sharded_masters(monkeypatch)
    for choice in rc.DP_OPTIMIZERS:
        with pytest.raises(mc.MilesConfigError, match="drop --rl-optimizer-offload-fraction"):
            mc.translate_run_config(_dp_full_parameter_config(choice, offload=1.0), AlgorithmSpec())


@pytest.mark.usefixtures("dp_replicas")
def test_dp_optimizer_falls_back_to_whole_masters_without_the_sharded_pin(monkeypatch):
    # S27 G-BE: a pin before the sharded masters keeps the non-distributed optimizer.
    from yeto.rl.adapters.miles import run_config_rules

    _pin_without_sharded_masters(monkeypatch)
    assert "d50fb04dd or later (distributed optimizer at DP>1)" in run_config_rules.full_parameter_dp_sharded_master_gap()
    for choice in ("auto", "replicated"):
        attrs = mc.translate_run_config(_dp_full_parameter_config(choice), AlgorithmSpec()).runtime_attrs
        assert attrs["use_distributed_optimizer"] is False
    with pytest.raises(mc.MilesConfigError, match="--rl-dp-optimizer distributed at DP 2 needs Miles d50fb04dd"):
        mc.translate_run_config(_dp_full_parameter_config("distributed"), AlgorithmSpec())


@pytest.mark.usefixtures("dp_replicas")
def test_dp_optimizer_keeps_the_distributed_optimizer_with_the_sharded_pin():
    # The pinned image (S27 eff, 7402a53-2fa8801) has the sharded masters, so auto
    # keeps Miles' default distributed optimizer at DP>1.
    from yeto.rl.adapters.miles import run_config_rules

    assert run_config_rules.full_parameter_dp_sharded_master_gap() is None
    for choice in ("auto", "distributed"):
        attrs = mc.translate_run_config(_dp_full_parameter_config(choice), AlgorithmSpec()).runtime_attrs
        assert "use_distributed_optimizer" not in attrs  # Miles' default for Adam: distributed
    attrs = mc.translate_run_config(_dp_full_parameter_config("replicated"), AlgorithmSpec()).runtime_attrs
    assert attrs["use_distributed_optimizer"] is False


def test_train_memory_reads_the_dp_optimizer():
    assert rc._train_memory(SimpleNamespace(seq_len=8)).dp_optimizer == "auto"
    assert rc._train_memory(SimpleNamespace(seq_len=8, rl_dp_optimizer="replicated")).dp_optimizer == "replicated"
    with pytest.raises(ValueError, match="--rl-dp-optimizer"):
        rc._train_memory(SimpleNamespace(seq_len=8, rl_dp_optimizer="sharded"))


def test_dp_replica_gap_follows_the_pinned_lineage(monkeypatch):
    from yeto.rl.adapters.miles import pins, run_config_rules
    from yeto.rl.engine.run_config import full_parameter_layout_problem

    # The pinned image (S27 eff, 7402a53-2fa8801) has the replicas.
    assert run_config_rules.full_parameter_dp_replica_gap() is None
    assert full_parameter_layout_problem(data_parallel=4, actor_num_nodes=1, rollout_gpus=4,
                                         rollout_gpus_per_engine=1) is None
    # A later fork commit keeps them.
    monkeypatch.setattr(pins, "MILES_FORK_LINEAGE", (*pins.MILES_FORK_LINEAGE, "f" * 40))
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", "f" * 40)
    assert run_config_rules.full_parameter_dp_replica_gap() is None
    # An earlier one has none: DP>1 with engines on the island is refused.
    _pin_before_dp_replicas(monkeypatch)
    assert "921cde8c7 or later" in run_config_rules.full_parameter_dp_replica_gap()
    assert "DP>1 needs" in full_parameter_layout_problem(data_parallel=4, actor_num_nodes=1, rollout_gpus=4,
                                                         rollout_gpus_per_engine=1)


@pytest.mark.usefixtures("dp_replicas")
def test_trainer_only_full_parameter_island_starts_no_engine():
    # S25 T-INTEG2: a full-parameter trainer that reads the store (DP>1, no engine).
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    cfg = sub(cfg, "parallel", dedicated_rollout_gpus=0, visible_gpus_per_node=2)
    assert not cfg.parallel.colocated
    request = mc.placement_request(cfg)
    assert request.trainer_only
    argv = mc.translate_run_config(cfg, AlgorithmSpec()).argv
    assert "--debug-train-only" in argv
    assert "--colocate" not in argv and "--rollout-num-gpus" not in argv


@pytest.mark.usefixtures("dp_replicas")
def test_full_parameter_mode_turns_the_distributed_optimizer_off():
    # Miles' set_default_megatron_args forces the distributed optimizer on for
    # Adam, which leaves each FP32 master a flat slice; the whole-master path
    # (--rl-dp-optimizer replicated) turns it off. The override rides preprocess_args.
    launch = mc.translate_run_config(_dp_full_parameter_config("replicated"), AlgorithmSpec())
    assert launch.runtime_attrs["use_distributed_optimizer"] is False
    assert "--use-distributed-optimizer" not in launch.argv
    namespace = SimpleNamespace(use_distributed_optimizer=True)
    mc.apply_runtime_attrs(namespace, launch)
    assert namespace.use_distributed_optimizer is False


def test_no_sync_full_parameter_trainer_keeps_the_distributed_optimizer():
    # N7.1d: only the full-parameter sync reads whole masters; a store-fed trainer
    # island shards the optimizer state over DP and may offload it to the host.
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    cfg = dataclasses.replace(cfg, train_memory=rc.TrainMemoryConfig(optimizer_offload_fraction=1.0))
    launch = mc.translate_run_config(cfg, AlgorithmSpec(), full_parameter_sync=False)
    assert "use_distributed_optimizer" not in launch.runtime_attrs
    joined = " ".join(launch.argv)
    assert ("--use-distributed-optimizer --use-precision-aware-optimizer --optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d --optimizer-offload-fraction 1.0") in joined
    mc.validate_parsed_args(_parsed(use_distributed_optimizer=True), launch, num_cells=lambda a: 1)


def test_optimizer_offload_with_the_full_parameter_sync_needs_the_distopt_reader(monkeypatch):
    # S27 G-B: Miles reads DP=1 distributed-optimizer masters (incl. the
    # HybridDeviceOptimizer's offloaded FP32 copies) from 56190a5a8 on; an
    # older pinned image is refused before launch.
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    cfg = sub(cfg, "parallel", data_parallel=1)  # the DP=1 distributed-optimizer reader
    cfg = dataclasses.replace(cfg, train_memory=rc.TrainMemoryConfig(optimizer_offload_fraction=0.5))
    monkeypatch.setattr(mc, "MILES_NEXT_COMMIT", "ed5f64fd7a183cccc8ab7ca3da90fba570108c2b")
    with pytest.raises(mc.MilesConfigError, match="56190a5a8 or later"):
        mc.translate_run_config(cfg, AlgorithmSpec())


def test_optimizer_offload_with_the_full_parameter_sync_keeps_the_distributed_optimizer():
    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    cfg = sub(cfg, "parallel", data_parallel=1)  # the DP=1 distributed-optimizer reader
    cfg = dataclasses.replace(cfg, train_memory=rc.TrainMemoryConfig(optimizer_offload_fraction=1.0))
    launch = mc.translate_run_config(cfg, AlgorithmSpec())
    assert "use_distributed_optimizer" not in launch.runtime_attrs
    assert ("--use-distributed-optimizer --use-precision-aware-optimizer --optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d --optimizer-offload-fraction 1.0") in " ".join(launch.argv)
    mc.validate_parsed_args(_parsed(use_distributed_optimizer=True), launch, num_cells=lambda a: 1)


def test_build_ports_launch_marks_a_no_sync_trainer_island(monkeypatch):
    from yeto.rl.adapters.miles import island_entry

    seen = {}

    def translate(config, algorithm, **kw):
        seen.update(kw)
        return "launch"

    monkeypatch.setattr(mc, "translate_run_config", translate)
    args = SimpleNamespace(rl_single_island_no_sync=True, rl_sample_role="trainer", rl_max_policy_age=0)
    assert island_entry.build_ports_launch(args, make_config()) == "launch"
    assert seen["full_parameter_sync"] is False
    args.rl_single_island_no_sync = False
    island_entry.build_ports_launch(args, make_config())
    assert seen["full_parameter_sync"] is True


def test_lora_mode_leaves_the_distributed_optimizer_to_miles():
    launch = mc.translate_run_config(make_config(), AlgorithmSpec())
    assert "use_distributed_optimizer" not in launch.runtime_attrs


@pytest.mark.usefixtures("dp_replicas")
def test_validate_parsed_args_refuses_a_reenabled_distributed_optimizer():
    launch = mc.translate_run_config(_dp_full_parameter_config("replicated"), AlgorithmSpec())
    with pytest.raises(mc.MilesConfigError, match="distributed optimizer"):
        mc.validate_parsed_args(
            _parsed(use_distributed_optimizer=True), launch, num_cells=lambda a: 1
        )

def _deepseek_v4_expert_full_config():
    cfg = make_config()
    cfg = sub(cfg, "model_recipe", name=rc.RECIPE_DEEPSEEK_V4_FLASH)
    cfg = sub(cfg, "parallel", data_parallel=1)
    return sub(cfg, "trainable", expert_full_count=4)


def test_deepseek_v4_expert_full_is_a_full_parameter_lora_hybrid():
    # P0 1.7b.2: the recipe flags plus Adam and the Miles LoRA-hybrid transport.
    argv = list(mc.translate_run_config(_deepseek_v4_expert_full_config(), AlgorithmSpec()).argv)

    assert argv[argv.index("--model-name") + 1] == "deepseekv4"
    assert argv[argv.index("--optimizer") + 1] == "adam"
    assert argv[argv.index("--moe-router-bias-update-rate") + 1] == "0"
    assert "--full-parameter-lora-hybrid" in argv
    assert "--lora-base-cpu-backup" not in argv  # Miles refuses it for the hybrid
    assert argv[argv.index("--lora-rank") + 1] == "8"
    flags = [token for token in argv if token.startswith("--")]
    assert len(flags) == len(set(flags)), "a recipe flag is emitted twice"


def test_deepseek_v4_recipe_without_expert_full_has_no_hybrid_flags():
    cfg = sub(_deepseek_v4_expert_full_config(), "trainable", expert_full_count=0)
    argv = mc.translate_run_config(cfg, AlgorithmSpec()).argv

    assert "--full-parameter-lora-hybrid" not in argv
    assert "--optimizer" not in argv


def test_deepseek_v4_clone_only_lora_uses_packed_expert_adapters():
    # P0 1.7b.2b: standard LoRA keeps one packed adapter per grouped-expert linear.
    cfg = sub(_deepseek_v4_expert_full_config(), "trainable", expert_full_count=0,
              lora_targets="attention-routed-experts")
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)

    assert argv[argv.index("--lora-type") + 1] == "lora"
    assert "--no-sglang-lora-use-virtual-experts" in argv
    assert "--full-parameter-lora-hybrid" not in argv
    plain = mc.translate_run_config(sub(cfg, "trainable", lora_targets="attention"), AlgorithmSpec()).argv
    assert plain[list(plain).index("--lora-type") + 1] == "canonical_lora"
    assert "--no-sglang-lora-use-virtual-experts" not in plain


def test_unknown_config_field_rejected_by_name():
    Extended = dataclasses.make_dataclass(
        "ServingConfig", [("new_knob", int, dataclasses.field(default=7))],
        bases=(rc.ServingConfig,), frozen=True,
    )
    cfg = make_config()
    cfg = dataclasses.replace(cfg, serving=Extended(**dataclasses.asdict(cfg.serving)))
    with pytest.raises(mc.UnmappedConfigError, match=r"serving\.new_knob"):
        mc.translate_run_config(cfg, AlgorithmSpec())


def test_leaf_policy_matches_run_config_fields():
    cfg = make_config(with_eval=True, with_moe=True)
    leaves = {path for path, _ in mc.iter_config_leaves(cfg)}
    leaves |= {"eval", "geometry.moe", "algorithm.lr_schedule"}  # the None variants
    # rl-algo-critic-family 2.4: critic run config (None here) and its leaves
    leaves |= {"algorithm.critic", "algorithm.critic.critic_load", "algorithm.critic.init_sha256"}
    # L4.6: the sample buffer config (None here) and its leaves
    leaves |= {f"sample_buffer.{f.name}" for f in dataclasses.fields(rc.SampleBufferConfig)}
    assert leaves == set(mc.LEAF_POLICY)


@pytest.mark.parametrize(
    "flag", ["--indep-dp", "--use-fault-tolerance", "--ft-components", "--api-server-port=1",
             "--mini-ft-controller-enable"],
)
def test_ft_extra_argv_rejected(flag):
    with pytest.raises(mc.FaultToleranceArgsError):
        mc.translate_run_config(make_config(), AlgorithmSpec(), extra_argv=[flag])


def test_owned_flag_override_rejected_and_passthrough_allowed():
    with pytest.raises(mc.MilesConfigError, match="--colocate"):
        mc.translate_run_config(make_config(), AlgorithmSpec(), extra_argv=["--colocate"])
    argv = mc.translate_run_config(
        make_config(), AlgorithmSpec(), extra_argv=["--micro-batch-size", "1"]
    ).argv
    assert argv[-2:] == ("--micro-batch-size", "1")


def _parsed(**kw):
    base = dict(
        indep_dp=False, use_fault_tolerance=False, ft_components=[], api_server_port=None,
        mini_ft_controller_enable=False, enable_witness=False, fully_async=False,
        colocate=True, actor_num_nodes=1, actor_num_gpus_per_node=2, rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1, debug_rollout_only=False, debug_train_only=False,
        rollout_external=False, eval_num_gpus=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_validate_parsed_args():
    launch = mc.translate_run_config(make_config(), AlgorithmSpec())
    mc.validate_parsed_args(_parsed(), launch, num_cells=lambda a: 1)
    with pytest.raises(mc.FaultToleranceArgsError, match="indep_dp"):
        mc.validate_parsed_args(_parsed(indep_dp=True), launch, num_cells=lambda a: 1)
    with pytest.raises(mc.FaultToleranceArgsError, match="ft_components"):
        mc.validate_parsed_args(_parsed(ft_components=["rollout"]), launch, num_cells=lambda a: 1)
    with pytest.raises(mc.SingleCellError):
        mc.validate_parsed_args(_parsed(), launch, num_cells=lambda a: 2)
    with pytest.raises(PlacementRewriteError):
        mc.validate_parsed_args(_parsed(colocate=False), launch, num_cells=lambda a: 1)


def test_default_cell_count_without_miles():
    launch = mc.translate_run_config(make_config(), AlgorithmSpec())
    try:
        import miles.ray.specs.train  # noqa: F401
    except ImportError:
        mc.validate_parsed_args(_parsed(), launch)
    else:  # pragma: no cover - exercised only where upstream Miles imports
        pytest.skip("upstream cell counter importable; covered by parse_args check")


def test_import_is_cheap():
    code = (
        "import sys, yeto.rl.adapters.miles as m\n"
        "from yeto.rl.adapters.miles import config, rollout, rollout_meta_hook, trainer,"
        " state, state_plugin, publish, placement\n"
        "bad=[k for k in ('torch','ray','miles','megatron','sglang') if k in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


_TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
    "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 4096,
    "vocab_size": 1000, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "attention_bias": False,
}


def test_train_memory_defaults_keep_argv_and_flags_translate():
    default = mc.translate_run_config(make_config(), AlgorithmSpec()).argv
    for flag in ("--recompute-granularity", "--use-dynamic-batch-size", "--max-tokens-per-gpu"):
        assert flag not in default
    memory = rc.TrainMemoryConfig(activation_recompute=True, max_tokens_per_gpu=32768)
    argv = mc.translate_run_config(make_config(train_memory=memory), AlgorithmSpec()).argv
    joined = " ".join(argv)
    assert ("--recompute-granularity full --recompute-method uniform "
            "--recompute-num-layers 1") in joined
    assert "--use-dynamic-batch-size --max-tokens-per-gpu 32768" in joined


def test_train_memory_from_args_refuses_a_budget_below_seq_len():
    args = SimpleNamespace(seq_len=32768, rl_activation_recompute=True, rl_max_tokens_per_gpu=16384)
    with pytest.raises(ValueError, match="below --seq-len"):
        rc._train_memory(args)
    args.rl_max_tokens_per_gpu = 32768
    assert rc._train_memory(args) == rc.TrainMemoryConfig(True, 32768)
    assert rc._train_memory(SimpleNamespace(seq_len=8)) == rc.TrainMemoryConfig()
    assert rc._train_memory(SimpleNamespace(seq_len=8, rl_optimizer_offload_fraction=1.0)) == \
        rc.TrainMemoryConfig(optimizer_offload_fraction=1.0)
    with pytest.raises(ValueError, match="offload-fraction"):
        rc._train_memory(SimpleNamespace(seq_len=8, rl_optimizer_offload_fraction=1.5))


@pytest.mark.ray_local  # upstream parse_args can start Ray on this machine
@pytest.mark.parametrize("colocated", [True, False])
def test_upstream_parse_args_accepts_translation(tmp_path, colocated):
    """Runs only where upstream Miles + Megatron + SGLang import (e.g. the pinned image)."""

    pytest.importorskip("miles.utils.arguments")
    pytest.importorskip("megatron.training")
    import json

    (tmp_path / "config.json").write_text(json.dumps(_TINY_QWEN3))
    (tmp_path / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
    cfg = dataclasses.replace(
        make_config(colocated=colocated), hf_checkpoint=str(tmp_path), ref_load=str(tmp_path)
    )
    cfg = sub(cfg, "data", prompt_path=str(tmp_path / "p.jsonl"))
    if not colocated:
        # LoRA fixed partition keeps the trainer resident (config.py guard).
        cfg = sub(cfg, "serving", offload_train=False)
    # HF target names avoid the Megatron-Bridge round trip parse_args needs for Megatron names.
    cfg = sub(cfg, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    spec = AlgorithmSpec(
        dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER, dynamic_sampling_max_replacements=2
    )
    launch = mc.translate_run_config(cfg, spec)
    args = mc.parse_miles_args(launch)  # also runs validate_parsed_args
    assert args.colocate is colocated and not args.indep_dp and not args.ft_components
    assert args.rollout_all_samples_process_path == mc.ROLLOUT_META_HOOK_PATH
    assert args.yeto_rl_dynamic_sampling_max_replacements == 2


# --- LR schedule (head decoupled attempt4: implicit linear decay hit 0) -----


def _upstream_lr(argv, step):
    """LR Megatron applies at optimizer step ``step`` (0-based).

    Transcribes miles/backends/megatron_utils/model.py (upstream :80-106,
    legacy fork :85-105) and megatron.core OptimizerParamScheduler.get_lr: the
    scheduler counts samples (``global_batch`` per step) and is stepped after
    each optimizer step, so step ``k`` trains with ``get_lr`` at
    ``num_steps = k * global_batch``.
    """

    def value(flag, default=None):
        return flag_value(argv, flag) if flag in argv else default

    max_lr = float(value("--lr"))
    gbs = int(value("--global-batch-size"))
    train_iters = (
        int(value("--num-rollout")) * int(value("--rollout-batch-size"))
        * int(value("--n-samples-per-prompt")) // gbs
    )
    decay_steps = int(value("--lr-decay-iters", train_iters)) * gbs
    warmup_steps = int(value("--lr-warmup-iters", 0)) * gbs
    style = value("--lr-decay-style", "linear")
    min_lr = float(value("--min-lr", 0.0))
    assert decay_steps > 0  # Megatron: lr_decay_steps > 0
    num_steps = step * gbs
    if warmup_steps > 0 and num_steps <= warmup_steps:
        return max_lr * num_steps / warmup_steps
    if style == "constant":
        return max_lr
    if num_steps > decay_steps:
        return min_lr
    assert style == "linear"
    ratio = (num_steps - warmup_steps) / (decay_steps - warmup_steps)
    return min_lr + (1.0 - ratio) * (max_lr - min_lr)


def _schedule_args(**kw):
    base = dict(
        global_rounds=4, optimizer_steps=1, sync_preset="strict-avg", eval_only=False,
        rollout_batch_size=4, n_samples_per_prompt=4,
    )
    base.update(kw)
    base.setdefault(
        "global_batch",
        base["rollout_batch_size"] * base["n_samples_per_prompt"] // base["optimizer_steps"],
    )
    return base


def test_decoupled_lr_is_explicitly_constant_past_global_rounds():
    # Head attempt4: 4 global rounds, 12 local rounds per island.
    schedule = rc.resolve_lr_schedule(**_schedule_args(sync_preset="decoupled"))
    cfg = make_config(batch=dataclasses.replace(make_config().batch, global_rounds=4))
    cfg = sub(cfg, "algorithm", lr_schedule=schedule)
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert flag_value(argv, "--lr-decay-style") == "constant"
    assert flag_value(argv, "--num-rollout") == "4"
    local_rounds_upper = 1000  # decoupled local rounds are not bounded by global_rounds
    total = 4 * cfg.batch.optimizer_steps
    assert all(
        _upstream_lr(argv, s) == 1e-5 for s in range(total + local_rounds_upper)
    )


def test_implicit_default_would_decay_decoupled_to_zero():
    # Regression witness for the pre-fix argv (no schedule flags).
    cfg = sub(make_config(), "algorithm", lr_schedule=None)
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert "--lr-decay-style" not in argv
    total = cfg.batch.global_rounds * cfg.batch.optimizer_steps
    assert _upstream_lr(argv, total - 1) > 0.0
    # Pre-fix: from local step global_rounds * optimizer_steps on, the LR is 0
    # (head attempt4: 4 global rounds, 12 local rounds per island).
    assert all(_upstream_lr(argv, s) == 0.0 for s in range(total, total + 12))


@pytest.mark.parametrize("optimizer_steps", [1, 2])
def test_strict_lr_schedule_matches_previous_implicit_default(optimizer_steps):
    batch = dataclasses.replace(
        make_config().batch, optimizer_steps=optimizer_steps,
        global_batch=16 // optimizer_steps,
    )
    implicit = list(mc.translate_run_config(
        sub(make_config(batch=batch), "algorithm", lr_schedule=None), AlgorithmSpec()
    ).argv)
    schedule = rc.resolve_lr_schedule(
        **_schedule_args(global_rounds=batch.global_rounds, optimizer_steps=optimizer_steps)
    )
    assert schedule == rc.LrSchedule("linear", batch.global_rounds * optimizer_steps)
    explicit = list(mc.translate_run_config(
        sub(make_config(batch=batch), "algorithm", lr_schedule=schedule), AlgorithmSpec()
    ).argv)
    total = batch.global_rounds * optimizer_steps
    for step in range(total + 3):
        assert _upstream_lr(explicit, step) == _upstream_lr(implicit, step)
    # every strict step trains with a nonzero LR (the last one included)
    assert all(_upstream_lr(explicit, s) > 0 for s in range(total))
    assert _upstream_lr(explicit, total - 1) > 0


def test_eval_only_emits_no_lr_schedule():
    assert rc.resolve_lr_schedule(**_schedule_args(eval_only=True)) is None
    with pytest.raises(ValueError):
        rc.LrSchedule("cosine", 4)
    with pytest.raises(ValueError):
        rc.LrSchedule("linear", 0)


@pytest.mark.parametrize("preset", ["strict-avg", "full-parameter"])
def test_strict_lr_schedule_is_linear_over_rounds_times_steps(preset):
    assert rc.resolve_lr_schedule(
        **_schedule_args(sync_preset=preset, global_rounds=3, optimizer_steps=2)
    ) == rc.LrSchedule("linear", 6)
    assert rc.resolve_lr_schedule(
        **_schedule_args(sync_preset="decoupled", global_rounds=3, optimizer_steps=2)
    ) == rc.LrSchedule("constant", 6)


def test_strict_lr_schedule_rejects_rounds_that_are_not_optimizer_steps():
    # 4*3 samples do not split into 2 optimizer steps of 5: Miles' implicit
    # horizon would differ from global_rounds * optimizer_steps.
    with pytest.raises(ValueError, match="global_batch \\* optimizer_steps"):
        rc.resolve_lr_schedule(**_schedule_args(
            n_samples_per_prompt=3, optimizer_steps=2, global_batch=5,
        ))
    # decoupled never derives a horizon from the batch shape
    assert rc.resolve_lr_schedule(**_schedule_args(
        sync_preset="decoupled", n_samples_per_prompt=3, optimizer_steps=2, global_batch=5,
    )).decay_style == "constant"


def test_lr_schedule_flags_are_adapter_owned():
    for flag in lrs.LR_SCHEDULE_FLAGS:
        with pytest.raises(mc.MilesConfigError):
            mc.check_extra_argv((flag, "1"))


# ---------------------------------------------------------------- append roles are decided by the --tito-model template

def test_explicit_append_roles_are_refused():
    """Since S26 E0.5 no signed Codex agent may state the append roles."""
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.config import LEAF_POLICY

    check = LEAF_POLICY["agent.tito_allowed_append_roles"]
    cfg = SimpleNamespace(agent=SimpleNamespace(custom_agent_function_path="x:y", tito_model="qwen35"))
    assert check(None, cfg) is None
    assert "upstream Miles has no --tito-allowed-append-roles" in check(("tool", "user"), cfg)


# --- S17 N16: --rl-lr-schedule ----------------------------------------------------
def test_elastic_island_lr_schedule_is_constant():
    """S17 M1 (N5 a73ab1b2, folded into the auto rule): an elastic island runs until the
    syncer's final outer version, so a linear horizon could reach 0 before it ends."""
    assert rc.resolve_lr_schedule(**_schedule_args(sync_preset="strict-avg", global_rounds=6),
                                  island_scheduling="elastic") == rc.LrSchedule("constant", 6)
    assert rc.resolve_lr_schedule(**_schedule_args(sync_preset="strict-avg", global_rounds=6),
                                  island_scheduling="legacy") == rc.LrSchedule("linear", 6)


def test_async_full_parameter_island_lr_schedule_is_constant():
    """An asynchronous full-parameter island trains until the syncer stops it
    (H local rounds per fragment round, more when it is fast under a partial
    quorum), so a linear horizon of global_rounds could decay the LR to 0
    while it still trains. The strict sweep keeps the linear schedule."""
    args = _schedule_args(sync_preset="full-parameter", global_rounds=6)
    assert rc.resolve_lr_schedule(**args, full_parameter_async=True) == rc.LrSchedule("constant", 6)
    assert rc.resolve_lr_schedule(**args) == rc.LrSchedule("linear", 6)
    with pytest.raises(ValueError, match="asynchronous full-parameter"):
        rc.resolve_lr_schedule(**args, full_parameter_async=True, requested="linear")


@pytest.mark.parametrize("preset,scheduling", [("strict-avg", "legacy"), ("decoupled", "legacy"),
                                               ("strict-avg", "elastic")])
def test_rl_lr_schedule_constant_in_every_mode(preset, scheduling):
    got = rc.resolve_lr_schedule(**_schedule_args(sync_preset=preset, global_rounds=5),
                                 island_scheduling=scheduling, requested="constant")
    assert got == rc.LrSchedule("constant", 5)


def test_rl_lr_schedule_auto_is_unchanged_and_linear_refused_without_horizon():
    args = _schedule_args(sync_preset="strict-avg", global_rounds=5)
    assert rc.resolve_lr_schedule(**args) == rc.resolve_lr_schedule(**args, requested="auto") \
        == rc.resolve_lr_schedule(**args, requested="linear") == rc.LrSchedule("linear", 5)
    for kw in ({"sync_preset": "decoupled"}, {"island_scheduling": "elastic"}):
        a = _schedule_args(**{k: v for k, v in kw.items() if k == "sync_preset"})
        with pytest.raises(ValueError, match="known local step count"):
            rc.resolve_lr_schedule(**a, requested="linear",
                                   **{k: v for k, v in kw.items() if k != "sync_preset"})
    with pytest.raises(ValueError, match="must be one of"):
        rc.resolve_lr_schedule(**args, requested="cosine")
    assert rc.resolve_lr_schedule(**_schedule_args(eval_only=True), requested="constant") is None


def test_rl_lr_schedule_flag_reaches_the_island_and_the_miles_argv():
    """cli -> launcher island flags -> island_entry parser -> run config -> Miles flags."""
    import argparse
    from types import SimpleNamespace

    from yeto import launcher
    from yeto.cli import _add_launch_args
    from yeto.rl.adapters.miles import island_entry

    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    base = ["--model", "qwen35-9b", "--data", "d"]
    assert parser.parse_args(base).rl_lr_schedule == "auto"
    assert parser.parse_args(base + ["--rl-lr-schedule", "constant"]).rl_lr_schedule == "constant"
    _, flags = launcher._ports_infra_flags(SimpleNamespace(rl_lr_schedule="constant"))
    assert " --rl-lr-schedule constant" in flags
    _, flags = launcher._ports_infra_flags(SimpleNamespace(rl_lr_schedule="auto"))
    assert "--rl-lr-schedule" not in flags
    seen = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, argv=None, namespace=None):
        if self.prog.endswith("island_entry"):
            action = next(a for a in self._actions if "--rl-lr-schedule" in a.option_strings)
            seen.update(default=action.default, choices=tuple(action.choices))
            raise SystemExit(0)
        return original(self, argv, namespace)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            island_entry.parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = original
    assert seen == {"default": "auto", "choices": ("auto", "linear", "constant")}
    assert lrs.lr_schedule_argv(rc.LrSchedule("constant", 5))[:2] == ("--lr-decay-style", "constant")


@pytest.mark.ray_local  # upstream parse_args can start Ray on this machine
@pytest.mark.parametrize("family", ["sao", "vapo", "compactionrl"])
def test_upstream_parse_args_accepts_critic_family_on_pinned_image(tmp_path, family):
    """S19 #1 (critic 6.3 / supplement 2.5): the pinned image's Miles parses the
    critic-family fork flags with the full ports translation.  Runs only in the
    image (Miles + Megatron + SGLang import)."""

    pytest.importorskip("miles.utils.arguments")
    pytest.importorskip("megatron.training")
    from yeto.rl.adapters.miles.overlay import spec_fork_flags
    from yeto.rl.algos import sao
    from yeto.rl.algos.compactionrl import compactionrl_spec
    from yeto.rl.algos.vapo import vapo_spec

    spec = {"sao": lambda: sao.sao_algorithm_spec("coding"), "vapo": vapo_spec,
            "compactionrl": compactionrl_spec}[family]()
    flags = spec_fork_flags(spec.canonical_json())
    assert flags
    (tmp_path / "config.json").write_text(json.dumps(_TINY_QWEN3))
    (tmp_path / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
    cfg = dataclasses.replace(make_config(colocated=True), hf_checkpoint=str(tmp_path), ref_load=str(tmp_path))
    cfg = sub(cfg, "data", prompt_path=str(tmp_path / "p.jsonl"))
    cfg = sub(cfg, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    # A legal run config for a critic-family spec: the estimator comes from the
    # spec, the LR decay horizon (50) exceeds the critic LR warm-up (10), and
    # init='copy_actor_backbone' + warmup_steps > 0 families load the warm-up
    # stage product (design D5; sao has warmup_steps=0 and takes no critic_load).
    cfg = sub(cfg, "algorithm", advantage_estimator=spec.advantage.estimator,
              lr_schedule=rc.LrSchedule("linear", 50),
              critic=None if family == "sao"
              else rc.CriticRunConfig(critic_load=str(tmp_path), init_sha256="0" * 64))
    launch = mc.translate_run_config(cfg, spec)
    args = mc.parse_miles_args(launch)
    # Fork aliases share one argparse dest (e07e51c07: --critic-updates-per-step
    # is an alias of --num-critic-epochs, dest num_critic_epochs).
    alias_dest = {"--critic-updates-per-step": "num_critic_epochs"}
    missing = [f for f in flags
               if not hasattr(args, alias_dest.get(f, f.lstrip("-").replace("-", "_")))]
    assert not missing, missing
    if spec.critic is not None and spec.critic.critic_updates_per_step not in (None, 1):
        assert args.num_critic_epochs == spec.critic.critic_updates_per_step


def test_dynamic_batching_refused_for_bshd_gated_delta_net_models():
    cfg = make_config(train_memory=rc.TrainMemoryConfig(max_tokens_per_gpu=32768))
    recipe = dataclasses.replace(cfg.model_recipe, gdn=rc.GdnRecipe(gated_delta_net=True))
    with pytest.raises(mc.MilesConfigError, match="qkv-format bshd"):
        mc.translate_run_config(dataclasses.replace(cfg, model_recipe=recipe), AlgorithmSpec())
    recompute_only = dataclasses.replace(
        cfg, model_recipe=recipe, train_memory=rc.TrainMemoryConfig(activation_recompute=True))
    assert "--recompute-granularity" in mc.translate_run_config(recompute_only, AlgorithmSpec()).argv


def test_gradient_accumulation_fusion_is_opt_in():
    # S26 E0.1: the Miles recipes train with the fusion on; yeto keeps it off by default
    off = " ".join(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--no-gradient-accumulation-fusion" in off
    cfg = dataclasses.replace(make_config(), train_memory=rc.TrainMemoryConfig(gradient_accumulation_fusion=True))
    assert "--no-gradient-accumulation-fusion" not in mc.translate_run_config(cfg, AlgorithmSpec()).argv
    assert rc._train_memory(SimpleNamespace(seq_len=8, rl_gradient_accumulation_fusion=True)) \
        .gradient_accumulation_fusion


def test_generic_recipe_trains_without_dropout():
    cfg = make_config()
    if cfg.model_recipe.name != rc.RECIPE_GENERIC:
        cfg = sub(cfg, "model_recipe", name=rc.RECIPE_GENERIC)
    joined = " ".join(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert "--attention-dropout 0.0 --hidden-dropout 0.0" in joined


def test_observe_training_entropy_reaches_the_miles_argv():
    plain = list(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--observe-training-entropy" not in plain
    cfg = sub(make_config(), "algorithm", observe_training_entropy=True)
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert argv.count("--observe-training-entropy") == 1
    assert [value for value in argv if value != "--observe-training-entropy"] == plain


def test_mamba_training_ssm_state_dtype_reaches_the_miles_argv():
    plain = list(mc.translate_run_config(make_config(), AlgorithmSpec()).argv)
    assert "--mamba-training-ssm-states-dtype" not in plain
    cfg = sub(make_config(), "algorithm", mamba_training_ssm_states_dtype="fp32")
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    at = argv.index("--mamba-training-ssm-states-dtype")
    assert argv[at + 1] == "fp32" and argv[:at] + argv[at + 2:] == plain
