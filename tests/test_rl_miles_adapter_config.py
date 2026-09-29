"""CPU tests for miles_adapter.config (task 3.1)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AlgorithmSpec,
)
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.engine.miles_adapter.placement import PlacementRewriteError


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
    launch = mc.translate_run_config(make_config(colocated=False), AlgorithmSpec())
    argv = list(launch.argv)
    assert "--colocate" not in argv
    assert flag_value(argv, "--rollout-num-gpus") == "2"
    assert launch.placement.kind == "fixed-partition"
    assert launch.placement.expected_layout() == (4, 2)


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
    assert flag_value(list(launch.argv), "--dynamic-sampling-filter-path") == BOUNDED_NONZERO_STD_FILTER
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
        ("trainable", "parameter_mode", "full", "trainable.parameter_mode"),
        ("trainable", "expert_full_count", 4, "trainable.expert_full_count"),
        ("trainable", "lora_targets", "attention-routed-experts", "trainable.lora_targets"),
        ("model_recipe", "name", rc.RECIPE_DEEPSEEK_V4_FLASH, "model_recipe.name"),
        ("serving", "tp_size", 4, "serving.tp_size"),
    ],
)
def test_unmappable_values_rejected_by_name(section, field, value, path):
    cfg = sub(make_config(), section, **{field: value})
    with pytest.raises(mc.UnmappedConfigError) as err:
        mc.translate_run_config(cfg, AlgorithmSpec())
    assert err.value.path == path
    assert path in str(err.value)


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
        "import sys, yeto.rl.engine.miles_adapter as m\n"
        "from yeto.rl.engine.miles_adapter import config, rollout, rollout_meta_hook, trainer,"
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


@pytest.mark.parametrize("preset", ["strict-avg", "dense-full"])
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
    for flag in rc.LR_SCHEDULE_FLAGS:
        with pytest.raises(mc.MilesConfigError):
            mc.check_extra_argv((flag, "1"))
