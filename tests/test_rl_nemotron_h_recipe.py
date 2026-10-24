"""Nemotron-H (Mamba-2 + MoE hybrid) Miles recipe: provider-driven model flags
reach the Miles argv; MTP is refused in full-parameter mode. CPU only; the
provider is a stand-in carrying the fields Megatron-Bridge 0.6.2
``nemotron_h_bridge.py`` sets."""

from __future__ import annotations

import ast
import json
import re
from types import SimpleNamespace

import pytest

from test_s25_g1_stage1mini import REPO

from yeto.rl.adapters.miles.models import nemotron_h


def _provider(**kw):
    # Nemotron-3 Nano 30B-A3B (HF config.json) as the Bridge provider sees it
    base = dict(
        is_hybrid_model=True, hybrid_layer_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        num_layers=52, hidden_size=2688, num_attention_heads=32, num_query_groups=2, kv_channels=128,
        ffn_hidden_size=1856, seq_length=262144, vocab_size=131072, layernorm_epsilon=1e-5,
        normalization="RMSNorm", position_embedding_type="none", share_embeddings_and_output_weights=False,
        add_bias_linear=False, gated_linear_unit=False,
        num_moe_experts=128, moe_ffn_hidden_size=1856, moe_router_topk=6, moe_layer_freq=1,
        moe_shared_expert_intermediate_size=3712, moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True, moe_router_load_balancing_type="seq_aux_loss",
        moe_router_dtype="fp32", moe_grouped_gemm=True, moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1, moe_router_group_topk=1, moe_latent_size=None, mtp_num_layers=0,
        mamba_num_heads=64, mamba_head_dim=64, mamba_state_dim=128, mamba_num_groups=8)
    return SimpleNamespace(**{**base, **kw})


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_model_flags_from_provider():
    flags = list(nemotron_h.model_flags(_provider()))
    assert _flag(flags, "--moe-router-score-function") == "sigmoid"
    assert _flag(flags, "--moe-router-topk-scaling-factor") == "2.5"
    assert _flag(flags, "--moe-router-bias-update-rate") == "0"
    assert _flag(flags, "--moe-aux-loss-coeff") == "0"
    assert "--moe-router-enable-expert-bias" in flags and "--moe-grouped-gemm" in flags
    assert "--moe-latent-size" not in flags
    # 120B Super: latent MoE and 512 experts come from the provider, not a table
    super_flags = list(nemotron_h.model_flags(_provider(moe_latent_size=1024, num_moe_experts=512)))
    assert _flag(super_flags, "--moe-latent-size") == "1024"


def test_check_provider_refuses_mtp_in_full_parameter_and_bad_pattern():
    nemotron_h.check_provider(_provider(), parameter_mode="full")
    with pytest.raises(nemotron_h.NemotronHRecipeError, match="MTP"):
        nemotron_h.check_provider(_provider(mtp_num_layers=1), parameter_mode="full")
    nemotron_h.check_provider(_provider(mtp_num_layers=1), parameter_mode="lora")
    with pytest.raises(nemotron_h.NemotronHRecipeError, match="experts"):
        nemotron_h.check_provider(_provider(num_moe_experts=None), parameter_mode="full")


def test_nemotron_h_provider_reaches_the_miles_argv(tmp_path):
    from yeto.rl.adapters.miles.island_entry import build_ports_launch, parse_args
    from yeto.rl.engine.run_config import RECIPE_NEMOTRON_H, resolve_rl_run_config
    from yeto.rl.stage1 import expand, load_plan

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"messages": [{"role": "user", "content": "hi"}], "label": "1"}\n')
    argv = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"][3:]
    argv = [str(model) if a == "/data/models/glm5next-mini-bf16-7fb7883" else a for a in argv]
    argv = [str(REPO / a) if a.startswith("examples/rl_algorithms/") else a for a in argv]
    if "--rl-model-recipe" in argv:
        argv[argv.index("--rl-model-recipe") + 1] = "generic"
    args = parse_args(argv)
    config = resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=_provider(),
                                   target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    assert config.model_recipe.name == RECIPE_NEMOTRON_H
    assert config.model_recipe.training_attention_backend == "auto"
    miles = list(build_ports_launch(args, config, []).argv)
    assert _flag(miles, "--megatron-to-hf-mode") == "bridge"
    assert _flag(miles, "--position-embedding-type") == "none"
    assert _flag(miles, "--num-experts") == "128"
    assert _flag(miles, "--moe-router-bias-update-rate") == "0"
    assert "--swiglu" not in miles  # relu2 experts
    assert miles.count("--moe-router-score-function") == 1
    # S27 G-AN training defaults; no DeepEP on this CPU host -> all-to-all. The
    # plan's own --rl-activation-recompute keeps full recompute.
    assert args.rl_activation_recompute and _flag(miles, "--recompute-granularity") == "full"
    assert "--recompute-modules" not in miles
    assert _flag(miles, "--moe-token-dispatcher-type") == "alltoall" and "--moe-permute-fusion" in miles
    assert "--use-dynamic-batch-size" in miles


def _train_flags(**kw):
    values = {"activation_recompute": False, "max_tokens_per_gpu": None, "tensor_parallel": 1,
              "context_parallel": 1, "seq_len": 4096, "deepep": False}
    return list(nemotron_h.train_flags(nemotron_h.model_flags(_provider()), **{**values, **kw}))


def test_train_flags_follow_the_measured_recipe():
    flags = _train_flags()
    assert flags == ["--recompute-granularity", "selective", "--recompute-modules", "moe", "layernorm",
                     "--use-dynamic-batch-size", "--max-tokens-per-gpu", "8192",
                     "--moe-token-dispatcher-type", "alltoall", "--moe-permute-fusion"]
    deepep = _train_flags(deepep=True)
    assert deepep[-5:] == ["--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", "deepep",
                           "--moe-permute-fusion"]
    # 8192 tokens per TP rank, and never below one sample per packed micro batch.
    assert _flag(_train_flags(tensor_parallel=2), "--max-tokens-per-gpu") == "16384"
    assert _flag(_train_flags(seq_len=20000, context_parallel=2), "--max-tokens-per-gpu") == "10000"
    assert _flag(_train_flags(seq_len=20000), "--max-tokens-per-gpu") == "20000"


def test_train_flags_step_aside_for_explicit_options():
    flags = _train_flags(activation_recompute=True, max_tokens_per_gpu=4096)
    assert "--recompute-granularity" not in flags and "--max-tokens-per-gpu" not in flags
    dense = nemotron_h.train_flags(nemotron_h.model_flags(_provider(num_moe_experts=None)),
                                   activation_recompute=False, max_tokens_per_gpu=None, tensor_parallel=1,
                                   context_parallel=1, seq_len=4096, deepep=True)
    assert "--moe-token-dispatcher-type" not in dense and "--moe-permute-fusion" not in dense


def test_explicit_megatron_flags_override_the_defaults(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles.config import TRAIN_EXTRA_ENV
    from yeto.rl.adapters.miles.island_entry import build_ports_launch, parse_args
    from yeto.rl.engine.run_config import resolve_rl_run_config
    from yeto.rl.stage1 import expand, load_plan

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"messages": [{"role": "user", "content": "hi"}], "label": "1"}\n')
    argv = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"][3:]
    argv = [str(model) if a == "/data/models/glm5next-mini-bf16-7fb7883" else a for a in argv]
    argv = [str(REPO / a) if a.startswith("examples/rl_algorithms/") else a for a in argv]
    if "--rl-model-recipe" in argv:
        argv[argv.index("--rl-model-recipe") + 1] = "generic"
    args = parse_args(argv)
    config = resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=_provider(),
                                   target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    monkeypatch.setattr(nemotron_h, "deepep_available", lambda: True)
    monkeypatch.setenv(TRAIN_EXTRA_ENV, "--moe-token-dispatcher-type alltoall --recompute-granularity full")
    miles = list(build_ports_launch(args, config, []).argv)
    # Megatron's argparse keeps the last value: the island's own flags win.
    dispatchers = [miles[i + 1] for i, a in enumerate(miles) if a == "--moe-token-dispatcher-type"]
    granularity = [miles[i + 1] for i, a in enumerate(miles) if a == "--recompute-granularity"]
    assert dispatchers == ["flex", "alltoall"] and granularity[-1] == "full"


def test_train_warnings_name_the_costly_settings():
    assert nemotron_h.train_warnings(samples_per_step=512, optimizer_offload_fraction=0.0, trainer_gpus=8) == ()
    (few,) = nemotron_h.train_warnings(samples_per_step=128, optimizer_offload_fraction=0.0, trainer_gpus=8)
    assert "128 samples" in few
    (offload,) = nemotron_h.train_warnings(samples_per_step=512, optimizer_offload_fraction=1.0, trainer_gpus=2)
    assert "offloaded" in offload and "2 trainer GPU" in offload
    assert nemotron_h.train_warnings(samples_per_step=512, optimizer_offload_fraction=1.0, trainer_gpus=16) == ()
    assert nemotron_h.train_warnings(samples_per_step=512, optimizer_offload_fraction=0.5, trainer_gpus=2) == ()


def test_mtp_refusal_names_the_miles_bridge_and_the_keep_switch():
    with pytest.raises(nemotron_h.NemotronHRecipeError, match="MILES_NEMOTRONH_KEEP_MTP"):
        nemotron_h.check_provider(_provider(mtp_num_layers=1), parameter_mode="full")
    nemotron_h.check_provider(_provider(mtp_num_layers=None), parameter_mode="full")  # Miles bridge drops MTP


@pytest.mark.parametrize(("config", "imported"), [
    ({"architectures": ["NemotronH_Omni_Reasoning_V3"]}, True),
    ({"architectures": ["NemotronHForCausalLM"], "num_nextn_predict_layers": 1}, True),  # 120B Super text
    ({"architectures": ["NemotronHForCausalLM"]}, False),  # Nano: upstream provider, as validated
    ({"architectures": ["Qwen3MoeForCausalLM"], "num_nextn_predict_layers": 1}, False),
    (None, False),
])
def test_island_reads_the_120b_checkpoints_with_the_miles_bridges(tmp_path, monkeypatch, config, imported):
    calls = []

    def import_module(name):
        calls.append(name)
        return SimpleNamespace(NEMOTRON_H_OMNI_ARCHITECTURE="NemotronH_Omni_Reasoning_V3")

    monkeypatch.setattr(nemotron_h.importlib, "import_module", import_module)
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    assert nemotron_h.register_miles_bridges(tmp_path) is imported
    assert calls[:1] == (["miles_plugins.megatron_bridge"] if imported else [])


def test_vl_checkpoint_refuses_a_miles_without_the_vl_bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(nemotron_h.importlib, "import_module", lambda name: SimpleNamespace())
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["NemotronH_Omni_Reasoning_V3"]}))
    with pytest.raises(nemotron_h.NemotronHRecipeError, match="language-only VL bridge"):
        nemotron_h.register_miles_bridges(tmp_path)


def _set(argv, flag, value=None):
    """Set (or add) one island flag; ``value=None`` removes it with its value."""
    if flag in argv:
        i = argv.index(flag)
        del argv[i:i + 2]
    if value is not None:
        argv += [flag, str(value)]


def test_120b_vl_multinode_island_reaches_the_miles_argv(tmp_path):
    """1 island = 4 trainer nodes (TP8/EP8/PP4, DP1) + 1 engine node (2 x TP4), no offload."""
    from yeto.rl.adapters.miles.island_entry import build_ports_launch, parse_args
    from yeto.rl.engine.run_config import RECIPE_NEMOTRON_H, resolve_rl_run_config
    from yeto.rl.stage1 import expand, load_plan

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["NemotronH_Omni_Reasoning_V3"]}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"messages": [{"role": "user", "content": "hi"}], "label": "1"}\n')
    argv = list(expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"][3:])
    argv = [str(model) if a == "/data/models/glm5next-mini-bf16-7fb7883" else a for a in argv]
    argv = [str(REPO / a) if a.startswith("examples/rl_algorithms/") else a for a in argv]
    for flag, value in (("--rl-model-recipe", "generic"), ("--rl-sample-role", None), ("--rl-resume-store", None),
                        ("--megatron-ref-load", None), ("--rl-optimizer-offload-fraction", None),
                        ("--tensor-parallel", 8), ("--pipeline-parallel", 4), ("--expert-parallel", 8),
                        ("--actor-num-nodes", 4), ("--actor-num-gpus-per-node", 8),
                        ("--rl-placement", "fixed-partition"), ("--rollout-num-gpus", 8),
                        ("--rollout-num-gpus-per-engine", 4), ("--rl-island-gpus-per-node", 8)):
        _set(argv, flag, value)
    args = parse_args(argv)
    pattern = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
    # Miles' bridge (MilesNemotronHBridge) gives the provider a per-layer MoE mask.
    provider = _provider(hybrid_layer_pattern=pattern, moe_layer_freq=[int(c == "E") for c in pattern],
                         num_layers=88, hidden_size=4096, ffn_hidden_size=2688, num_moe_experts=512,
                         moe_ffn_hidden_size=2688, moe_router_topk=22, moe_shared_expert_intermediate_size=5376,
                         moe_latent_size=1024, moe_router_topk_scaling_factor=5.0, mtp_num_layers=None,
                         mamba_num_heads=128)
    config = resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=provider,
                                   target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    assert config.model_recipe.name == RECIPE_NEMOTRON_H
    miles = list(build_ports_launch(args, config, []).argv)
    for flag, value in (("--actor-num-nodes", "4"), ("--actor-num-gpus-per-node", "8"), ("--rollout-num-gpus", "8"),
                        ("--rollout-num-gpus-per-engine", "4"), ("--tensor-model-parallel-size", "8"),
                        ("--pipeline-model-parallel-size", "4"), ("--expert-model-parallel-size", "8"),
                        ("--num-experts", "512"), ("--moe-router-topk", "22"), ("--moe-latent-size", "1024"),
                        ("--num-layers", "88"), ("--num-gpus-per-node", "8")):
        assert _flag(miles, flag) == value, flag
    assert "--colocate" not in miles and "--optimizer-cpu-offload" not in miles
    # Megatron's moe_freq_type evaluates a bracketed pattern: one mask entry per layer.
    # Its _eval_pattern allows only digits, commas, brackets, +, * and parentheses (no
    # spaces): s27-aw-a failed on "[0, 1, ...]" (argument --moe-layer-freq: invalid value).
    freq = _flag(miles, "--moe-layer-freq")
    assert not re.search(r"[^,\d\[\]\(\)\+\*]", freq), freq
    assert ast.literal_eval(freq) == [int(c == "E") for c in pattern]


def test_megatron_moe_layer_freq_has_no_spaces():
    from yeto.rl.adapters.miles.config import megatron_moe_layer_freq

    assert megatron_moe_layer_freq(1) == "1"
    assert megatron_moe_layer_freq([0, 1, 1]) == "[0,1,1]"
    assert megatron_moe_layer_freq((1, 0)) == "[1,0]"
