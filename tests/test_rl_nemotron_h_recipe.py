"""Nemotron-H (Mamba-2 + MoE hybrid) Miles recipe: provider-driven model flags
reach the Miles argv; MTP is refused in full-parameter mode. CPU only; the
provider is a stand-in carrying the fields Megatron-Bridge 0.6.2
``nemotron_h_bridge.py`` sets."""

from __future__ import annotations

import json
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
