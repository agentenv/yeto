"""DeepSeek-V4.1-Flash LoRA recipe: provider view, Miles argv, constraints, island translation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.adapters.miles.models.deepseek_v41 import ports_recipe, provider

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "examples/k31/algorithm-spec.async-age2-tis.json"

TEXT = {
    "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64,
    "n_routed_experts": 384, "num_experts_per_tok": 6, "n_shared_experts": 1,
    "moe_intermediate_size": 2304, "vocab_size": 129280, "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-6, "q_lora_rank": 1280, "head_dim": 512, "qk_rope_head_dim": 64,
}


def _config(nested: bool = False, **overrides) -> dict:
    text = {**TEXT, **overrides}
    if nested:
        return {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {"model_type": "deepseek_v41_text", **text}}
    return {"model_type": "deepseek_v41", **text}


def _value(argv, flag):
    return argv[list(argv).index(flag) + 1]


# -- provider view ------------------------------------------------------------


@pytest.mark.parametrize("nested", [False, True])
def test_provider_view_reads_flat_and_nested_configs(nested):
    view = provider.provider_view(_config(nested=nested))
    assert (view.num_layers, view.hidden_size, view.num_attention_heads) == (40, 5120, 64)
    assert (view.num_moe_experts, view.moe_router_topk, view.moe_ffn_hidden_size) == (384, 6, 2304)
    assert view.moe_shared_expert_intermediate_size == 2304
    assert view.moe_layer_freq == (1,) * 40 and view.first_k_dense_replace == 0
    assert (view.kv_lora_rank, view.v_head_dim, view.qk_head_dim) == (512, 512, 512)
    assert view.multi_latent_attention and not view.share_embeddings_and_output_weights


def test_provider_view_refuses_foreign_or_dense_configs():
    with pytest.raises(ValueError, match="not a DeepSeek-V4.1"):
        provider.provider_view({**TEXT, "model_type": "deepseek_v3"})
    with pytest.raises(ValueError, match="every DeepSeek-V4.1 layer is MoE"):
        provider.provider_view(_config(first_k_dense_replace=1))


def test_is_deepseek_v41(tmp_path):
    assert not provider.is_deepseek_v41(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(_config(nested=True)))
    assert provider.is_deepseek_v41(tmp_path)
    assert provider.deepseek_v41_provider(tmp_path).num_layers == 40


# -- recipe argv --------------------------------------------------------------


def test_model_argv_matches_the_miles_model_args():
    argv = ports_recipe.model_argv()
    assert argv[argv.index("--spec") + 1: argv.index("--spec") + 3] == list(ports_recipe.SPEC)
    for flag, value in (("--num-layers", "40"), ("--hidden-size", "5120"), ("--num-experts", "384"),
                        ("--moe-router-topk", "6"), ("--num-residual-streams", "4"), ("--o-groups", "8"),
                        ("--moe-router-score-function", "sqrtsoftplus"), ("--q-lora-rank", "1280"),
                        ("--make-vocab-size-divisible-by", "64"), ("--activation-func-clamp-value", "10")):
        assert _value(argv, flag) == value, flag
    assert _value(argv, "--csa-compress-ratios") == (
        "[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]")
    assert _value(argv, "--moe-layer-freq") == "[" + ",".join(["1"] * 40) + "]"


@pytest.mark.parametrize(("tp", "divisor"), [(1, 64), (2, 64), (4, 64), (8, 32)])
def test_vocab_divisor_follows_tensor_parallel(tp, divisor):
    assert ports_recipe.make_vocab_size_divisible_by(tp) == divisor
    assert _value(ports_recipe.model_argv(tensor_parallel=tp), "--make-vocab-size-divisible-by") == str(divisor)


def test_cut_down_checkpoint_takes_its_layer_count():
    argv = ports_recipe.model_argv(num_layers=4)
    assert _value(argv, "--num-layers") == "4"
    assert _value(argv, "--csa-compress-ratios") == "[0,0,2,2]"


def test_target_presets_never_adapt_routed_experts():
    assert ports_recipe.target_modules("attention") == ("attn",)
    assert ports_recipe.target_modules("auto") == ("attn",)
    assert ports_recipe.target_modules("all-linear") == ("attn", "mlp")
    assert _value(ports_recipe.RECIPE_ARGV, "--exclude-modules") == "*.ffn.experts.*"
    with pytest.raises(ports_recipe.DeepseekV41RecipeError):
        ports_recipe.target_modules("attention-routed-experts")


def test_geometry_check():
    ports_recipe.check_geometry(hidden_size=5120, num_attention_heads=64, num_experts=384,
                                router_topk=6, vocab_size=129280)
    with pytest.raises(ports_recipe.DeepseekV41RecipeError, match="num_experts=256"):
        ports_recipe.check_geometry(hidden_size=5120, num_attention_heads=64, num_experts=256,
                                    router_topk=6, vocab_size=129280)


def test_apply_recipe_replaces_geometry_and_recompute():
    base = ["train.py", "--num-layers", "40", "--num-query-groups", "64", "--kv-channels", "512",
            "--max-position-embeddings", "1048576", "--recompute-granularity", "full",
            "--recompute-method", "block", "--seq-length", "65536", "--model-name", "x",
            "--lora-base-cpu-backup"]
    out = ports_recipe.apply_recipe(base, ports_recipe.recipe_argv())
    assert out[0] == "train.py" and _value(out, "--seq-length") == "65536"
    for flag in ("--num-query-groups", "--kv-channels", "--max-position-embeddings", "--lora-base-cpu-backup"):
        assert flag not in out
    assert out.count("--num-layers") == 1 and out.count("--recompute-method") == 1
    assert _value(out, "--recompute-method") == "uniform" and _value(out, "--model-name") == "deepseekv41"
    with pytest.raises(ports_recipe.DeepseekV41RecipeError, match="full recompute"):
        ports_recipe.apply_recipe(["train.py", "--recompute-granularity", "selective"], ports_recipe.recipe_argv())


# -- island translation (the 2 x 8 B300 trainer layout) -----------------------


def _island_args(tmp_path, *extra):
    from yeto.rl.adapters.miles import island_entry as ie

    tokens = [
        "--model", "yeto-local/DeepSeek-V4.1-Flash-bf16", "--model-revision", "0" * 40,
        "--rl-model-recipe", "deepseek_v41", "--data", "d", "--data-revision", "1" * 40,
        "--rl-single-island-no-sync", "--learner-id", "0",
        "--reward-function", "yeto.rl.math_reward:reward_func", "--reward-sha256", "0" * 64,
        "--source-sha256", "0" * 64, "--global-rounds", "4", "--sync-preset", "strict-avg",
        "--fragments", "1", "--pipeline", "1", "--local-horizon", "1", "--total-fragment-steps", "4",
        "--groups-per-round", "8", "--samples-per-group", "8", "--over-sampling-batch-size", "8",
        "--optimizer-steps", "1", "--rollout-max-response-len", "61440",
        "--completed-groups-path", str(tmp_path / "groups.pt"), "--event-tape", str(tmp_path / "tape.jsonl"),
        "--actor-num-nodes", "2", "--actor-num-gpus-per-node", "8", "--rl-island-gpus-per-node", "8",
        "--tensor-parallel", "4", "--pipeline-parallel", "2", "--expert-parallel", "8",
        "--rollout-num-gpus-per-engine", "1", "--lora-r", "16", "--lora-targets", "attention",
        "--inner-lr", "1e-5", "--seq-len", "65536", "--rl-activation-recompute",
        "--rl-algorithm-spec", str(SPEC), "--rl-max-policy-age", "2", "--rl-sample-role", "trainer",
        "--rl-buffer-shards", "s-0,s-1", "--rl-lr-schedule", "constant", "--rl-round-cuts", "off",
        "--rl-resume-store", str(tmp_path / "store"), "--seed", "0", "--miles-root", str(tmp_path), *extra,
    ]
    return ie, ie.parse_args(tokens)


def _translate(tmp_path, *extra):
    from yeto.rl.engine.run_config import resolve_rl_run_config

    ie, args = _island_args(tmp_path, *extra)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_config(nested=True)))
    cfg = resolve_rl_run_config(
        args, model_path=model, prompt_path=tmp_path / "p.jsonl",
        provider=provider.deepseek_v41_provider(model),
        target_modules=list(ports_recipe.target_modules(args.lora_targets)),
        yeto_policy_sync=False, verify_ref_load=False)
    return list(ie.build_ports_launch(args, cfg).argv), str(model)


def test_trainer_argv_of_the_b300_layout(tmp_path):
    argv, model = _translate(tmp_path)
    assert _value(argv, "--hf-checkpoint") == _value(argv, "--ref-load") == model
    for flag, value in (("--tensor-model-parallel-size", "4"), ("--pipeline-model-parallel-size", "2"),
                        ("--expert-model-parallel-size", "8"), ("--megatron-to-hf-mode", "raw"),
                        ("--model-name", "deepseekv41"), ("--dsv4-impl", "miles"), ("--qkv-format", "bshd"),
                        ("--lora-type", "canonical_lora"), ("--target-modules", "attn"),
                        ("--exclude-modules", "*.ffn.experts.*"), ("--recompute-granularity", "full"),
                        ("--transformer-impl", "transformer_engine"), ("--seq-length", "65536")):
        assert _value(argv, flag) == value, flag
    for flag in ("--sequence-parallel", "--debug-train-only", "--moe-router-freeze-gate",
                 "--freeze-e-score-correction-bias", "--no-gradient-accumulation-fusion"):
        assert flag in argv, flag
    for flag in ("--num-query-groups", "--kv-channels", "--max-position-embeddings",
                 "--use-dynamic-batch-size", "--lora-base-cpu-backup"):
        assert flag not in argv, flag
    flags = [t for t in argv if t.startswith("--")]
    duplicated = {f for f in flags if flags.count(f) > 1}
    assert not duplicated, duplicated


def test_trainer_refuses_dynamic_batch_for_bshd(tmp_path):
    from yeto.rl.adapters.miles.config import MilesConfigError

    with pytest.raises(MilesConfigError, match="bshd"):
        _translate(tmp_path, "--rl-max-tokens-per-gpu", "65536")


def test_run_config_refuses_full_parameter_and_routed_targets(tmp_path):
    from yeto.rl.engine.run_config import resolve_rl_run_config

    _, args = _island_args(tmp_path)
    view = provider.provider_view(_config())
    args.lora_targets = "attention-routed-experts"
    with pytest.raises(ValueError):
        resolve_rl_run_config(args, model_path=tmp_path, prompt_path=tmp_path / "p", provider=view,
                              target_modules=["attn"], yeto_policy_sync=False, verify_ref_load=False)


def test_registry_has_the_family():
    from yeto.rl.model_registry import families

    entry = families.DEEPSEEK_V41_FLASH
    assert entry in families.FAMILIES
    assert "DeepseekV41ForCausalLM" in entry.match_architectures and "deepseek_v41" in entry.match_model_types
    recipe = entry.recipes[0]
    assert recipe.rl_model_recipe == "deepseek_v41" and tuple(recipe.tunings) == ("lora",)
