"""Kimi-K3.1 partial-experts recipe: constants, argv snapshot, constraints, manifest, runtime."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from yeto.rl.adapters.miles.models.kimi_k31 import model, ports_recipe, ports_runtime, release
from yeto.rl.adapters.miles.models.partial_experts import config as pe_config
from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection, SelectionError


def _selection(ep=8, layers=range(1, 93), model_name="kimi_k31", n=896, k=128):
    return ExpertSelection.synthetic("uniform", model=model_name, num_experts=n, num_trainable=k,
                                     ep_size=ep, layers=layers)


# -- constants ----------------------------------------------------------------


def test_model_constants_match_the_checkpoint():
    assert (model.NUM_EXPERTS, model.TOPK, model.NUM_MOE_LAYERS, model.MOE_FFN_HIDDEN_SIZE,
            model.MOE_LATENT_SIZE, model.HIDDEN_SIZE) == (896, 16, 92, 3072, 3584, 7168)
    assert model.NUM_TRAINABLE == 128 and model.EP_SIZE_DEFAULT == 8
    assert model.EXPERT_PARAMETERS == 33_030_144
    assert model.moe_layers() == tuple(range(1, 93))
    assert model.hf_expert_weight_name(5, 895, 3) == (
        "language_model.model.layers.5.block_sparse_moe.experts.895.w3.weight")
    with pytest.raises(ValueError):
        model.hf_expert_weight_name(0, 1, 1)  # layer 0 is dense
    cfg = pe_config.model_config("kimi_k31")
    assert cfg is model.MODEL_CONFIG
    assert cfg.allow_distributed_optimizer and not cfg.requires_complete_master
    assert cfg.trainable_param_fragment == ".mlp.experts.trainable."
    assert cfg.routed_experts_dtype == "uint16" and not cfg.needs_engine_patch


def _hf_config(**overrides):
    text = {
        "model_type": "kimi_k31", "num_hidden_layers": 93, "first_k_dense_replace": 1,
        "num_experts": 896, "num_experts_per_token": 16, "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584, "hidden_size": 7168, "num_expert_group": 1, "topk_group": 1,
    }
    text.update(overrides)
    return {"model_type": "kimi_k31", "text_config": text}


def test_moe_layer_count_matches_the_sampler_derivation():
    assert model.moe_layer_count(_hf_config()) == 92 == model.NUM_MOE_LAYERS == len(model.moe_layers())
    assert model.moe_layer_count({"num_hidden_layers": 4, "first_k_dense_replace": 1}) == 3  # K3-4layer
    with pytest.raises(ValueError, match="lacks num_hidden_layers"):
        model.moe_layer_count({})


def test_hf_config_check():
    assert model.check_hf_config(_hf_config())["num_experts"] == 896
    with pytest.raises(ValueError, match="num_experts=512"):
        model.check_hf_config(_hf_config(num_experts=512))
    with pytest.raises(ValueError, match="grouped top-k"):
        model.check_hf_config(_hf_config(num_expert_group=8))
    with pytest.raises(ValueError, match="not Kimi-K3.1"):
        model.check_hf_config({"model_type": "deepseek_v4"})
    with pytest.raises(ValueError, match="lacks"):
        model.check_hf_config(_hf_config(hidden_size=None))


# -- recipe argv --------------------------------------------------------------


def test_recipe_argv_snapshot():
    argv = ports_recipe.recipe_argv(selection_path="/run/expert_selection.json", ep_size=8,
                                    context_parallel_size=1, seq_length=8192)
    assert argv[:3] == ["--spec", "miles_plugins.models.kimi_k31", "get_kimi_k31_spec"]
    assert argv[-6:] == [
        "--megatron-to-hf-mode", "raw",
        "--use-rollout-routing-replay",
        "--no-enable-sample-ownership-checker",
        "--partial-experts-selection", "/run/expert_selection.json",
    ]
    text = " ".join(argv)
    for fragment in (
        "--num-experts 896", "--moe-router-topk 16", "--moe-ffn-hidden-size 3072", "--moe-latent-size 3584",
        "--hidden-size 7168", "--num-layers 93", "--moe-layer-freq [0]+[1]*92",
        "--moe-router-bias-update-rate 0", "--moe-router-enable-expert-bias",
        "--moe-router-score-function sigmoid", "--moe-token-dispatcher-type alltoall",
    ):
        assert fragment in text, fragment
    assert text.count("--moe-router-bias-update-rate") == 1
    assert "--num-layers-per-virtual-pipeline-stage" not in text
    # once, in a fixed order: the snapshot other workflows build on
    assert len(argv) == len(ports_recipe.MODEL_ARGV) + len(ports_recipe.RECIPE_ARGV) + 2
    short = ports_recipe.recipe_argv(selection_path="s.json", seq_length=4096, include_model_argv=False)
    assert short == [*ports_recipe.RECIPE_ARGV, "--partial-experts-selection", "s.json"]


@pytest.mark.parametrize("ep", [1, 2, 4, 8, 16, 32, 64, 128])
def test_recipe_accepts_power_of_two_ep_dividing_128(ep):
    assert ports_recipe.check_expert_parallel(ep) == ep
    summary = ports_recipe.trainable_summary(ep)
    assert summary["local_trainable"] * ep == 128 and summary["local_experts"] * ep == 896


@pytest.mark.parametrize("ep", [3, 7, 256, 0])
def test_recipe_refuses_other_ep(ep):
    with pytest.raises(ValueError):
        ports_recipe.check_expert_parallel(ep)


def test_recipe_zigzag_cp_and_vpp_constraints():
    assert ports_recipe.check_zigzag_context_parallel(1, 1000) == 1
    assert ports_recipe.check_zigzag_context_parallel(4, 8192) == 4
    with pytest.raises(ports_recipe.KimiK31RecipeError, match="divisible by 8"):
        ports_recipe.check_zigzag_context_parallel(4, 8190)
    with pytest.raises(ports_recipe.KimiK31RecipeError, match="power of two"):
        ports_recipe.check_zigzag_context_parallel(3, 8192)
    with pytest.raises(ports_recipe.KimiK31RecipeError, match="VPP"):
        ports_recipe.recipe_argv(selection_path="s", seq_length=8192,
                                 extra_argv=["--num-layers-per-virtual-pipeline-stage", "2"])
    with pytest.raises(ports_recipe.KimiK31RecipeError, match="VPP"):
        ports_recipe.refuse_virtual_pipeline(["--num-virtual-stages-per-pipeline-rank=3"])
    with pytest.raises(ports_recipe.KimiK31RecipeError, match="selection file"):
        ports_recipe.recipe_argv(selection_path="", seq_length=8192)


# -- release manifest ---------------------------------------------------------


def test_release_manifest_fields_and_check():
    selection = _selection()
    fields = release.release_manifest_fields(selection)
    assert fields["expert_selection_sha256"] == selection.sha256
    assert fields["partial_experts"]["layers"] == 92
    assert fields["partial_experts"]["trainable_sha256"] == selection.trainable_sha256
    release.check_release_manifest(fields, expected_selection_sha256=selection.sha256)
    with pytest.raises(ValueError, match="!= running selection"):
        release.check_release_manifest(fields, expected_selection_sha256="0" * 64)
    with pytest.raises(ValueError, match="lacks expert_selection_sha256"):
        release.check_release_manifest({}, expected_selection_sha256=selection.sha256)


def test_release_refuses_foreign_or_partial_selection():
    with pytest.raises(ValueError, match="not 'kimi_k31'"):
        release.release_manifest_fields(_selection(model_name="toy"))
    with pytest.raises(ValueError, match="not 'kimi_k31'"):
        release.release_manifest_fields(_selection(model_name="toy"))
    with pytest.raises(SelectionError, match="model MoE layers"):
        release.release_manifest_fields(_selection(layers=range(1, 92)))  # layer 92 missing
    with pytest.raises(SelectionError, match="model MoE layers"):
        release.release_manifest_fields(_selection(layers=range(0, 92)))  # dense layer 0 included


@pytest.mark.parametrize("k", [8, 32, 64, 128])
def test_release_accepts_a_trainable_count_that_splits_over_ep(k):
    fields = release.release_manifest_fields(_selection(k=k))
    assert fields["partial_experts"]["num_trainable"] == k


def test_release_refuses_a_trainable_count_that_does_not_split_over_ep():
    with pytest.raises(ValueError):  # the selection itself or the recipe refuses it
        release.release_manifest_fields(_selection(k=36))


def test_raw_hf_mode_exception_covers_glm5_and_k31():
    assert release.megatron_to_hf_mode("kimi_k31") == "raw"
    assert release.megatron_to_hf_mode("glm5_next") == "raw"
    assert release.megatron_to_hf_mode("generic") == "bridge"
    assert release.megatron_to_hf_mode("deepseek-v4-flash") == "bridge"
    from yeto.rl.engine.run_config import RECIPE_GLM5_NEXT, RECIPE_KIMI_K31

    assert release.uses_raw_hf_mode(RECIPE_KIMI_K31) and release.uses_raw_hf_mode(RECIPE_GLM5_NEXT)


# -- runtime ------------------------------------------------------------------


def _megatron_stubs(monkeypatch):
    opt_module = ModuleType("megatron.core.optimizer")
    opt_module.get_standard_config_overrides = lambda config: {}
    cfg_module = ModuleType("megatron.core.optimizer.optimizer_config")
    cfg_module.ParamKey = lambda *, attr: ("ParamKey", attr)
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
    monkeypatch.setitem(sys.modules, opt_module.__name__, opt_module)
    monkeypatch.setitem(sys.modules, cfg_module.__name__, cfg_module)


def test_runtime_allows_distributed_optimizer_and_groups_experts(monkeypatch):
    _megatron_stubs(monkeypatch)
    monkeypatch.setattr(ports_runtime, "validate_optimizer_masters", lambda *_: None)
    calls = []

    class Model:
        @staticmethod
        def get_megatron_optimizer(**kwargs):
            calls.append(kwargs)
            return "optimizer"

    ports_runtime.install_on_model(Model)
    ports_runtime.install_on_model(Model)  # idempotent
    monkeypatch.delenv(ports_runtime.EXPERT_LR_ENV, raising=False)
    assert Model.get_megatron_optimizer(config=SimpleNamespace(use_distributed_optimizer=True), model_chunks=[]) == "optimizer"
    assert calls[-1]["config_overrides"] == {}
    monkeypatch.setenv(ports_runtime.EXPERT_LR_ENV, "2e-6")
    Model.get_megatron_optimizer(config=SimpleNamespace(use_distributed_optimizer=True), model_chunks=[])
    assert calls[-1]["config_overrides"] == {("ParamKey", "partial_expert"): {"max_lr": 2e-6, "min_lr": 2e-6}}
    monkeypatch.setenv(ports_runtime.EXPERT_LR_ENV, "-1")
    with pytest.raises(RuntimeError, match="positive"):
        Model.get_megatron_optimizer(config=SimpleNamespace(use_distributed_optimizer=False), model_chunks=[])


def test_runtime_master_check_accepts_sharded_masters():
    import torch

    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    parameter.main_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))  # DP shard
    parameter.main_param_sharded = True
    chunk = torch.nn.Module()
    chunk.register_parameter("w", parameter)
    optimizer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": [parameter.main_param]}]))
    ports_runtime.validate_optimizer_masters([chunk], optimizer)
    parameter.main_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="FP32 master"):
        ports_runtime.validate_optimizer_masters([chunk], optimizer)


def test_install_arms_only_the_model_module(monkeypatch):
    armed = []
    monkeypatch.setattr(ports_runtime, "install_or_defer", lambda name, installer: armed.append(name))
    ports_runtime.install()
    assert armed == ["miles.backends.megatron_utils.model"]


# -- the adapter's argv -------------------------------------------------------


def test_translate_run_config_emits_raw_for_k31_and_bridge_otherwise():
    """config.py reads the raw-mode exception from the K3.1 release module."""

    import inspect

    from yeto.rl.adapters.miles import config as mc

    source = inspect.getsource(mc.translate_run_config)
    assert "kimi_k31_release.megatron_to_hf_mode(recipe.name)" in source


def test_a_kimi_k3_checkpoint_takes_the_k3_spec_and_bridge():
    """A 4-layer Kimi-K3 cut (HF model_type kimi_k3, no DSA indexer) runs the recipe
    with the K3 spec and bridge; the K3.1 ones read text_config.dsa_config (B300 run16)."""
    from yeto.rl.adapters.miles.models.kimi_k31 import ports_recipe as r

    assert r.spec_for("kimi_k31") == r.SPEC and r.model_name_for("kimi_k31") == "kimi_k31"
    assert r.spec_for("kimi_k3") == ("miles_plugins.models.kimi_k3", "get_kimi_k3_spec")
    assert r.model_name_for("kimi_k3") == "kimi_k3"
    argv = r.recipe_argv(selection_path="/s.json", ep_size=8, seq_length=4096, num_layers=4,
                         first_k_dense_replace=1, hf_model_type="kimi_k3")
    assert argv[:3] == ["--spec", "miles_plugins.models.kimi_k3", "get_kimi_k3_spec"]
    assert r.recipe_argv(selection_path="/s.json", ep_size=8, seq_length=4096)[:3] == ["--spec", *r.SPEC]


def test_release_accepts_a_cut_down_checkpoint_expert_count():
    # SC2's cut-down checkpoint: 64 routed experts, 16 trainable over EP 2, MoE layers 1..3
    selection = _selection(ep=2, layers=range(1, 4), n=64, k=16)
    fields = release.release_manifest_fields(selection, layers=range(1, 4), num_experts=64)
    assert fields["partial_experts"]["num_experts"] == 64
    assert fields["partial_experts"]["num_trainable"] == 16
    assert fields["expert_selection_sha256"] == selection.sha256
    # without the checkpoint's count the Pro default (896) still applies
    with pytest.raises(ValueError, match="896 experts"):
        release.release_manifest_fields(selection, layers=range(1, 4))
    # a selection of another expert count than the checkpoint is refused
    with pytest.raises(ValueError, match="64 experts"):
        release.release_manifest_fields(_selection(layers=range(1, 4)), layers=range(1, 4), num_experts=64)
    with pytest.raises(ValueError):
        release.release_manifest_fields(_selection(ep=2, layers=range(1, 4), n=64, k=15),
                                        layers=range(1, 4), num_experts=64)


def test_recipe_argv_carries_the_checkpoint_expert_count():
    pro = ports_recipe.recipe_argv(selection_path="s.json", seq_length=4096)
    assert pro.count("--num-experts") == 1 and pro[pro.index("--num-experts") + 1] == "896"
    cut = ports_recipe.recipe_argv(selection_path="s.json", seq_length=4096, ep_size=2, num_layers=4,
                                   num_experts=64)
    assert cut.count("--num-experts") == 1 and cut[cut.index("--num-experts") + 1] == "64"
    with pytest.raises(ports_recipe.KimiK31RecipeError):
        ports_recipe.model_argv(num_experts=0)


def test_full_recompute_is_turned_on_for_the_frozen_mxfp4_experts():
    from yeto.rl.adapters.miles import config as miles_config

    assert miles_config._full_recompute_for_mxfp4_experts(["--x", "1"]) == [
        "--x", "1", *miles_config.FULL_RECOMPUTE_ARGV]
    kept = ["--recompute-granularity", "full", "--recompute-method", "block"]
    assert miles_config._full_recompute_for_mxfp4_experts(kept) == kept
    with pytest.raises(miles_config.MilesConfigError, match="selective"):
        miles_config._full_recompute_for_mxfp4_experts(["--recompute-granularity", "selective"])
