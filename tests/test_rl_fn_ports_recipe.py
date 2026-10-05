"""G1: Qwen3.8-Flash-Next on the ports path (``--rl-engine ports``) renders the
same model + native-LoRA flags as ``scripts/run_qwen3_8_next.py`` (CPU only)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.profiles import qwen3_8_next as q

from test_rl_miles_adapter_config import flag_value, make_config


def _provider(**kw):
    base = dict(experimental_attention_variant="gated_delta_net", num_moe_experts=512,
                hidden_size=2560, num_layers=48)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("model,variant", [
    ("Qwen/Qwen3.8-Flash-Next", "full"), ("/root/models/Qwen3.8-Flash-Next/", "full"),
    ("CharyZeng/Qwen3.8-Flash-Next-4layer", "4layer"),
])
def test_variant_from_model_id(model, variant):
    assert rc.qwen3_8_next_variant(SimpleNamespace(model=model), SimpleNamespace()) == variant


def test_variant_from_provider_fingerprint_and_qwen35_untouched():
    args = SimpleNamespace(model="/root/models/local-ckpt")
    assert rc.qwen3_8_next_variant(args, _provider()) == "full"
    assert rc.qwen3_8_next_variant(args, _provider(num_layers=4)) == "4layer"
    # Qwen3.5 / 3.6 GDN hybrids (no 512-expert MoE) keep the qwen3_5 recipe
    assert rc.qwen3_8_next_variant(args, _provider(num_moe_experts=None)) is None
    assert rc.qwen3_8_next_variant(args, _provider(experimental_attention_variant=None)) is None


def _fn_config(layers=48, expert_rank=None):
    cfg = make_config(colocated=False, with_moe=True)
    cfg = dataclasses.replace(
        cfg,
        model_recipe=dataclasses.replace(cfg.model_recipe, name=rc.RECIPE_QWEN3_8_NEXT,
                                         gdn=rc.GdnRecipe(gated_delta_net=True)),
        geometry=dataclasses.replace(cfg.geometry, num_layers=layers),
        serving=dataclasses.replace(cfg.serving, offload_train=False),
    )
    return cfg


def _flags(argv):
    return [t for t in argv if t.startswith("--")]


def test_full_recipe_matches_profile_model_args_and_native_lora():
    argv = list(mc.translate_run_config(_fn_config(), AlgorithmSpec()).argv)
    flags = _flags(argv)
    dup = {f for f in flags if flags.count(f) > 1}
    assert not dup, dup
    ma = q.model_args("full")
    i = argv.index(ma[0])
    assert tuple(argv[i:i + len(ma)]) == ma  # verbatim, 48 layers, 512 experts
    assert flag_value(argv, "--num-layers") == "48" and flag_value(argv, "--num-experts") == "512"
    assert flag_value(argv, "--model-name") == "qwen4_exp"
    assert flag_value(argv, "--custom-model-provider-path") == q.PORTS_PROVIDER_PATH
    assert flag_value(argv, "--qkv-format") == "thd"
    assert flag_value(argv, "--megatron-to-hf-mode") == "raw"
    assert flag_value(argv, "--lora-expert-rank") == "8"
    targets = flag_value(argv, "--target-modules").split(",")
    assert targets == list(q.LORA_TARGET_MODULES) and len(targets) == 14
    # MoE experts + shared + GDN + QSA all covered
    for leaf in ("mlp.experts.gate_up_proj", "mlp.shared_expert.down_proj",
                 "linear_attn.in_proj_qkv", "self_attn.q_proj"):
        assert any(t.endswith(leaf) for t in targets)
    assert "--lora-type" not in argv and "--lora-base-cpu-backup" not in argv
    assert "--colocate" not in argv and flag_value(argv, "--update-weight-transfer-mode") == "broadcast"


def test_4layer_variant_and_expert_rank():
    cfg = _fn_config(layers=4)
    cfg = dataclasses.replace(cfg, trainable=dataclasses.replace(cfg.trainable, lora_rank=16))
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert flag_value(argv, "--num-layers") == "4" and flag_value(argv, "--lora-expert-rank") == "16"
    with pytest.raises(mc.MilesConfigError):
        mc.translate_run_config(_fn_config(layers=7), AlgorithmSpec())


def test_apply_ports_recipe_drops_values_too():
    out = q.apply_ports_recipe(["train.py", "--num-layers", "4", "--swiglu", "--lora-type", "c", "--x", "1"],
                               ["--num-layers", "48"])
    assert out == ("train.py", "--swiglu", "--x", "1", "--num-layers", "48")
