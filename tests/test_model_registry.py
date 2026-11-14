"""Model registry (yeto/rl/model_registry) and `yeto adapt`: golden recommendations for
Qwen3-0.6B, Qwen3-30B-A3B, DeepSeek-V3 and Kimi-K3.1 config.json fragments on NVIDIA and
Ascend, pin references, CLI-flag compatibility, and the off-by-default contract."""

from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path

import pytest

from yeto import memory_estimate
from yeto.rl.model_registry import Constraints, facts_from_config, recommend
from yeto.rl.model_registry.cli import advise_launch
from yeto.rl.model_registry.families import ALL_ENTRIES
from yeto.rl.model_registry.recommend import hardware_facts
from yeto.rl.model_registry.render import as_dict, cli_flags, config_draft

DATA = Path(__file__).parent / "data" / "model_registry"
GOLDEN = DATA / "golden.json"
MODELS = ("qwen3_0p6b", "qwen3_30b_a3b", "deepseek_v3", "kimi_k31")
HARDWARE = ("nvidia", "ascend")


def _cfg(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def _rec(name: str, hardware: str, **kw):
    return recommend(facts_from_config(_cfg(name)), hardware_facts(hardware), Constraints(**kw),
                     card_explicit=False)


def _snapshot() -> dict:
    out = {}
    for m in MODELS:
        for h in HARDWARE:
            d = as_dict(_rec(m, h), m)
            d["config_draft"] = config_draft(_rec(m, h), m)
            out[f"{m}/{h}"] = d
    return json.loads(json.dumps(out))


def test_golden_snapshot():
    """Edit an entry -> regenerate with YETO_UPDATE_GOLDEN=1 and review the diff."""
    snap = _snapshot()
    if os.environ.get("YETO_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    assert snap == json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name,family,total_b,active_b", [
    ("qwen3_0p6b", "qwen3-dense", 0.596, 0.596),
    ("qwen3_30b_a3b", "qwen3-moe", 30.53, 3.35),
    ("deepseek_v3", "deepseek-v3", 671.0, 37.5),
    ("kimi_k31", "kimi_k31", 2757.0, 82.6),
])
def test_facts_and_family(name, family, total_b, active_b):
    rec = _rec(name, "nvidia")
    assert rec.family == family
    assert rec.facts.total_params_b == pytest.approx(total_b, rel=0.01)
    assert rec.facts.active_params_b == pytest.approx(active_b, rel=0.01)


def test_qwen3_0p6b():
    nv, asc = _rec("qwen3_0p6b", "nvidia"), _rec("qwen3_0p6b", "ascend")
    assert nv.supported and nv.recipe.name == "qwen3-dense/cuda/miles" and nv.recipe.status == "verified"
    assert asc.supported and asc.recipe.trainer == "verl" and asc.recipe.sampler == "vllm-ascend"
    assert asc.trainer_parallel["engine"] == "fsdp2" and asc.trainer_parallel["world"] == 1


def test_qwen3_moe_picks_ep_that_divides_experts():
    rec = _rec("qwen3_30b_a3b", "nvidia")
    ep = rec.trainer_parallel["expert_parallel"]
    assert rec.supported and 128 % ep == 0 and "--expert-parallel" in rec.to_cli_flags()


def test_deepseek_v3_nvidia_lora_and_full_refused():
    rec = _rec("deepseek_v3", "nvidia")
    assert rec.supported and rec.facts.attention == "mla" and rec.facts.first_k_dense == 3
    assert rec.trainer_parallel["nodes"] > 1 and rec.sampler_parallel["nodes"] > 1
    full = _rec("deepseek_v3", "nvidia", tuning="full")
    assert not full.supported and any("optimizer state" in why for _, why in full.rejected)
    assert full.alternatives


def test_unsupported_on_ascend_has_reason_and_alternative():
    for name in ("deepseek_v3", "kimi_k31"):
        rec = _rec(name, "ascend")
        assert not rec.supported and rec.recipe.status == "unsupported"
        assert rec.recipe.reason and rec.alternatives and cli_flags(rec) == []


def test_kimi_k31_sizes_to_the_delivered_six_b300_nodes():
    rec = _rec("kimi_k31", "nvidia")
    t = rec.trainer_parallel
    assert rec.supported and rec.hardware.card == "B300"
    assert (t["nodes"], t["expert_parallel"], t["tuning"]) == (6, 8, "full")
    assert rec.recipe.trainer_image.endswith(":MILES_K31_FULL_IMAGE")
    assert rec.recipe.sampler_image.endswith(":VLLM_SAMPLER_IMAGE_ROCM")
    assert rec.sampler_parallel["card"] == "MI355X" and rec.sampler_parallel["nodes"] == 1
    assert "--rl-expert-selection" in rec.to_cli_flags()
    lora = _rec("kimi_k31", "nvidia", tuning="lora")
    assert lora.recipe.trainer_image.endswith(":MILES_K31_IMAGE")


def test_120b_dense_on_ascend_gets_a_multi_node_layout():
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 12288,
           "intermediate_size": 33024, "num_hidden_layers": 64, "num_attention_heads": 96,
           "num_key_value_heads": 8, "vocab_size": 128256}
    rec = recommend(facts_from_config(cfg), hardware_facts("ascend"), Constraints(tuning="full"))
    assert 100 < rec.facts.total_params_b < 140
    t = rec.trainer_parallel
    assert rec.supported and t["nodes"] >= 5 and t["tensor_parallel"] * t["pipeline_parallel"] > 1
    assert rec.memory["verl_rank0_fp32_load_gib_unpatched"] > 300
    gloo = recommend(facts_from_config(cfg), hardware_facts("ascend", interconnect="gloo-only"),
                     Constraints(tuning="full"))
    assert not gloo.supported


def test_unknown_model_falls_back_with_a_warning():
    cfg = {"architectures": ["FooForCausalLM"], "model_type": "foo", "hidden_size": 1024,
           "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 1000}
    rec = recommend(facts_from_config(cfg), hardware_facts("nvidia"))
    assert rec.family == "generic-dense" and any("no family entry" in w for w in rec.warnings)


def test_every_image_reference_resolves_to_a_pin():
    for entry in ALL_ENTRIES:
        for r in entry.recipes:
            for ref in (r.trainer_image, r.sampler_image):
                if not ref:
                    assert r.status == "unsupported"
                    continue
                module, const = ref.split(":")
                value = getattr(importlib.import_module(module), const)
                assert isinstance(value, str) and value
                if not module.endswith("verl.pins"):  # verl images are local build tags / CANN base
                    assert "@sha256:" in value, ref


def test_recipes_are_existing_rl_model_recipe_choices():
    from yeto.rl.adapters.miles.run_flags import RL_FLAGS
    choices = next(f.kwargs["choices"] for flags in RL_FLAGS.values() for f in flags
                   if "--rl-model-recipe" in f.options)
    for entry in ALL_ENTRIES:
        for r in entry.recipes:
            assert r.rl_model_recipe is None or r.rl_model_recipe in choices
            assert (r.rl_model_recipe is None) == (r.status == "unsupported")


@pytest.mark.parametrize("name", ["qwen3_0p6b", "qwen3_30b_a3b", "deepseek_v3"])
def test_cli_flags_parse_with_the_launch_parser(name):
    from yeto.cli import parse_args
    for hardware in HARDWARE:
        flags = _rec(name, hardware).to_cli_flags()
        if not flags:
            continue
        args = parse_args(["--gpu", "aws:1xa100@us-east-1", "--data", "org/data", *flags])
        assert args.training_mode == "rl" and args.rl_model_auto is False


def test_memory_estimate_default_unchanged_and_precise_is_opt_in():
    ds = _cfg("deepseek_v3")
    # the historical upper bound (unchanged); precise counts MLA, shared, dense-first
    assert memory_estimate.arch_from_hf_config(ds).params == 705_256_839_168
    assert memory_estimate.arch_from_hf_config(_cfg("qwen3_0p6b")).params == 654_761_984
    precise = memory_estimate.arch_from_hf_config(ds, precise=True).params
    assert precise == pytest.approx(671.0e9, rel=0.005)
    assert memory_estimate.arch_from_hf_config(ds, precise=True).layers == 61


def test_adapt_cli_exit_codes(capsys, tmp_path):
    from yeto.cli import main
    out = tmp_path / "draft.json"
    assert main(["adapt", str(DATA / "qwen3_0p6b.json"), "--hardware", "nvidia",
                 "--write-config", str(out)]) == 0
    assert json.loads(out.read_text())["trainer"]["argv"][:2] == ["--training-mode", "rl"]
    capsys.readouterr()
    assert main(["adapt", str(DATA / "kimi_k31.json"), "--hardware", "ascend", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["supported"] is False and report["alternatives"] and report["config_draft"] is None
    assert main(["adapt", str(tmp_path / "missing"), "--hardware", "nvidia"]) == 2


def test_rl_model_auto_is_off_by_default_and_advisory():
    from tests.test_rl_engine_selection import _cli
    args = _cli()
    assert args.rl_model_auto is False
    before = dict(vars(args))
    buf = io.StringIO()
    advise_launch(args, stream=buf)          # org/model has no config.json: a note, no raise
    assert "no recommendation" in buf.getvalue() and vars(args) == before
    args.model = str(DATA / "qwen3_0p6b.json")
    args.rl_model_recipe = "kimi_k31"
    buf = io.StringIO()
    advise_launch(args, stream=buf)
    assert "qwen3-dense" in buf.getvalue() and "explicit flag is used" in buf.getvalue()
    assert args.rl_model_recipe == "kimi_k31"
