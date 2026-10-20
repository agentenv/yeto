"""Rollout quant tiers on the engine sampler island (S26 E6): the spec written by the
launch layer, the settings filled on the node, the dense expert count."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_rl_quant_tiers import write_checkpoint
from tests.test_rl_quant_scope import dense_shapes, moe_shapes
from yeto.rl.adapters.sampler_engines import quant_tier
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, apply_rollout_quant
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.quant import tiers
from yeto.rl.quant.schemes import FP8_FFN_CHANNEL, release_dir_for


def _spec(base: Path, tier: int, adapter: str = "sglang-server", **server) -> dict:
    return {"engine": {"factory": "x:y", "adapter": adapter,
                       "server": {"base_dir": str(base), "quant_scheme": "bf16", "precision": "bf16",
                                  "extra_args": [], "extra_env": {}, **server}},
            "base_dir": str(base), "release_dir": "policy-releases", "checks": {},
            "event_tape": str(base.parent / "tape" / "events.jsonl"),
            **({"rollout_quant": s} if (s := quant_tier.spec_section(tier)) else {})}


@pytest.fixture
def dense(tmp_path):
    return write_checkpoint(tmp_path / "dense", dense_shapes(), {"num_hidden_layers": 2})


def test_spec_section_and_tier_0_is_unchanged(dense):
    assert quant_tier.spec_section(0) is None
    assert quant_tier.spec_section(1) == {"tier": 1, "apply": quant_tier.APPLY}
    with pytest.raises(quant_tier.QuantTierSpecError):
        quant_tier.spec_section(3)
    spec = _spec(dense, 0)
    assert apply_rollout_quant(spec) is spec


def test_tier_1_fills_sglang_flags_and_ignore_list(dense):
    out = apply_rollout_quant(_spec(dense, 1))
    server = out["engine"]["server"]
    assert server["extra_args"] == ["--quantization", "fp8", "--kv-cache-dtype", "fp8_e4m3"]
    assert "model.layers.0.self_attn.q_proj" in server["extra_env"][tiers.SGLANG_IGNORE_ENV]
    assert server["quant_scheme"] == "bf16" and server["precision"] == "fp8"
    assert out["release_dir"] == "policy-releases" and out["rollout_quant"]["plan"]["quantized_modules"] == 6


def test_tier_1_on_vllm_and_double_quantization_are_refused(dense):
    with pytest.raises(SamplerSpecError, match="no ignore list"):
        apply_rollout_quant(_spec(dense, 1, adapter="vllm-server"))
    with pytest.raises(SamplerSpecError, match="already sets"):
        apply_rollout_quant(_spec(dense, 1, extra_args=["--quantization", "w8a8_fp8"]))
    with pytest.raises(SamplerSpecError, match="already sets"):
        apply_rollout_quant(_spec(dense, 2, quant_scheme="mxfp4-g32-rtn"))


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_tier_2_converts_the_base_and_sets_scheme_format_and_expert_count(tmp_path, kind):
    shapes, config = (dense_shapes(), {"num_hidden_layers": 2}) if kind == "dense" else \
        (moe_shapes(), {"num_hidden_layers": 2, "num_experts": 2})
    base = write_checkpoint(tmp_path / kind, shapes, config)
    out = apply_rollout_quant(_spec(base, 2, adapter="vllm-server"))
    server = out["engine"]["server"]
    converted = Path(server["base_dir"])
    assert converted == tmp_path / "tape" / "fp8-base" and out["base_dir"] == str(converted)
    assert tiers.check_release_scope(converted) == 6
    assert (server["quant_scheme"], server["expected_format"]) == (FP8_FFN_CHANNEL, "float-quantized")
    assert out["release_dir"] == release_dir_for(FP8_FFN_CHANNEL)
    assert out["checks"]["expected_expert_matrices"] == (0 if kind == "dense" else 6)
    assert out["rollout_quant"]["bf16_base"] == str(base)


def test_tier_2_refuses_a_contradicting_expert_count(dense):
    spec = _spec(dense, 2)
    spec["checks"] = {"expected_expert_matrices": 36288}
    with pytest.raises(SamplerSpecError, match="36288"):
        apply_rollout_quant(spec)


def test_dense_expert_count_zero_passes_without_engine_report():
    engine = FakeEngine(expert_matrices=None)
    result = PostLoadChecks().expert_count(engine, 0)
    assert result["passed"] and "dense" in result["detail"]
    assert not PostLoadChecks().expert_count(engine, 6)["passed"]  # MoE still needs a report


def test_stage1_writes_the_tier_into_the_sampler_spec():
    from yeto.rl import stage1

    doc = {"rollout_quant_tier": 1}
    plan = type("P", (), {"doc": doc})()
    assert stage1.quant_spec_section(plan) == {"tier": 1, "apply": quant_tier.APPLY}
    plan.doc = {}
    assert stage1.quant_spec_section(plan) is None
