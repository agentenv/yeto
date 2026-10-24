"""Nemotron-H parameter rules: frozen vision, f32 Mamba vectors, expert coverage,
and the verl Megatron plan built from the official Nano v3 example."""

from __future__ import annotations

import pytest

from yeto.fragments import build_layout
from yeto.rl import hybrid_params as hp
from yeto.rl.adapters.verl import megatron_recipe as mr

# Names as SGLang's nano_nemotron_vl loader and HF nemotron_h define them.
VL_NAMES = {
    "vision_model.radio_model.model.blocks.0.attn.qkv.weight": 1280 * 3840,
    "mlp1.1.weight": 20480 * 5120,
    "language_model.backbone.embeddings.weight": 131072 * 64,
    "language_model.backbone.layers.0.mixer.A_log": 128,
    "language_model.backbone.layers.0.mixer.in_proj.weight": 64 * 64,
    "language_model.backbone.layers.1.mixer.experts.up_proj": 512 * 8 * 4,
    "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight": 64 * 16,
    "language_model.lm_head.weight": 131072 * 64,
}


def test_frozen_vision_names():
    assert hp.is_frozen_vision("vision_model.radio_model.x.weight")
    assert hp.is_frozen_vision("module.module.vision_projection.encoder.linear_fc1.weight")
    assert hp.is_frozen_vision("mlp1.0.weight")
    assert not hp.is_frozen_vision("language_model.backbone.layers.1.mixer.experts.up_proj")
    assert not hp.is_frozen_vision("decoder.layers.3.mlp.linear_fc1.weight")


def test_layout_excludes_frozen_vision():
    named = list(VL_NAMES.items())
    kept = hp.layout_named_numels(named, freeze_vision=True)
    assert {n for n, _ in kept} == {n for n in VL_NAMES if not n.startswith(("vision_model.", "mlp1."))}
    layout = build_layout(kept, 4, avg_name_regex=hp.HYBRID_AVG_NAME_REGEX)
    names = set(layout.tensor_names())
    assert not any(hp.is_frozen_vision(n) for n in names)
    assert len(names) == len(kept)
    # A_log is vector-like: avg fragment
    avg = layout.fragments[0]
    assert "language_model.backbone.layers.0.mixer.A_log" in dict(avg.tensors)
    assert hp.layout_named_numels(named, freeze_vision=False) == named
    with pytest.raises(ValueError):
        hp.layout_named_numels([("vision_model.a", 1)], freeze_vision=True)
    hp.check_frozen_excluded([n for n, _ in kept])
    with pytest.raises(hp.HybridParamError):
        hp.check_frozen_excluded(["mlp1.1.weight"])


@pytest.mark.parametrize("name", [
    "backbone.layers.0.mixer.A_log", "backbone.layers.0.mixer.D", "backbone.layers.0.mixer.dt_bias",
    "module.decoder.layers.2.mixer.A_log", "backbone.layers.1.mixer.gate.e_score_correction_bias",
    "decoder.layers.1.mlp.router.expert_bias",
])
def test_requires_f32(name):
    assert hp.requires_f32(name)


def test_check_f32():
    assert not hp.requires_f32("backbone.layers.0.mixer.in_proj.weight")
    assert not hp.requires_f32("backbone.layers.0.mixer.Dx")
    hp.check_f32({"backbone.layers.0.mixer.A_log": "torch.float32", "x.weight": "torch.bfloat16"})
    with pytest.raises(hp.HybridParamError, match="A_log"):
        hp.check_f32({"backbone.layers.0.mixer.A_log": "torch.bfloat16"})


def test_expert_coverage_hf_and_mcore():
    cov = hp.check_expert_coverage({"backbone.layers.1.mixer.experts.up_proj": (512, 8, 4),
                                    "backbone.layers.1.mixer.experts.down_proj": (512, 4, 8)}, num_experts=512)
    assert cov.moe_layers == (1,)
    with pytest.raises(hp.HybridParamError, match="511"):
        hp.check_expert_coverage({"backbone.layers.1.mixer.experts.up_proj": (511, 8, 4)}, num_experts=512)
    mcore = {f"decoder.layers.3.mlp.experts.{p}.weight{e}": (4, 4) for p in ("linear_fc1", "linear_fc2")
             for e in range(8)}
    assert hp.check_expert_coverage(mcore, num_experts=8).moe_layers == (3,)
    mcore.pop("decoder.layers.3.mlp.experts.linear_fc2.weight5")
    with pytest.raises(hp.HybridParamError, match=r"missing \[5\]"):
        hp.check_expert_coverage(mcore, num_experts=8)
    # latent projections and shared experts are dense, not routed experts
    with pytest.raises(hp.HybridParamError, match="no routed"):
        hp.check_expert_coverage({"backbone.layers.1.mixer.fc1_latent_proj.weight": (16, 64),
                                  "backbone.layers.1.mixer.shared_experts.up_proj.weight": (8, 64)},
                                 num_experts=8)


def test_layer_counts():
    nano = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"  # Nemotron-3 Nano config.json
    assert hp.layer_counts(nano) == {"mamba": 23, "moe": 23, "attention": 6, "mlp": 0}
    with pytest.raises(ValueError):
        hp.layer_counts("MEX")


def test_tiny_hf_nemotron_h_real_names():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "NemotronHConfig"):
        pytest.skip("transformers without nemotron_h")
    cfg = transformers.NemotronHConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=3, hybrid_override_pattern="ME*",
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, mamba_num_heads=4, mamba_head_dim=16,
        ssm_state_size=8, n_groups=2, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32, moe_latent_size=16, intermediate_size=32, use_mamba_kernels=False)
    model = transformers.NemotronHForCausalLM(cfg).to(torch.bfloat16)
    named = dict(model.named_parameters())
    f32 = [n for n in named if hp.requires_f32(n)]
    assert {n.rsplit(".", 1)[1] for n in f32} >= {"A_log", "D", "dt_bias"}
    # a plain .to(bf16) breaks the rule; restoring the f32 vectors passes it
    with pytest.raises(hp.HybridParamError):
        hp.check_f32({n: str(p.dtype) for n, p in named.items()})
    for n in f32:
        named[n].data = named[n].data.float()
    hp.check_f32({n: str(p.dtype) for n, p in named.items()})
    cov = hp.check_expert_coverage({n: tuple(p.shape) for n, p in named.items()}, num_experts=8)
    assert cov.moe_layers == (1,)
    assert any("fc1_latent_proj" in n for n in named)


def _plan(**kw):
    base = dict(model_path="/m", train_file="/t.parquet", val_file="/v.parquet", out_dir="/o")
    return mr.NemotronMegatronPlan(**{**base, **kw})


def test_megatron_plan_official_and_scaled():
    o = mr.build_overrides(_plan())
    assert "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8" in o
    assert "actor_rollout_ref.actor.megatron.expert_model_parallel_size=8" in o
    assert "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32" in o
    assert "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True" in o
    small = mr.build_overrides(_plan(n_gpus=2, tp=2, ep=2, rollout_tp=2, groups_per_round=8, samples_per_group=4))
    assert "trainer.n_gpus_per_node=2" in small
    assert "actor_rollout_ref.rollout.n=4" in small
    no_off = mr.build_overrides(_plan(offload=False))
    assert not any("optimizer_cpu_offload" in x for x in no_off)


@pytest.mark.parametrize("kw,match", [
    (dict(architecture="NemotronH_Omni_Reasoning_V3"), "视觉"),
    (dict(n_gpus=2, tp=4, ep=2, rollout_tp=2), "TP"),
    (dict(num_experts=128, ep=3, tp=1, n_gpus=3, rollout_tp=1), "EP"),
    (dict(n_gpus=2, tp=2, ep=2, rollout_tp=4), "rollout"),
])
def test_megatron_plan_refusals(kw, match):
    with pytest.raises(mr.NemotronPlanError, match=match):
        mr.validate(_plan(**kw))


def test_image_commands_pin_mamba_kernels():
    cmds = mr.image_commands()
    assert any("--extra megatron" in c for c in cmds)
    assert any("causal-conv1d==" in c for c in cmds) and any("mamba-ssm==" in c for c in cmds)


# Nemotron-3 Nano 30B-A3B config.json (HF, fields the FLOPs formula reads).
NANO = {"hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME", "hidden_size": 2688,
        "vocab_size": 131072, "head_dim": 128, "num_attention_heads": 32, "num_key_value_heads": 2,
        "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4,
        "n_routed_experts": 128, "num_experts_per_tok": 6, "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712, "n_shared_experts": 1, "moe_latent_size": None,
        "intermediate_size": 1856}


def test_hybrid_flops_matches_active_params():
    from yeto.rl import hybrid_flops as hf
    per_token = hf.forward_flops(NANO, [1])
    # 2 x active parameters (model card: ~3.5B active incl. embeddings, ~0.35B of them)
    assert 3.0e9 < per_token / 2 < 3.5e9
    assert hf.train_flops(NANO, [10, 20]) == 3 * hf.forward_flops(NANO, [10, 20])
    # attention adds an s^2 term only for the 6 attention layers
    extra = hf.forward_flops(NANO, [4096]) - 4096 * per_token
    assert extra == pytest.approx(6 * 2 * 32 * 128 * (4096 ** 2 - 4096))
    assert hf.mfu(NANO, [1000], 1.0, 1) == pytest.approx(hf.train_flops(NANO, [1000]) / 989.4e12)
    latent = {**NANO, "moe_latent_size": 1024}
    assert hf.forward_flops(latent, [1]) < per_token  # experts run at the latent width
