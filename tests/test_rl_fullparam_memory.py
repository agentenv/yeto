"""S25 N7.1: full-parameter memory estimate and the refusal before launch."""

from __future__ import annotations

from dataclasses import replace

import pytest

from yeto.rl.fullparam.memory import (
    GIB,
    MemoryPlanError,
    ModelShape,
    TrainPlan,
    check,
    estimate,
    plan_for_nodes,
    sweep,
)

# GLM5.3-Flash text model counted on the meta device (transformers 5.16.1, 2026-10-16).
GLM = ModelShape(params=313_326_799_870, largest_layer_params=7_412_629_750, num_layers=45, hidden_size=4096,
                 vocab_size=154_880, residual_streams=4, head_counts=(32, 64), dsa_layers=11, index_heads=32,
                 index_kpool=4)
PPU = TrainPlan(cards=64, cards_per_node=8, hbm_gib=144, host_memory_gib=1024, seq_len=8192)


def test_sharded_state_is_16_bytes_per_parameter_over_the_group():
    est = estimate(GLM, PPU)
    state = sum(est.items[k].bytes for k in ("params_shard", "grads_shard", "optimizer_shard"))
    assert state == pytest.approx(16 * GLM.params / 64)
    assert est.total("host") == 0


def test_offload_moves_state_to_the_host_per_node():
    opt = estimate(GLM, replace(PPU, offload="optimizer"))
    assert opt.items["host_master_and_optimizer"].bytes == pytest.approx(12 * GLM.params / 64 * 8)
    full = estimate(GLM, replace(PPU, offload="full"))
    assert full.total("host") == pytest.approx(16 * GLM.params / 64 * 8)
    assert full.total("device") < opt.total("device") < estimate(GLM, PPU).total("device")


def test_ulysses_divides_activations_but_not_the_hf_dsa_mask():
    a = estimate(GLM, replace(PPU, seq_len=32768))
    b = estimate(GLM, replace(PPU, seq_len=32768, ulysses=8))
    assert b.items["layer_inputs"].bytes == pytest.approx(a.items["layer_inputs"].bytes / 8)
    assert b.items["hf_dsa_dense_terms"].bytes == a.items["hf_dsa_dense_terms"].bytes
    assert "hf_dsa_dense_terms" not in estimate(GLM, replace(PPU, dsa_path="sparse_kernel")).items


def test_layer_inputs_formula():
    est = estimate(GLM, replace(PPU, seq_len=8192, ulysses=2))
    assert est.items["layer_inputs"].bytes == 45 * 4096 * 4 * 4096 * 2


def test_check_refuses_a_plan_that_does_not_fit_and_names_the_items():
    npu = TrainPlan(cards=32, cards_per_node=8, hbm_gib=64, host_memory_gib=1024, seq_len=8192)
    with pytest.raises(MemoryPlanError, match=r"device .* exceeds .*optimizer_shard"):
        check(GLM, npu)
    with pytest.raises(MemoryPlanError, match="host"):
        check(GLM, replace(npu, offload="full", hbm_gib=1000))


def test_check_accepts_a_plan_that_fits():
    est = check(GLM, replace(PPU, offload="full", ulysses=8, dsa_path="sparse_kernel"))
    assert est.fits and est.summary()["label"].startswith("derived")


@pytest.mark.parametrize("change, match", [
    ({"offload": "half"}, "offload"),
    ({"ulysses": 3}, "ulysses"),
    ({"ulysses": 128, "cards": 128}, "head count"),
    ({"fsdp_size": 3}, "fsdp_size"),
    ({"cards": 60}, "cards_per_node"),
    ({"state_bytes": 3}, "state_bytes"),
])
def test_malformed_plans_are_refused(change, match):
    with pytest.raises(MemoryPlanError, match=match):
        estimate(GLM, replace(PPU, **change))


def test_hsdp_shards_over_the_group_only():
    est = estimate(GLM, replace(PPU, fsdp_size=32))
    assert est.items["optimizer_shard"].bytes == pytest.approx(8 * GLM.params / 32)


class _Node:
    def __init__(self, **kw):
        self.__dict__.update(dict(device_family="ppu", card="PPU", cards_per_node=8, hbm_gib=144.0,
                                  host_memory_gib=2048.0) | kw)


def test_plan_for_nodes_requires_one_node_kind():
    plan = plan_for_nodes([_Node(), _Node()], seq_len=4096)
    assert (plan.cards, plan.cards_per_node, plan.host_memory_gib) == (16, 8, 2048.0)
    with pytest.raises(MemoryPlanError, match="differ"):
        plan_for_nodes([_Node(), _Node(hbm_gib=96.0)], seq_len=4096)
    with pytest.raises(MemoryPlanError, match="no trainer nodes"):
        plan_for_nodes([], seq_len=4096)


def test_sweep_rows():
    rows = sweep(GLM, PPU, [2048, 8192], offloads=["none", "full"])
    assert [(r["offload"], r["seq_len"]) for r in rows] == [("none", 2048), ("none", 8192),
                                                             ("full", 2048), ("full", 8192)]
    assert all(r["device_gib_per_card"] > 0 for r in rows)


def test_shape_from_hf_config_counts_a_tiny_model():
    pytest.importorskip("transformers")
    from yeto.rl.fullparam.memory import shape_from_hf_config

    cfg = {"model_type": "llama", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
           "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2}
    shape = shape_from_hf_config(cfg)
    layer = (32 * 32 * 2 + 32 * 16 * 2) + 3 * 32 * 64 + 2 * 32  # q,o + k,v + mlp + norms
    assert shape.largest_layer_params == layer
    assert shape.params == 2 * layer + 2 * 64 * 32 + 32  # embed + lm_head + final norm
    assert shape.head_counts == (2, 4) and shape.dsa_layers == 0 and GIB == 1024**3


# --- N7.1b Megatron variant -------------------------------------------------

# Synthetic MoE: 1000 params, 600 of them in 8 routed experts, 4 layers.
MOE = ModelShape(params=1000, largest_layer_params=250, num_layers=4, hidden_size=8, vocab_size=16,
                 residual_streams=4, head_counts=(4,), expert_params=600, num_experts=8)
MEG = TrainPlan(cards=16, cards_per_node=8, hbm_gib=144, host_memory_gib=1024, seq_len=64,
                framework="megatron", tp=2, pp=2, ep=4, chunked_loss=True, activation_margin_gib=0.0)


def test_megatron_state_splits_dense_by_tp_pp_and_experts_by_ep_pp():
    est = estimate(MOE, MEG)
    dense, experts = 400 / 2 / 2, 600 / 4 / 2  # (P - E) / TP / PP, E / EP / PP
    assert est.items["params_local_bf16"].bytes == pytest.approx(2 * (dense + experts))
    assert est.items["main_grads_local"].bytes == pytest.approx(4 * (dense + experts))
    # 4 B master + 8 B Adam; dense DP = 16 / (2*2) = 4, expert DP = 16 / (4*2) = 2
    assert est.items["optimizer_shard"].bytes == pytest.approx(12 * (dense / 4 + experts / 2))
    plain = estimate(MOE, replace(MEG, distributed_optimizer=False))
    assert plain.items["optimizer_local"].bytes == pytest.approx(12 * (dense + experts))
    assert "gathered_layers" not in est.items and est.total("host") == 0


def test_megatron_activations_count_pp_micro_batches_and_mhc_streams():
    est = estimate(MOE, MEG)
    # 2 stage layers x 2 in flight x 64/(TP 2 x CP 1) tokens x 4 streams x 8 hidden x 2 B
    assert est.items["layer_inputs"].bytes == 2 * 2 * 32 * 4 * 8 * 2
    one_stream = estimate(replace(MOE, residual_streams=1), MEG)
    assert est.items["layer_inputs"].bytes == 4 * one_stream.items["layer_inputs"].bytes


def test_megatron_uneven_pipeline_uses_the_largest_stage():
    est = estimate(replace(MOE, num_layers=5, largest_layer_params=200), replace(MEG, pp=2))
    assert "stage of 3/5 layers" in est.items["params_local_bf16"].formula


def test_megatron_output_is_labelled_derived():
    summary = estimate(MOE, MEG).summary()
    assert summary["basis"] == "推算" and summary["framework"] == "megatron"
    assert estimate(GLM, PPU).summary()["basis"] == "推算"


@pytest.mark.parametrize("change, match", [
    ({"ep": 3}, "ep x pp = 6 does not divide cards 16"),
    ({"tp": 3}, "tp x pp x cp = 6 does not divide"),
    ({"offload": "full"}, "offload is an FSDP2 mode"),
    ({"optimizer_offload_fraction": 1.5}, r"must be in \[0, 1\]"),
    ({"ulysses": 2}, "use cp instead of ulysses"),
    ({"pp": 8, "ep": 2}, "pp 8 exceeds the 4 layers"),
    ({"framework": "deepspeed"}, "framework must be one of"),
])
def test_malformed_megatron_plans_are_refused(change, match):
    with pytest.raises(MemoryPlanError, match=match):
        estimate(MOE, replace(MEG, **change))


def test_megatron_optimizer_offload_moves_state_and_master_to_the_host():
    # N7.1c: fraction F of (8 B Adam + 4 B master) leaves the card; host = per card x cards_per_node.
    dense, experts = 400 / 2 / 2, 600 / 4 / 2
    owned = dense / 4 + experts / 2
    full = estimate(MOE, replace(MEG, optimizer_offload_fraction=1.0))
    assert full.items["optimizer_shard"].bytes == pytest.approx(0)
    assert full.items["host_optimizer_offload"].bytes == pytest.approx(12 * owned * 8)
    assert full.items["host_optimizer_offload"].where == "host"
    half = estimate(MOE, replace(MEG, optimizer_offload_fraction=0.5, offload_master=False))
    assert half.items["optimizer_shard"].bytes == pytest.approx((12 - 4) * owned)
    assert half.total("host") == pytest.approx(4 * owned * 8)
    assert "optimizer offload 0.5" in half.summary()["plan_status"]
    assert "incl. FP32 master" in full.summary()["plan_status"]


def test_megatron_refuses_cp_with_dsa_and_fsdp2_refuses_megatron_knobs():
    with pytest.raises(MemoryPlanError, match="DSA layer does not support cp"):
        estimate(replace(MOE, dsa_layers=1, index_heads=4), replace(MEG, cp=2, pp=1, ep=8))
    with pytest.raises(MemoryPlanError, match="megatron only"):
        estimate(GLM, replace(PPU, tp=2))
    with pytest.raises(MemoryPlanError, match="optimizer_offload_fraction applies to framework megatron"):
        estimate(GLM, replace(PPU, optimizer_offload_fraction=0.5))
    with pytest.raises(MemoryPlanError, match="go together"):
        replace(MOE, num_experts=0)


def test_megatron_glm_on_32_ppu_cards_needs_bf16_optimizer_state():
    """GLM5.3-Flash on 4 PPU nodes (32 x 144 GiB), TP4 EP32, 32K tokens. The
    routed-expert share (95 % of parameters) is an assumption of this test, not
    counted: expert DP is 32 / (EP 32 x PP 1) = 1, so FP32 Adam on the experts
    is not divided by the distributed optimizer."""
    glm = replace(GLM, expert_params=int(GLM.params * 0.95), num_experts=288)
    plan = TrainPlan(cards=32, cards_per_node=8, hbm_gib=144, host_memory_gib=2048, seq_len=32768,
                     framework="megatron", tp=4, ep=32, dsa_path="sparse_kernel", chunked_loss=True)
    fp32 = estimate(glm, plan)
    assert not fp32.fits and "optimizer_shard" in fp32.problems()[0]
    assert estimate(glm, replace(plan, param_bytes=2, state_bytes=4)).fits
