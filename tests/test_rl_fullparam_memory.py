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
