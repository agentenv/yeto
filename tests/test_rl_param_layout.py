"""S27 R-5: param layout record and the offline TP/PP/EP merge (pure torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.engine.param_layout import (ParamLayoutError, build_param_layout, gather_with_stride,  # noqa: E402
                                         merge_shards, required_coverage, sequential_expert_global_name)


def _ranks(tp=0, tp_size=2, ep=0, ep_size=1):
    return {"tp_rank": tp, "tp_size": tp_size, "pp_rank": 0, "pp_size": 1, "ep_rank": ep, "ep_size": ep_size,
            "etp_rank": 0, "etp_size": 1}


def _layout(tp, tensors, *, stride=1, ep=0, ep_size=1, num_experts=None):
    named = list(tensors.items())
    return build_param_layout(
        named, global_names={n: f"module.module.{n}" for n, _ in named},
        partition_of=lambda g, p: (True, 0, stride) if "fc1" in g else (False, -1, 1),
        is_expert=lambda g: ".experts." in g, ranks=_ranks(tp, ep=ep, ep_size=ep_size), num_experts=num_experts)


def test_stride_two_interleaves_gate_and_up():
    full_gate, full_up = torch.arange(4.0), torch.arange(10.0, 14.0)
    parts = [torch.cat([full_gate[:2], full_up[:2]]), torch.cat([full_gate[2:], full_up[2:]])]
    assert torch.equal(gather_with_stride(parts, 0, 2), torch.cat([full_gate, full_up]))


def test_merge_gathers_tp_and_checks_replicas():
    a = {"mlp.linear_fc1.weight": torch.tensor([1.0, 2.0]), "norm.weight": torch.ones(2)}
    b = {"mlp.linear_fc1.weight": torch.tensor([3.0, 4.0]), "norm.weight": torch.ones(2)}
    merged = merge_shards([(_layout(0, a), a), (_layout(1, b), b)])
    assert torch.equal(merged["module.module.mlp.linear_fc1.weight"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    bad = {**b, "norm.weight": torch.zeros(2)}
    with pytest.raises(ParamLayoutError, match="replicas differ"):
        merge_shards([(_layout(0, a), a), (_layout(1, bad), bad)])
    with pytest.raises(ParamLayoutError, match="TP slices"):
        merge_shards([(_layout(0, a), a)])
    assert required_coverage([_layout(0, a)]) == ["pp0/tp1"]


def test_sequential_experts_get_global_indices():
    name = "module.module.decoder.layers.3.mlp.experts.local_experts.1.linear_fc2.weight"
    assert (sequential_expert_global_name(name, ep_rank=1, num_local_experts=2)
            == "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight3")
    t = {"decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight": torch.zeros(1)}
    lay = _layout(0, t, ep=1, ep_size=2, num_experts=4)
    assert next(iter(lay["params"].values()))["global_name"].endswith("experts.linear_fc2.weight2")
    with pytest.raises(ParamLayoutError, match="rank coordinates"):
        build_param_layout([], global_names={}, partition_of=None, is_expert=None, ranks={})
