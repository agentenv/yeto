"""Generic (in-place) partial-expert selections and the expert-subset layout check.

An ``in_place`` selection keeps Megatron's expert order: EP rank ``r`` owns
original experts ``[r*n/ep, (r+1)*n/ep)`` and every rank trains the same
number of them. Its geometry comes from the checkpoint's ``config.json``
(:mod:`yeto.rl.moe_geometry`), so the tool works for any MoE family.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.local_learner import ParameterSpec
from yeto.rl.moe_layout import ExpertSubset, check_expert_partition

QWEN3_MOE = {"model_type": "qwen3_moe", "num_hidden_layers": 3, "num_experts": 8,
             "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [0]}
DEEPSEEK = {"model_type": "deepseek_v3", "num_hidden_layers": 4, "n_routed_experts": 16,
            "num_experts_per_tok": 4, "first_k_dense_replace": 1}


def _in_place(mode="uniform", **kw):
    args = dict(model="qwen3_moe", num_experts=8, num_trainable=4, ep_size=2, layers=(1, 2))
    args.update(kw)
    return sel.ExpertSelection.in_place(mode, **args)


@pytest.mark.parametrize("mode", ["uniform", "random"])
def test_synthetic_in_place_selection_is_ep_balanced_and_round_trips(mode):
    selection = _in_place(mode, seed=3)
    for layer, ids in selection.trainable.items():
        assert sel.in_place_imbalance(ids, num_experts=8, ep_size=2) == [2, 2]
        assert selection.slot_orders[layer] == tuple(range(8))
    doc = selection.to_json()
    assert doc["layout"] == "in_place"
    assert set(doc["layers"]["1"]) == {"trainable"}
    sel.validate_document(doc)
    again = sel.ExpertSelection.from_json(json.loads(json.dumps(doc)))
    assert again == selection
    assert again.sha256 == selection.sha256


def test_routing_frequency_picks_each_ranks_hottest():
    counts = {1: [9, 1, 1, 8, 0, 7, 6, 0], 2: [0, 0, 5, 5, 1, 1, 9, 9]}
    selection = _in_place("routing_frequency", counts=counts)
    assert selection.trainable == {1: (0, 3, 5, 6), 2: (2, 3, 6, 7)}
    assert selection.coverage()["1"] == pytest.approx(30 / 32)


def test_validator_refuses_an_unbalanced_or_permuted_in_place_document():
    doc = _in_place().to_json()
    bad = json.loads(json.dumps(doc))
    bad["layers"]["1"]["trainable"] = [0, 1, 2, 3]  # all on EP rank 0
    with pytest.raises(sel.SelectionError, match="every EP rank must own 2"):
        sel.validate_document(bad)
    bad = json.loads(json.dumps(doc))
    bad["layers"]["1"]["slot_order"] = list(range(8))
    with pytest.raises(sel.SelectionError, match="exactly 'trainable'"):
        sel.validate_document(bad)
    bad = json.loads(json.dumps(doc))
    bad["sha256"] = "0" * 64
    with pytest.raises(sel.SelectionError, match="does not match"):
        sel.validate_document(bad)


def test_with_ep_size_keeps_a_balanced_in_place_set_and_refuses_otherwise():
    selection = _in_place(num_experts=8, num_trainable=4, ep_size=4)
    assert selection.with_ep_size(2).ep_size == 2
    lopsided = sel.ExpertSelection.in_place("uniform", model="m", num_experts=8, num_trainable=2, ep_size=2,
                                            layers=(1,))
    with pytest.raises(sel.SelectionError, match="not balanced"):
        lopsided.with_ep_size(1).with_ep_size(4)


def test_slot_order_documents_are_unchanged():
    """K3.1 selections keep their schema and digests (no layout key)."""
    doc = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=16, num_trainable=4,
                                        ep_size=2, layers=(1, 2)).to_json()
    assert "layout" not in doc
    assert doc["sha256"] == sel.canonical_layers_sha256(doc["layers"])


@pytest.mark.parametrize("config, layers, experts", [
    (QWEN3_MOE, [1, 2], 8),
    (DEEPSEEK, [1, 2, 3], 16),
])
def test_cli_builds_from_any_hf_config(tmp_path, config, layers, experts):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "selection.json"
    rc = sel.main(["--layout", "in_place", "--hf-config", str(path), "--num-trainable", "4",
                   "--ep-size", "2", "--mode", "uniform", "--output", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["model"] == config["model_type"]
    assert doc["num_experts"] == experts
    assert sorted(int(k) for k in doc["layers"]) == layers
    assert sel.main(["--validate", str(out)]) == 0


def test_cli_counts_from_routed_experts_follow_the_config_layers(tmp_path):
    """The routed-experts array's MoE axis maps onto the config's MoE layers, not an offset."""
    rng = np.random.default_rng(0)
    routed = rng.integers(0, 8, size=(64, 2, 2), dtype=np.int64)
    counts = sel.counts_from_routed_experts(routed, num_experts=8, moe_layers=(1, 2))
    assert sorted(counts) == [1, 2]
    with pytest.raises(sel.SelectionError, match="2 MoE layers"):
        sel.counts_from_routed_experts(routed, num_experts=8, moe_layers=(1, 2, 3))


def _expert_specs(subset_ids, *, layers=(1, 2), ep=2, experts=8, router=False):
    specs = []
    for layer in layers:
        base = f"module.module.decoder.layers.{layer}.mlp."
        if router:
            specs.append(ParameterSpec("actor", base + "router.weight", (experts, 4), "float32", experts * 4,
                                       "tp0-of-1.pp0-of-1.ep0-of-2.cp0-of-1.dp0-of-1"))
        for expert in subset_ids[layer]:
            rank = expert // (experts // ep)
            owner = f"tp0-of-1.pp0-of-1.ep{rank}-of-{ep}.cp0-of-1.dp0-of-1"
            specs.append(ParameterSpec("actor", base + f"experts.linear_fc1.weight{expert}", (6, 4), "float32", 24,
                                       owner))
            specs.append(ParameterSpec("actor", base + f"experts.linear_fc2.weight{expert}", (4, 3), "float32", 12,
                                       owner))
    return specs


def test_partial_manifests_need_the_declared_subset():
    selection = _in_place()
    specs = _expert_specs(selection.trainable)
    # Without the subset the frozen experts look missing.
    with pytest.raises(ValueError, match="experts are not 0..3"):
        check_expert_partition(specs)
    subset = ExpertSubset.of(8, selection.trainable)
    partition = check_expert_partition(specs, subset=subset)
    assert partition.num_experts == 8 and partition.moe_layers == (1, 2)
    # A frozen expert that still has a master, or a selected one that lacks it, is refused.
    extra = {1: (*selection.trainable[1], 7) if 7 not in selection.trainable[1] else selection.trainable[1],
             2: selection.trainable[2]}
    with pytest.raises(ValueError, match="unexpected"):
        check_expert_partition(_expert_specs(extra), subset=subset)
    with pytest.raises(ValueError, match="hold no trainable expert"):
        check_expert_partition(_expert_specs(selection.trainable, layers=(1,)), subset=subset)
    # A trainable router keeps the full expert width.
    check_expert_partition(_expert_specs(selection.trainable, router=True), subset=subset)


def test_slot_order_digests_are_pinned():
    """Kimi-K3.1 (slot_order) selections keep the digests main produced before the in-place layout."""
    synthetic = sel.ExpertSelection.synthetic("random", model="kimi_k31", num_experts=32, num_trainable=8,
                                              ep_size=4, layers=range(1, 5), seed=7)
    assert synthetic.sha256 == "82e0173d694da879f2a5a8fe3325763054060e29d06b98bdfc4435baa2f55d79"
    assert synthetic.trainable_sha256 == "27e1c6447f09c874562f4a13111a0b2ba66b84788970c683431ba73c9f81ef50"
    counts = {layer: [(layer * 31 + expert * 17) % 97 for expert in range(32)] for layer in range(1, 5)}
    lpt = sel.ExpertSelection.from_counts(counts, model="kimi_k31", num_trainable=8, ep_size=4, placement="lpt")
    assert lpt.sha256 == "fad83285eb69496cf3a858a279074031b29bb3daa8885e106e17946a5565b69c"
    assert lpt.trainable_sha256 == "2d8b14bc501bbcf11f1216187bbf9c34892553f65b5cf6df89bd047058fc6fc4"
    assert lpt.with_ep_size(2).sha256 == "de3cc5b50429f9892a48880f188cbef7bb1b7e5a8d72c17e19dd56520c8ba0b6"
