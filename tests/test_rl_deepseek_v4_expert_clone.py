from __future__ import annotations

import hashlib
import json

import pytest
import torch

import yeto.rl.deepseek_v4_expert_clone as expert_clone
from yeto.rl.deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
    contract_from_config,
    contract_from_selection,
    expand_routes_torch,
    remap_topk_ids_torch,
    split_bucket,
    use_clone,
)


def _selection():
    return {
        "schema": 1,
        "bucket": "always",
        "selection_scope": "32 independent source experts per decoder layer",
        "task_count": 654,
        "contract_sha256": "1" * 64,
        "eligibility": {
            "rounds_total": 3,
            "rounds_real": 3,
            "rounds_solved": 3,
            "env_failures": 0,
            "prompt_tier": "l2",
        },
        "layers": [
            {
                "layer": layer,
                "selected": [
                    {
                        "source_expert_id": (rank * 7 + layer) % ORIGINAL_EXPERTS,
                        "clone_expert_id": ORIGINAL_EXPERTS + rank,
                    }
                    for rank in range(CLONES_PER_LAYER)
                ],
            }
            for layer in range(NUM_LAYERS)
        ],
    }


def test_selection_round_trips_through_strict_config_contract(tmp_path):
    path = tmp_path / "selection.json"
    payload = json.dumps(_selection(), sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()

    contract = contract_from_selection(
        path,
        expected_selection_sha256=expected_hash,
    )
    config = {
        "n_routed_experts": TOTAL_EXPERTS,
        "num_hidden_layers": NUM_LAYERS,
        "num_experts_per_tok": 6,
        "num_nextn_predict_layers": 0,
        "yeto_routed_expert_clone": contract.config_value(),
    }

    assert contract_from_config(config) == contract
    assert contract.clone_ids == tuple(range(ORIGINAL_EXPERTS, TOTAL_EXPERTS))
    assert len(contract.source_experts_by_layer) == NUM_LAYERS


def test_nonstandard_expert_count_without_contract_is_rejected():
    assert contract_from_config({"n_routed_experts": ORIGINAL_EXPERTS}) is None
    with pytest.raises(ValueError, match="no Yeto clone contract"):
        contract_from_config({"n_routed_experts": TOTAL_EXPERTS})


def test_token_split_is_stable_and_non_degenerate():
    values = [use_clone(token, 17, 203) for token in range(10_000)]
    assert values == [
        split_bucket(token, 17, 203) < 1_073_741_824
        for token in range(10_000)
    ]
    assert 0.49 < sum(values) / len(values) < 0.51


def test_training_expert_layout_is_a_balanced_ep8_bijection():
    physical_ids = [
        expert_clone.logical_to_training_expert_id(logical_id)
        for logical_id in range(TOTAL_EXPERTS)
    ]

    assert sorted(physical_ids) == list(range(TOTAL_EXPERTS))
    assert [
        expert_clone.training_to_logical_expert_id(physical_id)
        for physical_id in physical_ids
    ] == list(range(TOTAL_EXPERTS))
    for rank in range(8):
        local_logical_ids = tuple(
            expert_clone.training_to_logical_expert_id(physical_id)
            for physical_id in range(rank * 36, (rank + 1) * 36)
        )
        assert local_logical_ids == (
            *range(rank * 32, (rank + 1) * 32),
            *range(ORIGINAL_EXPERTS + rank * 4, ORIGINAL_EXPERTS + (rank + 1) * 4),
        )


def test_expert_parameter_names_round_trip_between_logical_and_training_ids():
    logical = "base_model.model.model.layers.3.mlp.experts.256.gate_proj.weight"
    training = "base_model.model.model.layers.3.mlp.experts.32.gate_proj.weight"

    assert expert_clone.logical_to_training_expert_name(logical) == training
    assert expert_clone.training_to_logical_expert_name(training) == logical
    assert (
        expert_clone.training_to_logical_expert_name(
            "model.layers.3.self_attn.q_proj.weight"
        )
        == "model.layers.3.self_attn.q_proj.weight"
    )
    with pytest.raises(ValueError, match="outside"):
        expert_clone.logical_to_training_expert_name(
            "model.layers.3.mlp.experts.288.down_proj.weight"
        )


def test_route_expansion_preserves_topk_probs_and_mutual_exclusion():
    sources = tuple(range(CLONES_PER_LAYER))
    token_ids = torch.arange(128, dtype=torch.long)
    base_map = torch.zeros((128, ORIGINAL_EXPERTS), dtype=torch.bool)
    base_probs = torch.zeros((128, ORIGINAL_EXPERTS), dtype=torch.float32)
    for row in range(128):
        chosen = torch.tensor([row % 32, 40, 80, 120, 160, 200])
        base_map[row, chosen] = True
        base_probs[row, chosen] = torch.tensor([1, 2, 3, 4, 5, 6]) / 21

    probs, routing_map = expand_routes_torch(
        base_probs,
        base_map,
        token_ids,
        layer_id=9,
        source_experts=sources,
    )

    assert probs.shape == routing_map.shape == (128, TOTAL_EXPERTS)
    assert torch.equal(routing_map.sum(dim=1), torch.full((128,), 6))
    assert torch.allclose(probs.sum(dim=1), base_probs.sum(dim=1))
    for rank, source in enumerate(sources):
        clone = ORIGINAL_EXPERTS + rank
        training_source = expert_clone.logical_to_training_expert_id(source)
        training_clone = expert_clone.logical_to_training_expert_id(clone)
        assert not torch.any(
            routing_map[:, training_source] & routing_map[:, training_clone]
        )
        assert torch.equal(
            routing_map[:, training_source] | routing_map[:, training_clone],
            base_map[:, source],
        )
        assert torch.equal(
            probs[:, training_source] + probs[:, training_clone],
            base_probs[:, source],
        )
    training_clone_ids = torch.tensor(
        [
            expert_clone.logical_to_training_expert_id(expert)
            for expert in range(ORIGINAL_EXPERTS, TOTAL_EXPERTS)
        ]
    )
    assert routing_map.index_select(1, training_clone_ids).any()
    assert (~routing_map.index_select(1, training_clone_ids)).any()


def test_compact_topk_remap_matches_dense_route_expansion():
    sources = tuple(range(CLONES_PER_LAYER))
    token_ids = torch.arange(128, dtype=torch.long)
    base_ids = torch.stack(
        [
            token_ids.remainder(32),
            torch.full_like(token_ids, 40),
            torch.full_like(token_ids, 80),
            torch.full_like(token_ids, 120),
            torch.full_like(token_ids, 160),
            torch.full_like(token_ids, 200),
        ],
        dim=1,
    ).to(torch.int32)
    base_map = torch.zeros((128, ORIGINAL_EXPERTS), dtype=torch.bool)
    base_map.scatter_(1, base_ids.long(), True)
    base_probs = base_map.to(torch.float32) / 6

    remapped = remap_topk_ids_torch(
        base_ids,
        token_ids,
        layer_id=9,
        source_experts=sources,
    )
    _, expanded_map = expand_routes_torch(
        base_probs,
        base_map,
        token_ids,
        layer_id=9,
        source_experts=sources,
    )
    compact_map = torch.zeros_like(expanded_map)
    compact_map.scatter_(1, remapped.long(), True)
    logical_dense_map = expanded_map.index_select(
        1,
        torch.tensor(
            [
                expert_clone.logical_to_training_expert_id(expert)
                for expert in range(TOTAL_EXPERTS)
            ]
        ),
    )

    assert torch.equal(compact_map, logical_dense_map)
    assert torch.any(remapped >= ORIGINAL_EXPERTS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_topk_remap_is_cuda_graph_capture_safe():
    sources = tuple(range(CLONES_PER_LAYER))
    token_ids = torch.arange(128, dtype=torch.long, device="cuda")
    base_ids = torch.stack(
        [
            token_ids.remainder(32),
            torch.full_like(token_ids, 40),
            torch.full_like(token_ids, 80),
            torch.full_like(token_ids, 120),
            torch.full_like(token_ids, 160),
            torch.full_like(token_ids, 200),
        ],
        dim=1,
    ).to(torch.int32)
    source_ids = torch.tensor(sources, dtype=base_ids.dtype, device="cuda")

    # Populate allocator/kernel caches outside capture, matching SGLang's
    # eager warmup before it records the decode graph.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        remap_topk_ids_torch(
            base_ids,
            token_ids,
            layer_id=9,
            source_experts=sources,
            source_expert_ids=source_ids,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = remap_topk_ids_torch(
            base_ids,
            token_ids,
            layer_id=9,
            source_experts=sources,
            source_expert_ids=source_ids,
        )
    graph.replay()
    torch.cuda.synchronize()

    expected = remap_topk_ids_torch(
        base_ids,
        token_ids,
        layer_id=9,
        source_experts=sources,
        source_expert_ids=source_ids,
    )
    assert torch.equal(captured, expected)
