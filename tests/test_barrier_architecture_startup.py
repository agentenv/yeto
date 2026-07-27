"""CPU/meta regressions for barrier startup on production model layouts."""

from __future__ import annotations

import struct
from collections import Counter

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout, build_layout
from yeto.protocol import MSG_BCAST_FRAGMENT, SyncerClient


def _qwen_2_5_7b_config() -> Qwen2Config:
    # Qwen/Qwen2.5-7B, revision d1497293...; only structure-bearing fields
    # are needed because the model is instantiated entirely on meta tensors.
    return Qwen2Config(
        vocab_size=152_064,
        hidden_size=3_584,
        intermediate_size=18_944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=131_072,
        tie_word_embeddings=False,
    )


def _smollm2_1_7b_config() -> LlamaConfig:
    # HuggingFaceTB/SmolLM2-1.7B; tied embeddings and bias-free projections
    # are the architecture controls relevant to fragment derivation.
    return LlamaConfig(
        vocab_size=49_152,
        hidden_size=2_048,
        intermediate_size=8_192,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=64,
        max_position_embeddings=8_192,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
    )


@pytest.mark.parametrize(
    ("label", "config_factory", "tensor_count", "fragment_numels", "streams"),
    [
        pytest.param(
            "Qwen2.5-7B",
            _qwen_2_5_7b_config,
            339,
            (1_089_994_752, 2_176_319_488, 2_174_651_392, 2_174_650_880),
            1,
            id="qwen2.5-7b",
        ),
        pytest.param(
            "SmolLM2-1.7B",
            _smollm2_1_7b_config,
            218,
            (100_663_296, 536_905_728, 536_903_680, 536_903_680),
            0,
            id="smollm2-1.7b",
        ),
    ],
)
def test_reconnect_replay_yields_unique_version_zero_fragments_for_architecture(
    label, config_factory, tensor_count, fragment_numels, streams
):
    """A full startup replay must still expose each architecture fragment once."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config_factory())
    params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    layout = build_layout(
        [(name, parameter.numel()) for name, parameter in params.items()],
        4,
        "binpack",
    )

    # First disprove an architecture partitioning collision: every trainable
    # tensor name occurs exactly once and all four derived IDs are covered.
    derived_names = layout.tensor_names()
    assert len(params) == tensor_count, label
    assert Counter(derived_names) == Counter(params.keys()), label
    assert tuple(fragment.numel for fragment in layout.fragments) == fragment_numels
    expected = [(fragment_id, 0) for fragment_id in range(4)]
    assert len(expected) == len(set(expected)) == layout.num_fragments

    # Model the protocol's reconnect contract: the syncer sends the complete
    # current state again while the client retains the first delivery. Qwen's
    # >2 GiB bulk fragments also select chunking; SmolLM2 remains unstriped.
    client = SyncerClient(
        ("127.0.0.1", 1),
        learner_id=0,
        layout=layout,
        num_streams=0,
    )
    assert client.num_streams == streams, label
    for _connection_generation in range(2):
        for fragment_id in range(layout.num_fragments):
            client._dispatch(
                0,
                MSG_BCAST_FRAGMENT,
                struct.pack("<IQ", fragment_id, 0),
            )
    observed = [
        (broadcast.fragment_id, broadcast.version)
        for broadcast in client.drain_updates()
    ]
    assert observed == expected, label


def test_broadcast_inbox_discards_equal_and_older_reconnect_replays():
    layout = FragmentLayout([Fragment(MERGE_RDA, [("model.weight", 1)])])
    client = SyncerClient(("127.0.0.1", 1), 0, layout, num_streams=0)
    for version in (2, 2, 1, 3):
        client._dispatch(
            0,
            MSG_BCAST_FRAGMENT,
            struct.pack("<IQ", 0, version) + b"payload",
        )
    assert [item.version for item in client.drain_updates()] == [2, 3]
