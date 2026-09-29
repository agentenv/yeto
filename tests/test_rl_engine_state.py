import pytest
import torch

from yeto.rl.core import CanonicalLoraState, policy_hash, policy_tensor_hash
from yeto.rl.engine.trainable_state import (
    IMPLEMENTED_LAYOUTS,
    TRAINABLE_LAYOUTS,
    TrainableState,
    UnsupportedLayoutError,
    require_supported_layout,
)


def _lora():
    g = torch.Generator().manual_seed(0)
    tensors = {
        "base_model.model.layers.0.q.lora_B.weight": torch.randn(4, 2, generator=g),
        "base_model.model.layers.0.q.lora_A.weight": torch.randn(2, 4, generator=g),
    }
    return CanonicalLoraState("a" * 40, "b" * 64, "c" * 64, 3, tensors)


def test_lora_hash_matches_legacy():
    legacy = _lora()
    wrapped = TrainableState.from_lora(legacy)
    assert wrapped.policy_hash() == policy_hash(legacy)
    assert wrapped.policy_tensor_hash() == policy_tensor_hash(legacy)
    fresh = TrainableState("lora", "a" * 40, "b" * 64, "c" * 64, 3, dict(legacy.tensors))
    assert fresh.policy_hash() == policy_hash(legacy)
    assert fresh.tensor_names == tuple(sorted(legacy.tensors))
    assert fresh.to_lora().specs == legacy.specs


def test_layouts_representable_but_unimplemented_rejected():
    assert TRAINABLE_LAYOUTS == {"lora", "full", "fragments"}
    assert IMPLEMENTED_LAYOUTS == {"lora"}
    for layout in ("full", "fragments"):
        with pytest.raises(UnsupportedLayoutError, match=layout):
            require_supported_layout(layout)
        with pytest.raises(UnsupportedLayoutError):
            TrainableState(layout, "a" * 40, "b" * 64, "c" * 64, 0, {})
    with pytest.raises(ValueError, match="unknown"):
        require_supported_layout("sparse")


def test_lora_validation_is_legacy_validation():
    with pytest.raises(ValueError, match="immutable commit"):
        TrainableState("lora", "main", "b" * 64, "c" * 64, 0, dict(_lora().tensors))
