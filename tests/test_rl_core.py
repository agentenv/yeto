import hashlib

import pytest
import torch

from yeto.fragments import MERGE_AVG
from yeto.rl.cache import ResultCache
from yeto.rl.core import (
    PolicyIdentity,
    build_avg_layout,
    canonical_state,
    flat_tensor,
    policy_delta,
    tensors_from_flat,
)


def lora_tensors(reverse=False):
    items = [
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
            torch.arange(6, dtype=torch.float32).reshape(3, 2),
        ),
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
        ),
    ]
    return dict(reversed(items) if reverse else items)


def test_canonical_lora_is_order_independent_and_uses_one_avg_fragment():
    first = canonical_state(3, lora_tensors())
    second = canonical_state(3, lora_tensors(reverse=True))
    assert first.identity == second.identity
    assert first.specs == second.specs
    layout = build_avg_layout(first.specs)
    assert layout.num_fragments == 1
    assert layout.fragments[0].merge_mode == MERGE_AVG
    assert [name for name, _ in layout.fragments[0].tensors] == sorted(lora_tensors())

    restored = tensors_from_flat(flat_tensor(first.tensors, first.specs), first.specs)
    assert all(torch.equal(restored[name], first.tensors[name]) for name in restored)


def test_policy_hash_and_delta_are_exact_f32():
    base = canonical_state(0, lora_tensors())
    local = canonical_state(0, {name: value + 2 for name, value in lora_tensors().items()})
    assert torch.equal(policy_delta(local, base), torch.full((14,), 2.0))
    expected = hashlib.sha256(
        b"yeto-rl-policy-v1\0"
        + bytes.fromhex(base.layout_fingerprint)
        + flat_tensor(base.tensors).numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    assert base.policy_hash == expected


@pytest.mark.parametrize(
    "tensors,match",
    [
        ({"model.weight": torch.ones(1)}, "PEFT LoRA"),
        (
            {"base_model.model.x.lora_A.weight": torch.tensor([float("nan")])},
            "NaN or Inf",
        ),
    ],
)
def test_canonical_lora_fails_closed(tensors, match):
    with pytest.raises((TypeError, ValueError), match=match):
        canonical_state(0, tensors)


def test_local_result_cache_is_identity_bound_and_corruption_is_not_reused(tmp_path):
    state = canonical_state(4, lora_tensors())
    cache = ResultCache(
        tmp_path,
        run_manifest_sha256="ab" * 32,
        learner_id=1,
        layout_fingerprint=state.layout_fingerprint,
    )
    delta = torch.arange(14, dtype=torch.float32)
    saved = cache.save(
        base_identity=state.identity,
        target_step=5,
        delta=delta,
        stats={"groups": 2},
    )
    loaded = cache.load(
        base_identity=state.identity,
        target_step=5,
        expected_numel=14,
    )
    assert loaded is not None
    assert loaded.delta_sha256 == saved.delta_sha256
    assert torch.equal(loaded.delta, delta)
    first_push = cache.record_push(loaded)
    second_push = cache.record_push(first_push)
    assert second_push.push_attempts == 2
    assert second_push.first_push_unix_ns == first_push.first_push_unix_ns

    wrong = PolicyIdentity(4, "cd" * 32)
    assert cache.load(base_identity=wrong, target_step=5, expected_numel=14) is None
    assert not cache.metadata_path.exists()

    cache.save(
        base_identity=state.identity,
        target_step=5,
        delta=delta,
        stats={},
    )
    cache.delta_path.write_bytes(b"broken")
    assert (
        cache.load(base_identity=state.identity, target_step=5, expected_numel=14)
        is None
    )
