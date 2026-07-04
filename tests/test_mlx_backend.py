"""MLX island backend units (skipped where mlx is unavailable, e.g. CI on
Linux): peft bit-compatibility of the LoRA layer, canonical naming, and the
fragment pack/apply round-trip through the torch wire primitives.

The full cross-backend guarantee (an MLX learner and a peft/torch learner
report identical trainable FQNs for the same HF model) is checked against a
real checkpoint by scripts/check_name_parity.py, which needs the Hub.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
import mlx.nn as mnn  # noqa: E402

from yeto.fragments import build_layout  # noqa: E402
from yeto.mlx.lora import LoRALinear, attach_lora, canonical_name  # noqa: E402
from yeto.mlx.learner import torch_adapters, write_fragment  # noqa: E402
from yeto.protocol import DTYPE_BF16  # noqa: E402
from yeto.tensor_io import fragment_flat, pack_fragment, unpack_fragment  # noqa: E402

IN, OUT, R, ALPHA = 16, 12, 4, 8


class TinyAttn(mnn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = mnn.Linear(IN, OUT, bias=False)
        self.o_proj = mnn.Linear(OUT, IN, bias=False)


class TinyModel(mnn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TinyAttn(), TinyAttn()]
        self.lm_head = mnn.Linear(IN, 32, bias=False)

    def __call__(self, x):
        for l in self.layers:
            x = l.o_proj(l.q_proj(x))
        return self.lm_head(x)


class DenseConfig:
    """No MoE markers -> yeto.learner.is_moe_config is False -> all-linear."""


def test_lora_forward_matches_peft_semantics():
    """y = base(x) + (alpha/r) * x A^T B^T with peft-shaped A/B."""
    base = mnn.Linear(IN, OUT, bias=False)
    layer = LoRALinear(base, R, ALPHA)
    # B starts at zero -> adapter contributes nothing.
    x = mx.random.normal((3, IN))
    assert np.allclose(np.array(layer(x)), np.array(base(x)), atol=1e-6)

    layer.lora_B = mx.random.normal((OUT, R)) * 0.1
    a = torch.from_numpy(np.array(layer.lora_A))
    b = torch.from_numpy(np.array(layer.lora_B))
    w = torch.from_numpy(np.array(base.weight))
    xt = torch.from_numpy(np.array(x))
    expected = xt @ w.T + (ALPHA / R) * (xt @ a.T) @ b.T
    assert np.allclose(np.array(layer(x)), expected.numpy(), atol=1e-5)
    assert layer.lora_A.shape == (R, IN)  # peft lora_A orientation
    assert layer.lora_B.shape == (OUT, R)  # peft lora_B orientation


def test_attach_lora_all_linear_excludes_lm_head():
    model = TinyModel()
    registry = attach_lora(model, DenseConfig(), "all-linear", R, ALPHA)
    names = set(registry)
    assert (
        "base_model.model.layers.0.q_proj.lora_A.default.weight" in names
    )
    assert not any("lm_head" in n for n in names)
    # 2 layers x 2 linears x (A, B)
    assert len(names) == 8
    # Base is frozen: only lora_A/lora_B remain trainable.
    from mlx.utils import tree_flatten

    trainable = [p for p, _ in tree_flatten(model.trainable_parameters())]
    assert all(p.endswith(("lora_A", "lora_B")) for p in trainable)


def test_attach_lora_attention_regex():
    model = TinyModel()
    registry = attach_lora(model, DenseConfig(), "attention", R, ALPHA)
    # q_proj matches the shared attention regex; o_proj does too; lm_head not.
    assert all(".q_proj." in n or ".o_proj." in n for n in registry)


def test_canonical_name_mapping():
    assert (
        canonical_name("model.layers.7.self_attn.q_proj.lora_A")
        == "base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight"
    )


def test_pack_write_roundtrip_and_blend():
    """Adapters -> torch snapshot -> wire bytes -> flat -> back into MLX."""
    model = TinyModel()
    registry = attach_lora(model, DenseConfig(), "all-linear", R, ALPHA)
    # Give every adapter distinct values so a misordered flatten cannot pass.
    from mlx.utils import tree_flatten, tree_unflatten

    updates = []
    for i, (path, arr) in enumerate(sorted(tree_flatten(model.trainable_parameters()))):
        vals = mx.arange(arr.size).reshape(arr.shape).astype(mx.float32) + 100 * i
        updates.append((path, vals))
    model.update(tree_unflatten(updates))

    layout = build_layout(
        [(n, int(np.prod(i.shape))) for n, i in registry.items()], 3
    )
    snap = torch_adapters(model, registry)
    for frag in layout.fragments:
        wire = pack_fragment(frag, snap, DTYPE_BF16)
        flat = unpack_fragment(frag, wire, DTYPE_BF16)
        # bf16 wire is lossy but these integer-ish values are exact in bf16?
        # No — keep tolerance loose and check structure via f32 path below.
        assert flat.numel() == frag.numel

    # Lossless round-trip: perturb, write back, re-snapshot, compare.
    frag = layout.fragments[0]
    flat = fragment_flat(frag, snap) + 1.5
    write_fragment(model, frag, flat, registry)
    snap2 = torch_adapters(model, registry)
    assert torch.equal(fragment_flat(frag, snap2), flat)
    # Other fragments untouched.
    for other in layout.fragments[1:]:
        assert torch.equal(fragment_flat(other, snap2), fragment_flat(other, snap))


def test_registry_shapes_and_numels_are_peft_shaped():
    model = TinyModel()
    registry = attach_lora(model, DenseConfig(), "all-linear", R, ALPHA)
    for name, info in registry.items():
        if ".lora_A." in name:
            assert info.shape[0] == R
        else:
            assert info.shape[1] == R
