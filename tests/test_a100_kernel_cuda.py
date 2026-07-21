"""CUDA parity gate for the pinned fused linear cross-entropy primitive."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch
import torch.nn.functional as F

from yeto.causal_kernels import apply_liger_fused_linear_ce, liger_sft_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_liger_fused_linear_cross_entropy_loss_and_gradients_match_native():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the optimized lane is scoped to A100/SM80")
    try:
        version = metadata.version("liger-kernel")
    except metadata.PackageNotFoundError:
        pytest.skip("liger-kernel is not installed")
    if version != "0.8.0":
        pytest.skip(f"requires liger-kernel==0.8.0, found {version}")
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except Exception as exc:
        pytest.skip(f"the pinned Liger package is not importable: {exc}")

    torch.manual_seed(7)
    device = torch.device("cuda")
    hidden_native = torch.randn(
        32, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    weight_native = torch.randn(
        257, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    hidden_liger = hidden_native.detach().clone().requires_grad_(True)
    weight_liger = weight_native.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 257, (32,), device=device)
    labels[::5] = -100

    native = F.cross_entropy(
        hidden_native.float() @ weight_native.float().t(),
        labels,
        ignore_index=-100,
        reduction="sum",
    )
    liger = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
        accum_dtype=torch.float32,
    )(weight_liger, hidden_liger, labels)
    native.backward()
    liger.backward()

    torch.testing.assert_close(liger.float(), native.float(), rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(
        hidden_liger.grad.float(), hidden_native.grad.float(), rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(
        weight_liger.grad.float(), weight_native.grad.float(), rtol=5e-2, atol=5e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_instance_only_qwen2_fused_sft_forward_matches_native():
    """Exercise the production instance patch and FP32-accumulation call."""
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the optimized lane is scoped to A100/SM80")
    try:
        version = metadata.version("liger-kernel")
    except metadata.PackageNotFoundError:
        pytest.skip("liger-kernel is not installed")
    if version != "0.8.0":
        pytest.skip(f"requires liger-kernel==0.8.0, found {version}")

    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    native = Qwen2ForCausalLM(config)
    fused = Qwen2ForCausalLM(config)
    fused.load_state_dict(native.state_dict())
    application = apply_liger_fused_linear_ce(fused)
    assert application["patch_scope"] == "model-instance-forward-only"
    assert all(application["invariants"].values())

    device = torch.device("cuda")
    native = native.to(device=device, dtype=torch.bfloat16).train()
    fused = fused.to(device=device, dtype=torch.bfloat16).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    weights[:, 0] = 0
    weights[:, 3::5] = 0
    labels = input_ids.masked_fill(weights != 1, -100)

    native_output = native(input_ids=input_ids, use_cache=False)
    native_loss = F.cross_entropy(
        native_output.logits[:, :-1].float().reshape(-1, config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    fused_loss, target_tokens = liger_sft_forward(fused, input_ids, weights)
    assert int(target_tokens.item()) == int((labels[:, 1:] != -100).sum().item())
    native_loss.backward()
    fused_loss.backward()

    torch.testing.assert_close(
        fused_loss.float(), native_loss.float(), rtol=5e-2, atol=5e-3
    )
    native_gradients = dict(native.named_parameters())
    fused_gradients = dict(fused.named_parameters())
    assert native_gradients.keys() == fused_gradients.keys()
    for name, native_parameter in native_gradients.items():
        assert native_parameter.grad is not None, name
        assert fused_gradients[name].grad is not None, name
        torch.testing.assert_close(
            fused_gradients[name].grad.float(),
            native_parameter.grad.float(),
            rtol=5e-2,
            atol=5e-3,
            msg=lambda message, name=name: f"{name}: {message}",
        )
