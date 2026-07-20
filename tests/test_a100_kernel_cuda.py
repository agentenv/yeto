"""CUDA parity gate for the pinned fused linear cross-entropy primitive."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch
import torch.nn.functional as F


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
