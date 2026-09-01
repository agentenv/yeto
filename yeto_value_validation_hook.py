"""Validation-only Miles hook: numeric compatibility without the Yeto island.

Offline held-out validation replays buckets through the critic without an
optimizer or syncer. Older Miles checkouts can still construct an optimizer
for this path, so the BF16 static-unscale and homogeneous mixed-dtype grad
compatibility hooks remain as a fallback. The MilesValueIsland itself
(fragments, anchors, syncer client) is deliberately not constructed.
"""

from typing import Any

from yeto.megatron.miles_value_island import (
    _install_bf16_static_unscale_compat,
    _install_mixed_dtype_grad_compat,
)


def after_model_init(
    args: Any,
    role: str,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    del model, opt_param_scheduler
    if role != "critic":
        return
    # Frozen held-out replay deliberately constructs no optimizer. The two
    # compatibility patches below are needed only by a real optimizer step.
    if optimizer is None:
        return
    _install_bf16_static_unscale_compat(args, optimizer)
    _install_mixed_dtype_grad_compat(args)
