"""Validation-only Miles hook: numeric compatibility without the Yeto island.

Offline held-out validation replays buckets through the critic with bit-frozen
parameters and no syncer. The BF16 static-unscale and homogeneous mixed-dtype
grad norm/clip compatibility still must be installed: without them the
optimizer step (even at lr ~ 0) crashes on BF16 grads or logs corrupted
grad norms. The MilesValueIsland itself (fragments, anchors, syncer client)
is deliberately not constructed.
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
    _install_bf16_static_unscale_compat(args, optimizer)
    _install_mixed_dtype_grad_compat(args)
