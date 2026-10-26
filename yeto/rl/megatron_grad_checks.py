"""Megatron's per-bucket gradient check, on by default on a yeto island (S27 G-FL).

Megatron-Core's ``DistributedDataParallelConfig`` defaults
``check_for_nan_in_grad`` and ``check_for_large_grads`` to ``False``
(``megatron/core/distributed/distributed_data_parallel_config.py:51,57``), and
with both off ``ParamAndGradBuffer.check_grads`` never runs
(``param_and_grad_buffer.py:706``).  A non-finite gradient then reaches
``clip_grads``, where ``clip_coeff = max_norm / (total_norm + 1e-6)`` has no
non-finite guard (``clip_grads.py:183``): a NaN total norm makes ``clip_coeff``
NaN and ``multi_tensor_scale`` multiplies **every** parameter's gradient by it,
so the optimizer writes NaN into the whole model.  bf16 training has no
gradient scaler, so Megatron's other non-finite check (inside ``if
self.grad_scaler:``) does not run either.  The failure is silent weight
corruption, not a skipped step (S27 G-FJ, G-FK).

With ``check_for_nan_in_grad=True`` the same situation is loud:
``check_grads`` takes the L2 norm of each bucket's gradient buffer before the
data-parallel collective and calls ``rerun_state_machine.validate_result(...,
fatal=True)``, which raises ``RuntimeError`` even in the default
``RerunMode.DISABLED`` (``rerun_state_machine.py:520-534``).  ms-swift sets the
same flag unconditionally (``megatron_lm_utils.py:674``), which is why the
Ascend SFT recipe never carried this exposure; verl's config path never sets it
(``verl/utils/megatron_utils.py:98-105`` builds the DDP config from
``use_distributed_optimizer``, ``grad_reduce_in_fp32`` and
``overlap_grad_reduce`` only), so a yeto island has to ask for it.

Cost: one extra L2 norm over each bucket's gradient buffer per step, before a
collective that already reads the same buffer.  **Not measured here.**

``check_for_large_grads`` stays off: its rejection is ``fatal=False`` (a warning
only) and its threshold is a fixed 10x, which is a training-quality heuristic,
not a corruption guard.

torch and Megatron are not imported here; this module reads attributes off the
objects the backend hands it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# The default of every yeto island that trains on Megatron. Turned off per run by
# ``--rl-no-grad-nan-check`` (the launcher flag), never silently.
CHECK_FOR_NAN_IN_GRAD = True
# Megatron's DDP field this module owns.
DDP_FIELD = "check_for_nan_in_grad"


class MegatronGradCheckError(RuntimeError):
    """The trainer does not carry the gradient check the island asked for."""


def ddp_overrides(*, check_for_nan_in_grad: bool = CHECK_FOR_NAN_IN_GRAD) -> dict[str, bool]:
    """The Megatron DDP settings the island asks for; ``{}`` when the check is off.

    Off means "keep Megatron's own default", so a run that turns the check off
    reads exactly like a run from before this module."""

    return {DDP_FIELD: True} if check_for_nan_in_grad else {}


def _ddp_config(chunk: Any) -> Any | None:
    """The ``DistributedDataParallelConfig`` of one model chunk, or None when the
    chunk is not DDP-wrapped (``forward_only`` engines are not)."""

    seen = 0
    while chunk is not None and seen < 4:
        config = getattr(chunk, "ddp_config", None)
        if config is not None:
            return config
        chunk, seen = getattr(chunk, "module", None), seen + 1
    return None


def report(chunks: Sequence[Any]) -> dict[str, Any]:
    """What the DDP wrappers of this rank actually carry (no collective).

    ``wrapped`` counts the chunks that have a DDP config at all; ``check_for_nan_in_grad``
    is True only when every wrapped chunk has it."""

    configs = [config for config in (_ddp_config(chunk) for chunk in chunks) if config is not None]
    flags = [bool(getattr(config, DDP_FIELD, False)) for config in configs]
    return {"chunks": len(chunks), "wrapped": len(configs), DDP_FIELD: bool(flags) and all(flags),
            "check_for_large_grads": all(bool(getattr(c, "check_for_large_grads", False)) for c in configs)
            if configs else False}


def check_reports(reports: Sequence[dict], *, wanted: bool) -> None:
    """Refuse at island start when the island asked for the check and a rank does not
    carry it (an override that Hydra dropped would otherwise pass unnoticed)."""

    if not wanted:
        return
    bad = sorted(int(row.get("rank", -1)) for row in reports
                 if row.get("wrapped") and not row.get(DDP_FIELD))
    unwrapped = sorted(int(row.get("rank", -1)) for row in reports if not row.get("wrapped"))
    if bad:
        raise MegatronGradCheckError(
            f"the trainer runs without {DDP_FIELD} on rank(s) {bad[:8]}: a non-finite gradient would be "
            f"scaled into every parameter instead of raising (yeto.rl.megatron_grad_checks). Pass "
            f"--rl-no-grad-nan-check to run without it on purpose")
    if unwrapped:
        raise MegatronGradCheckError(
            f"rank(s) {unwrapped[:8]} have no Megatron DDP config, so {DDP_FIELD} cannot be checked")
