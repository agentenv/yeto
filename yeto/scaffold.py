"""SCAFFOLD-lite inner control variates for DiLoCo (candidate #5,
docs/OTHER_OPTIMIZERS.md).

Endpoint-derived control variates that reduce cross-worker client drift while
the outer optimizer stays SGD-0.28. Each worker i keeps a local control vector
``c_i``; the syncer maintains the token-weighted mean ``c`` and broadcasts it;
the next inner window applies the corrected gradient ``grad_i - c_i + c`` at
every optimizer step. The controls are derived entirely from the window
ENDPOINT (the same delta the learner already pushes) — there is NO extra
forward/backward pass.

One token-normalized formula everywhere (no per-H tuning)
--------------------------------------------------------
Over a window a worker moves the fragment from its exact pushed base anchor
``A_i`` to its endpoint ``theta_i`` while consuming ``T_i`` raw tokens. Define the
worker's token-normalized control (units: parameter-move per token):

    c_i = (theta_i - A_i) / T_i                                          (1)

The syncer holds every worker's endpoint and token count, so it forms the
token-weighted mean control from the SAME per-worker deltas it already merges:

    c   = ( sum_i (theta_i - A_i) ) / ( sum_i T_i )
        = sum_i (T_i / sum_j T_j) * c_i                                  (2)

i.e. ``c`` is the token-weighted mean of the ``c_i`` (equation 2 is exactly the
token-normalized merged pseudo-move, which the syncer already computes). ``c``
is broadcast back to every learner.

Per inner optimizer step (which consumes ``tokens_per_step`` raw tokens at
inner learning rate ``eta``) the SCAFFOLD correction ``grad <- grad - c_i + c``
is realized in gradient units as

    grad_delta = (c_i - c) * tokens_per_step / eta                       (3)

added to ``.grad`` before clipping/stepping. Derivation: a token-normalized
control ``c`` corresponds to a per-token gradient ``ghat = -c / eta`` (an SGD
step of ``eta`` tokens moves the parameter by ``-eta * ghat``); SCAFFOLD's
per-step correction ``ghat_c - ghat_i`` over ``tokens_per_step`` tokens is
``(ghat_c - ghat_i) * tokens_per_step = (c_i - c) * tokens_per_step / eta``.

Correctness envelope
--------------------
The zero-sum/unbiased identity for (3) is an identity of constant-learning-rate
plain SGD with equal token counts. It does *not* extend through AdamW, warmup,
weight decay, or gradient clipping: Adam's second moment, independently evolved
worker state, decoupled weight decay, and clipping are nonlinear transforms of
the corrected gradient. The production learner therefore exposes an explicit
plain-SGD correctness mode. The older AdamW/clipped path remains available as
an experiment, but is deliberately not described as unbiased.

Controls are also versioned. A learner may use a pair only when its local
``c_i`` and the server mean ``c`` came from the same completed fragment round.
This prevents an early local push or a late control broadcast from combining
controls from different endpoints.

Why token normalization (not per-step)
--------------------------------------
Normalizing by tokens (not by microstep count) makes ``c_i`` invariant to the
sync horizon H: a window twice as long has twice the delta AND twice the
tokens, so ``c_i`` is unchanged. The correction ``grad_delta`` is what carries
the horizon tilt: a fixed per-token control drives a per-step correction
proportional to ``tokens_per_step``, so the accumulated correction over an
H-step window scales with H — small at H16, larger at H256 (crossover-safe).

Under IID data every worker's endpoint delta matches the consensus, so
``c_i == c`` for all i and the correction (3) is identically zero: SCAFFOLD-lite
adds nothing to fix when there is no cross-worker drift.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("scaffold")

__all__ = [
    "local_control",
    "mean_control",
    "grad_correction",
    "effective_step_gradient",
    "VersionedControlPairs",
    "zero_sum_step_diagnostics",
]


class VersionedControlPairs:
    """Match local and mean controls atomically by fragment version."""

    def __init__(self) -> None:
        self._local: dict[int, torch.Tensor] = {}
        self._mean: dict[int, torch.Tensor] = {}

    def add_local(self, version: int, control: torch.Tensor) -> None:
        self._local[int(version)] = control

    def add_mean(self, version: int, control: torch.Tensor) -> None:
        self._mean[int(version)] = control

    def get(self, version: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        version = int(version)
        local = self._local.get(version)
        mean = self._mean.get(version)
        if local is None or mean is None:
            return None
        return local, mean

    def discard_before(self, version: int) -> None:
        """Drop halves that can no longer match the applied fragment version."""
        version = int(version)
        self._local = {v: c for v, c in self._local.items() if v >= version}
        self._mean = {v: c for v, c in self._mean.items() if v >= version}


def local_control(
    anchor: torch.Tensor, endpoint: torch.Tensor, tokens: float
) -> torch.Tensor:
    """Token-normalized worker control ``c_i`` (equation 1).

    ``anchor`` is the fragment value at the window start (the last global the
    learner applied); ``endpoint`` is the fragment value the learner pushes at
    the window end. ``tokens`` is the raw token count of the window (``c_tokens``
    on the wire). Reuses the already-materialized endpoint — no extra forward.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    return (endpoint.detach().float() - anchor.detach().float()) / float(tokens)


def mean_control(
    controls: list[torch.Tensor], token_counts: list[float]
) -> torch.Tensor:
    """Token-weighted mean control ``c`` (equation 2).

    This is what the syncer broadcasts. Given per-worker controls ``c_i`` and
    their token counts ``T_i`` it returns ``sum_i T_i c_i / sum_i T_i``, which
    equals ``sum_i (theta_i - A_i) / sum_i T_i`` — the token-normalized merged
    pseudo-move the syncer forms from the per-worker deltas it already has.
    """
    if len(controls) != len(token_counts):
        raise ValueError("controls and token_counts must be the same length")
    if not controls:
        raise ValueError("need at least one worker control")
    total = float(sum(token_counts))
    if total <= 0:
        raise ValueError("total tokens must be positive")
    acc = torch.zeros_like(controls[0], dtype=torch.float32)
    for c_i, t_i in zip(controls, token_counts):
        acc = acc + c_i.detach().float() * float(t_i)
    return acc / total


def grad_correction(
    local_c: torch.Tensor,
    mean_c: torch.Tensor,
    tokens_per_step: float,
    inner_lr: float,
) -> torch.Tensor:
    """SCAFFOLD per-step gradient correction (equation 3), added to ``.grad``.

    ``(c_i - c) * tokens_per_step / eta``. Zero when ``c_i == c`` (IID). The
    inner learning rate ``eta`` converts the token-normalized parameter-move
    control into gradient units so the result can be summed onto ``.grad``
    before clipping.
    """
    if inner_lr <= 0:
        raise ValueError(f"inner_lr must be positive, got {inner_lr}")
    scale = float(tokens_per_step) / float(inner_lr)
    return (local_c.detach().float() - mean_c.detach().float()) * scale


def effective_step_gradient(
    grad: torch.Tensor,
    local_c: torch.Tensor,
    mean_c: torch.Tensor,
    tokens_per_step: float,
    inner_lr: float,
) -> torch.Tensor:
    """The corrected per-step gradient ``grad - c_i + c`` (in gradient units).

    Convenience wrapper used to reason about the effective update; the learner
    adds :func:`grad_correction` onto ``.grad`` in place instead.
    """
    return grad.detach().float() + grad_correction(
        local_c, mean_c, tokens_per_step, inner_lr
    )


def zero_sum_step_diagnostics(
    corrections: list[torch.Tensor],
    parameters_before: list[torch.Tensor],
    parameters_after: list[torch.Tensor],
) -> dict[str, float]:
    """Measure aggregate correction and real optimizer displacement.

    Corrections are captured after gradient addition but before clipping, while
    the parameter snapshots bracket the real ``opt.step()``. This is intended
    for the equal-token, constant-LR plain-SGD correctness run.
    """
    if not corrections:
        raise ValueError("need at least one worker correction")
    if not (
        len(corrections) == len(parameters_before) == len(parameters_after)
    ):
        raise ValueError("corrections and parameter snapshots must have equal length")
    correction_sum = torch.stack([c.detach().float() for c in corrections]).sum(0)
    displacement_sum = torch.stack(
        [
            after.detach().float() - before.detach().float()
            for before, after in zip(parameters_before, parameters_after)
        ]
    ).sum(0)
    diagnostics = {
        "correction_sum_l2": float(correction_sum.norm().item()),
        "correction_sum_max_abs": float(correction_sum.abs().max().item()),
        "displacement_sum_l2": float(displacement_sum.norm().item()),
        "displacement_sum_max_abs": float(displacement_sum.abs().max().item()),
    }
    log.info(
        "SCAFFOLD plain-SGD zero-sum before_clip correction_sum_l2=%.9g "
        "correction_sum_max_abs=%.9g after_step displacement_sum_l2=%.9g "
        "displacement_sum_max_abs=%.9g",
        diagnostics["correction_sum_l2"],
        diagnostics["correction_sum_max_abs"],
        diagnostics["displacement_sum_l2"],
        diagnostics["displacement_sum_max_abs"],
    )
    return diagnostics
