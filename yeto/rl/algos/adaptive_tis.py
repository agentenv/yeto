"""Adaptive truncated importance sampling (S26 E2, opt-in, default off).

A Miles ``--custom-tis-function-path`` function (same call shape as Miles'
``vanilla_tis_function``) that sets the strength of the rollout/trainer
correction per mini-batch instead of always applying the full truncated weight.
It follows AIS (Zhou et al., "AIS: Adaptive Importance Sampling for Quantized
RL", arXiv:2605.13907, Eqs. 12-18):

* ``rho = exp(logp_train - logp_rollout)`` per token, ``w = clamp(rho, lower, C)``
  with ``C`` = ``--tis-clip`` (``correction.upper``) and ``lower`` =
  ``--tis-clip-low``;
* reliability gate ``a_ess = (1 + CV(w)^2) ** -0.5`` (square root of the
  effective-sample-size ratio);
* necessity gate ``a_mis = min(1, mean|logp_train - logp_rollout| / delta)``;
* variance gate ``a_var = max(0, (dsigma - gamma) / gamma)`` with
  ``dsigma = std(x * w) / (std(x) + eps)``;
* ``alpha = clip(a_ess - beta * a_var, 0, 1) * a_mis``; the token weight is
  ``1 + alpha * (w - 1)``: ``alpha = 0`` is the uncorrected gradient,
  ``alpha = 1`` is plain TIS.

Differences from the paper, on purpose:

* ``x`` in the variance gate is the per-token policy-gradient loss Miles hands
  the function (``pg_loss``), not the advantage: the TIS hook does not receive
  the advantages. Near on-policy ``pg_loss = -A * clip(ratio) ~ -A``, so the ratio
  of standard deviations is close to the paper's; this is not verified on GPU.
* ``beta`` is not given in the paper; :data:`BETA` = 1.0 is our choice.
* statistics are taken over this rank's masked tokens of the mini-batch (no
  data-parallel all-reduce), so ranks may use slightly different ``alpha``.

QuRL's Adaptive Clipping Range (Li et al., ICLR 2026) widens the PPO clip upper
bound to ``(1 + eps) / r`` for truncated tokens; it changes the PPO clip, which
Miles computes before this hook, so it is not implemented here.

Use: ``correction.method = "custom"``, ``correction.function.path =`` :data:`PATH`,
``correction.upper`` (paper: 5.0) and ``correction.lower`` (0.0). The metrics
keep Miles' names (``tis``, ``tis_clipfrac``, ``tis_abs``) and add ``ais_alpha``
and the three gates, so reports and the offline parsers read them unchanged.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

PATH = "yeto.rl.algos.adaptive_tis.ais_tis_function"

# AIS Appendix C.4 (tuned on GSM8K, fixed across tasks): delta 0.02, gamma 1.2, eps 1e-6.
DELTA = 0.02
GAMMA = 1.2
BETA = 1.0
EPS = 1e-6


@dataclass(frozen=True)
class AisGates:
    """The per-mini-batch gates and the resulting mixing coefficient."""

    ess: float
    mis: float
    var: float
    alpha: float


def ais_gates(weights: Any, log_diff: Any, signal: Any, *, delta: float = DELTA,
              gamma: float = GAMMA, beta: float = BETA, eps: float = EPS) -> AisGates:
    """Gates over 1-D tensors of the masked tokens: truncated weights, the log
    ratio ``logp_train - logp_rollout`` and the per-token signal ``x``."""

    import torch

    if weights.numel() == 0:
        return AisGates(ess=1.0, mis=0.0, var=0.0, alpha=0.0)
    w = weights.double()
    mean = w.mean()
    var_w = w.var(unbiased=False)
    cv2 = var_w / (mean * mean).clamp_min(eps)
    a_ess = float((1.0 + cv2).rsqrt())
    a_mis = min(1.0, float(log_diff.double().abs().mean()) / delta)
    x = signal.double()
    dsigma = float((x * w).std(unbiased=False) / (x.std(unbiased=False) + eps))
    a_var = max(0.0, (dsigma - gamma) / gamma)
    alpha = float(torch.tensor(a_ess - beta * a_var).clamp(0.0, 1.0)) * a_mis
    return AisGates(ess=a_ess, mis=a_mis, var=a_var, alpha=alpha)


def ais_tis_function(
    args: Namespace,
    *,
    pg_loss: Any,
    train_log_probs: list[Any],
    rollout_log_probs: list[Any],
    loss_masks: list[Any],
    **kwargs: Any,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Miles TIS hook: weight ``pg_loss`` by ``1 + alpha * (clamp(rho) - 1)``.

    ``loss_masks`` is returned unchanged; metrics report the pre-clamp ratio,
    as in ``vanilla_tis_function``."""

    import torch

    log_diff = torch.cat(train_log_probs, dim=0) - torch.cat(rollout_log_probs, dim=0)
    rho = torch.exp(log_diff)
    truncated = torch.clamp(rho, min=args.tis_clip_low, max=args.tis_clip)
    with torch.no_grad():
        mask = torch.cat(loss_masks, dim=0).bool()
        gates = ais_gates(truncated[mask], log_diff[mask], pg_loss.detach()[mask])
    weights = 1.0 + gates.alpha * (truncated - 1.0)
    metrics = {
        "tis": rho.clone().detach(),
        "tis_clipfrac": (truncated != rho).float(),
        "tis_abs": (rho - 1.0).abs().detach(),
        "ais_alpha": torch.full_like(rho, gates.alpha),
        "ais_gate_ess": torch.full_like(rho, gates.ess),
        "ais_gate_mis": torch.full_like(rho, gates.mis),
        "ais_gate_var": torch.full_like(rho, gates.var),
    }
    return pg_loss * weights.detach(), loss_masks, metrics
