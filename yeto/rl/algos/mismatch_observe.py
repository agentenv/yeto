"""Observe-only train/inference mismatch plugin (change ``rl-algo-mismatch-correction`` D2).

A Miles custom-TIS function (``--custom-tis-function-path``; called by Miles
``loss_hub/losses.py`` whenever ``--use-tis`` or ``--get-mismatch-metrics`` is
set). It returns ``pg_loss`` and ``loss_masks`` unchanged -- every importance
weight is 1 -- and only reports mismatch metrics, computed exactly like
Miles' built-in ``vanilla_tis_function`` (``loss_hub/corrections.py``):

* ``tis``      = exp(train_old - rollout) per token (pre-clamp ratio);
* ``tis_abs``  = |exp(train_old - rollout) - 1|;
* ``mismatch_outside_0p5_5`` = 1 where the ratio is outside the reference
  interval [0.5, 5] (the IcePop reference interval, tasks 7.4), else 0;
* ``tis_abs_p50`` / ``tis_abs_p90`` / ``tis_abs_p99``: quantiles of
  ``tis_abs`` over the loss-masked tokens of this call (one micro-batch),
  broadcast to every token so that Miles' per-sample mean reducer reports
  them unchanged; the logged value is their mean over micro-batches.

Miles itself adds ``ois``, ``train_rollout_kl``,
``train_rollout_logprob_abs_diff`` and ``ess_ratio`` to the same reported
loss dict. The source SHA256 of this file enters the algorithm identity
(``PluginRef``), so keep it free of unrelated code.
"""

from __future__ import annotations

from typing import Any

REFERENCE_INTERVAL = (0.5, 5.0)
QUANTILES = (0.5, 0.9, 0.99)
QUANTILE_NAMES = ("p50", "p90", "p99")


def observe_mismatch(
    args: Any,
    *,
    pg_loss: Any,
    train_log_probs: list,
    rollout_log_probs: list,
    loss_masks: list,
    **kwargs: Any,
):
    import torch

    rollout = torch.cat(rollout_log_probs, dim=0)
    old = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old - rollout)
    tis_abs = (torch.exp(old - rollout) - 1).abs()
    low, high = REFERENCE_INTERVAL
    outside = (~((tis >= low) & (tis <= high))).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
        "mismatch_outside_0p5_5": outside.clone().detach(),
    }
    active = tis_abs.detach()[torch.cat(loss_masks, dim=0).to(tis_abs.device) > 0].float()
    levels = torch.tensor(QUANTILES, dtype=torch.float32, device=tis_abs.device)
    values = torch.quantile(active, levels) if active.numel() else torch.zeros_like(levels)
    for level, value in zip(QUANTILE_NAMES, values):
        metrics[f"tis_abs_{level}"] = value.to(tis_abs.dtype).expand_as(tis_abs).clone()
    return pg_loss, loss_masks, metrics
