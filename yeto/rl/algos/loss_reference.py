"""Neutral per-token loss reference functions (yeto-framework-decoupling task
4.7, design D8 step 2).

Pure torch on per-token tensors and per-sample scalars only (log-probs,
old/rollout log-probs, advantages, masks); no framework import. They are the
written-down answer that each backend's implementation is compared with on
CPU (``tests/test_rl_loss_reference.py`` checks the Miles fork functions at
the pinned commit). They are **not** called by training: the Miles fork keeps
its own loss dispatch (D8: moving the fork onto these functions is deferred).

Notation: ``log_ratio = logp - old_logp`` (new minus old), ``ratio =
exp(log_ratio)`` with the fork's safety clamp (NaN -> 0, |log_ratio| <= 20).
Every loss returns ``(per_token_loss, clipfrac)``.

Started from ``tests/rl_loss_variant_reference.py`` (paper formulas for CISPO,
SAPO, GMPO, TIS/IcePop); PPO clip / dual-clip and the KL estimators follow
Schulman's KL note and the PPO paper.
"""

from __future__ import annotations

import torch

LOG_RATIO_CLAMP = 20.0


def safe_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """float32, NaN -> 0, +-inf and |x| > 20 clamped to +-20."""

    log_ratio = torch.nan_to_num(log_ratio.float(), nan=0.0, posinf=LOG_RATIO_CLAMP,
                                 neginf=-LOG_RATIO_CLAMP)
    return torch.clamp(log_ratio, min=-LOG_RATIO_CLAMP, max=LOG_RATIO_CLAMP)


def ratio(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    return safe_log_ratio(logp - old_logp).exp()


# --------------------------------------------------------------------------
# policy losses
# --------------------------------------------------------------------------


def ppo_clip(logp, old_logp, adv, eps_low: float, eps_high: float,
             dual_clip_c: float | None = None):
    """PPO clipped surrogate; ``dual_clip_c`` (> 1) adds the dual clip for A < 0."""

    r = ratio(logp, old_logp)
    unclipped = -r * adv
    clipped = -r.clamp(1 - eps_low, 1 + eps_high) * adv
    loss = torch.maximum(unclipped, clipped)
    clipfrac = torch.gt(clipped, unclipped).float()
    if dual_clip_c is not None:
        if not dual_clip_c > 1.0:
            raise ValueError(f"dual-clip constant must be > 1, got {dual_clip_c}")
        loss = torch.where(adv < 0, torch.min(-dual_clip_c * adv, loss), loss)
    return loss, clipfrac


def cispo(logp, old_logp, adv, eps_low: float, eps_high: float):
    """CISPO: -sg(clip(ratio, 1-eps_low, 1+eps_high)) * A * logp (every token keeps a gradient)."""

    r = ratio(logp, old_logp).detach()
    weight = r.clamp(1 - eps_low, 1 + eps_high)
    return -weight * adv * logp, torch.ne(weight, r).float()


def sapo(logp, old_logp, adv, tau_pos: float = 1.0, tau_neg: float = 1.05):
    """SAPO soft gate: -sigmoid(tau (ratio - 1)) * 4 / tau * A; no hard clip."""

    r = ratio(logp, old_logp)
    tau = torch.where(adv > 0, torch.full_like(r, tau_pos), torch.full_like(r, tau_neg))
    loss = -(torch.sigmoid(tau * (r - 1)) * 4 / tau) * adv
    return loss, torch.zeros_like(loss)


# --------------------------------------------------------------------------
# KL estimators (Schulman, http://joschu.net/blog/kl-approx.html)
# --------------------------------------------------------------------------

KL_ESTIMATORS = ("k1", "k2", "k3", "low_var_kl")


def kl(logp, ref_logp, estimator: str, importance_ratio: torch.Tensor | None = None):
    """Per-token KL(pi || ref) estimate; ``low_var_kl`` = k3 with clamps."""

    log_ratio = logp.float() - ref_logp.float()
    if estimator == "k1":
        value = log_ratio
    elif estimator == "k2":
        value = log_ratio ** 2 / 2.0
    elif estimator in ("k3", "low_var_kl"):
        neg = -log_ratio
        if estimator == "low_var_kl":
            neg = safe_log_ratio(neg)
        value = neg.exp() - 1 - neg
    else:
        raise ValueError(f"unknown KL estimator {estimator!r} (known: {KL_ESTIMATORS})")
    if importance_ratio is not None:
        value = importance_ratio * value
    if estimator == "low_var_kl":
        value = torch.clamp(value, min=-10, max=10)
    return value


# --------------------------------------------------------------------------
# train/rollout mismatch corrections (weights multiply the per-token loss)
# --------------------------------------------------------------------------


def tis_weight(old_logp, rollout_logp, clip_low: float, clip_high: float):
    """Truncated IS: clamp(exp(old - rollout), low, high); clipfrac on the pre-clamp ratio."""

    w = torch.exp(old_logp - rollout_logp)
    clamped = torch.clamp(w, min=clip_low, max=clip_high)
    return clamped, (clamped != w).float()


def icepop_weight(old_logp, rollout_logp, low: float, high: float):
    """IcePop: keep the ratio inside [low, high], zero the token outside."""

    w = torch.exp(old_logp - rollout_logp)
    kept = torch.where((w >= low) & (w <= high), w, torch.zeros_like(w))
    return kept, (kept != w).float()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def token_mean(losses, mask):
    m = mask.to(losses.dtype)
    return (losses * m).sum() / m.sum().clamp_min(1)


def sample_mean(losses_per_seq, masks):
    return sum(token_mean(l, m) for l, m in zip(losses_per_seq, masks)) / len(losses_per_seq)
