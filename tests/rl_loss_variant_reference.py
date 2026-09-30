"""Independent reference formulas for rl-algo-loss-variants (design D3, task 2.1).

Written from the papers with plain torch on small tensors; imports neither
yeto nor miles, so it can check the Miles fork (``michaellchung/miles``
``--policy-loss-variant``) and yeto's translation without sharing code.

Notation: rho_t = exp(logp_t - old_logp_t), A_t the advantage, mask the loss
mask. Every function returns per-token losses; ``token_mean`` is the
per-token normalized aggregation.

Sources and choices (arXiv ids as in docs/research/rl-algorithms/research.md
§3; the papers were not re-read offline, so no equation numbers are cited --
the formulas are the ones research.md and design D3 quote):
* CISPO -- MiniMax-M1 technical report (arXiv:2506.13585): J = E[ sg(r_hat) * A * log pi_theta ], r_hat = clip(r, 1-eps_l,
  1+eps_h), normalized by the total token count. Whether the paper uses eps_l
  is open (design Open Questions); here both bounds come from the spec
  (loss.eps_clip / loss.eps_clip_high), as in the fork.
* SAPO -- Qwen "Soft Adaptive Policy Optimization" (arXiv:2511.20347), soft gate: f(r) = sigmoid(tau (r - 1)) * 4 / tau, tau = tau_pos for
  A > 0 and tau_neg otherwise (defaults 1.0 / 1.05). The loss is -f(r) * A;
  df/dr = 4 sigma (1 - sigma) equals 1 at r = 1 (PPO-like on-policy gradient).
  Token normalization as the fork (whether the paper normalizes per token
  is not re-checked; design Risks).
* GMPO -- "Geometric-Mean Policy Optimization" (arXiv:2507.20673):
  per token l_t = sign(A) * clip(sign(A) * log r_t, -delta_l, delta_h);
  sequence ratio exp(mean_t l_t) over the sequence's valid tokens; loss
  -ratio * A. Default delta = 0.4 (range (e^-0.4, e^0.4)).
* Combination order (design D3): variant token loss first, then multiplied by
  the (truncated) TIS / IcePop weight, then aggregated.
"""

from __future__ import annotations

import torch


def ratio(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(logp - old_logp)


def cispo(logp, old_logp, adv, eps_low, eps_high):
    weight = ratio(logp, old_logp).detach().clamp(1 - eps_low, 1 + eps_high)
    return -weight * adv * logp


def sapo(logp, old_logp, adv, tau_pos=1.0, tau_neg=1.05):
    r = ratio(logp, old_logp)
    tau = torch.where(adv > 0, torch.full_like(r, tau_pos), torch.full_like(r, tau_neg))
    gate = torch.sigmoid(tau * (r - 1)) * 4 / tau
    return -gate * adv


def gmpo_sequence(logp, old_logp, adv, mask, delta_low=0.4, delta_high=0.4):
    """One sequence (1-D tensors); ``adv`` is constant along the sequence (GRPO)."""

    sign = torch.sign(adv)
    signed = sign * (logp - old_logp)
    clipped = signed.clamp(-delta_low, delta_high)
    ell = sign * clipped
    m = mask.to(logp.dtype)
    seq_ratio = torch.exp((ell * m).sum() / m.sum().clamp_min(1))
    return -seq_ratio * adv, ((clipped != signed).to(m.dtype) * m).sum() / m.sum().clamp_min(1)


def gmpo_sharded(logp, old_logp, adv, mask, shards, delta_low=0.4, delta_high=0.4):
    """Context parallel: each shard reduces (sum l, count) and all-reduces them."""

    sign = torch.sign(adv)
    ell = sign * (sign * (logp - old_logp)).clamp(-delta_low, delta_high)
    m = mask.to(logp.dtype)
    bounds = torch.tensor_split(torch.arange(len(logp)), shards)
    partial = [((ell[i] * m[i]).sum(), m[i].sum()) for i in bounds]
    total = sum(p[0] for p in partial)
    count = sum(p[1] for p in partial)
    seq_ratio = torch.exp(total / count.clamp_min(1))
    return [(-seq_ratio * adv[i]) for i in bounds]


def tis_weight(old_logp, rollout_logp, clip_low, clip_high):
    return torch.exp(old_logp - rollout_logp).clamp(clip_low, clip_high)


def icepop_weight(old_logp, rollout_logp, low, high):
    w = torch.exp(old_logp - rollout_logp)
    return torch.where((w >= low) & (w <= high), w, torch.zeros_like(w))


def token_mean(losses, mask):
    m = mask.to(losses.dtype)
    return (losses * m).sum() / m.sum().clamp_min(1)
