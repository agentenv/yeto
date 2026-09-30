"""Independent reference formulas for rl-algo-loss-variants (design D3, task 2.1).

Written from the papers with plain torch on small tensors; imports neither
yeto nor miles, so it can check the Miles fork (``michaellchung/miles``
``--policy-loss-variant``) and yeto's translation without sharing code.

Notation: rho_t = exp(logp_t - old_logp_t), A_t the advantage, mask the loss
mask. Every function returns per-token losses; ``token_mean`` is the
per-token normalized aggregation.

Sources (each formula checked against the paper text on 2026-09-30 via
arxiv.org HTML; GMPO also against the official code):
* CISPO -- MiniMax-M1 (arXiv:2506.13585, section 3.1), eq. (4)
  J = E[ 1/sum_i |o_i| * sum_i sum_t sg(r_hat_it) A_it log pi_theta ] with
  eq. (5) r_hat = clip(r, 1-eps_low^IS, 1+eps_high^IS). Differences from the
  yeto/fork use, recorded: the paper imposes NO lower bound in practice
  ("setting eps_low^IS to a large value", only eps_high tuned) -- yeto takes
  both from loss.eps_clip / loss.eps_clip_high, so a paper-like run sets
  loss.eps_clip >= 1; the paper normalizes by the token count of the group,
  Miles' --calculate-per-token-loss (required by yeto for CISPO) by the token
  count of the whole micro/global batch -- equal up to how groups share a
  batch; the optional mask M of eq. (6)-(7) is not CISPO and not implemented.
* SAPO -- Qwen "Soft Adaptive Policy Optimization" (arXiv:2511.20347v2),
  eq. (5) J = E[ 1/G sum_i 1/|y_i| sum_t f_it(r_it) A_it ], eq. (6)
  f(x) = sigmoid(tau (x-1)) * 4/tau, tau = tau_pos if A > 0 else tau_neg
  (A = 0 uses tau_neg; irrelevant, the term is 0); defaults tau_pos 1.0,
  tau_neg 1.05 (section 5.1); no KL. Per-SEQUENCE mean, i.e. Miles' default
  sample-mean aggregation (``sample_mean`` below), not a token mean.
* GMPO -- "Geometric-Mean Policy Optimization" (arXiv:2507.20673v3, eq. (4)):
  J = 1/G sum_i { prod_t |min[rho_t A, clip(rho_t, e^-delta_l, e^delta_h) A]| }^(1/|o_i|)
  * sgn(A). The paper states that the product and the clipping are done in
  log space; in log space, per token (derived here from eq. (4), checked
  against the paper's pseudo-code ``torch.min(sgn_A*logr, clamp(...))``):
      l_t = sign(A) * min(sign(A) * log r_t, sign(A) * clamp(log r_t, -delta_l, delta_h))
  i.e. ONE-SIDED (PPO-pessimistic): for A > 0 only log r > delta_h is clipped
  (log r << -delta_l keeps its gradient); for A < 0 only log r < -delta_l.
  Sequence ratio exp(mean_t l_t) over valid tokens; with a per-sequence
  constant A, |A| * sgn(A) = A, so loss = -ratio * A. Default delta = 0.4.
  Official code (github.com/callsys/GMPO train_zero_math_gmpo.py:675-688,
  checked line by line by the main agent 2026-09-30) matches this form.
  Clip fraction (yeto D5 semantics): per round, sum over sequences of clipped
  valid A != 0 tokens / sum of valid A != 0 tokens (``gmpo_global_clip``;
  the fork reports the two sums as gmpo_clip_num / gmpo_clip_den).
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


def _gmpo_ell(logp, old_logp, adv, delta_low, delta_high):
    log_r = logp - old_logp
    sign = torch.sign(adv)
    unclipped = sign * log_r
    clipped = sign * log_r.clamp(-delta_low, delta_high)
    chosen = torch.minimum(unclipped, clipped)
    return sign * chosen, (chosen != unclipped) & (adv != 0)


def gmpo_sequence(logp, old_logp, adv, mask, delta_low=0.4, delta_high=0.4):
    """One sequence (1-D tensors); ``adv`` is constant along the sequence (GRPO).

    Returns (per-token loss, clip fraction among valid A != 0 tokens or None).
    """

    ell, is_clipped = _gmpo_ell(logp, old_logp, adv, delta_low, delta_high)
    m = mask.to(logp.dtype)
    seq_ratio = torch.exp((ell * m).sum() / m.sum().clamp_min(1))
    active = m * (adv != 0).to(m.dtype)
    frac = None if active.sum() == 0 else (is_clipped.to(m.dtype) * active).sum() / active.sum()
    return -seq_ratio * adv, frac


def gmpo_sharded(logp, old_logp, adv, mask, shards, delta_low=0.4, delta_high=0.4):
    """Context parallel: each shard reduces (sum l, count) and all-reduces them."""

    ell, _ = _gmpo_ell(logp, old_logp, adv, delta_low, delta_high)
    m = mask.to(logp.dtype)
    bounds = torch.tensor_split(torch.arange(len(logp)), shards)
    total = sum((ell[i] * m[i]).sum() for i in bounds)
    count = sum(m[i].sum() for i in bounds)
    seq_ratio = torch.exp(total / count.clamp_min(1))
    return [(-seq_ratio * adv[i]) for i in bounds]


def gmpo_global_clip(sequences, delta_low=0.4, delta_high=0.4):
    """sequences: [(logp, old_logp, adv, mask)] -> (num, den) over A != 0 valid tokens."""

    num = den = 0.0
    for logp, old_logp, adv, mask in sequences:
        _, is_clipped = _gmpo_ell(logp, old_logp, adv, delta_low, delta_high)
        active = mask.bool() & (adv != 0)
        num += float((is_clipped & active).sum())
        den += float(active.sum())
    return num, den


def tis_weight(old_logp, rollout_logp, clip_low, clip_high):
    return torch.exp(old_logp - rollout_logp).clamp(clip_low, clip_high)


def icepop_weight(old_logp, rollout_logp, low, high):
    w = torch.exp(old_logp - rollout_logp)
    return torch.where((w >= low) & (w <= high), w, torch.zeros_like(w))


def sample_mean(losses_per_seq, masks):
    """Mean over sequences of each sequence's masked token mean (SAPO eq. (5))."""

    return sum(token_mean(l, m) for l, m in zip(losses_per_seq, masks)) / len(losses_per_seq)


def token_mean(losses, mask):
    m = mask.to(losses.dtype)
    return (losses * m).sum() / m.sum().clamp_min(1)
