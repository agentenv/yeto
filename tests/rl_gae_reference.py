"""Independent torch reference implementation of the GAE variants (design D6).

This module MUST NOT import Miles / fork code; it is the oracle used to
cross-check the fork's ``--gae-variant`` dispatch (tasks.md 6.1/6.2).

All functions operate on one trajectory (1-D tensors of length T, already
restricted to trainable tokens) with zero bootstrap after the last token:
    delta_t = r_t + gamma * V_{t+1} - V_t,   V_T = 0
    A_t     = sum_l (gamma*lambda)^l delta_{t+l}               (Schulman et al. 2016)

Variants
- vanilla:          A as above, returns R_t = A_t + V_t.
- length_adaptive:  lambda = 1 - 1/(alpha * l), l = response length,
                    alpha default 1.5 (VAPO, Yue et al. 2025, Eq. "length-adaptive GAE").
- decoupled:        policy advantages use lambda_policy, value targets use
                    lambda_critic (VAPO "decoupled GAE", typically lambda_critic = 1):
                    A = GAE(lambda_policy), R = GAE(lambda_critic) + V.
- cross_segment:    CompactionRL.  For segment s with n_s tokens,
                    A^loc_{s,i} = sum_{l=0}^{n_s-i} (gamma*lambda)^l delta_{s,i+l},
                    where the bootstrap value after the last token of every segment
                    is 0 (no bootstrapping across a compaction boundary) and the
                    terminal reward sits at the end of the last segment; then
                    A_{s,i} = (gamma*lambda)^{N_{>s}} * A^loc_{s,i},  N_{>s} = sum_{j>s} n_j.
                    Returns R = A + V.
"""

from __future__ import annotations

import torch


def deltas(rewards: torch.Tensor, values: torch.Tensor, gamma: float) -> torch.Tensor:
    nxt = torch.cat([values[1:], values.new_zeros(1)])
    return rewards + gamma * nxt - values


def gae_vanilla(rewards, values, gamma: float, lambd: float):
    d = deltas(rewards, values, gamma)
    T = d.numel()
    adv = torch.zeros_like(d)
    for t in range(T):
        adv[t] = sum((gamma * lambd) ** k * d[t + k] for k in range(T - t))
    return adv, adv + values


def length_adaptive_lambda(response_length: int, alpha: float = 1.5) -> float:
    return 1.0 - 1.0 / (alpha * response_length)


def gae_length_adaptive(rewards, values, gamma: float, response_length: int, alpha: float = 1.5):
    return gae_vanilla(rewards, values, gamma, length_adaptive_lambda(response_length, alpha))


def gae_decoupled(rewards, values, gamma: float, lambd_policy: float, lambd_critic: float):
    adv, _ = gae_vanilla(rewards, values, gamma, lambd_policy)
    adv_c, _ = gae_vanilla(rewards, values, gamma, lambd_critic)
    return adv, adv_c + values


def gae_cross_segment(rewards, values, segment_ids, gamma: float, lambd: float):
    seg = [int(s) for s in segment_ids]
    assert seg == sorted(seg), "segment ids must be non-decreasing"
    order = list(dict.fromkeys(seg))
    adv = torch.zeros_like(values)
    for si, s in enumerate(order):
        pos = [t for t, x in enumerate(seg) if x == s]
        r, v = rewards[pos], values[pos]
        local, _ = gae_vanilla(r, v, gamma, lambd)  # bootstrap 0 at segment end
        n_after = sum(seg.count(o) for o in order[si + 1 :])
        adv[pos] = local * (gamma * lambd) ** n_after
    return adv, adv + values
