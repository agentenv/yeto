"""Loss functions for SFT and RL fine-tuning.

All losses SUM over tokens (not mean). Each takes `target_logprobs` —
log p_theta(x_t) for the target token at each position — plus per-loss
inputs, and returns a scalar loss tensor.

Available: cross_entropy | importance_sampling | ppo | cispo | dro
"""

from __future__ import annotations

import torch

LOSS_FUNCTIONS = ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro")


def cross_entropy(target_logprobs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """L = -sum_t w_t * log p_theta(x_t)."""
    return -(target_logprobs * weights).sum()


def importance_sampling(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """L = -sum_t (p_theta/q)(x_t) * A_t."""
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    return -(ratio * advantages).sum()


def ppo(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low_threshold: float = 0.8,
    clip_high_threshold: float = 1.2,
) -> torch.Tensor:
    """L = -sum_t min(r_t A_t, clip(r_t, low, high) A_t)."""
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped = ratio.clamp(clip_low_threshold, clip_high_threshold)
    return -torch.minimum(ratio * advantages, clipped * advantages).sum()


def cispo(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low_threshold: float = 0.0,
    clip_high_threshold: float = 4.0,
) -> torch.Tensor:
    """L = -sum_t sg(clip(r_t)) * log p_theta(x_t) * A_t (detached clipped ratio)."""
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    coef = ratio.clamp(clip_low_threshold, clip_high_threshold).detach()
    return -(coef * target_logprobs * advantages).sum()


def dro(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """L = -sum_t [log p_theta(x_t) A_t - (beta/2)(log p_theta(x_t) - log q(x_t))^2]."""
    kl_sq = (target_logprobs - sampling_logprobs) ** 2
    return -(target_logprobs * advantages - 0.5 * beta * kl_sq).sum()


def sft_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_function: str = "cross_entropy"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised loss for causal LM training.

    logits: (B, T, V); labels: (B, T) with -100 on masked positions.
    Returns (loss, num_target_tokens). Only cross_entropy is meaningful for
    SFT; RL losses need sampling_logprobs/advantages, which an offline chat
    dataset does not carry.
    """
    if loss_function != "cross_entropy":
        raise ValueError(
            f"loss function {loss_function!r} requires sampling logprobs and "
            f"advantages (RL data); SFT datasets support only cross_entropy"
        )
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    n_tokens = mask.sum()
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    target_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    loss = cross_entropy(target_logprobs, mask.to(logprobs.dtype))
    return loss, n_tokens
