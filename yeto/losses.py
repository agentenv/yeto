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


def load_custom_loss(spec: str):
    """Load a user-supplied loss from a ``custom:<file.py>[:<fn>]`` spec.

    The file must define ``<fn>`` (default name: ``loss_fn``) with signature
    ``fn(logits, input_ids, weights) -> (loss, num_tokens)``, where ``weights``
    is a (B, T) float tensor of per-token loss weights aligned with
    ``input_ids`` (e.g. 1.0 on assistant tokens, 0.0 elsewhere with
    --train-on assistant). Because the learner owns the forward pass, the
    callable receives full logits — no extra forward pass or logprob
    round-trip is needed. The file must live inside the repo so the workdir
    sync ships it to every learner.
    """
    import importlib.util
    from pathlib import Path

    body = spec.split(":", 1)[1]
    path, _, fn_name = body.partition(":")
    fn_name = fn_name or "loss_fn"
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"custom loss file {path!r} not found")
    module_spec = importlib.util.spec_from_file_location("yeto_custom_loss", file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"{path} does not define {fn_name}()")
    return fn


def dump_pickled_loss(fn, path) -> None:
    """Serialize a loss callable by value (closures included) for shipping
    to learners via the workdir sync. Requires matching library versions on
    both ends, which the pinned requirements.txt provides."""
    import cloudpickle

    with open(path, "wb") as f:
        cloudpickle.dump(fn, f)


def load_pickled_loss(spec: str):
    """Load a loss callable from a ``pickle:<file>`` spec."""
    import pickle

    with open(spec.split(":", 1)[1], "rb") as f:
        return pickle.load(f)


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_function: str = "cross_entropy",
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised loss for causal LM training.

    logits: (B, T, V); labels: (B, T) with -100 on masked positions;
    weights: optional (B, T) per-token loss weights aligned with labels.
    Weights are shifted alongside the labels, so weights[t] scales the loss
    of predicting token t; weight-0 positions contribute nothing. Returns
    (loss, num_target_tokens) where num_target_tokens counts positions with
    weight > 0 after the shift (all unmasked positions when weights is None).
    Only cross_entropy is meaningful for SFT; RL losses need
    sampling_logprobs/advantages, which an offline chat dataset does not
    carry. For anything else, use a custom loss
    (``--loss-function custom:file.py``).
    """
    if loss_function != "cross_entropy":
        raise ValueError(
            f"loss function {loss_function!r} requires sampling logprobs and "
            f"advantages (RL data); SFT datasets support cross_entropy or a "
            f"custom:<file.py> loss"
        )
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    target_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    w = mask.to(logprobs.dtype)
    if weights is not None:
        w = w * weights[:, 1:].to(logprobs.dtype)
    n_tokens = (w > 0).sum()
    loss = cross_entropy(target_logprobs, w)
    return loss, n_tokens
