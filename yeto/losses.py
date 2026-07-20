"""Loss functions for SFT and RL fine-tuning.

All losses SUM over tokens (not mean). Each takes `target_logprobs` —
log p_theta(x_t) for the target token at each position — plus per-loss
inputs, and returns a scalar loss tensor.

Available: cross_entropy | importance_sampling | ppo | cispo | dro | flow_matching
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_FUNCTIONS = ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro", "flow_matching")


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


def flow_matching_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow diffusion loss: MSE against the velocity target.

    ``timestep`` is part of the public signature so future weighting schemes
    can be added without changing the diffusion learner's loss interface.
    """
    del timestep
    per_elem = (model_output.float() - target.float()).pow(2)
    if weights is not None:
        weights = weights.to(device=per_elem.device, dtype=per_elem.dtype)
        while weights.ndim < per_elem.ndim:
            weights = weights.view(*weights.shape, *([1] * (per_elem.ndim - weights.ndim)))
        per_elem = per_elem * weights
        return per_elem.sum(), weights.expand_as(per_elem).sum().clamp(min=1)
    loss = per_elem.sum()
    return loss, torch.tensor(per_elem.numel(), device=per_elem.device)


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
            f"loss function {loss_function!r} is not a causal-LM SFT loss; "
            f"SFT datasets support cross_entropy or a custom:<file.py> loss"
        )
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    flat_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    per_token_loss = flat_loss.view_as(shift_labels)
    w = mask.to(per_token_loss.dtype)
    if weights is not None:
        w = w * weights[:, 1:].to(per_token_loss.dtype)
    n_tokens = (w > 0).sum()
    loss = (per_token_loss * w).sum()
    return loss, n_tokens
