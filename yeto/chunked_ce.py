"""Device-agnostic chunked linear + cross-entropy SFT loss.

The native loss path materializes the full ``[batch, seq, vocab]`` logits
tensor (plus its FP32 copy and the softmax intermediates of the backward
pass). For long sequences and large vocabularies that dominates peak memory.

This lane runs the model only up to its final hidden states and then computes
``hidden_chunk @ lm_head.weight^T -> cross_entropy`` over bounded token
chunks. Each chunk runs under non-reentrant activation checkpointing, so the
autograd graph retains only the chunk's hidden slice and its scalar loss; the
backward pass recomputes one chunk's logits at a time. The arithmetic mirrors
:func:`yeto.losses.sft_loss` exactly (same label shift, ignore index, token
weights, FP32 CE, local token-SUM loss and positive-weight token count).

Only plain PyTorch ops are used, so it runs on CUDA, Ascend NPU, MPS and CPU.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_LOSS_CHUNK_TOKENS = 4096
CHUNKED_LOSS_IMPLEMENTATION = "chunked-linear-cross-entropy"

# Config attributes that change how HF turns hidden states into logits.
# The chunked lane applies only a plain linear head, so it refuses these.
_UNSUPPORTED_HEAD_CONFIG = (
    "final_logit_softcapping",
    "logit_scale",
    "logits_scaling",
    "output_multiplier_scale",
)


def _unwrap(model):
    """Return the underlying HF causal-LM module (through DDP/FSDP/PEFT)."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
            continue
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
            continue
        break
    return model


def _decoder(causal_lm):
    get_decoder = getattr(causal_lm, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is None:
        decoder = getattr(causal_lm, "model", None)
    if not isinstance(decoder, torch.nn.Module):
        raise RuntimeError(
            "--kernel-backend chunked requires a causal LM exposing its decoder "
            "via get_decoder() or .model"
        )
    return decoder


def require_chunked_ce_support(model) -> dict:
    """Validate that the model's head is a plain linear layer we can chunk.

    Returns a small description for logging. Raises with a clear message on
    unsupported heads (soft-capped/scaled logits, adapted lm_head, models
    without ``logits_to_keep``).
    """
    causal_lm = _unwrap(model)
    config = getattr(causal_lm, "config", None)
    for name in _UNSUPPORTED_HEAD_CONFIG:
        value = getattr(config, name, None)
        if value not in (None, 1, 1.0):
            raise ValueError(
                f"--kernel-backend chunked does not support models with "
                f"config.{name}={value!r}; use --kernel-backend native"
            )
    head = causal_lm.get_output_embeddings()
    if type(head) is not torch.nn.Linear:
        raise ValueError(
            "--kernel-backend chunked requires lm_head to be a plain "
            f"torch.nn.Linear, got {type(head).__name__}; an adapted or "
            "quantized lm_head (e.g. LoRA targeting lm_head) is unsupported, "
            "use --kernel-backend native"
        )
    params = inspect.signature(causal_lm.forward).parameters
    if "logits_to_keep" not in params:
        raise ValueError(
            f"--kernel-backend chunked requires a forward() accepting "
            f"logits_to_keep; {type(causal_lm).__name__} does not"
        )
    _decoder(causal_lm)
    input_emb = causal_lm.get_input_embeddings()
    tied = input_emb is not None and getattr(input_emb, "weight", None) is head.weight
    return {
        "loss_backend": "chunked",
        "model_class": type(causal_lm).__name__,
        "tied_embeddings": bool(tied),
    }


def _chunk_loss(hidden, weight, bias, targets, token_weights):
    logits = F.linear(hidden, weight, bias).float()
    per_token = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    return (per_token * token_weights).sum()


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_tokens: int = DEFAULT_LOSS_CHUNK_TOKENS,
) -> torch.Tensor:
    """Sum of ``token_weights * CE(hidden @ weight^T + bias, targets)``.

    hidden: (N, H); targets: (N,) with -100 ignored; token_weights: (N,) FP32.
    At most ``chunk_tokens`` rows of logits exist at any time, in forward and
    in backward.
    """
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    total = hidden.new_zeros((), dtype=torch.float32)
    needs_grad = torch.is_grad_enabled() and (
        hidden.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    for start in range(0, hidden.shape[0], chunk_tokens):
        stop = start + chunk_tokens
        args = (hidden[start:stop], weight, bias, targets[start:stop], token_weights[start:stop])
        if needs_grad:
            part = checkpoint(_chunk_loss, *args, use_reentrant=False)
        else:
            part = _chunk_loss(*args)
        total = total + part
    return total


def chunked_sft_forward(
    model,
    input_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    chunk_tokens: int = DEFAULT_LOSS_CHUNK_TOKENS,
):
    """Run the model to its final hidden states and return the chunked SFT loss.

    Semantics match ``sft_loss(model(input_ids).logits, input_ids,
    "cross_entropy", weights)``: returns (local token-SUM loss, number of
    shifted positions with positive weight).

    The full model (including any DDP/FSDP/PEFT wrapper) is called so wrapper
    hooks still run; ``logits_to_keep`` is an empty index so the HF head
    produces no logits, and a forward hook on the decoder captures its final
    (normalized) hidden states.
    """
    causal_lm = _unwrap(model)
    decoder = _decoder(causal_lm)
    head = causal_lm.get_output_embeddings()
    captured = []

    def _capture(_module, _inputs, output):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden)

    handle = decoder.register_forward_hook(_capture)
    try:
        model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=torch.empty(0, dtype=torch.long, device=input_ids.device),
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one decoder forward, observed {len(captured)}"
        )
    hidden = captured[0]

    labels = input_ids
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    w = mask.to(torch.float32)
    if weights is not None:
        w = w * weights[:, 1:].to(torch.float32)
    n_tokens = (w > 0).sum()

    flat_w = w.reshape(-1)
    keep = torch.nonzero(flat_w != 0, as_tuple=True)[0]
    shift_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    loss = chunked_linear_cross_entropy(
        shift_hidden.index_select(0, keep),
        head.weight,
        shift_labels.reshape(-1).index_select(0, keep),
        flat_w.index_select(0, keep),
        bias=head.bias,
        chunk_tokens=chunk_tokens,
    )
    if keep.numel() == 0:
        # Keep the graph connected so backward() works on an all-masked batch.
        loss = loss + hidden.sum() * 0.0
    return loss, n_tokens
