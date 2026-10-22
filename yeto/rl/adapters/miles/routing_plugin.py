"""Count the tokens each MoE router sends to each expert, inside every Megatron rank.

Loaded through ``TrainGroup.run_plugin`` like :mod:`.cut_plugin`. Miles and
Megatron keep no per-expert token count that survives a training step:
``TopKRouter.local_tokens_per_expert`` exists only with the expert-bias load
balancer and is reset when the bias is updated. This plugin therefore adds a
forward hook to every ``TopKRouter`` and accumulates its routing map.

Counting follows Megatron's own rule for that buffer
(``TopKRouter._apply_expert_bias``): only forwards with autograd enabled
count, so log-probability passes (``no_grad``) and the no-grad first pass of
activation recomputation do not, while the recomputed forward of a training
step counts once. Padding tokens are dropped when the router gets a padding
mask.

The counts stay on the router's device between drains (no host sync per
step). :func:`drain_routed_token_counts` returns this rank's counts since the
previous drain, keyed by the 0-based global layer index that Miles global
parameter names use (``decoder.layers.L``; Megatron's router ``layer_number``
is that index plus one).

GPU-only (not run on CPU here): the hook on a real Megatron ``TopKRouter``,
the padding-mask shape under packed sequences, and the layer index under
pipeline parallelism. ``tests/test_rl_moe_expert_token_weights.py`` runs the
same code on a stand-in router module.
"""

from __future__ import annotations

from typing import Any

_MODULE = "yeto.rl.adapters.miles.routing_plugin"
INSTALL_ROUTED_TOKEN_COUNTER = f"{_MODULE}.install_routed_token_counter"
DRAIN_ROUTED_TOKEN_COUNTS = f"{_MODULE}.drain_routed_token_counts"

_COUNTS = "_yeto_routed_tokens"


def _router_class() -> type | None:
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return None
    return TopKRouter


def _chunks(model: Any) -> list[Any]:
    return list(model) if isinstance(model, (list, tuple)) else [model]


def routers(model: Any, router_class: type | None = None) -> list[Any]:
    """Every top-k router module of the model chunks, in module order."""

    router_class = router_class or _router_class()
    if router_class is None:
        return []
    return [
        module
        for chunk in _chunks(model)
        for module in chunk.modules()
        if isinstance(module, router_class)
    ]


def _count_routed_tokens(module: Any, args: tuple, kwargs: dict, output: Any) -> None:
    import torch

    if not torch.is_grad_enabled():
        return
    _probs, routing_map = output
    padding_mask = args[1] if len(args) > 1 else kwargs.get("padding_mask")
    with torch.no_grad():
        if padding_mask is not None:
            keep = ~padding_mask.reshape(-1)
            if keep.shape[0] != routing_map.shape[0]:
                raise RuntimeError(
                    f"router padding mask has {keep.shape[0]} tokens, routing map {routing_map.shape[0]}"
                )
            routing_map = routing_map & keep.unsqueeze(-1)
        counts = getattr(module, _COUNTS)
        if counts is None:
            counts = torch.zeros(routing_map.shape[-1], dtype=torch.int64, device=routing_map.device)
            setattr(module, _COUNTS, counts)
        counts += routing_map.sum(dim=0, dtype=torch.int64)


def install_hooks(model: Any, router_class: type | None = None) -> int:
    """Hook every router once (idempotent); returns the number of routers."""

    found = routers(model, router_class)
    for module in found:
        if not hasattr(module, _COUNTS):
            setattr(module, _COUNTS, None)
            module.register_forward_hook(_count_routed_tokens, with_kwargs=True)
    return len(found)


def drain(model: Any, router_class: type | None = None) -> dict[str, list[int]]:
    """This rank's counts since the previous drain, then reset them.

    Keys are 0-based global layer indices as strings (the result travels as
    JSON under the rpc worker backend). A hooked router that has not run yet
    reports nothing.
    """

    out: dict[str, list[int]] = {}
    for module in routers(model, router_class):
        counts = getattr(module, _COUNTS, None)
        if counts is None:
            continue
        layer = int(module.layer_number) - 1
        if str(layer) in out:
            raise RuntimeError(f"two routers claim MoE layer {layer}")
        out[str(layer)] = [int(value) for value in counts.tolist()]
        counts.zero_()
    return out


def install_routed_token_counter(actor: Any) -> int:
    return install_hooks(actor.model)


def drain_routed_token_counts(actor: Any) -> dict[str, list[int]]:
    return drain(actor.model)
