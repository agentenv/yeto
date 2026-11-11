"""Miles before-train-step hook: make the first pipeline stage's frozen embedding output require grad.

``--partial-experts-selection`` freezes everything but the trainable experts, the embedding included. Under
full recompute (Megatron ``tensor_parallel.checkpoint``, reentrant) the first stage's checkpoints then see
no input requiring grad, so that stage's trainable experts get no gradient; later stages receive a
requires-grad p2p tensor. Miles k31/stage0-grad installs the same hook at model build; this module carries
it to images that predate that fix.

Use: ``--custom-megatron-before-train-step-hook-path
yeto.rl.adapters.miles.models.kimi_k31.first_stage_grad.before_train_step``. Another before-train-step hook
keeps running when named in ``YETO_K31_BEFORE_TRAIN_STEP_CHAIN`` (``module.path.fn``, Miles'
``load_function`` form); it is called after this one with the same arguments.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

CHAIN_ENV = "YETO_K31_BEFORE_TRAIN_STEP_CHAIN"
_MARKER = "_yeto_first_stage_grad_hook"


def _unwrap(module: Any) -> Any:
    """DDP / Float16Module wrappers -> the module that owns ``embedding`` (or the innermost one)."""
    while getattr(module, "embedding", None) is None and getattr(module, "module", None) is not None:
        module = module.module
    return module


def _hook(module, _inputs, output):
    return output.requires_grad_(True) if module.training and not output.requires_grad else output


def install(model: Any) -> int:
    """Register the hook on every model chunk's embedding (first stage only has one); idempotent.

    Returns how many embeddings carry the hook. Skips an embedding Miles already hooked the same way.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    hooked = 0
    for chunk in chunks:
        embedding = getattr(_unwrap(chunk), "embedding", None)
        if embedding is None:
            continue
        if not getattr(embedding, _MARKER, False):
            embedding.register_forward_hook(_hook)
            setattr(embedding, _MARKER, True)
        hooked += 1
    return hooked


def _load(path: str):
    module, _, name = path.rpartition(".")
    if not module:
        raise ValueError(f"{CHAIN_ENV}={path!r}: expected module.path.function")
    return getattr(importlib.import_module(module), name)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
    install(model)
    from . import nan_inject  # TEST ONLY, off unless YETO_K31_TEST_NAN_INJECT_ROLLOUT is set

    nan_inject.maybe_arm(rollout_id, model)
    chained = os.environ.get(CHAIN_ENV, "").strip()
    if chained:
        _load(chained)(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)
