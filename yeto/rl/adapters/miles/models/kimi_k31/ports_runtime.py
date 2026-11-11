"""Kimi-K3.1 trainer-side patch on ports Miles.

Everything structural (freezing all but ``.mlp.experts.trainable.``, the slot
permutation, export filtering, routing replay) is in the Miles fork behind
``--partial-experts-selection`` (F2). Yeto adds one thing: after Miles builds
the optimizer it checks that every trainable parameter has an FP32,
optimizer-owned master -- allowing the distributed optimizer, since GRPO
streams no masters -- and, when ``YETO_K31_PARTIAL_EXPERT_LR`` is set, gives
the experts their own learning-rate group (for mixed expert + other-parameter
runs; with experts alone the global ``--lr`` is the expert LR).
"""

from __future__ import annotations

import os
from types import ModuleType

from yeto.rl.adapters.miles.models.kimi_k31.model import MODEL_CONFIG
from yeto.rl.adapters.miles.models.partial_experts import optimizer as _optimizer
from yeto.rl.adapters.miles.models.partial_experts.hooks import install_once, install_or_defer

ENV_FLAG = "YETO_K31_PARTIAL_EXPERTS_PORTS"
EXPERT_LR_ENV = "YETO_K31_PARTIAL_EXPERT_LR"
LABEL = "kimi-k31 partial-experts"

_MODEL = "miles.backends.megatron_utils.model"


def expert_lr() -> float | None:
    raw = os.environ.get(EXPERT_LR_ENV)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{EXPERT_LR_ENV} must be a float") from exc
    if value <= 0:
        raise RuntimeError(f"{EXPERT_LR_ENV} must be positive")
    return value


def validate_optimizer_masters(model_chunks, optimizer) -> None:
    _optimizer.validate_optimizer_masters(
        model_chunks, optimizer,
        require_complete=MODEL_CONFIG.requires_complete_master,
        label=LABEL,
    )


def wrap_optimizer(original):
    return _optimizer.wrap_megatron_optimizer(
        original,
        label=LABEL,
        allow_distributed_optimizer=MODEL_CONFIG.allow_distributed_optimizer,
        param_attr=MODEL_CONFIG.trainable_param_attr,
        lr=expert_lr,
        validate=lambda chunks, optimizer: validate_optimizer_masters(chunks, optimizer),
    )


def _patch_model(module: ModuleType) -> None:
    module.get_megatron_optimizer = wrap_optimizer(module.get_megatron_optimizer)


def install_on_model(module: ModuleType) -> None:
    install_once(module, _patch_model, namespace="yeto_k31_ports")


def install() -> None:
    """Patch the imported ports Miles model module or arm its import hook."""

    install_or_defer(_MODEL, install_on_model)
