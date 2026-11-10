"""Optimizer side of a partial-experts run.

* :func:`wrap_megatron_optimizer` wraps Miles' ``get_megatron_optimizer`` so
  the trainable experts get their own ``ParamKey(attr=...)`` learning-rate
  group and the resulting optimizer is validated;
* :func:`validate_optimizer_masters` requires every trainable parameter's
  FP32 master to exist, be FP32 and belong to the optimizer; with
  ``require_complete`` it must also be unsharded and full-size.

The distributed optimizer is refused unless the model's
:class:`~.config.PartialExpertModelConfig` allows it (DSV4 streams complete
masters and keeps refusing; K3.1 GRPO allows it).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .freeze import model_chunks

Validator = Callable[[Any, Any], None]


def optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


def validate_optimizer_masters(
    model_chunks_: Any,
    optimizer: Any,
    *,
    require_complete: bool,
    label: str,
) -> None:
    """Require an optimizer-owned FP32 master for every trainable parameter.

    ``require_complete``: the master must be unsharded (``main_param_sharded``
    False) and have the parameter's numel -- what streaming the masters
    (DiLoCo / outer sync) needs. Without it a DP-sharded master (Megatron's
    distributed optimizer) passes as long as it is FP32 and optimizer-owned.
    """

    import torch

    optimizer_parameters = set()
    for child in optimizer_children(optimizer):
        inner = getattr(child, "optimizer", None)
        param_groups = getattr(inner, "param_groups", None)
        if param_groups is None:
            param_groups = getattr(child, "param_groups", None)
        if param_groups is None:
            if getattr(child, "is_stub_optimizer", False):
                continue
            raise RuntimeError(f"{label} Adam child has no parameter groups")
        for group in param_groups:
            if not isinstance(group, Mapping) or "params" not in group:
                raise RuntimeError(f"{label} Adam parameter group is invalid")
            optimizer_parameters.update(id(parameter) for parameter in group["params"])

    trainable = []
    seen = set()
    for chunk in model_chunks(model_chunks_):
        for parameter in chunk.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                trainable.append(parameter)
                seen.add(id(parameter))
    if not trainable:
        raise RuntimeError(f"{label} optimizer has no trainable parameters")
    for parameter in trainable:
        main = getattr(parameter, "main_param", None)
        sharded = bool(getattr(parameter, "main_param_sharded", False))
        owned = main is not None and main.dtype == torch.float32 and id(main) in optimizer_parameters
        complete = main is not None and not sharded and main.numel() == parameter.numel()
        if not owned or (require_complete and not complete):
            kind = "complete, optimizer-owned" if require_complete else "optimizer-owned"
            raise RuntimeError(f"{label} trainable parameter lacks a {kind} FP32 master")


def wrap_megatron_optimizer(
    original,
    *,
    label: str,
    allow_distributed_optimizer: bool,
    param_attr: str | None,
    lr: Callable[[], float | None] | None,
    validate: Validator,
):
    """``get_megatron_optimizer`` with an expert LR group and master validation.

    Miles calls Megatron's factory with keyword arguments from its model
    module. ``lr()`` is read at call time (environment-driven); ``None`` adds
    no override group. ``validate(model_chunks, optimizer)`` runs on the
    result (late-bound so tests can replace it).
    """

    def get_megatron_optimizer(
        *,
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        **kwargs,
    ):
        from megatron.core.optimizer import get_standard_config_overrides
        from megatron.core.optimizer.optimizer_config import ParamKey

        if config.use_distributed_optimizer and not allow_distributed_optimizer:
            raise RuntimeError(
                f"{label} requires replicated FP32 optimizer masters; the distributed "
                "optimizer is refused for this model (allow_distributed_optimizer=False)"
            )
        overrides = (
            get_standard_config_overrides(config)
            if config_overrides is None
            else dict(config_overrides)
        )
        expert_lr = lr() if lr is not None else None
        if expert_lr is not None:
            if param_attr is None:
                raise RuntimeError(f"{label} expert LR override needs a trainable parameter attribute")
            overrides[ParamKey(attr=param_attr)] = {"max_lr": expert_lr, "min_lr": expert_lr}
        optimizer = original(
            config=config,
            model_chunks=model_chunks,
            config_overrides=overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            **kwargs,
        )
        validate(model_chunks, optimizer)
        return optimizer

    return get_megatron_optimizer
