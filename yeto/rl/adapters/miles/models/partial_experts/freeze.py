"""Per-expert freezing of a Megatron model's expert parameters.

The model runtime decides *which* parameters are experts and which of those
train (:class:`ExpertParamDecision`); this module applies the decision once per
parameter -- ``requires_grad``, the trainable mark the optimizer groups on and
any model-specific marks -- and refuses a second configuration of the same
parameter. Completeness checks (every local expert seen, every branch present)
stay with the model runtime, which knows the expected geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpertParamDecision:
    """What to do with one expert parameter."""

    trainable: bool
    # Extra attributes set on the parameter (layer, branch, expert id ...).
    marks: Mapping[str, Any] = field(default_factory=dict)


Classifier = Callable[[str, Any], ExpertParamDecision | None]


@dataclass(frozen=True)
class FrozenParam:
    parameter_name: str
    trainable: bool
    decision: ExpertParamDecision


def model_chunks(models: Any) -> tuple[Any, ...]:
    return tuple(models) if isinstance(models, (list, tuple)) else (models,)


def apply_expert_freeze(
    models: Any,
    classify: Classifier,
    *,
    configured_attr: str,
    trainable_attr: str,
) -> list[FrozenParam]:
    """Apply ``classify(name, parameter)`` to every parameter of ``models``.

    ``None`` leaves a parameter untouched (it is not an expert weight).
    Otherwise ``requires_grad`` follows the decision, ``configured_attr`` is set
    (a second call on the same parameter raises) and a trainable parameter gets
    ``trainable_attr = True`` for the optimizer's ``ParamKey(attr=...)`` group.
    Returned in traversal order.
    """

    out: list[FrozenParam] = []
    for chunk in model_chunks(models):
        for name, parameter in chunk.named_parameters():
            decision = classify(name, parameter)
            if decision is None:
                continue
            if getattr(parameter, configured_attr, False):
                raise RuntimeError(f"expert parameter was configured twice: {name}")
            setattr(parameter, configured_attr, True)
            for attr, value in decision.marks.items():
                setattr(parameter, attr, value)
            parameter.requires_grad_(bool(decision.trainable))
            if decision.trainable:
                setattr(parameter, trainable_attr, True)
            out.append(FrozenParam(name, bool(decision.trainable), decision))
    return out


def freeze_all_but_fragment(models: Any, fragment: str, *, trainable_attr: str) -> tuple[int, int]:
    """In-place strategy: freeze every parameter whose name lacks ``fragment``.

    Returns ``(trainable_count, frozen_count)``. Buffers (e.g. the router's
    ``expert_bias``) are not parameters and are unaffected.
    """

    trainable = frozen = 0
    for chunk in model_chunks(models):
        for name, parameter in chunk.named_parameters():
            if fragment in name:
                parameter.requires_grad_(True)
                setattr(parameter, trainable_attr, True)
                trainable += 1
            else:
                parameter.requires_grad_(False)
                frozen += 1
    if not trainable:
        raise RuntimeError(f"no parameter name contains {fragment!r}; nothing would train")
    return trainable, frozen
