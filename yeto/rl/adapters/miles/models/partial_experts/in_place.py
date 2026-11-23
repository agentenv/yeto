"""Generic partial-expert training on Miles for any MoE model (``in_place`` selections).

The trainer keeps Megatron's expert layout: EP rank ``r`` holds global experts
``[r*n/ep, (r+1)*n/ep)`` as individual weights (TE grouped GEMM with
``NVTE_GROUPED_LINEAR_SINGLE_PARAM=0``: ``mlp.experts.linear_fc1.weight<local>``;
``SequentialMLP``: ``mlp.experts.local_experts.<local>.linear_fc1.weight``).
Right after Miles builds each model chunk and before DDP wraps it, every
parameter is frozen except the selected experts' weights, so DDP's grad
buffers, the optimizer and Miles' full-parameter transport only ever see the
selected experts: the frozen experts and every dense parameter never reach a
syncer fragment. The router is frozen too, and its expert bias stays fixed
(``--moe-router-bias-update-rate 0``), so every island routes alike.

What is reused instead of rebuilt:

* the selection document and its validation (:mod:`.selection`, layout
  ``in_place``) and the MoE geometry of the checkpoint
  (:mod:`yeto.rl.moe_geometry`, cross-checked against the Megatron-Bridge
  provider the island builds);
* per-parameter freezing with double-configuration detection
  (:func:`.freeze.apply_expert_freeze`) and the import hook that patches a
  Miles module in every Ray worker (:func:`.hooks.install_or_defer`, armed
  from ``sitecustomize`` by :data:`ENV_SELECTION`);
* Miles' own ``full_parameter_state`` export of ``requires_grad`` masters
  under global expert names (``named_params_and_buffers``), and the layout
  check :func:`yeto.rl.moe_layout.check_expert_partition` with the selection
  as its :class:`~yeto.rl.moe_layout.ExpertSubset`.

Import-light: torch and megatron are imported inside the runtime functions.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

from yeto.rl.moe_layout import ExpertSubset

from .freeze import ExpertParamDecision, apply_expert_freeze, model_chunks
from .hooks import install_once, install_or_defer
from .selection import LAYOUT_IN_PLACE, ExpertSelection

# Ray workers get both through the job's runtime_env (worker_env): sitecustomize arms the
# model-provider hook when the first is set.
ENV_SELECTION = "YETO_PARTIAL_EXPERTS_SELECTION"
GROUPED_SINGLE_PARAM_ENV = "NVTE_GROUPED_LINEAR_SINGLE_PARAM"
# The Miles flag that keeps the router's expert bias fixed (a frozen router).
BIAS_UPDATE_ARGV = ("--moe-router-bias-update-rate", "0")
TRAINABLE_ATTR = "_yeto_partial_expert"
CONFIGURED_ATTR = "_yeto_partial_expert_configured"
LABEL = "partial-experts (in place)"

_GROUPED = re.compile(r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\."
                      r"(?P<branch>linear_fc[12])\.weight(?P<local>\d+)$")
_SEQUENTIAL = re.compile(r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\.local_experts\."
                         r"(?P<local>\d+)\.(?P<branch>linear_fc[12])\.weight$")
# A grouped expert linear holding every local expert in one parameter: no per-expert freezing.
_GROUPED_SINGLE = re.compile(r"(?:^|\.)decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12]\.weight$")

_PROVIDER_MODULE = "miles.backends.megatron_utils.model_provider"
_MODEL_MODULE = "miles.backends.megatron_utils.model"


class PartialExpertsError(ValueError):
    """The selection does not fit the model or the run."""


def load_in_place(path: str | os.PathLike) -> ExpertSelection:
    selection = ExpertSelection.load_file(path)
    if selection.layout != LAYOUT_IN_PLACE:
        raise PartialExpertsError(
            f"{path}: layout {selection.layout!r}; generic partial experts need an in_place selection "
            "(python -m yeto.rl.adapters.miles.models.partial_experts.selection --layout in_place ...)")
    return selection


def expert_subset(selection: ExpertSelection) -> ExpertSubset:
    """The selection as the layout check's subset (the same object on launcher, island and probe)."""

    return ExpertSubset.of(selection.num_experts, selection.trainable)


def check_against_geometry(selection: ExpertSelection, geometry: Any, *, ep_size: int | None) -> None:
    """``geometry``: a :class:`yeto.rl.moe_geometry.RoutedExpertGeometry` (HF config or Bridge provider)."""

    if geometry is None:
        raise PartialExpertsError("partial experts need a mixture-of-experts model (the config has no routed experts)")
    if selection.num_experts != geometry.num_experts:
        raise PartialExpertsError(
            f"the selection covers {selection.num_experts} experts per layer; the model has {geometry.num_experts}")
    try:
        selection.check_layers(geometry.moe_layers)
    except ValueError as exc:
        raise PartialExpertsError(f"the selection does not name the model's MoE layers: {exc}") from None
    if ep_size and int(ep_size) != selection.ep_size:
        raise PartialExpertsError(
            f"the selection was built for EP {selection.ep_size}, the run uses --expert-parallel {ep_size}; "
            "regenerate it (--input FILE --ep-size N keeps the set when it stays balanced)")


@lru_cache(maxsize=4)
def _cached_selection(path: str) -> ExpertSelection:
    return load_in_place(path)


def selection_from_env() -> ExpertSelection | None:
    path = os.environ.get(ENV_SELECTION)
    return None if not path else _cached_selection(path)


def subset_from_env() -> ExpertSubset | None:
    """The subset of the run this process belongs to (the layout probe reads it), or None."""

    selection = selection_from_env()
    return None if selection is None else expert_subset(selection)


# -- trainer-side freezing ----------------------------------------------------


@dataclass(frozen=True)
class RankCoordinates:
    layer_offset: int  # pipeline / virtual-pipeline offset of this chunk's first layer
    ep_rank: int
    ep_size: int


def classify(
    name: str,
    *,
    selection: ExpertSelection,
    coords: RankCoordinates,
) -> ExpertParamDecision | None:
    """The decision for one parameter of a model chunk: None for a non-expert parameter."""

    match = _GROUPED.search(name) or _SEQUENTIAL.search(name)
    if match is None:
        if _GROUPED_SINGLE.search(name):
            raise PartialExpertsError(
                f"{name} holds every local expert in one parameter; set {GROUPED_SINGLE_PARAM_ENV}=0 so each "
                "expert is its own parameter")
        return None
    n_local = selection.num_experts // coords.ep_size
    local = int(match.group("local"))
    if not 0 <= local < n_local:
        raise PartialExpertsError(f"{name}: local expert {local} outside the {n_local} of this EP rank")
    layer = int(match.group("layer")) + coords.layer_offset
    expert = coords.ep_rank * n_local + local
    trainable = selection.trainable.get(layer)
    if trainable is None:
        raise PartialExpertsError(f"MoE layer {layer} is not in the selection")
    marks = {"_yeto_expert_layer": layer, "_yeto_expert_id": expert, "_yeto_expert_branch": match.group("branch")}
    return ExpertParamDecision(expert in trainable, marks)


def freeze_chunk(model: Any, *, selection: ExpertSelection, coords: RankCoordinates) -> tuple[int, int]:
    """Freeze every parameter of one model chunk but the selected experts; ``(trainable, frozen)``.

    Every selected expert this rank holds in the chunk's MoE layers must have
    both projections; a chunk without MoE layers trains nothing.
    """

    if selection.ep_size != coords.ep_size:
        raise PartialExpertsError(f"selection EP {selection.ep_size} != trainer EP {coords.ep_size}")
    records = apply_expert_freeze(
        model,
        lambda name, _parameter: classify(name, selection=selection, coords=coords),
        configured_attr=CONFIGURED_ATTR,
        trainable_attr=TRAINABLE_ATTR,
    )
    configured = {record.parameter_name for record in records}
    frozen = 0
    for chunk in model_chunks(model):
        for name, parameter in chunk.named_parameters():
            if name not in configured:
                parameter.requires_grad_(False)
                frozen += 1
    seen: dict[tuple[int, int], set[str]] = {}
    for record in records:
        marks = record.decision.marks
        if record.trainable:
            seen.setdefault((marks["_yeto_expert_layer"], marks["_yeto_expert_id"]), set()).add(
                marks["_yeto_expert_branch"])
        else:
            frozen += 1
    n_local = selection.num_experts // coords.ep_size
    local_ids = range(coords.ep_rank * n_local, (coords.ep_rank + 1) * n_local)
    for layer in sorted({record.decision.marks["_yeto_expert_layer"] for record in records}):
        for expert in (e for e in selection.trainable[layer] if e in local_ids):
            branches = seen.get((layer, expert), set())
            if branches != {"linear_fc1", "linear_fc2"}:
                raise PartialExpertsError(
                    f"selected expert {expert} of layer {layer} has projections {sorted(branches)} on EP rank "
                    f"{coords.ep_rank}")
    return sum(1 for record in records if record.trainable), frozen


def _rank_coordinates(model: Any, vp_stage: int | None) -> RankCoordinates:
    import inspect

    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if "vp_stage" in inspect.signature(get_transformer_layer_offset).parameters:
        offset = get_transformer_layer_offset(model.config, vp_stage)
    else:
        offset = get_transformer_layer_offset(model.config)
    return RankCoordinates(
        layer_offset=int(offset),
        ep_rank=int(parallel_state.get_expert_model_parallel_rank()),
        ep_size=int(parallel_state.get_expert_model_parallel_world_size()),
    )


def _check_router_bias(model: Any) -> None:
    config = getattr(model, "config", None)
    if getattr(config, "moe_router_enable_expert_bias", False) and float(
        getattr(config, "moe_router_bias_update_rate", 0.0) or 0.0
    ):
        raise PartialExpertsError(
            "the router's expert bias would keep updating on each island; partial experts freeze the router "
            "(--moe-router-bias-update-rate 0)")


def wrap_model_provider(provider: Callable[..., Any], selection: ExpertSelection) -> Callable[..., Any]:
    """A Miles model provider that freezes each chunk it builds (before DDP wraps it)."""

    def model_provider(*args: Any, **kwargs: Any) -> Any:
        model = provider(*args, **kwargs)
        _check_router_bias(model)
        coords = _rank_coordinates(model, kwargs.get("vp_stage"))
        trainable, frozen = freeze_chunk(model, selection=selection, coords=coords)
        print(f"[yeto] {LABEL}: {trainable} trainable expert weights, {frozen} frozen parameters on EP rank "
              f"{coords.ep_rank} (layer offset {coords.layer_offset})", flush=True)
        return model

    model_provider._yeto_partial_experts = True
    return model_provider


def wrap_provider_factory(factory: Callable[..., Any], selection: ExpertSelection) -> Callable[..., Any]:
    """``get_model_provider_func(args, role)`` returning frozen-chunk providers for the actor."""

    def get_model_provider_func(*args: Any, **kwargs: Any) -> Any:
        provider = factory(*args, **kwargs)
        role = kwargs.get("role", args[1] if len(args) > 1 else "actor")
        return provider if role != "actor" else wrap_model_provider(provider, selection)

    return get_model_provider_func


def _patch(module: ModuleType) -> None:
    selection = selection_from_env()
    if selection is None:
        return
    module.get_model_provider_func = wrap_provider_factory(module.get_model_provider_func, selection)
    # ``model`` binds the name at import; patch that binding too when it already exists.
    import sys

    model = sys.modules.get(_MODEL_MODULE)
    if model is not None and hasattr(model, "get_model_provider_func"):
        model.get_model_provider_func = module.get_model_provider_func


def install_on_provider_module(module: ModuleType) -> None:
    install_once(module, _patch, namespace="yeto_partial_experts")


def install() -> None:
    """Patch Miles' model-provider module now, or as soon as it is imported (every Ray worker)."""

    install_or_defer(_PROVIDER_MODULE, install_on_provider_module)


def island_env(selection_path: str) -> Mapping[str, str]:
    """The switches the island driver sets for a selection (see :func:`worker_env`)."""

    return {ENV_SELECTION: str(selection_path), GROUPED_SINGLE_PARAM_ENV: "0"}


def worker_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The switches every Ray worker needs, read back from the driver's environment.

    The island configures partial experts after ``ray start``, and Ray workers
    inherit the raylet's environment, not the driver's: the island forwards
    these through the job's ``runtime_env``
    (:func:`yeto.rl.adapters.miles.entry.connect_island_ray`) so each worker's
    ``sitecustomize`` arms the model-provider hook. Empty without a selection.
    """

    path = environ.get(ENV_SELECTION)
    return {} if not path else dict(island_env(path))
