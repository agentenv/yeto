"""Per-rank full-parameter state and the published weight version of a Miles train actor.

Miles (k31/b2 and later) keeps a rank's full-parameter bookkeeping in one
dataclass, ``actor.full_parameter`` (``FullParameterRankState``: the installed
transport ``policy_version``, the ``apply_stats`` of the last export / prepare /
commit, ...), reached through
``miles.backends.megatron_utils.full_parameter_state.full_parameter_rank_state(actor)``;
the published weight version lives only in the model companion
(``model_weight_version`` / ``set_model_weight_version`` of the same module; the
weight updater has no counter any more). The ports-v1 fork kept the same values
as private attributes (``actor._external_full_policy_version``,
``actor._external_full_parameter_apply_stats``) and a ``weight_updater.weight_version``
counter. This module is the one place that knows both spellings; the cut, state
and rank-stats plugins read and write through it.

Dependencies on the Miles side (k31/b2), by exact signature:

* ``full_parameter_state.full_parameter_rank_state(actor) -> FullParameterRankState``
  with ``policy_version: int | None`` and ``apply_stats: dict | None`` fields;
* ``full_parameter_state.model_weight_version(actor) -> int | None``;
* ``full_parameter_state.set_model_weight_version(actor, version: int) -> int``
  (monotone: refuses a version below the installed one);
* ``TrainerController.apply_full_parameter_chunked_cut(cut, *, ..., weight_version: int | None = None)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# ports-v1 spellings (the fork before k31/b2)
LEGACY_POLICY_VERSION_ATTRIBUTE = "_external_full_policy_version"
LEGACY_APPLY_STATS_ATTRIBUTE = "_external_full_parameter_apply_stats"

_MISSING = object()


def _state_module() -> Any | None:
    try:
        from miles.backends.megatron_utils import full_parameter_state
    except ImportError:
        return None
    return full_parameter_state


def rank_state(actor: Any) -> Any | None:
    """The actor's ``FullParameterRankState`` (k31/b2 Miles), or the dataclass-like
    object already on ``actor.full_parameter``; None when this Miles keeps the
    legacy attributes instead (or the actor has no full-parameter tracking)."""
    module = _state_module()
    reader = getattr(module, "full_parameter_rank_state", None) if module is not None else None
    if callable(reader):
        return reader(actor)
    state = getattr(actor, "full_parameter", None)
    return state if state is not None and hasattr(state, "policy_version") else None


def rank_policy_version(actor: Any) -> int | None:
    """The transport's installed policy version on this rank; None without
    full-parameter tracking (a LoRA trainer)."""
    state = rank_state(actor)
    if state is not None:
        version = getattr(state, "policy_version", None)
        if version is not None:
            return int(version)
    version = getattr(actor, LEGACY_POLICY_VERSION_ATTRIBUTE, None)
    return None if version is None else int(version)


def rank_apply_stats(actor: Any) -> dict[str, dict[str, Any]]:
    """This rank's last export / materialize / prepare / commit stats by phase ({} when none)."""
    state = rank_state(actor)
    stats = getattr(state, "apply_stats", None) if state is not None else None
    if not isinstance(stats, Mapping):
        stats = getattr(actor, LEGACY_APPLY_STATS_ATTRIBUTE, None)
    if not isinstance(stats, Mapping):
        return {}
    return {phase: dict(values) for phase, values in stats.items() if isinstance(values, Mapping)}


def _companions(actor: Any) -> list[Any]:
    """The model companions of the actor's model chunks; [] without a model or
    on a process without Miles (CPU fakes: the legacy counter is the carrier)."""
    model = getattr(actor, "model", None)
    if model is None:
        return []
    try:
        from yeto.rl.adapters.miles.state_plugin import model_companions

        return list(model_companions(model))
    except ImportError:
        return []


def model_weight_version(actor: Any) -> int | None:
    """The published weight version: the model companion's ``weight_version``
    (k31/b2 helper when present, else read off the companions), else the ports-v1
    ``weight_updater.weight_version`` counter; None when the actor carries neither."""
    module = _state_module()
    reader = getattr(module, "model_weight_version", None) if module is not None else None
    if callable(reader):
        try:
            version = reader(actor)
        except Exception:  # noqa: BLE001 - an actor without a model falls back below
            version = None
        if version is not None:
            return int(version)
    companions = _companions(actor)
    if companions:
        versions = {int(companion.weight_version.item()) for companion in companions}
        if len(versions) != 1:
            raise RuntimeError(f"model companion weight versions diverged: {sorted(versions)}")
        return versions.pop()
    version = getattr(getattr(actor, "weight_updater", None), "weight_version", None)
    return None if version is None else int(version)


def can_set_model_weight_version(actor: Any) -> bool:
    """Whether :func:`set_model_weight_version` has a carrier on this actor."""
    if _companions(actor):
        return True
    return getattr(actor, "weight_updater", None) is not None


def set_model_weight_version(actor: Any, version: int) -> int:
    """Stamp the published weight version on this rank and return it.

    k31/b2 Miles: ``full_parameter_state.set_model_weight_version(actor, version)``
    (every companion and its CPU backup). Older Miles: the companions are filled
    (and the ``actor`` weights backup refreshed) by hand; a Miles whose updater
    still counts versions gets its counter set too.
    """
    version = int(version)
    if version < 0:
        raise ValueError(f"weight version must be >= 0 (got {version})")
    module = _state_module()
    setter = getattr(module, "set_model_weight_version", None) if module is not None else None
    if callable(setter) and _companions(actor):
        return int(setter(actor, version))
    companions = _companions(actor)
    if companions:
        for companion in companions:
            companion.weight_version.fill_(version)
        backuper = getattr(actor, "weights_backuper", None)
        if backuper is not None and "actor" in getattr(backuper, "backup_tags", ()):
            from yeto.rl.adapters.miles.state_plugin import is_companion_parameter

            for name, tensor in backuper.get("actor").items():
                if is_companion_parameter(name) and name.split(".")[-1] == "weight_version":
                    tensor.fill_(version)
    updater = getattr(actor, "weight_updater", None)
    if updater is not None and hasattr(updater, "weight_version"):
        updater.weight_version = version
    elif not companions:
        raise RuntimeError("the trainer model has no companion to carry the weight version")
    return version


__all__ = [
    "LEGACY_APPLY_STATS_ATTRIBUTE",
    "LEGACY_POLICY_VERSION_ATTRIBUTE",
    "can_set_model_weight_version",
    "model_weight_version",
    "rank_apply_stats",
    "rank_policy_version",
    "rank_state",
    "set_model_weight_version",
]
