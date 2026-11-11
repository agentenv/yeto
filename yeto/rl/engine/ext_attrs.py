"""yeto's extension context on Miles' configuration (``yeto_ext``).

Miles processes (rollout executors, custom hooks, trainer ranks) read yeto's
static values through one carrier: the opaque ``yeto_ext`` mapping that the
Miles fork's ``YetoExtensionConfig`` trait puts on every leaf config and hook
view. The driver installs it once, before any component config is sliced
(:func:`install_ext`); readers use :func:`ext_attr`, which also understands the
legacy carrier (one ``yeto_<key>`` attribute per value on a mutable namespace),
so the same code runs on the pinned Miles and on test fakes.

Keys are the attribute names without the ``yeto_`` prefix (``rl_learner_id``
is ``args.yeto_rl_learner_id`` in the legacy spelling). Only the keys in
:data:`WIRE_KEYS` cross a process boundary; driver-only values stay in
``IslandArgs`` and never reach a Miles process. This module is engine-neutral (no Miles, no Ray);
``yeto.rl.adapters.miles.ext_args`` re-exports it for the Miles-side code.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

EXT_FIELD = "yeto_ext"
ATTR_PREFIX = "yeto_"

_MISSING = object()

# Values yeto code reads inside a Miles process (B0 class X), unprefixed.
WIRE_KEYS: frozenset[str] = frozenset({
    # rollout executor / hooks
    "algo_plugins",
    "rl_codex_reasoning_in_loss",
    "rl_codex_reasoning_markers",
    "rl_dynamic_sampling_max_replacements",
    "rl_elastic_metadata",
    "rl_event_tape",
    "rl_expected_algorithm_sha256",
    # ports-v1 fork flags (external_policy_*), carried here on a Miles whose
    # config does not declare them (island_args.override_fork_field)
    "rl_external_policy_identity_setter_path",
    "rl_external_policy_sync_run_until_stop",
    "rl_learner_id",
    "rl_max_policy_age",
    "rl_mismatch_tape_max",
    "rl_mismatch_text_max",
    "rl_observe_timeline",
    "rl_policy_version",
    "rl_reward_sha256",
    "rl_rollout_ahead",
    "rl_seq_adv",
    "rl_trajectory_evidence_dir",
    "rl_trajectory_evidence_kind",
    "rl_trajectory_evidence_schema_version",
    "rl_trajectory_tape_max",
    # trainer ranks
    "rl_grad_audit_dir",
    "rl_native_lora_export",
    "rl_profile_trainer",
    "rl_trust_remote_code",
})


def ext_key(name: str) -> str:
    """The ``yeto_ext`` key of an attribute name (``yeto_rl_x`` -> ``rl_x``)."""
    return name[len(ATTR_PREFIX):] if name.startswith(ATTR_PREFIX) else name


def attr_name(key: str) -> str:
    """The legacy attribute name of a key (``rl_x`` -> ``yeto_rl_x``)."""
    return key if key.startswith(ATTR_PREFIX) else ATTR_PREFIX + key


def iter_ext_keys() -> Iterator[str]:
    """The keys that may cross into Miles processes, sorted."""
    return iter(sorted(WIRE_KEYS))


def _ext_mapping(args: Any) -> Mapping[str, Any] | None:
    ext = getattr(args, EXT_FIELD, None)
    return ext if isinstance(ext, Mapping) else None


def ext_attr(args: Any, name: str, default: Any = _MISSING) -> Any:
    """Read one yeto value off any args-like object.

    Order: ``args.yeto_ext[key]`` (new carrier), then ``args.yeto_<key>``
    (legacy carrier, test fakes), then ``default``; with no default a missing
    value raises :class:`AttributeError` like ``getattr`` would. ``name`` may
    be spelled with or without the ``yeto_`` prefix.
    """
    key = ext_key(name)
    ext = _ext_mapping(args)
    if ext is not None and key in ext:
        return ext[key]
    value = getattr(args, attr_name(key), _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise AttributeError(attr_name(key))
        return default
    return value


def ext_has(args: Any, name: str) -> bool:
    """``hasattr`` for :func:`ext_attr` (either carrier)."""
    key = ext_key(name)
    ext = _ext_mapping(args)
    if ext is not None and key in ext:
        return True
    return hasattr(args, attr_name(key))


def build_ext(values: Mapping[str, Any]) -> dict[str, Any]:
    """The ``yeto_ext`` payload for ``values`` (keys with or without prefix).

    Only :data:`WIRE_KEYS` are kept; the result must be JSON-serializable
    because the fork's RPC and payload transports carry it as JSON (a
    driver-only object such as a bridge config fails closed here).
    """
    ext = {ext_key(k): v for k, v in values.items() if ext_key(k) in WIRE_KEYS}
    try:
        json.dumps(ext)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{EXT_FIELD} values must be JSON-serializable: {error}") from error
    return dict(sorted(ext.items()))


def _is_pydantic(obj: Any) -> bool:
    return hasattr(type(obj), "model_fields") and hasattr(obj, "model_copy")


def install_ext(all_config: Any, mapping: Mapping[str, Any]) -> Any:
    """Install ``mapping`` (merged over any existing ``yeto_ext``) on a Miles config.

    Returns the config to hand to Miles: a pydantic leaf is immutable, so a
    copy with the field replaced comes back (``model_copy(update=...)``); a
    mutable namespace (pinned Miles, test fakes) is updated in place and
    returned. Call it once, before Miles slices the component configs.
    """
    merged = dict(_ext_mapping(all_config) or {})
    merged.update(build_ext(mapping))
    if _is_pydantic(all_config):
        if EXT_FIELD not in type(all_config).model_fields:
            raise TypeError(
                f"{type(all_config).__name__} has no {EXT_FIELD} field: this Miles build lacks the "
                "YetoExtensionConfig trait, yeto's values would never reach its processes"
            )
        return all_config.model_copy(update={EXT_FIELD: merged})
    setattr(all_config, EXT_FIELD, merged)
    return all_config
