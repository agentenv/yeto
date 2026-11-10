"""Process-local runtime state keyed off an args object (engine-neutral).

Miles' configuration objects are immutable, so the per-process scratch that
yeto's hooks used to hang on ``args`` (``args._yeto_*``) lives here instead,
scoped like those attributes were: to this process and to one args scope.
The scope is, in order:

* the run (``args.run_uuid``) when the engine stamped one -- every hook view
  of one run then shares the state, as the attributes on one namespace did;
* a mutable namespace itself (``argparse.Namespace``, ``SimpleNamespace``,
  fakes): the state dict is kept in its ``__dict__``, so it lives and dies
  with that object and independent namespaces never see each other;
* otherwise (frozen object without a run id) the object's identity.

:func:`take_runtime_stats` reads the Miles rollout statistics channel
(``miles.rollout.rollout_runtime_stats.take(kind)``, kinds ``abort_discard``,
``over_sampling``, ``suspend``, ``resume``) and falls back to the legacy
``args.rollout_<kind>_stats`` attribute only where the channel does not exist
(the ports-v1 fork). The backend adapter registers the channel
(:func:`register_runtime_stats_source`); this module imports no framework.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable

STORE_ATTR = "_yeto_proc_state"  # the state dict on a mutable namespace

_BY_RUN: dict[str, dict[str, Any]] = {}
_BY_ID: dict[int, dict[str, Any]] = {}
_REFS: dict[int, Any] = {}  # keeps the weakrefs alive so their callbacks fire


def _is_pydantic(obj: Any) -> bool:
    return hasattr(type(obj), "model_fields")


def _forget(ident: int) -> Callable[[Any], None]:
    def callback(_ref: Any) -> None:
        _REFS.pop(ident, None)
        _BY_ID.pop(ident, None)
    return callback


def _store(args: Any, create: bool) -> dict[str, Any] | None:
    run_uuid = getattr(args, "run_uuid", None)
    if isinstance(run_uuid, str) and run_uuid:
        return _BY_RUN.setdefault(run_uuid, {}) if create else _BY_RUN.get(run_uuid)
    attrs = getattr(args, "__dict__", None)
    if isinstance(attrs, dict) and not _is_pydantic(args):
        if create:
            return attrs.setdefault(STORE_ATTR, {})
        return attrs.get(STORE_ATTR)
    ident = id(args)
    if ident not in _REFS:
        try:
            _REFS[ident] = weakref.ref(args, _forget(ident))
        except TypeError:
            pass  # not weak-referenceable: keyed by id for the object's lifetime
    return _BY_ID.setdefault(ident, {}) if create else _BY_ID.get(ident)


def proc_state(args: Any, name: str, factory: Callable[[], Any] | None = None) -> Any:
    """The state ``name`` of this scope (``None`` if unset); ``factory`` creates it on first use."""
    store = _store(args, create=factory is not None)
    if store is None:
        return None
    if name not in store and factory is not None:
        store[name] = factory()
    return store.get(name)


def set_proc_state(args: Any, name: str, value: Any) -> None:
    _store(args, create=True)[name] = value  # type: ignore[index]


def pop_proc_state(args: Any, name: str, default: Any = None) -> Any:
    """Remove and return the state (replaces the old ``setattr(args, name, None)`` reset)."""
    store = _store(args, create=False)
    return default if store is None else store.pop(name, default)


def has_proc_state(args: Any, name: str) -> bool:
    store = _store(args, create=False)
    return store is not None and name in store


def reset_proc_state() -> None:
    """Forget every run- and identity-scoped entry (tests)."""
    _BY_RUN.clear()
    _BY_ID.clear()
    _REFS.clear()


RuntimeStatsSource = Callable[[str], Any]
_RUNTIME_STATS_SOURCE: RuntimeStatsSource | None = None
# What a source returns when the backend has no statistics channel at all (a
# Miles without ``rollout_runtime_stats``): only then is the legacy attribute read.
RUNTIME_STATS_UNAVAILABLE: Any = NotImplemented
RUNTIME_STATS_KINDS = ("abort_discard", "over_sampling", "suspend", "resume")


def register_runtime_stats_source(source: RuntimeStatsSource | None) -> None:
    """Install the backend's ``take(kind)`` channel (``None`` removes it). The
    source returns the tally, ``None`` (nothing this rollout) or
    :data:`RUNTIME_STATS_UNAVAILABLE` (the backend has no channel)."""
    global _RUNTIME_STATS_SOURCE
    _RUNTIME_STATS_SOURCE = source


def runtime_stats_kind(name: str) -> str:
    """The channel kind of ``name``: a kind itself, or its legacy attribute
    ``rollout_<kind>_stats`` (accepted so older callers keep working)."""
    if name.startswith("rollout_") and name.endswith("_stats"):
        name = name[len("rollout_"):-len("_stats")]
    if name not in RUNTIME_STATS_KINDS:
        raise ValueError(f"unknown runtime statistics kind {name!r}; known: {RUNTIME_STATS_KINDS}")
    return name


def legacy_runtime_stats_attr(kind: str) -> str:
    """The ``args`` attribute the ports-v1 fork set for ``kind``."""
    return f"rollout_{runtime_stats_kind(kind)}_stats"


def take_runtime_stats(args: Any, kind: str) -> dict[str, Any] | None:
    """Consume the backend's ``kind`` statistics (abort discards, over-sampling,
    suspend, resume) once per rollout.

    With a channel registered and present (Miles k31: ``take`` pops the tally),
    its answer is final, ``None`` included, and ``args`` is neither read nor
    written. Without one (ports-v1 fork, no Miles installed) the legacy
    ``args.rollout_<kind>_stats`` attribute is read and reset when the
    namespace allows it (a frozen config keeps no attribute to reset).
    """
    kind = runtime_stats_kind(kind)
    stats = (_RUNTIME_STATS_SOURCE(kind) if _RUNTIME_STATS_SOURCE is not None
             else RUNTIME_STATS_UNAVAILABLE)
    if stats is not RUNTIME_STATS_UNAVAILABLE:
        return stats
    attr = legacy_runtime_stats_attr(kind)
    legacy = getattr(args, attr, None)
    if legacy is not None:
        try:
            setattr(args, attr, None)
        except (AttributeError, TypeError, ValueError):
            pass
    return legacy
