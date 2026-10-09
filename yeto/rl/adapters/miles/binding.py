"""Neutral name -> Miles implementation, and the training-side binding check
(yeto-framework-decoupling 4.4 / 4.4a, design D6 / D6a).

4.4: the core names dynamic-sampling filters neutrally
(``nonzero_reward_std_bounded``, ``nonzero_reward_std``); this table spells
them as the Miles module paths Miles loads (``--dynamic-sampling-filter-path``).

4.4a: before launch, every neutral name the spec uses is checked three ways
against the pre-rename spelling (``LEGACY_FILTER_NAMES`` in the core), and any
mismatch fails the launch naming the item:

1. argv -- the Miles argv for the neutral name equals, token for token, the
   argv the pre-rename path produced;
2. identity -- the module path, function qualname and module source sha256
   the binding resolves to equal those of the pre-rename path;
3. call -- one fixed CPU input through the mapped loading path gives a result
   equal to the pre-rename path (skipped, and reported as skipped, when the
   implementation cannot be imported here, e.g. a ``miles.*`` built-in on a
   machine without Miles).

The result is a JSON-able dict for the runtime manifest.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    LEGACY_FILTER_NAMES,
    STOCK_NONZERO_STD_FILTER,
)

BINDING_SCHEMA = "yeto-backend-binding-v1"

# neutral name -> Miles dotted path (module.function)
FILTER_PATHS: dict[str, str] = {
    BOUNDED_NONZERO_STD_FILTER: "yeto.rl.filters.bounded_nonzero_reward_std",
    STOCK_NONZERO_STD_FILTER:
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
}


class BindingError(RuntimeError):
    """A neutral name does not bind to the same implementation as before the rename."""


def filter_path(name: str) -> str:
    try:
        return FILTER_PATHS[name]
    except KeyError:
        raise BindingError(f"dynamic sampling filter {name!r} has no Miles binding") from None


def filter_argv(name: str) -> list[str]:
    return ["--dynamic-sampling-filter-path", filter_path(name)]


def _legacy_argv(old_path: str) -> list[str]:
    # What the pre-4.4 translation emitted: the spec carried the path itself.
    return ["--dynamic-sampling-filter-path", old_path]


def identity(dotted: str) -> dict[str, Any]:
    """Module path, qualname and module source sha256 of ``dotted`` without importing it."""
    module, _, qualname = dotted.rpartition(".")
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    origin = getattr(spec, "origin", None) if spec is not None else None
    sha = None
    if origin and origin.endswith(".py"):
        with open(origin, "rb") as handle:
            sha = hashlib.sha256(handle.read()).hexdigest()
    return {"module": module, "qualname": qualname, "source_sha256": sha,
            "found": spec is not None}


def _load(dotted: str) -> Callable[..., Any]:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _fixed_filter_input() -> tuple[Any, list[Any]]:
    args = SimpleNamespace(reward_key=None, dynamic_sampling_max_replacements=2)
    samples = [SimpleNamespace(reward=r, group_index=0, index=i, metadata={}, label=None)
               for i, r in enumerate((0.0, 1.0, 1.0, 0.0))]
    return args, samples


def _call_result(fn: Callable[..., Any]) -> Any:
    args, samples = _fixed_filter_input()
    out = fn(args, samples)
    return (getattr(out, "keep", out), getattr(out, "reason", None))


def check_filter(name: str, *, paths: Mapping[str, str] | None = None,
                 legacy: Mapping[str, str] | None = None,
                 loader: Callable[[str], Callable[..., Any]] = _load,
                 identify: Callable[[str], dict[str, Any]] = identity) -> dict[str, Any]:
    """The three D6a checks for one neutral filter name; raises :class:`BindingError`."""
    paths = FILTER_PATHS if paths is None else paths
    legacy = LEGACY_FILTER_NAMES if legacy is None else legacy
    olds = [old for old, new in legacy.items() if new == name]
    if len(olds) != 1:
        raise BindingError(f"{name!r}: no unique pre-rename name to check against ({olds})")
    old = olds[0]
    if name not in paths:
        raise BindingError(f"{name!r}: no Miles binding")
    mapped = paths[name]
    result: dict[str, Any] = {"name": name, "legacy": old, "mapped": mapped}
    argv, expected = ["--dynamic-sampling-filter-path", mapped], _legacy_argv(old)
    if argv != expected:
        raise BindingError(f"{name!r} ① argv: {argv} != pre-rename {expected}")
    result["argv"] = "same"
    ident, ident_old = identify(mapped), identify(old)
    if ident != ident_old:
        raise BindingError(f"{name!r} ② identity: {ident} != pre-rename {ident_old}")
    result["identity"] = ident
    try:
        new_fn, old_fn = loader(mapped), loader(old)
    except ImportError as exc:
        result["call"] = f"skipped: {type(exc).__name__}: {exc}"
        return result
    got, want = _call_result(new_fn), _call_result(old_fn)
    if got != want or new_fn is not old_fn:
        raise BindingError(f"{name!r} ③ call: mapped path gave {got!r}, pre-rename path {want!r}")
    result["call"] = "same"
    return result


def check_spec(spec) -> dict[str, Any]:
    """Binding check of every neutral name ``spec`` uses (launch refuses on error)."""
    names = [spec.dynamic_sampling_filter] if spec.dynamic_sampling_filter else []
    return {"schema": BINDING_SCHEMA, "backend": "miles",
            "filters": [check_filter(n) for n in names]}
