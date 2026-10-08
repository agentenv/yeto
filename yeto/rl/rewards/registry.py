"""Registry and loader for neutral rewards (decoupling 3.10, design D9b).

A user reward is a plain function ``Trajectory -> RewardResult``; it needs no
training framework.  It is referenced either by a registered neutral name
(``register_reward("my_reward")``) or as ``"package.module:function"``.  Every
load records the plugin identity: module, qualified name and the sha256 of the
module's source file (same rule as ``AlgorithmSpec`` plugin refs), so editing
the reward's source changes its identity.

Backend adapters turn a loaded reward into their own callable (Miles:
``yeto.rl.engine.miles_adapter.rewards``).  An unknown name fails with the list
of registered names, before anything is launched.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from typing import Callable

from yeto.rl.rewards.types import NeutralReward


class RewardRegistryError(ValueError):
    pass


_REGISTRY: dict[str, NeutralReward] = {}


def register_reward(name: str, fn: NeutralReward | None = None):
    """Register ``fn`` under neutral ``name``; usable as a decorator."""

    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise RewardRegistryError(f"reward name {name!r} must be letters, digits, '_' or '-'")

    def _register(func: NeutralReward) -> NeutralReward:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not func:
            raise RewardRegistryError(f"reward name {name!r} is already registered to "
                                      f"{existing.__module__}.{existing.__qualname__}")
        _REGISTRY[name] = func
        return func

    return _register if fn is None else _register(fn)


def registered_names() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


@dataclass(frozen=True)
class RewardPlugin:
    """A loaded neutral reward and its identity."""

    ref: str
    fn: Callable
    module: str
    qualname: str
    source_sha256: str

    def identity(self) -> dict[str, str]:
        return {"module": self.module, "qualname": self.qualname,
                "source_sha256": self.source_sha256}


def _source_sha256(fn: Callable) -> str:
    path = inspect.getsourcefile(inspect.unwrap(fn))
    if path is None:
        raise RewardRegistryError(f"reward {fn!r} has no source file; cannot record its identity")
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_reward(ref: str) -> RewardPlugin:
    """Resolve a registered name or ``module:function`` to a :class:`RewardPlugin`."""

    _ensure_builtins()
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RewardRegistryError(f"reward {ref!r}: cannot import {module_name!r}: {exc}") from exc
        fn = module
        for part in attr.split("."):
            fn = getattr(fn, part, None)
            if fn is None:
                raise RewardRegistryError(f"reward {ref!r}: {module_name!r} has no {attr!r}")
    else:
        fn = _REGISTRY.get(ref)
        if fn is None:
            raise RewardRegistryError(
                f"reward {ref!r} is not registered; registered rewards: "
                f"{', '.join(sorted(_REGISTRY)) or '(none)'}; or use 'module:function'")
    if not callable(fn):
        raise RewardRegistryError(f"reward {ref!r} is not callable")
    return RewardPlugin(ref=ref, fn=fn, module=fn.__module__, qualname=fn.__qualname__,
                        source_sha256=_source_sha256(fn))


_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from yeto.rl.rewards import builtin

    for name, fn in (("math", builtin.math_reward), ("gsm8k", builtin.gsm8k_reward),
                     ("length", builtin.length_reward), ("gdpo", builtin.gdpo_reward),
                     ("gdpo_correctness", builtin.gdpo_correctness_reward)):
        register_reward(name, fn)
