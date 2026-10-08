"""Training-backend registry (yeto-framework-decoupling 5.4 + 4.11, design D11 方案 A).

One table maps a backend name to the adapter modules the neutral code needs.
The core, the launcher and the CLI look modules up here by *role* and load
them with ``importlib`` -- they never ``import`` an adapter statically, which
is what the boundary check (``tests/test_import_boundaries.py``) enforces.

Which backend is active:

* the launcher / CLI pass ``--rl-backend`` (default ``miles``);
* processes the adapter starts (rollout Ray actors, codex subprocesses) read
  the environment variable ``YETO_RL_BACKEND``; when it is unset the backend is
  ``miles``, exactly today's behaviour (主 agent 代拍板 2026-10-08 第 2 条;
  revisit once verl lands -- design D11 suggests requiring it then).

A name that is not in the table raises :class:`UnknownBackend` ("未注册"), e.g.
``verl`` until its adapter registers here (rl-verl-backend).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Mapping

BACKEND_ENV = "YETO_RL_BACKEND"
DEFAULT_BACKEND = "miles"
# Names the CLI accepts; a known-but-unregistered name fails with "未注册"
# at launch instead of argparse's generic "invalid choice".
KNOWN_BACKENDS = ("miles", "verl")


class UnknownBackend(RuntimeError):
    """The requested training backend has no registered adapter."""


@dataclass(frozen=True)
class BackendEntry:
    name: str
    package: str
    # role -> module path inside the adapter.  Roles used by neutral code:
    #   rollout_meta  policy token / round counters in the rollout process (4.11)
    #   entry         island entry, capabilities, Ray connect (launcher, stage W)
    #   placement     placement request type (launcher)
    #   elastic_hook  --rl-elastic recommend flags (CLI, launcher)
    #   rollout       rollout pool (test-only fault injection env names)
    #   publish       publisher (test-only fault injection env names)
    #   algorithm_flags  AlgorithmSpec -> backend argv (critic warmup)
    #   selection_argv   facts read from pass-through backend argv (--rl-engine matrix)
    #   binding       neutral name -> backend binding check (4.4a)
    #   identity      BackendIdentity of this adapter (phase 5)
    roles: Mapping[str, str] = field(default_factory=dict)


_MILES = "yeto.rl.adapters.miles"
_REGISTRY: dict[str, BackendEntry] = {
    "miles": BackendEntry("miles", _MILES, {
        "rollout_meta": f"{_MILES}.rollout_meta_hook",
        "entry": f"{_MILES}.entry",
        "placement": f"{_MILES}.placement",
        "elastic_hook": f"{_MILES}.elastic_hook",
        "rollout": f"{_MILES}.rollout",
        "publish": f"{_MILES}.publish",
        "algorithm_flags": f"{_MILES}.algorithm_flags",
        "selection_argv": f"{_MILES}.selection_argv",
        "binding": f"{_MILES}.binding",
        "identity": f"{_MILES}.identity",
    }),
}


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def active_name(name: str | None = None) -> str:
    """``name`` if given, else ``$YETO_RL_BACKEND``, else ``miles``."""

    return name or os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND


def get(name: str | None = None) -> BackendEntry:
    resolved = active_name(name)
    try:
        return _REGISTRY[resolved]
    except KeyError:
        raise UnknownBackend(
            f"训练后端 {resolved!r} 未注册（已注册：{', '.join(registered())}）；"
            f"见 yeto/rl/engine/backends.py") from None


def module(role: str, name: str | None = None) -> ModuleType:
    """Import the adapter module that plays ``role`` for backend ``name``."""

    entry = get(name)
    try:
        path = entry.roles[role]
    except KeyError:
        raise UnknownBackend(f"训练后端 {entry.name!r} 没有提供 {role!r}") from None
    return importlib.import_module(path)


def register(entry: BackendEntry) -> None:
    """Add a backend (tests; a new adapter adds its row to ``_REGISTRY``)."""

    _REGISTRY[entry.name] = entry


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)
