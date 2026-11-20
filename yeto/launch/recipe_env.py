"""Recipe environment defaults of the trainer island.

A training backend may declare, per model recipe, environment variables its trainer
island gets by default: the backend's ``recipe_env`` role (``yeto.rl.engine.backends``)
provides ``recipe_env_defaults(args) -> dict | None`` with

* ``name``: what the defaults are (shown in the launch line),
* ``doc``: where they are documented,
* ``applies``: whether this launch (or island, for island args) gets them,
* ``reason``: why or why not,
* ``env``: ``{KEY: VALUE}``.

The launcher only applies what is declared: every non-sampler island whose args apply
gets ``env`` less the names the launch sets itself (``--island-env``, ``--modal-env``, a
per-island override, the task's own environment; an ``ssh:`` inventory node's ``env``
takes back the names listed in :data:`DEFAULTED_ENV`, see
``yeto.launch.ssh_trainer_island.node_env``). ``--rl-recipe-env-defaults off`` applies
none. A backend without the role declares nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

FLAG = "--rl-recipe-env-defaults"
MODES = ("auto", "off")
# comma-separated names of the island environment the launcher defaulted
DEFAULTED_ENV = "YETO_ISLAND_ENV_DEFAULTS"


def mode(args: Any) -> str:
    value = getattr(args, "rl_recipe_env_defaults", None) or "auto"
    if value not in MODES:
        raise ValueError(f"{FLAG} must be one of {MODES}, got {value!r}")
    return value


def declared(args: Any) -> dict[str, Any] | None:
    """The backend's declaration for ``args`` (None: nothing declared)."""
    if getattr(args, "training_mode", "rl") != "rl":
        return None
    from ..rl.engine import backends
    from .rl_backend import _rl_backend_module

    try:
        module = _rl_backend_module(args, "recipe_env")
    except backends.UnknownBackend:
        return None
    record = module.recipe_env_defaults(args)
    return dict(record) if record else None


def user_env(args: Any) -> dict[str, str]:
    """What the launch sets itself for every island: ``--island-env`` then ``--modal-env``."""
    from .modal_islands import parse_modal_env

    return {**parse_modal_env(getattr(args, "island_env", None) or [], flag="--island-env"),
            **parse_modal_env(getattr(args, "modal_env", None) or [])}


def island_defaults(iargs: Any, taken: Mapping[str, str] | None = None) -> dict[str, str]:
    """The defaults one island gets (``iargs``: that island's args): the declared ``env``
    less the names in ``taken`` and in the launch's own environment, plus
    :data:`DEFAULTED_ENV`; {} for a sampler island, with ``off`` or when nothing applies."""
    from ..rl.engine import sample_roles

    if mode(iargs) == "off" or sample_roles.sample_role(iargs) == sample_roles.SAMPLER:
        return {}
    record = declared(iargs)
    if not record or not record.get("applies"):
        return {}
    skip = {**user_env(iargs), **(taken or {})}
    env = {str(k): str(v) for k, v in (record.get("env") or {}).items() if k not in skip}
    if env:
        env[DEFAULTED_ENV] = ",".join(sorted(env))
    return env


def effective_env(args: Any) -> dict[str, str]:
    """The launch's trainer environment the launch-time checks see: the defaults, then
    ``--island-env``, then ``--modal-env``."""
    defaults = island_defaults(args)
    defaults.pop(DEFAULTED_ENV, None)
    return {**defaults, **user_env(args)}


def summary(args: Any) -> dict[str, Any] | None:
    """The dry-run / manifest record (None: nothing declared)."""
    record = declared(args)
    if record is None:
        return None
    env = record.pop("env", None) or {}
    current = mode(args)
    users = user_env(args)
    kept = sorted(k for k in env if k in users)
    applies = bool(record.get("applies")) and current == "auto"
    record.update({"mode": current, "applies": applies, "declared": len(env),
                   "injected": len(env) - len(kept) if applies else 0, "user_set": kept,
                   "disable": f"{FLAG} off"})
    if current == "off" and env:
        record["reason"] = f"{FLAG} off"
    if applies:
        record["env"] = {k: v for k, v in env.items() if k not in users}
    return record


def launch_line(record: Mapping[str, Any]) -> str:
    name = record.get("name") or "recipe environment defaults"
    if record.get("applies"):
        kept = record.get("user_set") or []
        return (f"[launcher] {name}: {record['injected']} of {record['declared']} set in the trainer "
                f"environment by default ({record.get('doc') or 'the recipe'}"
                + (f"; {len(kept)} set by this launch kept: {', '.join(kept)}" if kept else "")
                + f"); turn them off with {record['disable']}")
    return f"[launcher] {name}: not set ({record.get('reason')})"


def report(args: Any, *, out=None) -> dict[str, Any] | None:
    """Print the launch line (stderr) and return the record; None when nothing is declared."""
    record = summary(args)
    if record is not None:
        print(launch_line(record), file=sys.stderr if out is None else out, flush=True)
    return record


__all__ = ["DEFAULTED_ENV", "FLAG", "MODES", "declared", "effective_env", "island_defaults",
           "launch_line", "mode", "report", "summary", "user_env"]
