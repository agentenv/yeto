"""Miles ``recipe_env`` role: the trainer environment a ``--rl-model-recipe`` declares as
its default (applied by ``yeto.launch.recipe_env``; a value the launch sets wins)."""

from __future__ import annotations

from typing import Any


def recipe_env_defaults(args: Any) -> dict[str, Any] | None:
    """``{"name", "doc", "applies", "reason", "env"}`` of the recipe of ``args``, or None
    when the recipe declares nothing for this launch."""
    if getattr(args, "rl_model_recipe", None) == "kimi_k31":
        from yeto.rl.adapters.miles.models.kimi_k31 import mfu_switches

        return mfu_switches.declaration(args)
    return None


__all__ = ["recipe_env_defaults"]
