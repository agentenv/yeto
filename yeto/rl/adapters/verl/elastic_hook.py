"""--rl-elastic recommend flags on the verl backend: not supported yet (design Non-Goals)."""

from __future__ import annotations

from typing import Any


def check_recommend_flags(args: Any) -> None:
    mode = getattr(args, "rl_recommend_mode", None) or "disabled"
    if mode != "disabled" or getattr(args, "rl_elastic", False):
        raise ValueError("the verl backend does not support in-island elasticity (--rl-elastic / --rl-recommend-mode)")


def recommend_flags(args: Any) -> str:
    return ""
