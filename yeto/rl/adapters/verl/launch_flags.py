"""verl island-entry spellings the launcher emits (rl-verl-backend; decoupling 5.5 role).

The verl island runs one node, one GPU, colocated (first step), so the
multi-node layout flags have no verl spelling yet and are refused instead of
being silently dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .pins import VERL_COMMIT, VERL_IMAGE

ISLAND_ENTRY_MODULE = "yeto.rl.adapters.verl.island_entry"
NEEDS_MILES_SOURCE = False
DEFAULT_RL_IMAGE = VERL_IMAGE
_IMAGE = re.compile(r"verl-build:(?P<commit>[0-9a-f]{40})")


def rl_image_ok(rl_image: str) -> bool:
    """``verl-build:<pinned commit>`` (Modal-built, see ``image``); other commits are refused."""
    match = _IMAGE.fullmatch(rl_image or "")
    return bool(match) and match["commit"] == VERL_COMMIT


def bundle_map_flag(bundle_map: Mapping[str, Sequence[int]]) -> str:
    raise ValueError("verl 后端第一版只支持单节点单卡岛，没有 bundle map")


def cross_node_flags(allow_cross_node_tp: bool, allow_cross_node_engine_tp: bool) -> str:
    if allow_cross_node_tp or allow_cross_node_engine_tp:
        raise ValueError("verl 后端第一版不支持跨节点 TP")
    return ""
