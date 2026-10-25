"""verl island-entry spellings the launcher emits (rl-verl-backend; decoupling 5.5 role).

A verl island is one node with any number of cards (S26 E9 G3): the
island-local Ray sees every card of the node and verl's colocated hybrid
engine puts the trainer and the rollout replicas on all of them
(``island_entry.island_gpus``).  A single-node multi-card island therefore
needs no layout flag: the launcher emits a bundle map only for a
fixed-partition placement (``--rl-placement fixed-partition`` with a
``placement`` in the initial config, ``launch.topology.rl_island_layout``),
and verl places its own workers, so it cannot honour a role -> card map.
Such a map and the cross-node switches have no verl spelling and are refused
instead of being silently dropped.
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
    """Refuse a role -> card map: verl places its trainer and rollout workers itself.

    Only a fixed-partition placement reaches here (the launcher emits no map
    for a single-node colocated island, whatever its card count)."""
    roles = ", ".join(f"{role}={list(slots)}" for role, slots in sorted(bundle_map.items()))
    raise ValueError(
        "verl 岛是单节点同卡共置（colocated）：训练与 rollout 用本节点全部卡，由 verl 自己放置，"
        f"不接受按角色指定卡号的 bundle map（收到 {roles}）。单节点多卡直接用 --gpu <provider>:<N>x<卡型>，"
        "不要给 fixed-partition 的 placement")


def cross_node_flags(allow_cross_node_tp: bool, allow_cross_node_engine_tp: bool) -> str:
    if allow_cross_node_tp or allow_cross_node_engine_tp:
        raise ValueError("verl 岛只在一个节点内：不支持跨节点 TP")
    return ""
