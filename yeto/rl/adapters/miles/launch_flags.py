"""Miles island-entry flag spellings the launcher emits (yeto-framework-decoupling 5.5, audit P2).

The launcher computes the island layout neutrally (``yeto.rl.engine.multinode``:
nodes x GPUs, role -> logical slot map); how that layout is spelled on the
Miles island entry's command line lives here.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence


def bundle_map_flag(bundle_map: Mapping[str, Sequence[int]]) -> str:
    """`` --rl-island-bundle-map JSON`` (Miles placement-group logical bundles)."""
    payload = json.dumps({k: list(v) for k, v in bundle_map.items()}, sort_keys=True,
                         separators=(",", ":"))
    return f" --rl-island-bundle-map {shlex.quote(payload)}"


def cross_node_flags(allow_cross_node_tp: bool, allow_cross_node_engine_tp: bool) -> str:
    return ((" --rl-allow-cross-node-tp" if allow_cross_node_tp else "")
            + (" --rl-allow-cross-node-engine-tp" if allow_cross_node_engine_tp else ""))
