"""Miles island-entry flag spellings the launcher emits (yeto-framework-decoupling 5.5).

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


# Module the launcher runs on every island node (rl-verl-backend: per backend).
ISLAND_ENTRY_MODULE = "yeto.rl.adapters.miles.island_entry"
NEEDS_MILES_SOURCE = True


def offline_full_parameter_layout(config: Mapping, **layout_args):
    """The full-parameter layout a Miles island builds, computed by the launcher from
    the model's ``config.json`` (:func:`.models.dense_gpt_layout.offline_layout`:
    ``tp``, ``pp``, ``minimum_fragments``, ``max_fragment_bytes``, ``matrix_merge``,
    ``model_revision``, ``config_sha256``); None for a model or parallelism the offline
    layout does not model. The launcher checks ``--fragments`` against it
    (``yeto.launch.fragment_preflight``)."""
    from .models.dense_gpt_layout import DenseLayoutError, offline_layout

    try:
        return offline_layout(config, **layout_args)
    except (DenseLayoutError, RuntimeError, ValueError):
        return None
