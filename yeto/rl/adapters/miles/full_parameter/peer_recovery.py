"""Peer recovery of a Miles full-parameter island (S27 E-3b).

The engine-neutral exchange is :mod:`yeto.rl.peer_recovery`, and the rank
list, the layout tag, the digest check and the stream island are in
:mod:`yeto.rl.engine.stream_recovery` (S27 G-ED, shared with the verl island);
this module is the Miles trainer behind them (the ``cut_plugin`` verbs).

Snapshot content. Every Megatron rank writes one cut shard through
:mod:`yeto.rl.adapters.miles.cut_plugin` (its model copy, its inner optimizer
state with the FP32 main copy and moments, the LR scheduler, RNG) into the
snapshot directory, and the driver adds :data:`RANKS` (every shard's file,
sha256, size and rank coordinate). The rank shards are written in place on a
shared snapshot directory, or into a staging directory that every rank of the
island sees (one node, or a node-shared path); a shard that did not land
there refuses the snapshot (the syncer then asks another donor). Each rank's
entry also keeps the cut summary digests (state, RNG, LR scheduler position);
after the restore every rank must re-export exactly those, as the
single-island cut restore checks, else the restore fails.

Layout tag. The parameter layout hash plus the trainer parallel layout read
back from the ranks (world, TP, PP, CP, EP, DP). Only a donor with the same
tag can serve; the shard restore checks every rank coordinate again.

Replay. After the restore the island reports "nothing held", so the syncer
resends the current value of every fragment. Broadcasts overwrite (no update
log; see the E-3 design), and the resent fragments become the island's
anchors in one apply. The donor's local progress since its last broadcast
is therefore not copied: the donor still pushes it itself, and the
recovering island never counts it a second time. What the island takes from
the donor is the inner optimizer state (moments, step, LR schedule) and RNG.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.engine.stream_recovery import (  # noqa: F401 - re-exported (Miles import path)
    RANKS,
    PeerRecoveryOptions,
    RankShardSnapshot,
    StreamRecoveryIsland,
    snapshot_layout_tag,
)
from yeto.rl.peer_recovery import SnapshotRefused

RANKS_SCHEMA = "yeto.miles_peer_snapshot/v1"
SNAPSHOT_CUT_ID = "peer-snapshot"
_LAYOUT_KEYS = ("world", "tp", "pp", "cp", "ep", "dp")
TAG_PREFIX = "miles-"


def parallel_layout(coords: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The trainer layout from every rank's ``cut_plugin.rank_coords``."""
    if not coords:
        raise RuntimeError("trainer reported no ranks")
    first = coords[0]
    return {
        "world": len(coords),
        "tp": int(first.get("tp_size", 1)),
        "pp": int(first.get("pp_size", 1)),
        "cp": int(first.get("cp_size", 1)),
        "ep": int(first.get("ep_size", 1)),
        "dp": int(first.get("dp_size", 1)),
    }


def layout_tag(parameter_layout_hash: str, layout: Mapping[str, int]) -> str:
    return snapshot_layout_tag(TAG_PREFIX, RANKS_SCHEMA, parameter_layout_hash,
                               {key: int(layout[key]) for key in _LAYOUT_KEYS})


def _refusals(results: Sequence[Mapping[str, Any]]) -> list[str]:
    from yeto.rl.adapters.miles.cut_plugin import REFUSAL_KIND, REFUSED

    return sorted({f"[{r.get(REFUSAL_KIND, 'refused')}] {r[REFUSED]}" for r in results if REFUSED in r})


class MilesRankShards(RankShardSnapshot):
    """The per-rank shards of one Miles trainer group (``cut_plugin`` cut shards).

    ``group``: the Miles train group (``run_plugin`` runs a plugin function
    on every rank and returns their results). ``parameter_layout_hash``: the
    role's fragment layout hash."""

    schema = RANKS_SCHEMA
    tag_prefix = TAG_PREFIX

    def __init__(self, group: Any, parameter_layout_hash: str, **kwargs: Any) -> None:
        super().__init__(parameter_layout_hash, **kwargs)
        self.group = group

    async def rank_layout(self) -> dict[str, int]:
        from yeto.rl.adapters.miles.cut_plugin import RANK_COORDS

        coords = [dict(c) for c in await self.group.run_plugin(RANK_COORDS, {})]
        return {key: value for key, value in parallel_layout(coords).items() if key in _LAYOUT_KEYS}

    async def save_ranks(self, directory: Path) -> list[dict[str, Any]]:
        from yeto.rl.adapters.miles.cut_plugin import SAVE_CUT_SHARD

        results = [dict(r) for r in await self.group.run_plugin(
            SAVE_CUT_SHARD, {"directory": str(directory), "cut_id": SNAPSHOT_CUT_ID})]
        refused = _refusals(results)
        if refused:
            raise SnapshotRefused("rank refused the peer snapshot: " + "; ".join(refused))
        return results

    async def restore_ranks(self, directory: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from yeto.rl.adapters.miles.cut_plugin import RESTORE_CUT_SHARD

        results = [dict(r) for r in await self.group.run_plugin(
            RESTORE_CUT_SHARD, {"directory": str(directory), "files": files, "cut_id": SNAPSHOT_CUT_ID})]
        refused = _refusals(results)
        if refused:
            # The trainer may be partly written: the caller must not train on it.
            raise RuntimeError("rank refused the peer snapshot restore: " + "; ".join(refused))
        return results


# The role stream's island (the neutral one; the Miles name stays for imports).
MilesPeerRecoveryIsland = StreamRecoveryIsland
