"""Peer recovery of a verl full-parameter island (S27 G-ED, VERL-PARITY #17).

The exchange with the syncer and the snapshot store is
:mod:`yeto.rl.peer_recovery`; the rank list, the layout tag, the digest check
and the stream island are :mod:`yeto.rl.engine.stream_recovery` (the same code
as on a Miles island). This module only gives the verl FSDP2 trainer's three
verbs, over the round-cut rank files of :mod:`.cut`:

* save: every FSDP rank writes ``rank<NNNNN>.pt`` (its local parameter shards,
  which are the FP32 masters, its Adam state, the LR scheduler and RNG) into
  the snapshot directory (``cut.remote_save_rank``);
* restore: every rank of the fresh trainer loads its file
  (``cut.remote_restore_rank``: same world size, file sha256, and a trainer
  without optimizer state) and reports the digests of what it now holds;
* layout: ``{"world", "dp"}`` (FSDP2: every rank is a data-parallel shard).

The digests are the cut's ``state_digest`` (parameters and Adam state the
optimizer steps), ``rng_digest`` and ``scheduler_samples``; the restore fails
unless every rank re-exports the donor's.

After a restore the trainer holds the donor's parameters, not the cut the
adapter last captured or applied, so the adapter forgets its resident cut:
the stream's first apply (the syncer's resent fragments) then writes and
proves every fragment.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from yeto.rl.engine.cut import CutError
from yeto.rl.engine.stream_recovery import RankShardSnapshot
from yeto.rl.peer_recovery import SnapshotRefused

from . import cut as vcut

SCHEMA = "yeto.verl_peer_snapshot/v1"
SNAPSHOT_CUT_ID = "peer-snapshot"
TAG_PREFIX = "verl-"


class VerlRankShards(RankShardSnapshot):
    """The FSDP2 rank files of one verl island (``group.island.call_workers``
    runs ``fn(worker, *args)`` on every actor worker and returns their results)."""

    schema = SCHEMA
    tag_prefix = TAG_PREFIX

    def __init__(self, group: Any, adapter: Any, **kwargs: Any) -> None:
        super().__init__(adapter.layout.layout_hash, **kwargs)
        self.group = group
        self.adapter = adapter

    def _call(self, fn: Any, *args: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in self.group.island.call_workers(fn, *args)]

    async def rank_layout(self) -> dict[str, int]:
        return vcut.common_layout(await asyncio.to_thread(self._call, vcut.remote_rank_layout))

    async def save_ranks(self, directory: Path) -> list[dict[str, Any]]:
        try:
            results = await asyncio.to_thread(self._call, vcut.remote_save_rank, str(directory),
                                              SNAPSHOT_CUT_ID)
        except CutError as exc:
            raise SnapshotRefused(f"verl rank refused the peer snapshot: {exc}") from exc
        worlds = {int(r["coord"]["world"]) for r in results}
        if len(worlds) != 1 or len(results) != next(iter(worlds)):
            raise SnapshotRefused(f"{len(results)} verl rank results for world sizes {sorted(worlds)}")
        return results

    async def restore_ranks(self, directory: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The trainer leaves the adapter's resident cut whatever happens next.
        self.adapter.resident = None
        return await asyncio.to_thread(self._call, vcut.remote_restore_rank, str(directory), files,
                                       SNAPSHOT_CUT_ID)
