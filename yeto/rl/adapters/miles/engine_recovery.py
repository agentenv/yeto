"""Miles :class:`~yeto.rl.engine.engine_outage.EngineRecoveryPort` (S27 G-DI design v2).

The engines of a two-cluster island run on the engine cluster's Ray nodes; the trainer
and the driver run on the Ray head (D3 head pin). This port tells the driver whether
those nodes are alive and restarts the engines on the nodes that replace them.

Mechanism (all fork verbs already used by E1 membership, ``rollout.MilesRolloutPool``):

* ``engines_alive``: every island node with GPUs is alive in the island's Ray
  (``ray.nodes()``; the head never dies without the driver).
* ``abort_generation``: ``MilesRolloutPool.abort`` (``controller.abort_all``).
* ``nodes_ready``: the alive GPU nodes are back to the island's node count AND no
  placement group of the island's Ray is PENDING or RESCHEDULING (Ray reschedules the
  bundles of a dead node onto a node that joins with the same resources).
* ``restore_engines``: ``remove_engines`` (fork ``stop_cells``) for every member, then
  ``add_engines`` (fork ``start_cells`` + ``wait_cells_tracked``) for the same declared
  cells at the next fork epoch. The cells start on the rescheduled bundles. The driver
  then publishes the policy to them with the normal full publication.

Requires declared rollout cells (``--rl-elastic-declare-cells --rl-elastic-cells``; the
launcher adds them for a two-cluster island).

NOT verified on GPUs (S27 G-DI): whether ``stop_cells`` accepts cells whose actors are
dead; whether the restarted cells see the right local GPUs on the new node (the startup
GPU-id view may need ``bind_members``); whether a full ``update_weights`` reaches cells
started after the weight-update NCCL group was created.
"""

from __future__ import annotations

from typing import Any, Callable


class MilesEngineRecovery:
    def __init__(self, pool: Any, *, nodes: int, placement_group: Callable[[], Any] | None = None,
                 ray_module: Any = None) -> None:
        self.pool, self.nodes = pool, int(nodes)
        self._pg = placement_group
        self._ray = ray_module
        self.members: frozenset[str] = frozenset()

    def _ray_mod(self):
        if self._ray is None:
            import ray

            return ray
        return self._ray

    def _alive_gpu_nodes(self) -> int:
        return sum(1 for n in self._ray_mod().nodes()
                   if n.get("Alive") and float((n.get("Resources") or {}).get("GPU", 0) or 0) > 0)

    def engines_alive(self) -> bool:
        return self._alive_gpu_nodes() >= self.nodes

    def is_engine_loss(self, error: BaseException) -> bool:
        return not self.engines_alive()

    def abort_generation(self) -> None:
        self.pool.abort()

    def nodes_ready(self) -> bool:
        if self._alive_gpu_nodes() < self.nodes:
            return False
        util = self._ray_mod().util
        if self._pg is not None:
            return str(util.placement_group_table(self._pg()).get("state", "")).upper() == "CREATED"
        # every placement group of the island's Ray placed again (none pending/rescheduling)
        states = [str(info.get("state", "")).upper() for info in (util.placement_group_table() or {}).values()]
        return not any(s in ("PENDING", "RESCHEDULING") for s in states)

    def restore_engines(self) -> dict[str, Any]:
        members = frozenset(self.pool.members()) or self.members
        status = self.pool.membership_status()
        epoch = int(status.get("epoch", 0))
        stop_error = None
        if members:
            try:
                self.pool.remove_engines(members, epoch=epoch)
                epoch += 1
            except Exception as exc:  # noqa: BLE001 - dead cells may refuse a clean stop
                stop_error = f"{type(exc).__name__}: {exc}"[:300]
                epoch = int(self.pool.membership_status().get("epoch", epoch))
        declared = self.pool._require_declared()
        cells = frozenset(f"engine:{c}" for c in declared) if not members else members
        started = self.pool.add_engines(len(cells), epoch=epoch, members=cells)
        self.members = frozenset(started)
        return {"cells": sorted(started), "fork_epoch": epoch + 1,
                **({"stop_error": stop_error} if stop_error else {})}
