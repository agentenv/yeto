"""``Placement.reconfigure`` for E1 rollout reconfiguration (rl-infra-spec 3.4/3.4a).

Wraps the startup :class:`~.placement.MilesPlacement` (fork M1 explicit
role->bundle map). E1 never moves a trainer GPU: a plan must keep the trainer
GPUs, its rollout GPUs must be pool GPUs outside the trainer set, and the
epoch must be the next config epoch. The physical engines are started and
stopped by :class:`~.rollout.MilesRolloutPool`; this object records the
committed role->GPU description the driver and the controller report.
Trainer DP changes / role transfer are E3 (4.7) and are refused here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..ports import Placement, PlacementDescription


class PlacementPlanError(ValueError):
    pass


class ElasticPlacement:
    def __init__(self, base: Placement, *, pool_gpus: tuple[str, ...] | None = None,
                 epoch: int = 0) -> None:
        self._base = base
        self._current: PlacementDescription = base.describe()
        described = tuple(self._current.trainer_gpus) + tuple(self._current.rollout_gpus)
        standby = tuple(self._current.extra.get("standby_gpus", ()) or ())
        self._pool = tuple(pool_gpus) if pool_gpus is not None else described + standby
        self.epoch = int(epoch)

    def describe(self) -> PlacementDescription:
        return self._current

    def restore_committed(self, rollout_gpus: tuple[str, ...], *, epoch: int) -> PlacementDescription:
        """After a learner restart: adopt the committed config's rollout GPUs (journal authority)."""
        if epoch < self.epoch:
            raise PlacementPlanError(f"committed epoch {epoch} is behind {self.epoch}")
        self.epoch = epoch - 1
        return self.reconfigure(replace(self._current, rollout_gpus=tuple(rollout_gpus)), epoch=epoch)

    def reconfigure_trainer(self, plan: PlacementDescription, *, epoch: int) -> PlacementDescription:
        """E3 (4.7): record a committed trainer DP change / role transfer (nested trainer GPU sets)."""
        current = self._current
        if epoch != self.epoch + 1:
            raise PlacementPlanError(f"placement epoch {epoch}, expected {self.epoch + 1}")
        if plan.kind != current.kind or current.kind != "fixed-partition":
            raise PlacementPlanError("trainer reconfiguration needs a fixed partition")
        old, new = set(current.trainer_gpus), set(plan.trainer_gpus)
        if not new or not (old <= new or new <= old):
            raise PlacementPlanError("trainer GPU sets must be non-empty and nested")
        rollout = tuple(plan.rollout_gpus)
        if set(rollout) & new or set(rollout) - set(self._pool) or new - set(self._pool):
            raise PlacementPlanError("trainer/rollout GPUs overlap or leave the pool")
        standby = tuple(g for g in self._pool if g not in rollout and g not in new)
        self._current = replace(plan, trainer_gpus=tuple(plan.trainer_gpus), rollout_gpus=rollout,
                                extra={**dict(plan.extra), "standby_gpus": standby, "config_epoch": epoch})
        self.epoch = epoch
        return self._current

    def reconfigure(self, plan: PlacementDescription, *, epoch: int) -> PlacementDescription:
        current = self._current
        if epoch != self.epoch + 1:
            raise PlacementPlanError(f"placement epoch {epoch}, expected {self.epoch + 1}")
        if plan.kind != current.kind or current.kind != "fixed-partition":
            raise PlacementPlanError("E1 reconfigures only a fixed partition")
        if tuple(plan.trainer_gpus) != tuple(current.trainer_gpus):
            raise PlacementPlanError("E1 keeps the trainer GPUs (role transfer is E3)")
        rollout = tuple(plan.rollout_gpus)
        if len(set(rollout)) != len(rollout):
            raise PlacementPlanError("duplicate rollout GPU")
        outside = sorted(set(rollout) - set(self._pool))
        if outside:
            raise PlacementPlanError(f"rollout GPUs {outside} are outside the pool")
        if set(rollout) & set(current.trainer_gpus):
            raise PlacementPlanError("rollout GPUs overlap the trainer")
        standby = tuple(g for g in self._pool
                        if g not in rollout and g not in current.trainer_gpus)
        extra: dict[str, Any] = {**dict(plan.extra), "standby_gpus": standby, "config_epoch": epoch}
        self._current = replace(plan, rollout_gpus=rollout, extra=extra)
        self.epoch = epoch
        return self._current
