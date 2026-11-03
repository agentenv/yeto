"""verl placement request (role ``placement``).

The verl islands have two placements: colocated (FSDP2 island: rollout and
trainer on the same cards) and fixed partition (Megatron island: a trainer
pool of whole leading nodes and a rollout pool on the rest, ``megatron_engine``).
:meth:`PlacementRequest.trainer_shape` accepts only that leading-node layout:
no standby cards, no explicit bundle map other than the leading one, no
cross-node TP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlacementRequest:
    kind: str
    trainer_gpus: int
    rollout_gpus: int
    gpus_per_engine: int = 1
    standby_gpus: int = 0
    gpus_per_node: int | None = None
    model_parallel: int = 1
    node_parallel: int = 1
    expert_parallel: int = 1
    bundle_map: Mapping[str, Sequence[int]] | None = None
    allow_cross_node_tp: bool = False
    allow_cross_node_engine_tp: bool = False
    extra: Any = None

    def trainer_shape(self) -> tuple[int, int]:
        """(trainer nodes, trainer GPUs per node) of the verl island."""
        if self.kind != "fixed-partition":
            raise ValueError(f"verl placement {self.kind!r}: only fixed-partition has a trainer pool shape")
        if self.standby_gpus:
            raise ValueError("a verl island has no standby cards")
        if self.allow_cross_node_tp or self.allow_cross_node_engine_tp:
            raise ValueError("a verl island does not support cross-node TP")
        if self.trainer_gpus <= 0 or self.rollout_gpus <= 0:
            raise ValueError("verl fixed-partition needs both trainer cards and rollout cards > 0")
        if self.rollout_gpus % max(1, self.gpus_per_engine):
            raise ValueError(f"rollout card count {self.rollout_gpus} is not a multiple of {self.gpus_per_engine} cards per engine")
        if self.trainer_gpus % max(1, self.model_parallel) or self.trainer_gpus % max(1, self.expert_parallel):
            raise ValueError("TP*PP*CP and EP must both divide the trainer card count")
        per_node = int(self.gpus_per_node or self.trainer_gpus)
        if self.trainer_gpus > per_node and self.trainer_gpus % per_node:
            raise ValueError(f"the verl trainer pool must be whole nodes: {self.trainer_gpus} cards, {per_node} cards per node")
        nodes, per = ((self.trainer_gpus // per_node, per_node) if self.trainer_gpus >= per_node
                      else (1, self.trainer_gpus))
        if self.bundle_map is not None:
            leading = {"trainer": list(range(self.trainer_gpus)),
                       "rollout": list(range(self.trainer_gpus, self.trainer_gpus + self.rollout_gpus)),
                       "standby": []}
            if {k: list(v) for k, v in self.bundle_map.items()} != leading:
                raise ValueError("a verl island accepts only the leading layout (trainer pool first, rollout after), no other bundle map")
        return nodes, per
