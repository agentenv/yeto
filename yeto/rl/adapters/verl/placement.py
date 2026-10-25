"""verl placement request (role ``placement``; S27 G-DC, V10).

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
            raise ValueError("verl 岛没有 standby 卡")
        if self.allow_cross_node_tp or self.allow_cross_node_engine_tp:
            raise ValueError("verl 岛不支持跨节点 TP")
        if self.trainer_gpus <= 0 or self.rollout_gpus <= 0:
            raise ValueError("verl fixed-partition 需要训练卡和 rollout 卡都 > 0")
        if self.rollout_gpus % max(1, self.gpus_per_engine):
            raise ValueError(f"rollout 卡数 {self.rollout_gpus} 不是每引擎 {self.gpus_per_engine} 卡的整数倍")
        if self.trainer_gpus % max(1, self.model_parallel) or self.trainer_gpus % max(1, self.expert_parallel):
            raise ValueError("TP*PP*CP 和 EP 都必须整除训练卡数")
        per_node = int(self.gpus_per_node or self.trainer_gpus)
        if self.trainer_gpus > per_node and self.trainer_gpus % per_node:
            raise ValueError(f"verl 训练池必须是整节点：{self.trainer_gpus} 卡、每节点 {per_node} 卡")
        nodes, per = ((self.trainer_gpus // per_node, per_node) if self.trainer_gpus >= per_node
                      else (1, self.trainer_gpus))
        if self.bundle_map is not None:
            leading = {"trainer": list(range(self.trainer_gpus)),
                       "rollout": list(range(self.trainer_gpus, self.trainer_gpus + self.rollout_gpus)),
                       "standby": []}
            if {k: list(v) for k, v in self.bundle_map.items()} != leading:
                raise ValueError("verl 岛只接前导布局（训练池在前，rollout 在后），不接其他 bundle map")
        return nodes, per
