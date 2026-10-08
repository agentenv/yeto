"""Backend-neutral trainer resharding plan (yeto-framework-decoupling 2.2). Import-light.

A DP change keeps TP/PP/CP/EP, the global batch size and the micro-batch
size fixed. The core builds the plan; whether a backend can execute it is
asked through the trainer group's optional ``reshard_problems`` method
(:class:`~yeto.rl.engine.ports.CuttableTrainer`); the Miles checks live in
``miles_adapter.reshard``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReshardPlan:
    source: Mapping[str, int]  # trainer_layout(): world/tp/pp/cp/ep/dp
    target: Mapping[str, int]
    global_batch_size: int
    micro_batch_size: int
    rng_policy: str = "keep_on_dp_change"

    @property
    def dp_changes(self) -> bool:
        return int(self.source["dp"]) != int(self.target["dp"])

    def to_dict(self) -> dict[str, Any]:
        return {"source": dict(self.source), "target": dict(self.target),
                "global_batch_size": self.global_batch_size, "micro_batch_size": self.micro_batch_size,
                "rng_policy": self.rng_policy}
