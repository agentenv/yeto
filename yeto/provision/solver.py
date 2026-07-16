"""Provider-neutral deterministic single-request/single-response solver.

This is not an ILP. It is a bounded O(number of curated offerings) filter and
stable rank over the pinned :class:`SupplySnapshotV1`. With one curated
offering per enabled provider the expected candidate set is at most two.

It never imports or calls ``yeto.shape``, ILP, model memory/MFU, AWS quota,
placement score, RunPod stock, boto, Sky, learner, launcher, head, or syncer
code, and it never re-derives identity: the AE-owned ``plan_item_id`` is
validated by the Task 10 contracts and echoed unchanged. There is no
automatic re-solve and no multi-attempt path.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from .contracts import (
    ComputeLaunchSpecV1,
    ContractError,
    FleetPlanItemV1,
    FleetPlanV1,
    SolveProvisionRequestV1,
    SupplyOfferingV1,
    assert_fresh_for_solve,
)

__all__ = [
    "NoFeasibleSupplyError",
    "matches_launch_spec",
    "selection_key",
    "solve_provision",
]

# available sorts before AWS unknown; unavailable is never a candidate.
_AVAILABILITY_RANK = {"available": 0, "unknown": 1}


class NoFeasibleSupplyError(ContractError):
    """No fresh offering satisfies the exact launch constraints."""

    def __init__(self, message: str = "no_feasible_supply"):
        super().__init__("no_feasible_supply", message, retryable=False)


def matches_launch_spec(offering: SupplyOfferingV1, launch: ComputeLaunchSpecV1) -> bool:
    """Exact, time-independent feasibility of one offering for one launch."""
    if not offering.create_ready:
        return False
    if offering.availability == "unavailable":
        return False
    # AWS readiness cannot promise capacity, so `unknown` stays selectable
    # only for AWS; unknown Verda supply fails closed.
    if offering.availability == "unknown" and offering.provider != "aws":
        return False
    if offering.max_count < launch.nodes:
        return False
    if launch.accelerator is not None and offering.accelerator != launch.accelerator:
        return False
    if launch.min_accelerators_per_node is not None and offering.accelerators_per_node < launch.min_accelerators_per_node:
        return False
    if launch.min_gpu_memory_gb is not None and offering.gpu_memory_gb < launch.min_gpu_memory_gb:
        return False
    if launch.allowed_providers and offering.provider not in launch.allowed_providers:
        return False
    if launch.allowed_regions and offering.region not in launch.allowed_regions:
        return False
    if not launch.required_topology_capabilities.issubset(offering.topology_capabilities):
        return False
    if launch.max_hourly_price_usd is not None:
        # The cap is explicitly the total Fleet hourly price.
        if launch.nodes * offering.hourly_price_usd > launch.max_hourly_price_usd:
            return False
    return True


def selection_key(
    offering: SupplyOfferingV1, nodes: int
) -> tuple[int, Decimal, str, str, str]:
    """Exact deterministic rank: two same-shape exact IDs never collide."""
    return (
        _AVAILABILITY_RANK[offering.availability],
        nodes * offering.hourly_price_usd,
        offering.provider,
        offering.region,
        offering.offering_id,
    )


def solve_provision(request: SolveProvisionRequestV1, now: datetime) -> FleetPlanV1:
    """Solve one request into exactly one exact homogeneous FleetPlan item."""
    assert_fresh_for_solve(request.supply, now)
    launch = request.launch
    candidates = [
        offering
        for offering in request.supply.offerings
        if offering.valid_until > now and matches_launch_spec(offering, launch)
    ]
    if not candidates:
        raise NoFeasibleSupplyError()
    selected = min(candidates, key=lambda offering: selection_key(offering, launch.nodes))
    return FleetPlanV1(
        schema_version=1,
        catalog_snapshot_id=request.supply.snapshot_id,
        catalog_content_etag=request.supply.content_etag,
        items=(
            FleetPlanItemV1(
                plan_item_id=request.plan_item_id,
                offering_id=selected.offering_id,
                nodes=launch.nodes,
            ),
        ),
    )
