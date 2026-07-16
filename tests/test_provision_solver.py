"""Deterministic selection tests for the provider-neutral provision solver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from yeto.provision.contracts import (
    ComputeLaunchSpecV1,
    FleetPlanItemV1,
    SolveProvisionRequestV1,
    SupplyExpiredError,
)
from yeto.provision.solver import (
    NoFeasibleSupplyError,
    matches_launch_spec,
    selection_key,
    solve_provision,
)

FIXTURES = Path(__file__).parent / "fixtures" / "agentenv_compute_v1"

REQUEST_PLAN_ITEM_ID = "ae-0123456789abcdef0123-000"
NOW = datetime(2026, 7, 15, 0, 15, 0, tzinfo=timezone.utc)
FRESH = "2026-07-15T00:30:00.000Z"
EXPIRED = "2026-07-15T00:10:00.000Z"


def make_offering(**overrides) -> dict:
    offering = {
        "offeringId": "ae:aws:h100:exact-a",
        "region": "us-east-1",
        "market": "on_demand",
        "accelerator": "H100",
        "acceleratorsPerNode": 8,
        "gpuMemoryGb": 80,
        "vcpus": 192,
        "topologyCapabilities": ["nvlink"],
        "sshUser": "ubuntu",
        "hourlyPriceUsd": "12.00",
        "maxCount": 4,
        "validUntil": FRESH,
        "provider": "aws",
        "createReady": True,
        "availability": "available",
    }
    offering.update(overrides)
    return offering


def make_request(offerings: list[dict], launch: dict | None = None) -> SolveProvisionRequestV1:
    return SolveProvisionRequestV1.from_dict(
        {
            "schemaVersion": 1,
            "planItemId": REQUEST_PLAN_ITEM_ID,
            "launch": {"schemaVersion": 1, "nodes": 2, **(launch or {})},
            "supply": {
                "schemaVersion": 1,
                "snapshotId": "snap-2",
                "contentEtag": "b" * 64,
                "generatedAt": "2026-07-15T00:00:00.000Z",
                "offerings": offerings,
            },
        }
    )


def solve(offerings: list[dict], launch: dict | None = None):
    return solve_provision(make_request(offerings, launch), NOW)


# ---------------------------------------------------------------------------
# Exact expected selection and determinism
# ---------------------------------------------------------------------------


def test_selects_exact_offering_and_echoes_plan_item_id():
    offerings = [
        make_offering(availability="unknown", provider="aws", region="us-east-1", offeringId="ae:aws:h100:exact-a"),
        make_offering(availability="available", provider="verda", region="eu-north-1", offeringId="ae:verda:h100:exact-b"),
    ]
    plan = solve(offerings)
    assert plan.items == (
        FleetPlanItemV1(
            plan_item_id=REQUEST_PLAN_ITEM_ID,
            offering_id="ae:verda:h100:exact-b",
            nodes=2,
        ),
    )
    assert plan.catalog_snapshot_id == "snap-2"
    assert plan.catalog_content_etag == "b" * 64


def test_result_is_stable_under_snapshot_reordering():
    offerings = [
        make_offering(offeringId="ae:aws:h100:exact-a", provider="aws", region="us-east-1", availability="unknown"),
        make_offering(offeringId="ae:verda:h100:exact-b", provider="verda", region="eu-north-1", availability="available"),
        make_offering(offeringId="ae:aws:h100:exact-c", provider="aws", region="us-west-2", availability="available", hourlyPriceUsd="12.50"),
    ]
    baseline = solve(offerings)
    assert baseline == solve(list(reversed(offerings)))
    assert baseline == solve([offerings[1], offerings[2], offerings[0]])
    # available beats aws-unknown; equal rank ties resolve on cheaper total.
    assert baseline.items[0].offering_id == "ae:verda:h100:exact-b"


def test_fleet_plan_uses_fixed_request_nodes():
    plan = solve([make_offering()], launch={"nodes": 3})
    assert plan.items[0].nodes == 3


# ---------------------------------------------------------------------------
# Launch-spec constraint filters
# ---------------------------------------------------------------------------


def test_provider_allowlist_filters():
    offerings = [
        make_offering(offeringId="ae:aws:h100:exact-a", provider="aws"),
        make_offering(offeringId="ae:verda:h100:exact-b", provider="verda", region="eu-north-1"),
    ]
    plan = solve(offerings, launch={"allowedProviders": ["verda"]})
    assert plan.items[0].offering_id == "ae:verda:h100:exact-b"


def test_region_allowlist_filters():
    offerings = [
        make_offering(offeringId="ae:aws:h100:exact-a", region="us-east-1"),
        make_offering(offeringId="ae:aws:h100:exact-c", region="us-west-2"),
    ]
    plan = solve(offerings, launch={"allowedRegions": ["us-west-2"]})
    assert plan.items[0].offering_id == "ae:aws:h100:exact-c"


def test_exact_accelerator_filter():
    offerings = [
        make_offering(offeringId="ae:aws:a10g:exact-a", accelerator="A10G"),
        make_offering(offeringId="ae:aws:h100:exact-c", region="us-west-2"),
    ]
    plan = solve(offerings, launch={"accelerator": "H100"})
    assert plan.items[0].offering_id == "ae:aws:h100:exact-c"
    with pytest.raises(NoFeasibleSupplyError):
        solve(offerings, launch={"accelerator": "H200"})


def test_min_accelerators_per_node_filter():
    offerings = [
        make_offering(offeringId="ae:aws:h100:small", acceleratorsPerNode=4),
        make_offering(offeringId="ae:aws:h100:big", region="us-west-2", acceleratorsPerNode=8),
    ]
    plan = solve(offerings, launch={"minAcceleratorsPerNode": 8})
    assert plan.items[0].offering_id == "ae:aws:h100:big"


def test_min_gpu_memory_filter():
    offerings = [
        make_offering(offeringId="ae:aws:a10g:exact-a", gpuMemoryGb=24),
        make_offering(offeringId="ae:aws:h100:exact-c", region="us-west-2", gpuMemoryGb=80),
    ]
    plan = solve(offerings, launch={"minGpuMemoryGb": 80})
    assert plan.items[0].offering_id == "ae:aws:h100:exact-c"


def test_required_topology_capability_subset_filter():
    offerings = [
        make_offering(offeringId="ae:aws:h100:plain", topologyCapabilities=[]),
        make_offering(
            offeringId="ae:aws:h100:fabric",
            region="us-west-2",
            topologyCapabilities=["efa", "nvlink"],
        ),
    ]
    plan = solve(offerings, launch={"requiredTopologyCapabilities": ["nvlink", "efa"]})
    assert plan.items[0].offering_id == "ae:aws:h100:fabric"


def test_max_count_must_cover_requested_nodes():
    offerings = [
        make_offering(offeringId="ae:aws:h100:tight", maxCount=1),
        make_offering(offeringId="ae:aws:h100:roomy", region="us-west-2", maxCount=2),
    ]
    plan = solve(offerings, launch={"nodes": 2})
    assert plan.items[0].offering_id == "ae:aws:h100:roomy"


def test_price_cap_is_total_fleet_hourly_price():
    # 2 nodes * 12.00 = 24.00 total per hour.
    offerings = [make_offering()]
    assert solve(offerings, launch={"maxHourlyPriceUsd": "24.00"}).items[0].nodes == 2
    with pytest.raises(NoFeasibleSupplyError):
        solve(offerings, launch={"maxHourlyPriceUsd": "23.99"})
    # A per-node reading (12.00 <= 13.00) must NOT make this feasible.
    with pytest.raises(NoFeasibleSupplyError):
        solve(offerings, launch={"maxHourlyPriceUsd": "13.00"})


# ---------------------------------------------------------------------------
# Supply-state filters
# ---------------------------------------------------------------------------


def test_not_create_ready_unavailable_and_zero_cap_are_filtered():
    offerings = [
        make_offering(offeringId="ae:aws:h100:draft", createReady=False),
        make_offering(offeringId="ae:aws:h100:gone", region="us-west-2", availability="unavailable"),
        make_offering(offeringId="ae:aws:h100:empty", region="eu-central-1", maxCount=0),
        make_offering(offeringId="ae:aws:h100:live", region="eu-west-1"),
    ]
    plan = solve(offerings)
    assert plan.items[0].offering_id == "ae:aws:h100:live"


def test_expired_offering_filtered_without_blocking_fresh_sibling():
    offerings = [
        make_offering(offeringId="ae:aws:h100:stale", validUntil=EXPIRED, hourlyPriceUsd="1.00"),
        make_offering(offeringId="ae:verda:h100:exact-b", provider="verda", region="eu-north-1"),
    ]
    plan = solve(offerings)
    assert plan.items[0].offering_id == "ae:verda:h100:exact-b"


def test_fully_expired_supply_fails_as_supply_expired():
    with pytest.raises(SupplyExpiredError) as err:
        solve([make_offering(validUntil=EXPIRED)])
    assert err.value.code == "supply_expired"


def test_missing_supply_fails_closed():
    with pytest.raises(SupplyExpiredError) as err:
        solve([])
    assert err.value.code == "missing_supply"


def test_unknown_availability_selectable_only_for_aws():
    offerings = [
        make_offering(offeringId="ae:verda:h100:mist", provider="verda", region="eu-north-1", availability="unknown"),
        make_offering(offeringId="ae:aws:h100:haze", availability="unknown"),
    ]
    plan = solve(offerings)
    assert plan.items[0].offering_id == "ae:aws:h100:haze"

    with pytest.raises(NoFeasibleSupplyError):
        solve([make_offering(offeringId="ae:verda:h100:mist", provider="verda", region="eu-north-1", availability="unknown")])


def test_aws_unknown_sorts_after_equivalent_available():
    equivalent_available = make_offering(
        offeringId="ae:verda:h100:exact-b", provider="verda", region="eu-north-1", availability="available"
    )
    aws_unknown = make_offering(offeringId="ae:aws:h100:exact-a", availability="unknown", hourlyPriceUsd="1.00")
    # Even though AWS unknown is far cheaper, available wins on rank.
    plan = solve([aws_unknown, equivalent_available])
    assert plan.items[0].offering_id == "ae:verda:h100:exact-b"


# ---------------------------------------------------------------------------
# Exact selection key
# ---------------------------------------------------------------------------


def test_selection_key_is_exact_tuple():
    from decimal import Decimal

    request = make_request([make_offering()])
    offering = request.supply.offerings[0]
    assert selection_key(offering, 2) == (
        0,
        Decimal("24.00"),
        "aws",
        "us-east-1",
        "ae:aws:h100:exact-a",
    )
    unknown_request = make_request([make_offering(availability="unknown")])
    assert selection_key(unknown_request.supply.offerings[0], 2)[0] == 1


def test_ranking_prefers_cheaper_total_then_provider_region_offering_id():
    cheap = make_offering(offeringId="ae:verda:h100:cheap", provider="verda", region="eu-north-1", hourlyPriceUsd="11.00")
    pricey = make_offering(offeringId="ae:aws:h100:pricey", hourlyPriceUsd="12.00")
    assert solve([pricey, cheap]).items[0].offering_id == "ae:verda:h100:cheap"

    # Equal totals: provider tiebreak (aws < verda).
    aws = make_offering(offeringId="ae:aws:h100:tie", provider="aws")
    verda = make_offering(offeringId="ae:verda:h100:tie", provider="verda", region="eu-north-1")
    assert solve([verda, aws]).items[0].offering_id == "ae:aws:h100:tie"

    # Equal totals + provider: region tiebreak.
    east = make_offering(offeringId="ae:aws:h100:east", region="us-east-1")
    west = make_offering(offeringId="ae:aws:h100:west", region="us-west-2")
    assert solve([west, east]).items[0].offering_id == "ae:aws:h100:east"


def test_two_same_shape_exact_ids_never_collide():
    first = make_offering(offeringId="ae:aws:h100:exact-a")
    twin = make_offering(offeringId="ae:aws:h100:exact-z")
    plan_ab = solve([first, twin])
    plan_ba = solve([twin, first])
    assert plan_ab == plan_ba
    assert plan_ab.items[0].offering_id == "ae:aws:h100:exact-a"
    assert selection_key(
        make_request([first]).supply.offerings[0], 2
    ) != selection_key(make_request([twin]).supply.offerings[0], 2)


def test_no_candidate_returns_typed_no_feasible_supply():
    with pytest.raises(NoFeasibleSupplyError) as err:
        solve([make_offering()], launch={"nodes": 8})
    assert err.value.code == "no_feasible_supply"
    assert err.value.retryable is False


# ---------------------------------------------------------------------------
# matches_launch_spec is pure feasibility (no ranking, no time)
# ---------------------------------------------------------------------------


def test_matches_launch_spec_direct():
    request = make_request([make_offering()], launch={"accelerator": "H100"})
    offering = request.supply.offerings[0]
    assert matches_launch_spec(offering, request.launch) is True
    mismatch = make_request([make_offering()], launch={"accelerator": "H200"})
    assert matches_launch_spec(offering, mismatch.launch) is False


def test_pinned_two_offering_fixture_solves_deterministically():
    data = json.loads((FIXTURES / "supply-snapshot.two-offerings.local.json").read_text())
    request = SolveProvisionRequestV1.from_dict(
        {
            "schemaVersion": 1,
            "planItemId": REQUEST_PLAN_ITEM_ID,
            "launch": {"schemaVersion": 1, "nodes": 2},
            "supply": data,
        }
    )
    plan = solve_provision(request, NOW)
    assert plan.items == (
        FleetPlanItemV1(
            plan_item_id=REQUEST_PLAN_ITEM_ID,
            offering_id="ae:verda:h100:exact-b",
            nodes=2,
        ),
    )
