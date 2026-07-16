"""Strict tests for the exact provision-only AgentEnv Compute V1 contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from yeto.provision.contracts import (
    ComputeLaunchSpecV1,
    ContractError,
    FleetPlanItemV1,
    FleetPlanV1,
    ProcessErrorV1,
    SolveProvisionRequestV1,
    SupplyExpiredError,
    SupplySnapshotV1,
    assert_fresh_for_solve,
    dump_envelope,
    error_envelope,
    parse_solve_envelope,
    success_envelope,
    validate_plan_item_id,
)

FIXTURES = Path(__file__).parent / "fixtures" / "agentenv_compute_v1"

SNAPSHOT_ID = "33333333-3333-4333-8333-333333333333"
CONTENT_ETAG = "a" * 64
PLAN_ITEM_ID = "ae-0123456789abcdef0123-000"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def two_offering_supply() -> dict:
    return load_fixture("supply-snapshot.two-offerings.local.json")


def valid_fleet_plan_dict() -> dict:
    return {
        "schemaVersion": 1,
        "catalogSnapshotId": SNAPSHOT_ID,
        "catalogContentEtag": CONTENT_ETAG,
        "items": [
            {
                "planItemId": PLAN_ITEM_ID,
                "offeringId": "ae:aws:h100:a",
                "nodes": 2,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Pinned AE fixture manifest
# ---------------------------------------------------------------------------


def test_pinned_fixtures_match_canonical_ae_manifest():
    manifest = {}
    for line in (FIXTURES / "manifest.sha256").read_text().splitlines():
        digest, name = line.split()
        manifest[name] = digest
    copied = sorted(p.name for p in FIXTURES.glob("*.json") if not p.name.endswith(".local.json"))
    assert copied == [
        "fleet-plan.valid.json",
        "solve-error.valid.json",
        "solve-request.valid.json",
        "solve-success.valid.json",
        "supply-snapshot.valid.json",
        "unknown-field.invalid.json",
    ]
    for name in copied:
        actual = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert actual == manifest[name], f"fixture drift for {name}; requires a versioned contract change"


# ---------------------------------------------------------------------------
# FleetPlanV1
# ---------------------------------------------------------------------------


def test_fleet_plan_roundtrip_exact():
    plan = FleetPlanV1.from_dict(
        {
            "schemaVersion": 1,
            "catalogSnapshotId": SNAPSHOT_ID,
            "catalogContentEtag": CONTENT_ETAG,
            "items": [
                {
                    "planItemId": "ae-0123456789abcdef0123-000",
                    "offeringId": "ae:aws:h100:a",
                    "nodes": 2,
                }
            ],
        }
    )
    assert FleetPlanV1.from_json(plan.to_json()) == plan
    assert plan.items[0].offering_id == "ae:aws:h100:a"


def test_fleet_plan_fixture_parses():
    plan = FleetPlanV1.from_dict(load_fixture("fleet-plan.valid.json"))
    assert plan.items[0].plan_item_id == PLAN_ITEM_ID
    assert plan.items[0].nodes == 2


def test_solve_success_fixture_fleet_plan_parses():
    fixture = load_fixture("solve-success.valid.json")
    plan = FleetPlanV1.from_dict(fixture["fleetPlan"])
    assert plan.items[0].plan_item_id == PLAN_ITEM_ID


def test_fleet_plan_unknown_field_fails():
    data = valid_fleet_plan_dict()
    data["provider"] = "aws"
    with pytest.raises(ContractError) as err:
        FleetPlanV1.from_dict(data)
    assert err.value.code == "unknown_field"


def test_fleet_plan_item_count_other_than_one_fails():
    data = valid_fleet_plan_dict()
    data["items"] = []
    with pytest.raises(ContractError) as err:
        FleetPlanV1.from_dict(data)
    assert err.value.code == "invalid_item_count"

    data = valid_fleet_plan_dict()
    data["items"] = data["items"] * 2
    with pytest.raises(ContractError) as err:
        FleetPlanV1.from_dict(data)
    assert err.value.code == "invalid_item_count"


@pytest.mark.parametrize("version", [0, 2, "1", 1.0, True, None])
def test_fleet_plan_bad_schema_version_fails(version):
    data = valid_fleet_plan_dict()
    data["schemaVersion"] = version
    with pytest.raises(ContractError):
        FleetPlanV1.from_dict(data)


@pytest.mark.parametrize("nodes", [0, -1, "2", 2.0, True, None, 2**53 + 1])
def test_fleet_plan_item_bad_nodes_fails(nodes):
    data = valid_fleet_plan_dict()
    data["items"][0]["nodes"] = nodes
    with pytest.raises(ContractError):
        FleetPlanV1.from_dict(data)


@pytest.mark.parametrize(
    "plan_item_id",
    [
        "",
        "ae-0123456789abcdef0123",  # missing item suffix
        "ae-0123456789abcdef01-000",  # 18 hex, too short
        "ae-0123456789ABCDEF0123-000",  # uppercase hex
        "ae-0123456789abcdef0123-001",  # AE derives exactly -000
        "xx-0123456789abcdef0123-000",
        "ae-0123456789abcdef0123-000 ",
        123,
        None,
    ],
)
def test_malformed_plan_item_id_fails(plan_item_id):
    with pytest.raises(ContractError) as err:
        validate_plan_item_id(plan_item_id)
    assert err.value.code == "invalid_plan_item_id"


def test_fleet_plan_parsing_is_time_independent_after_supply_expiry():
    """A persisted FleetPlan stays parseable after its offering expired."""
    supply = SupplySnapshotV1.from_dict(load_fixture("supply-snapshot.valid.json"))
    after_expiry = datetime(2030, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(SupplyExpiredError):
        assert_fresh_for_solve(supply, after_expiry)
    plan = FleetPlanV1.from_dict(load_fixture("fleet-plan.valid.json"))
    assert plan.items[0].offering_id == supply.offerings[0].offering_id
    assert FleetPlanV1.from_json(plan.to_json()) == plan


# ---------------------------------------------------------------------------
# SupplySnapshotV1 / SupplyOfferingV1
# ---------------------------------------------------------------------------


def test_supply_snapshot_fixture_parses_exactly():
    supply = SupplySnapshotV1.from_dict(load_fixture("supply-snapshot.valid.json"))
    assert supply.snapshot_id == "snap-1"
    offering = supply.offerings[0]
    assert offering.offering_id == "ae:aws:g5-xlarge:0123456789abcdef01234567"
    assert offering.provider == "aws"
    assert offering.market == "on_demand"
    assert offering.ssh_user == "ubuntu"
    assert offering.hourly_price_usd == Decimal("1.50")
    assert offering.topology_capabilities == frozenset({"nvlink"})
    assert offering.valid_until == datetime(2026, 7, 15, 0, 30, tzinfo=timezone.utc)
    assert offering.availability == "unknown"
    assert SupplySnapshotV1.from_json(supply.to_json()) == supply


def test_supply_unknown_field_fixture_fails():
    with pytest.raises(ContractError) as err:
        SupplySnapshotV1.from_dict(load_fixture("unknown-field.invalid.json"))
    assert err.value.code == "unknown_field"


def test_two_same_shape_offerings_with_different_exact_ids():
    supply = SupplySnapshotV1.from_dict(two_offering_supply())
    a, b = supply.offerings
    assert a.offering_id == "ae:aws:h100:exact-a"
    assert b.offering_id == "ae:verda:h100:exact-b"
    assert a.offering_id != b.offering_id
    assert (a.accelerator, a.accelerators_per_node, a.gpu_memory_gb) == (
        b.accelerator,
        b.accelerators_per_node,
        b.gpu_memory_gb,
    )


def test_duplicate_offering_ids_fail():
    data = two_offering_supply()
    data["offerings"][1]["offeringId"] = data["offerings"][0]["offeringId"]
    with pytest.raises(ContractError) as err:
        SupplySnapshotV1.from_dict(data)
    assert err.value.code == "duplicate_offering_id"


def test_supply_parsing_computes_no_implicit_defaults():
    data = two_offering_supply()
    for field in ("sshUser", "createReady", "availability", "maxCount", "provider", "market"):
        broken = copy.deepcopy(data)
        del broken["offerings"][0][field]
        with pytest.raises(ContractError) as err:
            SupplySnapshotV1.from_dict(broken)
        assert err.value.code == "missing_field"


@pytest.mark.parametrize("max_count", [-1, "3", 3.0, True, None, 2**53 + 1])
def test_negative_or_non_integer_max_count_fails(max_count):
    data = two_offering_supply()
    data["offerings"][0]["maxCount"] = max_count
    with pytest.raises(ContractError):
        SupplySnapshotV1.from_dict(data)


def test_zero_max_count_parses():
    data = two_offering_supply()
    data["offerings"][0]["maxCount"] = 0
    supply = SupplySnapshotV1.from_dict(data)
    assert supply.offerings[0].max_count == 0


@pytest.mark.parametrize("availability", ["available", "unavailable", "unknown"])
def test_all_availability_values_parse(availability):
    data = two_offering_supply()
    data["offerings"][0]["availability"] = availability
    supply = SupplySnapshotV1.from_dict(data)
    assert supply.offerings[0].availability == availability


def test_invalid_availability_fails():
    data = two_offering_supply()
    data["offerings"][0]["availability"] = "maybe"
    with pytest.raises(ContractError):
        SupplySnapshotV1.from_dict(data)


def test_create_ready_false_parses():
    data = two_offering_supply()
    data["offerings"][0]["createReady"] = False
    supply = SupplySnapshotV1.from_dict(data)
    assert supply.offerings[0].create_ready is False


@pytest.mark.parametrize("price", ["-1.50", "1.5.0", "1e2", ".5", "1.", "", " 1.50", "0x10", 1.5, Decimal("1.5"), None])
def test_invalid_decimal_price_fails(price):
    data = two_offering_supply()
    data["offerings"][0]["hourlyPriceUsd"] = price
    with pytest.raises(ContractError):
        SupplySnapshotV1.from_dict(data)


def test_decimal_price_never_rounds_through_float():
    data = two_offering_supply()
    data["offerings"][0]["hourlyPriceUsd"] = "0.1000000000000000000000000001"
    supply = SupplySnapshotV1.from_dict(data)
    assert supply.offerings[0].hourly_price_usd == Decimal("0.1000000000000000000000000001")
    assert str(supply.offerings[0].hourly_price_usd) != str(float("0.1000000000000000000000000001"))


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-07-15T00:30:00",  # naive
        "2026-07-15T00:30:00+02:00",  # not UTC
        "2026-07-15",
        "not-a-date",
        "",
        1721000000,
        None,
    ],
)
def test_invalid_utc_timestamp_fails(stamp):
    data = two_offering_supply()
    data["offerings"][0]["validUntil"] = stamp
    with pytest.raises(ContractError):
        SupplySnapshotV1.from_dict(data)


def test_bigint_like_values_are_rejected_not_coerced():
    data = two_offering_supply()
    data["offerings"][0]["vcpus"] = 2**63
    with pytest.raises(ContractError):
        SupplySnapshotV1.from_dict(data)


# ---------------------------------------------------------------------------
# assert_fresh_for_solve
# ---------------------------------------------------------------------------


def test_assert_fresh_for_solve_missing_supply():
    supply = SupplySnapshotV1.from_dict(load_fixture("solve-request.valid.json")["supply"])
    with pytest.raises(SupplyExpiredError) as err:
        assert_fresh_for_solve(supply, datetime(2026, 7, 15, tzinfo=timezone.utc))
    assert err.value.code == "missing_supply"


def test_assert_fresh_for_solve_expired_supply():
    supply = SupplySnapshotV1.from_dict(two_offering_supply())
    at_expiry = supply.offerings[0].valid_until
    with pytest.raises(SupplyExpiredError) as err:
        assert_fresh_for_solve(supply, at_expiry)
    assert err.value.code == "supply_expired"


def test_assert_fresh_for_solve_accepts_fresh_supply():
    supply = SupplySnapshotV1.from_dict(two_offering_supply())
    before_expiry = datetime(2026, 7, 15, 0, 15, tzinfo=timezone.utc)
    assert assert_fresh_for_solve(supply, before_expiry) is None


# ---------------------------------------------------------------------------
# ComputeLaunchSpecV1
# ---------------------------------------------------------------------------


def test_launch_spec_minimal_fixture_parses():
    launch = ComputeLaunchSpecV1.from_dict(load_fixture("solve-request.valid.json")["launch"])
    assert launch.nodes == 2
    assert launch.accelerator is None
    assert launch.allowed_providers == ()
    assert launch.allowed_regions == ()
    assert launch.required_topology_capabilities == frozenset()
    assert launch.max_hourly_price_usd is None
    assert ComputeLaunchSpecV1.from_json(launch.to_json()) == launch


def test_launch_spec_full_roundtrip_and_normalized_allowlists():
    launch = ComputeLaunchSpecV1.from_dict(
        {
            "schemaVersion": 1,
            "nodes": 2,
            "accelerator": "H100",
            "minAcceleratorsPerNode": 8,
            "minGpuMemoryGb": 80,
            "allowedProviders": ["verda", "aws", "verda"],
            "allowedRegions": ["us-east-1", "eu-north-1", "us-east-1"],
            "requiredTopologyCapabilities": ["nvlink"],
            "maxHourlyPriceUsd": "30.00",
        }
    )
    assert launch.allowed_providers == ("aws", "verda")
    assert launch.allowed_regions == ("eu-north-1", "us-east-1")
    assert launch.max_hourly_price_usd == Decimal("30.00")
    assert ComputeLaunchSpecV1.from_json(launch.to_json()) == launch


def test_launch_spec_contains_no_workload_fields():
    for forbidden in ("image", "command", "env", "workdir", "modelPath"):
        with pytest.raises(ContractError) as err:
            ComputeLaunchSpecV1.from_dict({"schemaVersion": 1, "nodes": 1, forbidden: "x"})
        assert err.value.code == "unknown_field"


@pytest.mark.parametrize("nodes", [0, -3, "1", 1.0, True])
def test_launch_spec_nonpositive_nodes_fails(nodes):
    with pytest.raises(ContractError):
        ComputeLaunchSpecV1.from_dict({"schemaVersion": 1, "nodes": nodes})


def test_launch_spec_unknown_provider_fails():
    with pytest.raises(ContractError):
        ComputeLaunchSpecV1.from_dict({"schemaVersion": 1, "nodes": 1, "allowedProviders": ["gcp"]})


# ---------------------------------------------------------------------------
# SolveProvisionRequestV1
# ---------------------------------------------------------------------------


def test_solve_request_fixture_parses_and_echoes_identity():
    request = SolveProvisionRequestV1.from_dict(load_fixture("solve-request.valid.json"))
    assert request.plan_item_id == PLAN_ITEM_ID
    assert request.launch.nodes == 2
    assert request.supply.offerings == ()
    assert SolveProvisionRequestV1.from_json(request.to_json()) == request


def test_solve_request_plan_item_id_is_required():
    data = load_fixture("solve-request.valid.json")
    del data["planItemId"]
    with pytest.raises(ContractError) as err:
        SolveProvisionRequestV1.from_dict(data)
    assert err.value.code == "missing_field"


def test_solve_request_rejects_non_sky_safe_plan_item_id():
    data = load_fixture("solve-request.valid.json")
    data["planItemId"] = "Not A Sky Safe Name!"
    with pytest.raises(ContractError) as err:
        SolveProvisionRequestV1.from_dict(data)
    assert err.value.code == "invalid_plan_item_id"


def test_solve_request_contains_no_orchestration_id():
    data = load_fixture("solve-request.valid.json")
    data["orchestrationId"] = "11111111-1111-4111-8111-111111111111"
    with pytest.raises(ContractError) as err:
        SolveProvisionRequestV1.from_dict(data)
    assert err.value.code == "unknown_field"


# ---------------------------------------------------------------------------
# Process envelope
# ---------------------------------------------------------------------------


def test_plan_item_id_preserved_byte_for_byte_through_success_envelope():
    request = SolveProvisionRequestV1.from_dict(load_fixture("solve-request.valid.json"))
    plan = FleetPlanV1(
        schema_version=1,
        catalog_snapshot_id=request.supply.snapshot_id,
        catalog_content_etag=request.supply.content_etag,
        items=(
            FleetPlanItemV1(
                plan_item_id=request.plan_item_id,
                offering_id="ae:aws:h100:a",
                nodes=request.launch.nodes,
            ),
        ),
    )
    text = dump_envelope(success_envelope(plan))
    raw = json.loads(text)
    assert raw == {"schemaVersion": 1, "ok": True, "value": plan.to_dict()}
    assert raw["value"]["items"][0]["planItemId"].encode() == request.plan_item_id.encode()
    parsed = parse_solve_envelope(text)
    assert parsed == plan


def test_error_envelope_roundtrip_and_bounded_message():
    error = ContractError("no_feasible_supply", "x" * 10_000 + "\x00\x1b[31m").to_process_error()
    text = dump_envelope(error_envelope(error))
    parsed = parse_solve_envelope(text)
    assert isinstance(parsed, ProcessErrorV1)
    assert parsed.code == "no_feasible_supply"
    assert parsed.retryable is False
    assert len(parsed.message) <= 512
    assert "\x00" not in parsed.message and "\x1b" not in parsed.message


def test_envelope_rejects_unknown_fields_and_bad_versions():
    with pytest.raises(ContractError):
        parse_solve_envelope(json.dumps({"schemaVersion": 2, "ok": False, "error": {}}))
    with pytest.raises(ContractError):
        parse_solve_envelope(json.dumps({"schemaVersion": 1, "ok": True, "value": {}, "extra": 1}))
    with pytest.raises(ContractError):
        parse_solve_envelope("not json")


# ---------------------------------------------------------------------------
# Dependency hygiene
# ---------------------------------------------------------------------------


def test_contracts_import_no_provider_sky_or_training_modules():
    forbidden = (
        "torch",
        "transformers",
        "boto3",
        "botocore",
        "sky",
        "yeto.cli",
        "yeto.shape",
        "yeto.launcher",
        "yeto.learner",
        "yeto.protocol",
        "yeto.models",
    )
    code = (
        "import sys\n"
        "import yeto.provision.contracts\n"
        f"forbidden = {forbidden!r}\n"
        "hits = [name for name in forbidden if name in sys.modules]\n"
        "assert not hits, f'forbidden imports: {hits}'\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
