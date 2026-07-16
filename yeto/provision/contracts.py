"""Exact provision-only wire contracts for the AgentEnv Compute V1 bundle.

The canonical wire authority is the frozen AE contract bundle
(``docs/contracts/compute/v1/compute-v1.schema.json`` in the AgentEnv
monorepo); the fixtures pinned under ``tests/fixtures/agentenv_compute_v1``
must match it byte-for-byte.

Rules implemented here:

- Wire keys are camelCase; Python attributes are snake_case.
- Parsing is strict: unknown keys are rejected, no field is defaulted
  implicitly, booleans are never accepted where integers are expected, and
  numbers outside the JSON-safe integer range are rejected rather than
  coerced.
- Prices parse via :class:`decimal.Decimal` from canonical non-negative
  decimal strings only; they never round through ``float``.
- Timestamps are RFC 3339 UTC instants (``Z`` or ``+00:00``); naive or
  non-UTC timestamps are rejected.
- Parsing is time-independent. Freshness is enforced only by
  :func:`assert_fresh_for_solve`, which only new solve calls; status/delete
  recovery paths may parse an expired persisted plan or snapshot.

This module has no dependency on training shape, model fit, MFU, head,
learner, launcher, syncer, provider SDK, or Sky code.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Union

__all__ = [
    "ContractError",
    "SupplyExpiredError",
    "ComputeLaunchSpecV1",
    "SupplyOfferingV1",
    "SupplySnapshotV1",
    "SolveProvisionRequestV1",
    "FleetPlanItemV1",
    "FleetPlanV1",
    "ProcessErrorV1",
    "PLAN_ITEM_ID",
    "validate_plan_item_id",
    "assert_fresh_for_solve",
    "success_envelope",
    "error_envelope",
    "dump_envelope",
    "parse_solve_envelope",
]

SCHEMA_VERSION = 1

PROVIDERS = ("aws", "verda")
AVAILABILITIES = ("available", "unavailable", "unknown")
MARKET = "on_demand"

# AE derives the ID as ``ae-<first 20 hex of sha256(orchestrationId)>-000``
# before solve. Yeto only validates the Sky-name-safe shape and echoes the
# value unchanged; it never re-derives or hashes identity.
PLAN_ITEM_ID = re.compile(r"ae-[0-9a-f]{20}-000")

# Largest integer exactly representable in JSON interoperability (2^53 - 1).
_MAX_SAFE_INT = 9007199254740991

_MAX_MESSAGE_CHARS = 512

_DECIMAL_RE = re.compile(r"[0-9]+(\.[0-9]+)?")

Provider = Literal["aws", "verda"]
Availability = Literal["available", "unavailable", "unknown"]


class ContractError(ValueError):
    """A typed, handled contract violation (maps to a process error)."""

    def __init__(self, code: str, message: str | None = None, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        self.message = _bound_message(message if message is not None else code)
        super().__init__(f"{self.code}: {self.message}")

    def to_process_error(self) -> "ProcessErrorV1":
        return ProcessErrorV1(code=self.code, retryable=self.retryable, message=self.message)


class SupplyExpiredError(ContractError):
    """Raised by :func:`assert_fresh_for_solve` for missing/expired supply."""


def _bound_message(message: str) -> str:
    sanitized = "".join(ch for ch in str(message) if ch.isprintable())
    return sanitized[:_MAX_MESSAGE_CHARS]


# ---------------------------------------------------------------------------
# Strict scalar parsing helpers
# ---------------------------------------------------------------------------


def _require_object(data: Any, contract: str) -> dict:
    if not isinstance(data, dict):
        raise ContractError("invalid_contract", f"{contract} must be a JSON object")
    return data


def _check_keys(data: dict, contract: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    for key in data:
        if not isinstance(key, str) or (key not in required and key not in optional):
            raise ContractError("unknown_field", f"{contract} has unknown field {key!r}")
    for key in required:
        if key not in data:
            raise ContractError("missing_field", f"{contract} requires field {key!r}")


def _schema_version(data: dict, contract: str) -> Literal[1]:
    value = data["schemaVersion"]
    if isinstance(value, bool) or value != SCHEMA_VERSION or not isinstance(value, int):
        raise ContractError("invalid_schema_version", f"{contract}.schemaVersion must be {SCHEMA_VERSION}")
    return SCHEMA_VERSION


def _parse_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError("invalid_field", f"{field} must be a non-empty trimmed string")
    return value


def _parse_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError("invalid_field", f"{field} must be a boolean")
    return value


def _parse_int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("invalid_field", f"{field} must be an integer")
    if value > _MAX_SAFE_INT:
        raise ContractError("invalid_field", f"{field} exceeds the JSON-safe integer range")
    if value < minimum:
        raise ContractError("invalid_field", f"{field} must be >= {minimum}")
    return value


def _parse_number(value: Any, field: str, *, minimum: float) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("invalid_field", f"{field} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("invalid_field", f"{field} must be finite")
    if isinstance(value, int) and value > _MAX_SAFE_INT:
        raise ContractError("invalid_field", f"{field} exceeds the JSON-safe integer range")
    if value < minimum:
        raise ContractError("invalid_field", f"{field} must be >= {minimum}")
    return value


def _parse_decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise ContractError("invalid_field", f"{field} must be a canonical non-negative decimal string")
    return Decimal(value)


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError("invalid_field", f"{field} must be an RFC 3339 UTC timestamp string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ContractError("invalid_field", f"{field} is not a valid RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ContractError("invalid_field", f"{field} must be an explicit UTC instant")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    timespec = "microseconds" if value.microsecond % 1000 else "milliseconds"
    return value.astimezone(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def _parse_enum(value: Any, field: str, allowed: tuple[str, ...]) -> str:
    parsed = _parse_str(value, field)
    if parsed not in allowed:
        raise ContractError("invalid_field", f"{field} must be one of {allowed}")
    return parsed


def _parse_str_array(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError("invalid_field", f"{field} must be an array of strings")
    return tuple(_parse_str(item, f"{field}[]") for item in value)


def _normalized_allowlist(value: Any, field: str) -> tuple[str, ...]:
    """Trim, deduplicate, and sort a wire allowlist."""
    return tuple(sorted({item.strip() for item in _parse_str_array(value, field)}))


def _loads(text: Union[str, bytes], contract: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        raise ContractError("invalid_json", f"{contract} is not valid JSON") from None


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def validate_plan_item_id(value: Any) -> str:
    if not isinstance(value, str) or not PLAN_ITEM_ID.fullmatch(value):
        raise ContractError("invalid_plan_item_id")
    return value


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeLaunchSpecV1:
    """Provision-only launch constraints.

    Contains no image, command, environment, model/data/output, source sync,
    workdir, or user workload. ``max_hourly_price_usd`` caps the *total Fleet*
    hourly price (``nodes * offering.hourly_price_usd``).
    """

    schema_version: Literal[1]
    nodes: int
    accelerator: str | None
    min_accelerators_per_node: int | None
    min_gpu_memory_gb: Union[int, float, None]
    allowed_providers: tuple[Provider, ...]
    allowed_regions: tuple[str, ...]
    required_topology_capabilities: frozenset[str]
    max_hourly_price_usd: Decimal | None

    @classmethod
    def from_dict(cls, data: Any) -> "ComputeLaunchSpecV1":
        obj = _require_object(data, "ComputeLaunchSpecV1")
        _check_keys(
            obj,
            "ComputeLaunchSpecV1",
            required=("schemaVersion", "nodes"),
            optional=(
                "accelerator",
                "minAcceleratorsPerNode",
                "minGpuMemoryGb",
                "allowedProviders",
                "allowedRegions",
                "requiredTopologyCapabilities",
                "maxHourlyPriceUsd",
            ),
        )
        allowed_providers = _normalized_allowlist(obj.get("allowedProviders", []), "launch.allowedProviders")
        for provider in allowed_providers:
            if provider not in PROVIDERS:
                raise ContractError("invalid_field", f"launch.allowedProviders contains unknown provider {provider!r}")
        return cls(
            schema_version=_schema_version(obj, "ComputeLaunchSpecV1"),
            nodes=_parse_int(obj["nodes"], "launch.nodes", minimum=1),
            accelerator=_parse_str(obj["accelerator"], "launch.accelerator") if "accelerator" in obj else None,
            min_accelerators_per_node=(
                _parse_int(obj["minAcceleratorsPerNode"], "launch.minAcceleratorsPerNode", minimum=1)
                if "minAcceleratorsPerNode" in obj
                else None
            ),
            min_gpu_memory_gb=(
                _parse_number(obj["minGpuMemoryGb"], "launch.minGpuMemoryGb", minimum=0)
                if "minGpuMemoryGb" in obj
                else None
            ),
            allowed_providers=allowed_providers,  # type: ignore[arg-type]
            allowed_regions=_normalized_allowlist(obj.get("allowedRegions", []), "launch.allowedRegions"),
            required_topology_capabilities=frozenset(
                _parse_str_array(obj.get("requiredTopologyCapabilities", []), "launch.requiredTopologyCapabilities")
            ),
            max_hourly_price_usd=(
                _parse_decimal(obj["maxHourlyPriceUsd"], "launch.maxHourlyPriceUsd")
                if "maxHourlyPriceUsd" in obj
                else None
            ),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"schemaVersion": self.schema_version, "nodes": self.nodes}
        if self.accelerator is not None:
            out["accelerator"] = self.accelerator
        if self.min_accelerators_per_node is not None:
            out["minAcceleratorsPerNode"] = self.min_accelerators_per_node
        if self.min_gpu_memory_gb is not None:
            out["minGpuMemoryGb"] = self.min_gpu_memory_gb
        if self.allowed_providers:
            out["allowedProviders"] = list(self.allowed_providers)
        if self.allowed_regions:
            out["allowedRegions"] = list(self.allowed_regions)
        if self.required_topology_capabilities:
            out["requiredTopologyCapabilities"] = sorted(self.required_topology_capabilities)
        if self.max_hourly_price_usd is not None:
            out["maxHourlyPriceUsd"] = str(self.max_hourly_price_usd)
        return out

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "ComputeLaunchSpecV1":
        return cls.from_dict(_loads(text, "ComputeLaunchSpecV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class SupplyOfferingV1:
    """One exact curated offering.

    ``ssh_user`` is part of the frozen AE wire contract
    (``SupplyOfferingV1.sshUser`` is required by ``compute-v1.schema.json``)
    even though the planning document's sketch omitted it.
    """

    offering_id: str
    provider: Provider
    region: str
    market: Literal["on_demand"]
    accelerator: str
    accelerators_per_node: int
    gpu_memory_gb: Union[int, float]
    vcpus: int
    topology_capabilities: frozenset[str]
    ssh_user: str
    create_ready: bool
    availability: Availability
    hourly_price_usd: Decimal
    max_count: int
    valid_until: datetime

    _FIELDS = (
        "offeringId",
        "region",
        "market",
        "accelerator",
        "acceleratorsPerNode",
        "gpuMemoryGb",
        "vcpus",
        "topologyCapabilities",
        "sshUser",
        "hourlyPriceUsd",
        "maxCount",
        "validUntil",
        "provider",
        "createReady",
        "availability",
    )

    @classmethod
    def from_dict(cls, data: Any) -> "SupplyOfferingV1":
        obj = _require_object(data, "SupplyOfferingV1")
        _check_keys(obj, "SupplyOfferingV1", required=cls._FIELDS)
        market = _parse_str(obj["market"], "offering.market")
        if market != MARKET:
            raise ContractError("invalid_field", f"offering.market must be {MARKET!r}")
        return cls(
            offering_id=_parse_str(obj["offeringId"], "offering.offeringId"),
            provider=_parse_enum(obj["provider"], "offering.provider", PROVIDERS),  # type: ignore[arg-type]
            region=_parse_str(obj["region"], "offering.region"),
            market=MARKET,
            accelerator=_parse_str(obj["accelerator"], "offering.accelerator"),
            accelerators_per_node=_parse_int(obj["acceleratorsPerNode"], "offering.acceleratorsPerNode", minimum=1),
            gpu_memory_gb=_parse_number(obj["gpuMemoryGb"], "offering.gpuMemoryGb", minimum=0),
            vcpus=_parse_int(obj["vcpus"], "offering.vcpus", minimum=0),
            topology_capabilities=frozenset(
                _parse_str_array(obj["topologyCapabilities"], "offering.topologyCapabilities")
            ),
            ssh_user=_parse_str(obj["sshUser"], "offering.sshUser"),
            create_ready=_parse_bool(obj["createReady"], "offering.createReady"),
            availability=_parse_enum(obj["availability"], "offering.availability", AVAILABILITIES),  # type: ignore[arg-type]
            hourly_price_usd=_parse_decimal(obj["hourlyPriceUsd"], "offering.hourlyPriceUsd"),
            max_count=_parse_int(obj["maxCount"], "offering.maxCount", minimum=0),
            valid_until=_parse_timestamp(obj["validUntil"], "offering.validUntil"),
        )

    def to_dict(self) -> dict:
        return {
            "offeringId": self.offering_id,
            "region": self.region,
            "market": self.market,
            "accelerator": self.accelerator,
            "acceleratorsPerNode": self.accelerators_per_node,
            "gpuMemoryGb": self.gpu_memory_gb,
            "vcpus": self.vcpus,
            "topologyCapabilities": sorted(self.topology_capabilities),
            "sshUser": self.ssh_user,
            "hourlyPriceUsd": str(self.hourly_price_usd),
            "maxCount": self.max_count,
            "validUntil": _format_timestamp(self.valid_until),
            "provider": self.provider,
            "createReady": self.create_ready,
            "availability": self.availability,
        }

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "SupplyOfferingV1":
        return cls.from_dict(_loads(text, "SupplyOfferingV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class SupplySnapshotV1:
    schema_version: Literal[1]
    snapshot_id: str
    content_etag: str
    generated_at: datetime
    offerings: tuple[SupplyOfferingV1, ...]

    @classmethod
    def from_dict(cls, data: Any) -> "SupplySnapshotV1":
        obj = _require_object(data, "SupplySnapshotV1")
        _check_keys(
            obj,
            "SupplySnapshotV1",
            required=("schemaVersion", "snapshotId", "contentEtag", "generatedAt", "offerings"),
        )
        raw_offerings = obj["offerings"]
        if not isinstance(raw_offerings, list):
            raise ContractError("invalid_field", "supply.offerings must be an array")
        offerings = tuple(SupplyOfferingV1.from_dict(item) for item in raw_offerings)
        seen: set[str] = set()
        for offering in offerings:
            if offering.offering_id in seen:
                raise ContractError("duplicate_offering_id", f"duplicate offeringId {offering.offering_id!r}")
            seen.add(offering.offering_id)
        return cls(
            schema_version=_schema_version(obj, "SupplySnapshotV1"),
            snapshot_id=_parse_str(obj["snapshotId"], "supply.snapshotId"),
            content_etag=_parse_str(obj["contentEtag"], "supply.contentEtag"),
            generated_at=_parse_timestamp(obj["generatedAt"], "supply.generatedAt"),
            offerings=offerings,
        )

    def to_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "snapshotId": self.snapshot_id,
            "contentEtag": self.content_etag,
            "generatedAt": _format_timestamp(self.generated_at),
            "offerings": [offering.to_dict() for offering in self.offerings],
        }

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "SupplySnapshotV1":
        return cls.from_dict(_loads(text, "SupplySnapshotV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class SolveProvisionRequestV1:
    """One bounded solve request: AE-owned identity + launch + supply.

    The request never contains an Orchestration ID, token, fence, or path;
    ``plan_item_id`` is AE-derived and only echoed by Yeto.
    """

    schema_version: Literal[1]
    plan_item_id: str
    launch: ComputeLaunchSpecV1
    supply: SupplySnapshotV1

    @classmethod
    def from_dict(cls, data: Any) -> "SolveProvisionRequestV1":
        obj = _require_object(data, "SolveProvisionRequestV1")
        _check_keys(
            obj,
            "SolveProvisionRequestV1",
            required=("schemaVersion", "planItemId", "launch", "supply"),
        )
        return cls(
            schema_version=_schema_version(obj, "SolveProvisionRequestV1"),
            plan_item_id=validate_plan_item_id(obj["planItemId"]),
            launch=ComputeLaunchSpecV1.from_dict(obj["launch"]),
            supply=SupplySnapshotV1.from_dict(obj["supply"]),
        )

    def to_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "planItemId": self.plan_item_id,
            "launch": self.launch.to_dict(),
            "supply": self.supply.to_dict(),
        }

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "SolveProvisionRequestV1":
        return cls.from_dict(_loads(text, "SolveProvisionRequestV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class FleetPlanItemV1:
    plan_item_id: str
    offering_id: str
    nodes: int

    @classmethod
    def from_dict(cls, data: Any) -> "FleetPlanItemV1":
        obj = _require_object(data, "FleetPlanItemV1")
        _check_keys(obj, "FleetPlanItemV1", required=("planItemId", "offeringId", "nodes"))
        return cls(
            plan_item_id=validate_plan_item_id(obj["planItemId"]),
            offering_id=_parse_str(obj["offeringId"], "item.offeringId"),
            nodes=_parse_int(obj["nodes"], "item.nodes", minimum=1),
        )

    def to_dict(self) -> dict:
        return {"planItemId": self.plan_item_id, "offeringId": self.offering_id, "nodes": self.nodes}

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "FleetPlanItemV1":
        return cls.from_dict(_loads(text, "FleetPlanItemV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class FleetPlanV1:
    """The minimal exact plan: exactly one homogeneous item.

    Parsing is time-independent so an already-persisted FleetPlan stays valid
    after its selected Supply offering expires (status/delete recovery).
    """

    schema_version: Literal[1]
    catalog_snapshot_id: str
    catalog_content_etag: str
    items: tuple[FleetPlanItemV1]

    @classmethod
    def from_dict(cls, data: Any) -> "FleetPlanV1":
        obj = _require_object(data, "FleetPlanV1")
        _check_keys(
            obj,
            "FleetPlanV1",
            required=("schemaVersion", "catalogSnapshotId", "catalogContentEtag", "items"),
        )
        raw_items = obj["items"]
        if not isinstance(raw_items, list) or len(raw_items) != 1:
            raise ContractError("invalid_item_count", "FleetPlanV1.items must contain exactly one item")
        items = (FleetPlanItemV1.from_dict(raw_items[0]),)
        return cls(
            schema_version=_schema_version(obj, "FleetPlanV1"),
            catalog_snapshot_id=_parse_str(obj["catalogSnapshotId"], "fleetPlan.catalogSnapshotId"),
            catalog_content_etag=_parse_str(obj["catalogContentEtag"], "fleetPlan.catalogContentEtag"),
            items=items,
        )

    def to_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "catalogSnapshotId": self.catalog_snapshot_id,
            "catalogContentEtag": self.catalog_content_etag,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "FleetPlanV1":
        return cls.from_dict(_loads(text, "FleetPlanV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class ProcessErrorV1:
    code: str
    retryable: bool
    message: str

    @classmethod
    def from_dict(cls, data: Any) -> "ProcessErrorV1":
        obj = _require_object(data, "ProcessErrorV1")
        _check_keys(obj, "ProcessErrorV1", required=("code", "retryable", "message"))
        message = obj["message"]
        if not isinstance(message, str):
            raise ContractError("invalid_field", "error.message must be a string")
        return cls(
            code=_parse_str(obj["code"], "error.code"),
            retryable=_parse_bool(obj["retryable"], "error.retryable"),
            message=_bound_message(message),
        )

    def to_dict(self) -> dict:
        return {"code": self.code, "retryable": self.retryable, "message": _bound_message(self.message)}

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "ProcessErrorV1":
        return cls.from_dict(_loads(text, "ProcessErrorV1"))

    def to_json(self) -> str:
        return _dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Solve freshness (time-dependent; only new solve calls this)
# ---------------------------------------------------------------------------


def assert_fresh_for_solve(snapshot: SupplySnapshotV1, now: datetime) -> None:
    if not snapshot.offerings:
        raise SupplyExpiredError("missing_supply")
    if all(offering.valid_until <= now for offering in snapshot.offerings):
        raise SupplyExpiredError("supply_expired")


# ---------------------------------------------------------------------------
# Shared one-request/one-response process envelope
# ---------------------------------------------------------------------------


def success_envelope(plan: FleetPlanV1) -> dict:
    return {"schemaVersion": SCHEMA_VERSION, "ok": True, "value": plan.to_dict()}


def error_envelope(error: ProcessErrorV1) -> dict:
    return {"schemaVersion": SCHEMA_VERSION, "ok": False, "error": error.to_dict()}


def dump_envelope(envelope: dict) -> str:
    return _dumps(envelope)


def parse_solve_envelope(text: Union[str, bytes]) -> Union[FleetPlanV1, ProcessErrorV1]:
    """Parse one solve process envelope back into its typed payload."""
    obj = _require_object(_loads(text, "process envelope"), "process envelope")
    if "ok" not in obj:
        raise ContractError("invalid_contract", "process envelope requires field 'ok'")
    ok = _parse_bool(obj["ok"], "envelope.ok")
    payload_key = "value" if ok else "error"
    _check_keys(obj, "process envelope", required=("schemaVersion", "ok", payload_key))
    _schema_version(obj, "process envelope")
    if ok:
        return FleetPlanV1.from_dict(obj["value"])
    return ProcessErrorV1.from_dict(obj["error"])
