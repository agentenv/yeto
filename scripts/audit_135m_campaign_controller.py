#!/usr/bin/env python3
"""Launch and supervise one sealed audit-135M acquisition suffix on GCP Spot.

This controller reuses the reviewed P1 GCP/evidence backend and its exact-ID
teardown implementation.  It adds the audit runtime authorization, current-seed
evaluation registry, stage-wide cost guard, survival-weighted zone rotation,
shape-cost eligibility, loss-blind whole-block retry, bounded assembly, and a
continuous spend ledger.  Scientific losses are never printed by this process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
P1_SESSION = Path("/private/tmp/yeto-p1r0-launcher/p1-adaptive-session")
P1_CONTROLLER = P1_SESSION / "p1ad_campaign_controller.py"
R0_CONTROLLER = Path("/private/tmp/yeto-p1r0-launcher/p1r0-session/p1r0_controller.py")
NOTE_PATH = Path("/private/tmp/audit-135m-note.md")
GCLOUD_CONFIG = "/private/tmp/yeto-gcloud-admin-codex"
PROTECTED_INSTANCE_ID = "3908640733128066700"
ZONE_ROTATION = (
    "us-east1-b",
    "us-west4-b",
    "us-west1-b",
    "us-west4-a",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)
AUDIT_BLOCK_WIDTH = 3
MAX_CONCURRENT_BLOCKS = 2
SLOTS = tuple(
    f"v{index}" for index in range(AUDIT_BLOCK_WIDTH * MAX_CONCURRENT_BLOCKS)
)
PRICE_PER_VM_HOUR = {
    "us-east1": {"a2-highgpu-1g": 1.928, "a2-highgpu-4g": 7.712},
    "us-west4": {"a2-highgpu-1g": 1.817, "a2-highgpu-4g": 7.267},
    "us-west1": {"a2-highgpu-1g": 1.928, "a2-highgpu-4g": 7.712},
    "us-central1": {"a2-highgpu-1g": 1.928, "a2-highgpu-4g": 7.712},
}
CELL_HOURS = {
    8: 1.001183440327381,
    16: 0.5683582358333333,
    64: 0.24238701958333334,
    256: 0.16299781333333335,
    512: 0.14866439438988095,
}
FINITE_KERNEL_EXTRA_HOURS = {8: 0.40, 16: 0.25, 64: 0.12, 256: 0.06, 512: 0.04}
# The A3/A4 hard ceilings are lower_bound * 1.30; A1 has the operator-authorized
# $140 correction.  The reviewed remaining-work forecast keeps 1.25 so reserve
# remains available for realized boot, transient-provider, and exact teardown.
SPOT_PREEMPTION_RESERVE_FACTOR = 1.25
WATCHDOG_POLL_SECONDS = 15
TEARDOWN_RESERVE_SECONDS = 900
TEARDOWN_RESERVE_FIXED_USD = 0.25
TRANSIENT_PROVIDER_POLL_SECONDS = 1.0
TRANSIENT_PROVIDER_DELETE_TIMEOUT_SECONDS = 900
GLOBAL_A100_CEILING = 16
PREFERRED_PROBE_A100S = 1
MAX_PREFERRED_PROBE_WIDTH = 4
ABORT_BURN_KILL_USD = 40.0
FUTURE_STAGE_CELL_COUNTS = {
    "a1d": {16: 36, 256: 36},
    "a1x": {16: 32, 256: 32},
    "a1c": {},
    "a3k": {8: 10, 512: 12},
    "a3r0": {8: 2, 512: 2},
    "a3x": {},
    "a4d": {16: 56, 256: 56},
    "a4b": {16: 48, 256: 48},
    "a4c": {16: 24, 256: 24},
    "a4x": {},
}


class ControllerError(RuntimeError):
    """The campaign cannot continue within its frozen operational contract."""


class HardCeilingStop(ControllerError):
    """The stage was killed loss-blindly before its registered dollar ceiling."""


class AbortBurnStop(ControllerError):
    """Pre-science aborted-launch spend exceeded the registered stage kill."""


class HiddenSurvivorPreempted(ControllerError):
    """The complete hidden batch lost its evaluator before any shared unblind."""


class TransientProviderGeneration(ControllerError):
    """A provider identity existed but vanished before normal evidence registration."""

    def __init__(self, lifecycle: Mapping[str, Any]) -> None:
        self.lifecycle = dict(lifecycle)
        super().__init__(
            "provider created and preempted exact physical generation "
            f"{self.lifecycle.get('run_id')} before normal registration"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControllerError(f"cannot load reviewed controller module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bind_reviewed_backend_workspace(base) -> None:
    tracked_parallel = REPO_ROOT / "scripts" / "run_parallel_phase_map.py"
    if not tracked_parallel.is_file():
        raise ControllerError("tracked parallel executor is missing from the source commit")
    base.WORKSPACE = REPO_ROOT


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ControllerError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_json_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise ControllerError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _cell_forecast_hours(cell: Mapping[str, Any]) -> float:
    h = int(cell["H"])
    if h not in CELL_HOURS:
        raise ControllerError(f"no registered runtime estimate for H={h}")
    hours = CELL_HOURS[h]
    if cell.get("finite_kernel_capture_required") is True:
        hours += FINITE_KERNEL_EXTRA_HOURS[h]
    return hours


def _remaining_cells(
    *,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
) -> list[dict[str, Any]]:
    cells = {
        str(row["cell_id"]): dict(row)
        for row in scientific.get("cells", [])
        if isinstance(row, Mapping)
    }
    ids: list[str] = []
    for wave in plan.get("waves", [])[planned_index:]:
        ids.extend(
            str(row["cell_id"])
            for row in wave.get("assigned_cells_in_dispatch_order", [])
        )
    if len(ids) != len(set(ids)) or any(cell_id not in cells for cell_id in ids):
        raise ControllerError("remaining parallel-plan cells are missing/duplicated")
    return [cells[cell_id] for cell_id in ids]


def _shape_stage_forecast(
    *,
    stage_code: str,
    machine_type: str,
    region: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
) -> float:
    try:
        price = PRICE_PER_VM_HOUR[region][machine_type]
        future = FUTURE_STAGE_CELL_COUNTS[stage_code]
    except KeyError as exc:
        raise ControllerError(
            f"no registered cost forecast for {stage_code}/{region}/{machine_type}"
        ) from exc
    hours = sum(
        _cell_forecast_hours(cell)
        for cell in _remaining_cells(
            plan=plan, scientific=scientific, planned_index=planned_index
        )
    )
    hours += sum(CELL_HOURS[h] * count for h, count in future.items())
    return hours * price * SPOT_PREEMPTION_RESERVE_FACTOR


def _registered_remaining_forecasts(
    *,
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
) -> list[dict[str, Any]]:
    rows = []
    registered_regions = {
        zone.rsplit("-", 1)[0] for zone in ZONE_ROTATION
    }
    for region in sorted(registered_regions):
        for machine_type in sorted(PRICE_PER_VM_HOUR[region]):
            rows.append(
                {
                    "region": region,
                    "machine_type": machine_type,
                    "forecast_usd": _shape_stage_forecast(
                        stage_code=stage_code,
                        machine_type=machine_type,
                        region=region,
                        plan=plan,
                        scientific=scientific,
                        planned_index=planned_index,
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["forecast_usd"]),
            str(row["region"]),
            str(row["machine_type"]),
        )
    )
    if not rows:
        raise ControllerError("no registered remaining-stage cost forecast exists")
    return rows


def _cost_eligible_zone_rotation(
    *,
    stage_code: str,
    machine_type: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    prior_stage_spend: float,
    current_campaign_spend: float,
    ceiling: float,
) -> tuple[str, ...]:
    candidates = []
    for index, zone in enumerate(ZONE_ROTATION):
        region = zone.rsplit("-", 1)[0]
        forecast = _shape_stage_forecast(
            stage_code=stage_code,
            machine_type=machine_type,
            region=region,
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
        )
        if prior_stage_spend + current_campaign_spend + forecast < ceiling:
            candidates.append((forecast, index, zone))
    candidates.sort()
    return tuple(zone for _forecast, _index, zone in candidates)


def _assert_remaining_stage_cost_feasible(
    *,
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    prior_stage_spend: float,
    current_campaign_spend: float,
    ceiling: float,
    phase: str,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    forecasts = _registered_remaining_forecasts(
        stage_code=stage_code,
        plan=plan,
        scientific=scientific,
        planned_index=planned_index,
    )
    cheapest = forecasts[0]
    projected = (
        prior_stage_spend
        + current_campaign_spend
        + float(cheapest["forecast_usd"])
    )
    feasible = projected < ceiling
    evidence = {
        "schema": "audit_135m_remaining_cost_feasibility_v1",
        "status": "PASS" if feasible else "HARD_CEILING_INFEASIBLE",
        "stage_code": stage_code,
        "phase": phase,
        "planned_index": planned_index,
        "loss_inspected": False,
        "prior_stage_spend_usd": round(prior_stage_spend, 6),
        "current_campaign_spend_usd": round(current_campaign_spend, 6),
        "cheapest_registered_remaining_forecast": {
            **cheapest,
            "forecast_usd": round(float(cheapest["forecast_usd"]), 6),
        },
        "registered_remaining_forecasts": [
            {**row, "forecast_usd": round(float(row["forecast_usd"]), 6)}
            for row in forecasts
        ],
        "estimated_stage_spend_if_continued_usd": round(projected, 6),
        "hard_ceiling_usd": ceiling,
        "provider_mutation_authorized": feasible,
        "evaluated_at_utc": utc_now(),
    }
    if feasible:
        return evidence
    if evidence_path is not None and not evidence_path.exists():
        write_json_create_only(evidence_path, evidence)
    raise HardCeilingStop(
        "remaining-stage cost preflight: durable prior spend "
        f"${prior_stage_spend:.6f} + current campaign burn "
        f"${current_campaign_spend:.6f} + cheapest registered forecast "
        f"${float(cheapest['forecast_usd']):.6f} = ${projected:.6f}, which "
        f"reaches/exceeds the ${ceiling:.2f} ceiling"
    )


def _generation_cost(
    provider: Mapping[str, Any], *, ended_at: str | None = None
) -> float:
    region = str(provider["region"])
    machine_type = str(provider["machine_type"])
    price = PRICE_PER_VM_HOUR[region][machine_type]
    start = parse_time(str(provider["creation_timestamp"]))
    end = parse_time(ended_at) if ended_at is not None else datetime.now(timezone.utc)
    seconds = max(0.0, (end - start).total_seconds())
    return seconds / 3600.0 * price


def _transient_provider_lifecycle_registry(campaign_root: Path) -> dict[str, Any]:
    root = campaign_root / "common" / "transient-provider-lifecycles"
    rows = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            lifecycle = load_json(path)
            if (
                lifecycle.get("schema")
                != "audit_135m_transient_provider_lifecycle_v1"
                or lifecycle.get("status")
                != "TRANSIENT_PROVIDER_PREEMPTED_AND_EXACT_IDS_ABSENT"
                or lifecycle.get("scientific_attempt_started") is not False
                or lifecycle.get("loss_inspected") is not False
            ):
                raise ControllerError(f"invalid transient provider lifecycle: {path}")
            rows.append(
                {
                    "run_id": str(lifecycle["run_id"]),
                    "slot": str(lifecycle["slot"]),
                    "generation": int(lifecycle["generation"]),
                    "instance_numeric_id": str(lifecycle["instance_numeric_id"]),
                    "boot_disk_numeric_id": str(
                        lifecycle["boot_disk_numeric_id"]
                    ),
                    "path": path.relative_to(campaign_root).as_posix(),
                    "raw_sha256": _sha256_file(path),
                }
            )
    identities = [(row["slot"], row["generation"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ControllerError("transient provider lifecycle identities are duplicated")
    return {
        "schema": "audit_135m_transient_provider_registry_v1",
        "status": "SEALED_EXACT_ID_EVIDENCE",
        "scientific_attempt_started": False,
        "loss_inspected": False,
        "count": len(rows),
        "lifecycles": rows,
    }


def _current_campaign_cost(executor, campaign_root: Path) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    total = 0.0
    for key, provider in sorted(executor.providers.items()):
        slot, generation = key
        lifecycle_path = (
            campaign_root
            / "vms"
            / slot
            / f"g{generation}"
            / "manifests"
            / "vm-lifecycle-final.json"
        )
        ended = None
        status = "ACTIVE"
        if lifecycle_path.is_file():
            lifecycle = load_json(lifecycle_path)
            ended = str(lifecycle["deletion_completed_at_utc"])
            status = "FINAL"
        cost = _generation_cost(provider, ended_at=ended)
        total += cost
        rows.append(
            {
                "slot": slot,
                "generation": generation,
                "run_id": provider["run_id"],
                "instance_numeric_id": provider["instance_numeric_id"],
                "region": provider["region"],
                "zone": provider["zone"],
                "machine_type": provider["machine_type"],
                "creation_timestamp": provider["creation_timestamp"],
                "ended_at_utc": ended,
                "status": status,
                "estimated_cost_usd": round(cost, 6),
            }
        )
    registered = set(executor.providers)
    transient_registry = _transient_provider_lifecycle_registry(campaign_root)
    for entry in transient_registry["lifecycles"]:
        key = (str(entry["slot"]), int(entry["generation"]))
        if key in registered:
            raise ControllerError(
                "transient lifecycle collides with a registered provider generation"
            )
        lifecycle = load_json(campaign_root / str(entry["path"]))
        cost = _generation_cost(
            lifecycle, ended_at=str(lifecycle["deletion_completed_at_utc"])
        )
        total += cost
        rows.append(
            {
                "slot": key[0],
                "generation": key[1],
                "run_id": lifecycle["run_id"],
                "instance_numeric_id": lifecycle["instance_numeric_id"],
                "region": lifecycle["region"],
                "zone": lifecycle["zone"],
                "machine_type": lifecycle["machine_type"],
                "creation_timestamp": lifecycle["creation_timestamp"],
                "ended_at_utc": lifecycle["deletion_completed_at_utc"],
                "status": "TRANSIENT_PRE_READY_FINAL",
                "estimated_cost_usd": round(cost, 6),
                "lifecycle_raw_sha256": entry["raw_sha256"],
            }
        )
    rows.sort(key=lambda row: (str(row["slot"]), int(row["generation"])))
    return total, rows


def _guard_provider_mutation_cost(
    *,
    executor,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    ceiling: float,
    phase: str,
    lock: threading.RLock | None = None,
) -> dict[str, Any]:
    def evaluate() -> dict[str, Any]:
        current, _rows = _current_campaign_cost(executor, campaign_root)
        return _assert_remaining_stage_cost_feasible(
            stage_code=stage_code,
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
            prior_stage_spend=float(stage_ledger["estimated_spend_usd"]),
            current_campaign_spend=current,
            ceiling=ceiling,
            phase=phase,
            evidence_path=(
                campaign_root / "campaign" / "cost-feasibility-stop.json"
            ),
        )

    if lock is None:
        return evaluate()
    with lock:
        return evaluate()


def _instance_a100_count(row: Mapping[str, Any]) -> int:
    accelerators = row.get("guestAccelerators") or []
    if isinstance(accelerators, list) and accelerators:
        return sum(
            int(accelerator.get("acceleratorCount", 0))
            for accelerator in accelerators
            if isinstance(accelerator, Mapping)
            and "A100" in str(accelerator.get("acceleratorType", "")).upper()
        )
    machine_type = str(row.get("machineType", "")).rsplit("/", 1)[-1]
    match = re.fullmatch(r"a2-(?:ultra)?highgpu-([1-9][0-9]*)g", machine_type)
    return 0 if match is None else int(match.group(1))


def _global_a100_census(backend) -> dict[str, Any]:
    result = backend.gcloud(
        "compute", "instances", "list", "--project=model-training-497007", "--format=json"
    )
    rows = json.loads(result.stdout)
    live = [row for row in rows if row.get("status") != "TERMINATED"]
    inventory = [
        {
            "name": row.get("name"),
            "instance_numeric_id": str(row.get("id")),
            "a100_count": _instance_a100_count(row),
            "campaign_tag": (row.get("labels") or {}).get("campaign-tag"),
        }
        for row in live
        if _instance_a100_count(row) > 0
    ]
    return {
        "schema": "audit_135m_global_a100_census_v1",
        "total_attached_a100_equivalent": sum(row["a100_count"] for row in inventory),
        "instances": inventory,
        "queried_at_utc": utc_now(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persist_create_only_provider_evidence(
    *, backend, path: Path, value: Mapping[str, Any], remote_uri: str
) -> None:
    created = False
    if path.exists():
        if load_json(path) != dict(value):
            raise ControllerError(f"existing provider sighting differs: {path}")
    else:
        backend.pexec.write_json_create_only(path, dict(value))
        created = True
    if created:
        backend.gcloud(
            "storage",
            "cp",
            str(path),
            remote_uri,
            "--if-generation-match=0",
            timeout=900,
        )


def _optimizer_launch_context(command: Sequence[str]) -> dict[str, Any] | None:
    values = list(command)
    if "yeto.optimizer_harness" not in values or "launch" not in values:
        return None
    launch_index = values.index("launch")
    if launch_index + 1 >= len(values):
        raise ControllerError("optimizer harness launch command has no spec path")
    spec_path = Path(values[launch_index + 1]).resolve()
    spec = load_json(spec_path)
    cloud = spec.get("cloud")
    artifacts = spec.get("artifacts")
    if not isinstance(cloud, Mapping) or not isinstance(artifacts, Mapping):
        raise ControllerError("optimizer harness launch spec lacks cloud/artifacts")
    try:
        nonce = values[values.index("--ownership-nonce") + 1]
    except (ValueError, IndexError) as exc:
        raise ControllerError("optimizer harness launch lacks ownership nonce") from exc
    labels = cloud.get("labels")
    if not isinstance(labels, Mapping):
        raise ControllerError("optimizer harness launch spec lacks labels")
    run_id = str(spec.get("run_id"))
    if run_id != str(cloud.get("instance_name")) or labels.get("run-id") != run_id:
        raise ControllerError("optimizer harness launch run identity differs")
    return {
        "spec_path": spec_path,
        "run_id": run_id,
        "slot": str(labels.get("logical-slot")),
        "generation": int(str(labels.get("physical-generation"))),
        "ownership_nonce": nonce,
        "zone": str(cloud.get("zone")),
        "region": str(cloud.get("zone")).rsplit("-", 1)[0],
        "machine_type": str(cloud.get("machine_type")),
        "a100_count": int(cloud.get("accelerator_count", 0)),
        "project": str(cloud.get("project")),
        "labels": {str(key): str(value) for key, value in labels.items()},
        "artifact_uri": str(artifacts.get("uri")).rstrip("/"),
    }


def _provider_operation_rows(backend, *, run_id: str, zone: str) -> list[dict[str, Any]]:
    result = backend.gcloud(
        "compute",
        "operations",
        "list",
        "--project=model-training-497007",
        f"--filter=zone:({zone}) AND targetLink~{run_id}",
        "--format=json",
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        return []
    values = json.loads(result.stdout)
    rows = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        rows.append(
            {
                field: value[field]
                for field in (
                    "name",
                    "operationType",
                    "status",
                    "statusMessage",
                    "targetId",
                    "targetLink",
                    "insertTime",
                    "startTime",
                    "endTime",
                    "error",
                )
                if value.get(field) is not None
            }
        )
    return rows


def _finalize_transient_provider_generation(
    *,
    backend,
    campaign_root: Path,
    context: Mapping[str, Any],
    instance_sighting: Mapping[str, Any],
    disk_sighting: Mapping[str, Any],
    original_error: BaseException,
) -> dict[str, Any]:
    run_id = str(context["run_id"])
    zone = str(context["zone"])
    instance_id = str(instance_sighting["instance_numeric_id"])
    disk_id = str(disk_sighting["boot_disk_numeric_id"])
    if instance_id == PROTECTED_INSTANCE_ID:
        raise ControllerError("transient provider generation is the protected instance")
    deletion_requested_at: str | None = None
    delete_returncode: int | None = None
    deadline = time.time() + TRANSIENT_PROVIDER_DELETE_TIMEOUT_SECONDS
    while time.time() < deadline:
        live = backend.describe_instance(run_id, check=False)
        if live is None:
            instance_not_found_at = utc_now()
            break
        if str(live.get("id")) != instance_id:
            raise ControllerError("transient instance name rebound to a different exact ID")
        labels = live.get("labels") or {}
        if labels.get("ownership-nonce") != context["ownership_nonce"]:
            raise ControllerError("transient instance ownership nonce differs")
        if deletion_requested_at is None:
            deletion_requested_at = utc_now()
            deleted = backend.gcloud(
                "compute",
                "instances",
                "delete",
                run_id,
                "--project=model-training-497007",
                f"--zone={zone}",
                "--quiet",
                check=False,
            )
            delete_returncode = deleted.returncode
        time.sleep(TRANSIENT_PROVIDER_POLL_SECONDS)
    else:
        raise ControllerError("transient exact instance did not become NOT_FOUND")

    disk_name = str(disk_sighting["boot_disk_name"])
    while time.time() < deadline:
        result = backend.gcloud(
            "compute",
            "disks",
            "describe",
            disk_name,
            "--project=model-training-497007",
            f"--zone={zone}",
            "--format=json",
            check=False,
        )
        if result.returncode:
            disk_not_found_at = utc_now()
            break
        disk = json.loads(result.stdout)
        if str(disk.get("id")) != disk_id:
            raise ControllerError("transient disk name rebound to a different exact ID")
        time.sleep(TRANSIENT_PROVIDER_POLL_SECONDS)
    else:
        raise ControllerError("transient exact disk did not become NOT_FOUND")

    census = _global_a100_census(backend)
    if any(
        str(row.get("instance_numeric_id")) == instance_id
        for row in census["instances"]
    ):
        raise ControllerError("transient exact generation still owns an attached A100")
    operations = _provider_operation_rows(backend, run_id=run_id, zone=zone)
    lifecycle = {
        "schema": "audit_135m_transient_provider_lifecycle_v1",
        "status": "TRANSIENT_PROVIDER_PREEMPTED_AND_EXACT_IDS_ABSENT",
        "stage_code": str(context["labels"].get("stage")),
        "run_id": run_id,
        "slot": str(context["slot"]),
        "generation": int(context["generation"]),
        "ownership_nonce": str(context["ownership_nonce"]),
        "labels": dict(context["labels"]),
        "region": str(instance_sighting["region"]),
        "zone": zone,
        "machine_type": str(instance_sighting["machine_type"]),
        "provisioning_model": "SPOT",
        "instance_numeric_id": instance_id,
        "boot_disk_name": disk_name,
        "boot_disk_numeric_id": disk_id,
        "source_image_numeric_id": str(
            disk_sighting.get("source_image_numeric_id")
        ),
        "creation_timestamp": str(instance_sighting["creation_timestamp"]),
        "first_observed_at_utc": str(instance_sighting["observed_at_utc"]),
        "deletion_requested_at_utc": deletion_requested_at,
        "deletion_completed_at_utc": max(instance_not_found_at, disk_not_found_at),
        "teardown_mode": (
            "OPERATOR_EXACT_DELETE"
            if deletion_requested_at is not None and delete_returncode == 0
            else "PROVIDER_SPOT_PREEMPTION_AUTO_DELETE"
        ),
        "provider_not_found_verification": {
            "instance": {
                "name": run_id,
                "provider_id": instance_id,
                "result": "NOT_FOUND",
                "verified_at_utc": instance_not_found_at,
            },
            "boot_disk": {
                "name": disk_name,
                "provider_id": disk_id,
                "result": "NOT_FOUND",
                "verified_at_utc": disk_not_found_at,
            },
        },
        "zero_attached_accelerator_proof": {
            "generation_attached_a100s": 0,
            "instance_numeric_id": instance_id,
            "verified_at_utc": str(census["queried_at_utc"]),
            "global_attached_a100_equivalent": int(
                census["total_attached_a100_equivalent"]
            ),
        },
        "provider_operations": operations,
        "provider_spot_preempted": True,
        "scientific_attempt_started": False,
        "loss_inspected": False,
        "original_error_type": type(original_error).__name__,
        "original_error_detail": str(original_error)[-4000:],
    }
    path = (
        campaign_root
        / "common"
        / "transient-provider-lifecycles"
        / f"{run_id}.json"
    )
    _persist_create_only_provider_evidence(
        backend=backend,
        path=path,
        value=lifecycle,
        remote_uri=(
            str(context["artifact_uri"])
            + "/manifests/transient-provider-lifecycle.json"
        ),
    )
    backend.audit_transient_generations.add(
        (str(context["slot"]), int(context["generation"]))
    )
    backend.audit_transient_lifecycle_paths.append(path)
    backend.note(
        f"TRANSIENT PROVIDER PREEMPTION PASS {context['slot']}/g"
        f"{context['generation']}: exact instance {instance_id} NOT_FOUND, exact "
        f"disk {disk_id} NOT_FOUND, attached A100s=0; the generation is consumed "
        "and may never be retried."
    )
    print(
        json.dumps(
            {
                "event": "TRANSIENT_PROVIDER_EXACT_ID_TEARDOWN_PASS",
                "slot": context["slot"],
                "generation": context["generation"],
                "run_id": run_id,
                "instance_id": instance_id,
                "boot_disk_id": disk_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return lifecycle


def _install_transient_provider_generation_monitor(
    *, backend, campaign_root: Path
) -> None:
    """Capture exact IDs while a blocking provider create operation is pending."""

    original_run = backend.run
    backend.audit_transient_generations = set()
    backend.audit_transient_lifecycle_paths = []

    def run(command, *, check=True, timeout=None):
        context = _optimizer_launch_context(command)
        if context is None:
            return original_run(command, check=check, timeout=timeout)
        holder: dict[str, Any] = {}
        stop = threading.Event()

        def observe() -> None:
            while not stop.is_set():
                try:
                    live = backend.describe_instance(context["run_id"], check=False)
                    if live is None:
                        stop.wait(TRANSIENT_PROVIDER_POLL_SECONDS)
                        continue
                    instance_id = str(live.get("id"))
                    labels = live.get("labels") or {}
                    expected_labels = {
                        **context["labels"],
                        "ownership-nonce": context["ownership_nonce"],
                    }
                    if instance_id == PROTECTED_INSTANCE_ID or any(
                        labels.get(key) != value
                        for key, value in expected_labels.items()
                    ):
                        raise ControllerError(
                            "pending provider instance exact ownership differs"
                        )
                    machine_type = str(live.get("machineType", "")).rsplit("/", 1)[-1]
                    scheduling = live.get("scheduling") or {}
                    if (
                        machine_type != context["machine_type"]
                        or scheduling.get("provisioningModel") != "SPOT"
                        or _instance_a100_count(live) != context["a100_count"]
                    ):
                        raise ControllerError(
                            "pending provider instance shape/Spot identity differs"
                        )
                    zone = str(live.get("zone", "")).rsplit("/", 1)[-1]
                    instance = {
                        "schema": "audit_135m_pending_instance_sighting_v1",
                        "run_id": context["run_id"],
                        "slot": context["slot"],
                        "generation": context["generation"],
                        "ownership_nonce": context["ownership_nonce"],
                        "instance_numeric_id": instance_id,
                        "region": zone.rsplit("-", 1)[0],
                        "zone": zone,
                        "machine_type": machine_type,
                        "a100_count": context["a100_count"],
                        "provisioning_model": "SPOT",
                        "creation_timestamp": str(live["creationTimestamp"]),
                        "observed_at_utc": utc_now(),
                        "labels": dict(labels),
                    }
                    instance_path = (
                        campaign_root
                        / "common"
                        / "provider-sightings"
                        / context["run_id"]
                        / "instance.json"
                    )
                    _persist_create_only_provider_evidence(
                        backend=backend,
                        path=instance_path,
                        value=instance,
                        remote_uri=(
                            context["artifact_uri"]
                            + "/provider/pending-instance-sighting.json"
                        ),
                    )
                    holder["instance"] = instance
                    disk = backend._disk_description(live)
                    disk_sighting = {
                        "schema": "audit_135m_pending_disk_sighting_v1",
                        "run_id": context["run_id"],
                        "slot": context["slot"],
                        "generation": context["generation"],
                        "ownership_nonce": context["ownership_nonce"],
                        "boot_disk_name": str(disk["name"]),
                        "boot_disk_numeric_id": str(disk["id"]),
                        "source_image_numeric_id": str(disk.get("sourceImageId")),
                        "zone": zone,
                        "observed_at_utc": utc_now(),
                    }
                    disk_path = instance_path.with_name("disk.json")
                    _persist_create_only_provider_evidence(
                        backend=backend,
                        path=disk_path,
                        value=disk_sighting,
                        remote_uri=(
                            context["artifact_uri"]
                            + "/provider/pending-disk-sighting.json"
                        ),
                    )
                    holder["disk"] = disk_sighting
                    return
                except BaseException as exc:
                    holder["monitor_error"] = exc
                    return

        monitor = threading.Thread(
            target=observe,
            name=f"audit-provider-sighting-{context['run_id']}",
            daemon=False,
        )
        monitor.start()
        result = None
        run_error: BaseException | None = None
        try:
            result = original_run(command, check=check, timeout=timeout)
        except BaseException as exc:
            run_error = exc
        finally:
            stop.set()
            monitor.join()
        monitor_error = holder.get("monitor_error")
        if monitor_error is not None and run_error is None:
            run_error = monitor_error
        if run_error is None:
            assert result is not None
            return result
        if holder.get("instance") is not None and holder.get("disk") is not None:
            lifecycle = _finalize_transient_provider_generation(
                backend=backend,
                campaign_root=campaign_root,
                context=context,
                instance_sighting=holder["instance"],
                disk_sighting=holder["disk"],
                original_error=run_error,
            )
            if isinstance(run_error, (KeyboardInterrupt, SystemExit)):
                raise run_error
            raise TransientProviderGeneration(lifecycle) from run_error
        operations = _provider_operation_rows(
            backend, run_id=context["run_id"], zone=context["zone"]
        )
        if any(str(row.get("targetId", "")).isdigit() for row in operations):
            raise ControllerError(
                "provider operation reports a physical target ID but the pending "
                "monitor lacks an exact disk sighting; refusing identity reuse"
            ) from run_error
        raise run_error

    backend.run = run


def _audit_provisional_identity(*, pexec, registry, backend, slot: str):
    prior = [
        identity.generation
        for identity in registry.identities
        if identity.slot == slot
    ]
    prior.extend(
        generation
        for seen_slot, generation in backend.audit_transient_generations
        if seen_slot == slot
    )
    generation = max(prior, default=0) + 1
    if generation == 1:
        nonce = registry._prebound_nonces[slot]
    else:
        used = {
            identity.ownership_nonce for identity in registry.identities
        }
        nonce = secrets.token_hex(16)
        while nonce in used:
            nonce = secrets.token_hex(16)
    return pexec.GenerationIdentity(
        stage_code=registry.stage_code,
        study_id=registry.study_id,
        roster_hash=registry.roster_hash,
        campaign_attempt=registry.campaign_attempt,
        slot=slot,
        generation=generation,
        ownership_nonce=nonce,
        campaign_state_root=registry.campaign_state_root,
        campaign_artifact_root=registry.campaign_artifact_root,
    )


def _audit_provision_loop(
    *,
    base,
    p1,
    pexec,
    executor,
    registry,
    backend,
    slot: str,
    stop_event: threading.Event,
    science_started: threading.Event,
    lock: threading.RLock,
    errors: list[BaseException],
    required: bool,
    pre_launch_guard: Callable[[], Any] | None = None,
) -> None:
    identity = _audit_provisional_identity(
        pexec=pexec, registry=registry, backend=backend, slot=slot
    )
    zone_index = 0
    tries = 0
    while required or not stop_event.is_set():
        started = time.time()
        zone = p1.ZONE_ROTATION[zone_index % len(p1.ZONE_ROTATION)]
        tries += 1
        try:
            if pre_launch_guard is not None:
                pre_launch_guard()
            p1.cache_zone_render(backend, identity, zone)
            backend.note(
                f"ADAPTIVE CAPACITY probe {tries} for {identity.slot}/g"
                f"{identity.generation}: region {pexec.region_for_zone(zone)}, zone "
                f"{zone}; exact run ID/nonce retained, Spot 1g preferred under the "
                "ceiling amendment; provider stockout rotates the registered zones."
            )
            provider = dict(backend.provision(identity))
            pexec.validate_provider_record(provider, identity.registry_row())
            with lock:
                p1.initialize_registered_generation(
                    base=base,
                    pexec=pexec,
                    executor=executor,
                    registry=registry,
                    identity=identity,
                    provider=provider,
                )
            ready_at = p1.mark_ready(
                pexec=pexec,
                executor=executor,
                registry=registry,
                identity=identity,
                provider=provider,
                active=False,
            )
            with lock:
                late_optional = (not required) and science_started.is_set()
                if not late_optional:
                    executor.active[identity.slot] = identity
            if late_optional:
                backend.note(
                    f"LATE OPTIONAL READY {identity.slot}/g{identity.generation} at "
                    f"{ready_at}; science already began, so exact-ID zero-attempt "
                    "teardown starts immediately rather than idle-burning."
                )
                executor._finalize_identity(identity, preempted=False)
            else:
                backend.note(
                    f"ADAPTIVE READY {identity.slot}/g{identity.generation}: exact "
                    f"instance {provider['instance_numeric_id']}, region "
                    f"{provider['region']}, zone {provider['zone']}, machine "
                    f"{provider['machine_type']}; available width is now "
                    f"{len(executor.active)}."
                )
            return
        except TransientProviderGeneration as transient:
            lifecycle = transient.lifecycle
            if (
                lifecycle.get("run_id") != identity.run_id
                or lifecycle.get("ownership_nonce") != identity.ownership_nonce
                or int(lifecycle.get("generation", 0)) != identity.generation
            ):
                errors.append(ControllerError("transient lifecycle identity differs"))
                return
            previous = identity
            zone_index += 1
            identity = _audit_provisional_identity(
                pexec=pexec, registry=registry, backend=backend, slot=slot
            )
            backend.note(
                f"PRE-SCIENTIFIC TRANSIENT PREEMPTION finalized for {slot}: "
                f"{previous.run_id} is permanently consumed; fresh identity "
                f"{identity.run_id} continues in the next survival-weighted zone."
            )
            continue
        except HardCeilingStop as error:
            backend.note(
                f"HARD CEILING PREFLIGHT STOP before provider mutation for "
                f"{identity.slot}/g{identity.generation}: {error}"
            )
            errors.append(error)
            return
        except Exception as error:
            text = str(error).casefold()
            absent = (
                backend.describe_instance(identity.run_id, check=False) is None
                and not (base.HARNESS_STATE_ROOT / f"{identity.run_id}.json").exists()
            )
            unsupported_zone = (
                "machine type with name" in text
                and "does not exist in zone" in text
                and absent
            )
            capacity = base._capacity_stockout(error) or (
                "operation was canceled by user" in text and absent
            )
            if unsupported_zone:
                zone_index += 1
                backend.note(
                    f"ADAPTIVE ZONE-CATALOG SKIP {identity.slot}/g"
                    f"{identity.generation}: {zone} has no A2 shape; no state/resource "
                    "was created and the next registered zone is tried immediately."
                )
                continue
            if capacity and absent:
                zone_index += 1
                next_zone = p1.ZONE_ROTATION[zone_index % len(p1.ZONE_ROTATION)]
                backend.note(
                    f"ADAPTIVE STOCKOUT {identity.slot}/g{identity.generation} try "
                    f"{tries} in {zone}; exact name/nonce and empty prefix retained; "
                    f"next zone {next_zone} after the registered {p1.RETRY_SECONDS}s "
                    "start-to-start cadence."
                )
                remaining = max(0.0, started + p1.RETRY_SECONDS - time.time())
                if required:
                    time.sleep(remaining)
                elif stop_event.wait(remaining):
                    return
                continue
            key = (identity.slot, identity.generation)
            if key in executor.providers and backend.describe_instance(
                identity.run_id, check=False
            ) is None:
                try:
                    executor._finalize_identity(identity, preempted=True)
                    identity = _audit_provisional_identity(
                        pexec=pexec,
                        registry=registry,
                        backend=backend,
                        slot=slot,
                    )
                    zone_index += 1
                    backend.note(
                        f"PRE-SCIENTIFIC PREEMPTION finalized for {slot}; fresh "
                        f"identity {identity.run_id} will continue capacity search."
                    )
                    continue
                except Exception as finalize_error:
                    errors.append(finalize_error)
                    return
            errors.append(error)
            return


def _install_launch_census_and_direct_fallback_patch(
    *, base, backend, campaign_root: Path
) -> None:
    """Re-census before every launch and reserve only unseen 1g probes."""

    original_provision = backend.provision
    lock = threading.RLock()
    backend.audit_pending_launch_a100s = 0
    backend.audit_pending_launches = {}
    backend.audit_direct_1g_authorized_zones = {}

    def release_pending(run_id: str) -> None:
        with lock:
            if run_id not in backend.audit_pending_launches:
                raise ControllerError("pending launch reservation is missing")
            del backend.audit_pending_launches[run_id]
            backend.audit_pending_launch_a100s = sum(
                int(value) for value in backend.audit_pending_launches.values()
            )

    def provision(identity):
        zone = backend.zone_for_name(identity.run_id)
        with lock:
            direct_parent = backend.audit_direct_1g_authorized_zones.get(zone)
            preferred_spec = backend._packet_paths(identity)[0]
            preferred_value = load_json(preferred_spec)
            preferred_machine = str(
                (preferred_value.get("cloud") or {}).get("machine_type")
            )
            if direct_parent is None and preferred_machine != "a2-highgpu-1g":
                raise ControllerError(
                    "ceiling-amended audit packet is not bound to preferred 1g"
                )
            requested_a100s = 1
            census = _global_a100_census(backend)
            visible_names = {
                str(row["name"])
                for row in census.get("instances", [])
                if row.get("name") is not None
            }
            unseen_pending_a100s = sum(
                int(value)
                for run_id, value in backend.audit_pending_launches.items()
                if run_id not in visible_names
            )
            projected = (
                int(census["total_attached_a100_equivalent"])
                + unseen_pending_a100s
                + requested_a100s
            )
            if projected > GLOBAL_A100_CEILING:
                raise ControllerError(
                    "loss-blind prelaunch census plus pending probes would exceed "
                    "the global 16-A100 ceiling"
                )
            if identity.run_id in backend.audit_pending_launches:
                raise ControllerError("launch already has a pending A100 reservation")
            backend.audit_pending_launches[identity.run_id] = requested_a100s
            backend.audit_pending_launch_a100s = sum(
                int(value) for value in backend.audit_pending_launches.values()
            )
        direct_authorization_path: Path | None = None
        try:
            if direct_parent is not None:
                preferred, _bootstrap, _runtime, _artifact = backend._packet_paths(identity)
                fallback = backend._fallback_spec(identity, preferred)
                backend.spec_by_key[(identity.slot, identity.generation)] = fallback
                direct_authorization_path = (
                    campaign_root
                    / "common"
                    / "direct-1g-fallback"
                    / f"{identity.run_id}.json"
                )
                direct_authorization = {
                        "schema": "audit_135m_direct_1g_fallback_authorization_v1",
                        "status": "AUTHORIZED_FROM_PROVIDER_CONFIRMED_4G_STOCKOUT",
                        "loss_inspected": False,
                        "zone": zone,
                        "run_id": identity.run_id,
                        "slot": identity.slot,
                        "generation": identity.generation,
                        "ownership_nonce": identity.ownership_nonce,
                        "parent_fallback_run_id": direct_parent["run_id"],
                        "parent_fallback_instance_numeric_id": direct_parent[
                            "instance_numeric_id"
                        ],
                        "parent_fallback_machine_type": "a2-highgpu-1g",
                        "authorized_at_utc": utc_now(),
                    }
                if direct_authorization_path.exists():
                    existing = load_json(direct_authorization_path)
                    stable_fields = set(direct_authorization) - {"authorized_at_utc"}
                    if any(
                        existing.get(field) != direct_authorization[field]
                        for field in stable_fields
                    ):
                        raise ControllerError(
                            "existing direct-1g fallback authorization differs"
                        )
                else:
                    backend.pexec.write_json_create_only(
                        direct_authorization_path, direct_authorization
                    )
                backend.note(
                    f"DIRECT 1G FALLBACK authorized for {identity.slot}/g"
                    f"{identity.generation} in {zone}: a prior exact provider launch "
                    f"({direct_parent['run_id']}, instance "
                    f"{direct_parent['instance_numeric_id']}) reached reviewed 1g only "
                    "after pre-creation 4g stockout; global census PASS."
                )
        except BaseException:
            release_pending(identity.run_id)
            raise
        try:
            try:
                provider = dict(original_provision(identity))
            except Exception as exc:
                context = exc.__context__
                if (
                    requested_a100s == 1
                    and "preferred spec is not the reviewed spot 4g shape"
                    in str(exc).casefold()
                    and context is not None
                    and base._capacity_stockout(context)
                ):
                    raise context
                raise
        finally:
            release_pending(identity.run_id)
        if direct_authorization_path is not None:
            backend.note(
                f"DIRECT 1G FALLBACK evidence sealed at "
                f"{direct_authorization_path}."
            )
        return provider

    backend.provision = provision


def _load_spend_ledger(path: Path, audit_stage: str, ceiling: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "audit_135m_stage_spend_ledger_v1",
            "audit_stage": audit_stage,
            "hard_ceiling_usd": ceiling,
            "abort_burn_kill_usd": ABORT_BURN_KILL_USD,
            "pre_science_aborted_launch_spend_usd": 0.0,
            "estimated_spend_usd": 0.0,
            "campaigns": [],
            "updated_at_utc": utc_now(),
        }
    value = load_json(path)
    if (
        value.get("schema") != "audit_135m_stage_spend_ledger_v1"
        or value.get("audit_stage") != audit_stage
        or float(value.get("hard_ceiling_usd", math.nan)) != ceiling
    ):
        raise ControllerError("stage spend ledger identity/ceiling differs")
    value.setdefault("abort_burn_kill_usd", ABORT_BURN_KILL_USD)
    value.setdefault("pre_science_aborted_launch_spend_usd", 0.0)
    if (
        float(value["abort_burn_kill_usd"]) != ABORT_BURN_KILL_USD
        or float(value["pre_science_aborted_launch_spend_usd"]) < 0.0
    ):
        raise ControllerError("stage abort-burn ledger identity differs")
    return value


def _campaign_scientific_attempt_started(campaign_root: Path) -> bool:
    for path in sorted(campaign_root.glob("vms/*/g*/manifests/vm-partial-manifest.json")):
        value = load_json(path)
        attempts = value.get("attempts", [])
        if isinstance(attempts, list) and attempts:
            return True
    return False


def _guard_abort_burn(
    *,
    executor,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    phase: str,
) -> float:
    prior = float(stage_ledger["pre_science_aborted_launch_spend_usd"])
    if _campaign_scientific_attempt_started(campaign_root):
        return prior
    current, _rows = _current_campaign_cost(executor, campaign_root)
    projected = prior + current
    kill = float(stage_ledger["abort_burn_kill_usd"])
    evidence = {
        "schema": "audit_135m_abort_burn_guard_v1",
        "status": "STOP" if projected > kill else "PASS",
        "phase": phase,
        "prior_pre_science_aborted_launch_spend_usd": round(prior, 6),
        "current_pre_science_campaign_spend_usd": round(current, 6),
        "projected_pre_science_aborted_launch_spend_usd": round(projected, 6),
        "abort_burn_kill_usd": kill,
        "scientific_attempt_started": False,
        "evaluated_at_utc": utc_now(),
    }
    write_json_atomic(campaign_root / "campaign" / "abort-burn-guard.json", evidence)
    if projected > kill:
        raise AbortBurnStop(
            f"pre-science aborted-launch spend ${projected:.6f} exceeds the "
            f"registered ${kill:.2f} stage abort-burn kill"
        )
    return projected


def _write_live_cost(
    *,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    current_cost: float,
    rows: Sequence[Mapping[str, Any]],
    stage_forecast: float,
    ceiling: float,
) -> None:
    prior = float(stage_ledger["estimated_spend_usd"])
    value = {
        "schema": "audit_135m_live_cost_guard_v1",
        "updated_at_utc": utc_now(),
        "prior_stage_spend_usd": prior,
        "current_campaign_spend_usd": round(current_cost, 6),
        "estimated_stage_spend_if_continued_usd": round(
            max(prior + stage_forecast, prior + current_cost), 6
        ),
        "hard_ceiling_usd": ceiling,
        "hard_kill_required": max(prior + stage_forecast, prior + current_cost)
        >= ceiling,
        "generations": list(rows),
    }
    write_json_atomic(campaign_root / "campaign" / "live-cost-guard.json", value)


def _cost_guard(
    *,
    executor,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    ceiling: float,
) -> float:
    current, rows = _current_campaign_cost(executor, campaign_root)
    if not executor.active:
        stage_forecast = 0.0
    else:
        forecasts = []
        for slot, identity in executor.active.items():
            provider = executor.providers[(slot, identity.generation)]
            forecasts.append(
                _shape_stage_forecast(
                    stage_code=stage_code,
                    machine_type=str(provider["machine_type"]),
                    region=str(provider["region"]),
                    plan=plan,
                    scientific=scientific,
                    planned_index=planned_index,
                )
            )
        stage_forecast = max(forecasts)
    _write_live_cost(
        campaign_root=campaign_root,
        stage_ledger=stage_ledger,
        current_cost=current,
        rows=rows,
        stage_forecast=stage_forecast,
        ceiling=ceiling,
    )
    projected = max(
        float(stage_ledger["estimated_spend_usd"]) + stage_forecast,
        float(stage_ledger["estimated_spend_usd"]) + current,
    )
    if projected >= ceiling:
        raise ControllerError(
            f"hard cost guard: projected stage spend ${projected:.2f} reaches/exceeds "
            f"the ${ceiling:.2f} ceiling"
        )
    return stage_forecast


def _active_teardown_reserve(executor) -> float:
    hourly = 0.0
    for slot, identity in list(executor.active.items()):
        provider = executor.providers[(slot, identity.generation)]
        hourly += PRICE_PER_VM_HOUR[str(provider["region"])][
            str(provider["machine_type"])
        ]
    return hourly * TEARDOWN_RESERVE_SECONDS / 3600.0 + TEARDOWN_RESERVE_FIXED_USD


def _install_operator_kill_lifecycle_patch(backend) -> None:
    backend.audit_operator_kills = {}
    backend.audit_finalizing_keys = set()
    original = backend.finalize_generation

    def finalize_generation(identity, provider_record, partial_manifest_sha256, *, preempted):
        key = (identity.slot, identity.generation)
        operator = backend.audit_operator_kills.get(key)
        lifecycle = dict(
            original(
                identity,
                provider_record,
                partial_manifest_sha256,
                preempted=(preempted or operator is not None),
            )
        )
        if operator is not None:
            lifecycle.update(
                {
                    "provider_spot_preempted": False,
                    "operator_hard_ceiling_kill": True,
                    "operator_hard_ceiling_kill_at_utc": operator["killed_at_utc"],
                    "operator_hard_ceiling_kill_state_raw_sha256": operator[
                        "deletion_state_raw_sha256"
                    ],
                }
            )
        return lifecycle

    backend.finalize_generation = finalize_generation


def _emergency_exact_delete_active(
    *, executor, backend, lifecycle_lock: threading.RLock
) -> list[dict[str, Any]]:
    deleted: list[dict[str, Any]] = []
    with lifecycle_lock:
        identities = list(executor.active.values())
        for identity in identities:
            key = (identity.slot, identity.generation)
            if (
                key in backend.audit_operator_kills
                or key in backend.audit_finalizing_keys
            ):
                continue
            provider = executor.providers[key]
            instance_id = str(provider["instance_numeric_id"])
            if instance_id == PROTECTED_INSTANCE_ID or not instance_id.isdigit():
                raise ControllerError("hard-stop deletion encountered an unsafe instance ID")
            if backend.describe_instance(identity.run_id, check=False) is None:
                continue
            spec = backend._packet_paths(identity)[0]
            result = backend.run(
                [
                    sys.executable,
                    "-m",
                    "yeto.optimizer_harness",
                    "--state-dir",
                    str(Path(backend.identity_plan["harness_state_root"])),
                    "delete",
                    str(spec),
                    "--instance-id",
                    instance_id,
                    "--yes",
                ]
            )
            state_path = Path(backend.identity_plan["harness_state_root"]) / (
                identity.run_id + ".json"
            )
            state = json.loads(result.stdout) if result.stdout.strip() else load_json(state_path)
            if (
                str(state.get("deleted_instance_id")) != instance_id
                or str(state.get("deleted_boot_disk_id"))
                != str(provider["boot_disk_numeric_id"])
            ):
                raise ControllerError("hard-stop deletion proof differs from exact provider IDs")
            killed_at = str(state["deleted_at_utc"])
            backend.audit_operator_kills[key] = {
                "killed_at_utc": killed_at,
                "deletion_state_path": str(state_path),
                "deletion_state_raw_sha256": executor.backend.pexec.sha256_file(
                    state_path
                ),
            }
            deleted.append(
                {
                    "slot": identity.slot,
                    "generation": identity.generation,
                    "run_id": identity.run_id,
                    "instance_numeric_id": instance_id,
                    "boot_disk_numeric_id": str(provider["boot_disk_numeric_id"]),
                    "deleted_at_utc": killed_at,
                    "deletion_state_raw_sha256": backend.audit_operator_kills[key][
                        "deletion_state_raw_sha256"
                    ],
                }
            )
    return deleted


class CostWatchdog:
    """Continuously reserve teardown burn and exact-delete before the ceiling."""

    def __init__(
        self,
        *,
        executor,
        backend,
        campaign_root: Path,
        stage_ledger: Mapping[str, Any],
        ceiling: float,
        lifecycle_lock: threading.RLock,
    ) -> None:
        self.executor = executor
        self.backend = backend
        self.campaign_root = campaign_root
        self.stage_ledger = stage_ledger
        self.ceiling = ceiling
        self.lifecycle_lock = lifecycle_lock
        self.stop_event = threading.Event()
        self.triggered = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="audit-135m-cost-watchdog", daemon=False
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    def raise_if_triggered(self) -> None:
        if self.error is not None:
            raise self.error
        if self.triggered.is_set():
            raise HardCeilingStop("hard ceiling watchdog stopped the campaign")

    def _run(self) -> None:
        try:
            prior = float(self.stage_ledger["estimated_spend_usd"])
            while not self.stop_event.is_set():
                global_census = _global_a100_census(self.backend)
                if (
                    int(global_census["total_attached_a100_equivalent"])
                    > GLOBAL_A100_CEILING
                ):
                    self.triggered.set()
                    deleted = _emergency_exact_delete_active(
                        executor=self.executor,
                        backend=self.backend,
                        lifecycle_lock=self.lifecycle_lock,
                    )
                    write_json_create_only(
                        self.campaign_root / "campaign" / "global-a100-stop.json",
                        {
                            "schema": "audit_135m_global_a100_stop_v1",
                            "status": "EXACT_ID_KILL_ISSUED",
                            "loss_inspected": False,
                            "global_ceiling": GLOBAL_A100_CEILING,
                            "global_census": global_census,
                            "deleted_campaign_generations": deleted,
                            "triggered_at_utc": utc_now(),
                        },
                    )
                    self.error = ControllerError(
                        "global attached A100-equivalent count exceeded 16"
                    )
                    return
                current, rows = _current_campaign_cost(
                    self.executor, self.campaign_root
                )
                reserve = _active_teardown_reserve(self.executor)
                projected = prior + current + reserve
                _write_live_cost(
                    campaign_root=self.campaign_root,
                    stage_ledger=self.stage_ledger,
                    current_cost=current,
                    rows=rows,
                    stage_forecast=max(current + reserve, 0.0),
                    ceiling=self.ceiling,
                )
                if projected >= self.ceiling:
                    self.triggered.set()
                    deleted = _emergency_exact_delete_active(
                        executor=self.executor,
                        backend=self.backend,
                        lifecycle_lock=self.lifecycle_lock,
                    )
                    evidence = {
                        "schema": "audit_135m_hard_ceiling_stop_v1",
                        "status": "EXACT_ID_KILL_ISSUED",
                        "loss_inspected": False,
                        "prior_stage_spend_usd": prior,
                        "current_campaign_spend_usd": round(current, 6),
                        "reserved_teardown_burn_usd": round(reserve, 6),
                        "projected_with_teardown_reserve_usd": round(projected, 6),
                        "hard_ceiling_usd": self.ceiling,
                        "deleted_generations": deleted,
                        "triggered_at_utc": utc_now(),
                    }
                    write_json_create_only(
                        self.campaign_root / "campaign" / "hard-ceiling-stop.json",
                        evidence,
                    )
                    self.backend.note(
                        "HARD CEILING STOP: loss-blind actual spend plus reserved exact-ID "
                        "teardown burn reached the registered threshold; every live campaign "
                        "instance received an exact-ID delete before the ceiling."
                    )
                    return
                self.stop_event.wait(WATCHDOG_POLL_SECONDS)
        except BaseException as exc:
            try:
                deleted = _emergency_exact_delete_active(
                    executor=self.executor,
                    backend=self.backend,
                    lifecycle_lock=self.lifecycle_lock,
                )
                self.backend.note(
                    "COST WATCHDOG FAILURE: fail-closed exact-ID deletion was issued "
                    f"for {len(deleted)} live generations without inspecting losses."
                )
            except BaseException as delete_error:
                self.error = ControllerError(
                    f"cost watchdog failed ({exc}) and fail-closed deletion also failed "
                    f"({delete_error})"
                )
            else:
                self.error = HardCeilingStop(
                    f"cost watchdog failed closed before the ceiling: {exc}"
                )
            self.triggered.set()


def _select_cost_eligible_shape(
    *,
    executor,
    stage_code: str,
    ceiling: float,
    prior_spend: float,
    current_spend: float,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    target_width: int,
) -> tuple[str, tuple[str, ...], float]:
    by_shape: dict[str, list[str]] = {}
    forecasts: dict[str, float] = {}
    for slot, identity in executor.active.items():
        provider = executor.providers[(slot, identity.generation)]
        shape = str(provider["machine_type"])
        by_shape.setdefault(shape, []).append(slot)
        forecast = _shape_stage_forecast(
            stage_code=stage_code,
            machine_type=shape,
            region=str(provider["region"]),
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
        )
        forecasts[slot] = forecast
    candidates = []
    for shape, slots in by_shape.items():
        eligible_slots = [
            slot
            for slot in slots
            if prior_spend + current_spend + forecasts[slot] < ceiling
        ]
        eligible_slots.sort(key=lambda slot: (forecasts[slot], slot))
        if eligible_slots:
            selected = eligible_slots[:target_width]
            candidates.append(
                (
                    min(len(selected), target_width),
                    -max(forecasts[slot] for slot in selected),
                    shape,
                    selected,
                )
            )
    if not candidates:
        detail = {
            slot: {
                "shape": executor.providers[(slot, executor.active[slot].generation)][
                    "machine_type"
                ],
                "forecast_usd": forecasts[slot],
            }
            for slot in sorted(executor.active)
        }
        raise ControllerError(f"no landed machine shape can finish within ceiling: {detail}")
    _width, _negative_cost, selected_shape, slots = max(candidates)
    slots = sorted(slots)
    stage_forecast = max(forecasts[slot] for slot in slots)
    return selected_shape, tuple(slots), stage_forecast


def _finalize_identity(
    *,
    executor,
    identity,
    preempted: bool,
    lifecycle_lock: threading.RLock,
) -> None:
    key = (identity.slot, identity.generation)
    with lifecycle_lock:
        if key in executor.backend.audit_finalizing_keys:
            raise ControllerError(f"generation is already finalizing: {key}")
        executor.backend.audit_finalizing_keys.add(key)
    try:
        executor._finalize_identity(identity, preempted=preempted)
    finally:
        with lifecycle_lock:
            executor.backend.audit_finalizing_keys.discard(key)


def _finalize_nonselected(
    *, executor, selected_slots: set[str], lifecycle_lock: threading.RLock
) -> None:
    for slot, identity in list(executor.active.items()):
        if slot in selected_slots:
            continue
        executor.backend.note(
            f"COST/PAIRING SHAPE FILTER: {slot}/g{identity.generation} is not in "
            "the selected homogeneous cost-eligible set; exact-ID zero-attempt teardown "
            "starts before science."
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=False,
            lifecycle_lock=lifecycle_lock,
        )


def _expand_homogeneous_initial_capacity(
    *,
    base,
    p1,
    pexec,
    executor,
    registry,
    backend,
    logical_slots: Sequence[str],
    selected_shape: str,
    preferred_zone: str,
    assembly_seconds: int,
    lock: threading.RLock,
    lifecycle_lock: threading.RLock,
    pre_launch_guard: Callable[[], Any],
) -> tuple[str, ...]:
    """Fill up to five three-VM block lanes without exceeding 16 A100s."""

    if selected_shape == "a2-highgpu-1g":
        preferred_region = preferred_zone.rsplit("-", 1)[0]
        ordered_zones = (
            preferred_zone,
            *[
                zone
                for zone in ZONE_ROTATION
                if zone != preferred_zone
                and zone.rsplit("-", 1)[0] == preferred_region
            ],
        )
        p1.ZONE_ROTATION = ordered_zones
        stop_event = threading.Event()
        science_started = threading.Event()
        errors: list[BaseException] = []
        threads: list[threading.Thread] = []
        for slot in logical_slots:
            if slot in executor.active:
                continue
            thread = threading.Thread(
                target=_audit_provision_loop,
                kwargs={
                    "base": base,
                    "p1": p1,
                    "pexec": pexec,
                    "executor": executor,
                    "registry": registry,
                    "backend": backend,
                    "slot": slot,
                    "stop_event": stop_event,
                    "science_started": science_started,
                    "lock": lock,
                    "errors": errors,
                    "required": False,
                    "pre_launch_guard": pre_launch_guard,
                },
                name=f"audit-expand-{slot}",
                daemon=False,
            )
            thread.start()
            threads.append(thread)
        deadline = time.time() + assembly_seconds
        while time.time() < deadline:
            if errors:
                break
            with lock:
                matching = [
                    slot
                    for slot, identity in executor.active.items()
                    if executor.providers[(slot, identity.generation)][
                        "machine_type"
                    ]
                    == selected_shape
                ]
            if len(matching) >= len(logical_slots):
                break
            time.sleep(2)
        science_started.set()
        stop_event.set()
        for thread in threads:
            thread.join()
        p1.ZONE_ROTATION = ZONE_ROTATION
        if errors:
            if isinstance(errors[0], HardCeilingStop):
                raise errors[0]
            raise ControllerError("concurrent-block capacity expansion failed") from errors[0]

    matching_slots = []
    for slot, identity in list(executor.active.items()):
        provider = executor.providers[(slot, identity.generation)]
        if provider["machine_type"] == selected_shape:
            matching_slots.append(slot)
            continue
        backend.note(
            f"CONCURRENT-BLOCK SHAPE FILTER: {slot}/g{identity.generation} landed "
            f"{provider['machine_type']} while {selected_shape} is selected; exact-ID "
            "zero-attempt teardown starts before science."
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=False,
            lifecycle_lock=lifecycle_lock,
        )
    matching_slots = sorted(matching_slots)
    usable_count = min(
        len(logical_slots),
        (len(matching_slots) // AUDIT_BLOCK_WIDTH) * AUDIT_BLOCK_WIDTH,
    )
    if usable_count < AUDIT_BLOCK_WIDTH:
        raise ControllerError(
            "capacity assembly did not retain one complete three-VM audit block"
        )
    selected = tuple(matching_slots[:usable_count])
    for slot in matching_slots[usable_count:]:
        identity = executor.active[slot]
        backend.note(
            f"CONCURRENT-BLOCK WIDTH FILTER: {slot}/g{identity.generation} is outside "
            "the largest complete three-slot lane set; exact-ID zero-attempt teardown "
            "starts before science."
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=False,
            lifecycle_lock=lifecycle_lock,
        )
    return selected


def _execute_concurrent_block_batches(
    *,
    base,
    p1,
    pexec,
    executor,
    registry,
    backend,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    selected_shape: str,
    selected_slots: Sequence[str],
    preferred_replacement_zone: str,
    lock: threading.RLock,
    lifecycle_lock: threading.RLock,
    watchdog: CostWatchdog,
    campaign_root: Path,
    spend_ledger: Mapping[str, Any],
    stage_code: str,
    ceiling: float,
) -> None:
    contract_hash = str(roster["audit_135m_design_contract_hash"])
    planned_index = 0
    actual_wave_index = 0
    concurrent_batch_index = 0
    selected_slot_names = tuple(sorted(selected_slots))
    while planned_index < len(plan["waves"]):
        watchdog.raise_if_triggered()
        remaining = len(plan["waves"]) - planned_index
        active_selected = tuple(
            slot for slot in selected_slot_names if slot in executor.active
        )
        concurrent_blocks = min(
            MAX_CONCURRENT_BLOCKS,
            remaining,
            len(active_selected) // AUDIT_BLOCK_WIDTH,
        )
        if concurrent_blocks < 1:
            raise ControllerError("no complete audit block lane remains READY")
        required_slot_count = concurrent_blocks * AUDIT_BLOCK_WIDTH
        batch_slots = active_selected[:required_slot_count]
        for slot in active_selected[required_slot_count:]:
            identity = executor.active[slot]
            backend.note(
                f"NO-IDLE WIDTH SHRINK: {slot}/g{identity.generation} is surplus "
                f"for the final {concurrent_blocks}-block batch; exact-ID teardown "
                "starts before the other slots enter scientific work."
            )
            _finalize_identity(
                executor=executor,
                identity=identity,
                preempted=False,
                lifecycle_lock=lifecycle_lock,
            )
        selected_slot_names = batch_slots
        pending = {
            block_index: {"retry_round": 1, "prior_rows": None}
            for block_index in range(
                planned_index, planned_index + concurrent_blocks
            )
        }
        while pending:
            watchdog.raise_if_triggered()
            _cost_guard(
                executor=executor,
                campaign_root=campaign_root,
                stage_ledger=spend_ledger,
                stage_code=stage_code,
                plan=plan,
                scientific=scientific,
                planned_index=min(pending),
                ceiling=ceiling,
            )
            lanes = {
                block_index: pexec.audit_concurrent_block_slots(
                    contract_hash=contract_hash,
                    block_index=block_index,
                    available_slots=batch_slots,
                )
                for block_index in pending
            }
            if len({slot for lane in lanes.values() for slot in lane}) != (
                len(lanes) * AUDIT_BLOCK_WIDTH
            ):
                raise ControllerError("deterministic concurrent block lanes overlap")
            for block_index, lane in lanes.items():
                for slot in lane:
                    if slot in executor.active:
                        continue
                    backend.note(
                        f"MISSING LANE SLOT before block {block_index}: {slot}; "
                        "fresh exact-generation replacement starts loss-blindly."
                    )
                    _provision_required_replacement(
                        base=base,
                        p1=p1,
                        pexec=pexec,
                        executor=executor,
                        registry=registry,
                        backend=backend,
                        slot=slot,
                        selected_shape=selected_shape,
                        preferred_zone=preferred_replacement_zone,
                        lock=lock,
                        stage_code=stage_code,
                        plan=plan,
                        scientific=scientific,
                        planned_index=block_index,
                        spend_ledger=spend_ledger,
                        campaign_root=campaign_root,
                        ceiling=ceiling,
                        lifecycle_lock=lifecycle_lock,
                    )
            assigned_actual_indices = {}
            for block_index in sorted(pending):
                assigned_actual_indices[block_index] = actual_wave_index
                actual_wave_index += 1
            retry_summary = ", ".join(
                f"{index}:{pending[index]['retry_round']}"
                for index in sorted(pending)
            )
            backend.note(
                f"CONCURRENT BLOCK BATCH {concurrent_batch_index}: planned blocks "
                f"{sorted(pending)}, slots {list(batch_slots)}, retry rounds "
                f"{{{retry_summary}}}; "
                "losses remain SEALED/BLINDED."
            )
            results: dict[int, list[dict[str, Any]]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(pending)
            ) as pool:
                futures = {
                    block_index: pool.submit(
                        executor._execute_wave,
                        planned_wave=plan["waves"][block_index],
                        retry_round=int(state["retry_round"]),
                        actual_wave_index=assigned_actual_indices[block_index],
                        concurrent_batch_index=concurrent_batch_index,
                        available_slots=lanes[block_index],
                        concurrent_batch_slot_set=batch_slots,
                        prior_rows=state["prior_rows"],
                    )
                    for block_index, state in pending.items()
                }
                for block_index, future in futures.items():
                    results[block_index] = future.result()
            watchdog.raise_if_triggered()
            completed_blocks = []
            for block_index in sorted(pending):
                rows = results[block_index]
                statuses = {str(row["status"]) for row in rows}
                if "FAILED" in statuses:
                    raise pexec.ScheduleError("nonretryable FAILED audit block")
                preempted_slots = sorted(
                    {
                        str(row["logical_slot"])
                        for row in rows
                        if row["status"] == "INFRA_FAILURE"
                        and row.get("failure_reason")
                        == "provider_spot_preemption"
                    }
                )
                for slot in preempted_slots:
                    identity = executor.active.get(slot)
                    if identity is not None:
                        _finalize_identity(
                            executor=executor,
                            identity=identity,
                            preempted=True,
                            lifecycle_lock=lifecycle_lock,
                        )
                if "INFRA_FAILURE" in statuses:
                    pending[block_index]["prior_rows"] = rows
                    pending[block_index]["retry_round"] = (
                        int(pending[block_index]["retry_round"]) + 1
                    )
                    backend.note(
                        f"WHOLE BLOCK RETRY authorized loss-blindly for block "
                        f"{block_index} after {sorted(statuses)}; fresh attempt "
                        "namespaces and any preempted physical generations are replaced."
                    )
                else:
                    completed_blocks.append(block_index)
            for block_index in completed_blocks:
                del pending[block_index]
        planned_index += concurrent_blocks
        concurrent_batch_index += 1


def _evaluation_registry(
    *, packet: Path, campaign_root: Path, seed_registry: Mapping[str, Any], bound
) -> dict[str, Any]:
    frozen_tar = packet / "inputs" / "frozen-seed347.tar.gz"
    common = campaign_root / "common" / "evaluation"
    common.mkdir(parents=True, exist_ok=True)
    registry = {}
    with tarfile.open(frozen_tar, "r:gz") as archive:
        for seed in sorted(seed_registry["seeds"], key=int):
            member = archive.getmember(f"seed-{seed}/parallel-eval-freeze.json")
            destination = common / f"seed-{seed}.json"
            handle = archive.extractfile(member)
            if handle is None:
                raise ControllerError(f"cannot extract evaluation freeze for seed {seed}")
            with destination.open("xb") as target:
                shutil.copyfileobj(handle, target)
            from scripts import run_parallel_phase_map as pexec

            registry[seed] = {
                "path": f"common/evaluation/seed-{seed}.json",
                "sha256": pexec.sha256_file(destination),
                **{field: bound["frozen"][field] for field in pexec.EVAL_BOUND_FIELDS},
            }
    write_json_atomic(campaign_root / "common" / "evaluation-registry.json", registry)
    return registry


def _final_outputs(
    *,
    base,
    p1,
    campaign_root: Path,
    packet: Path,
    backend,
    vm_registry,
    census,
    completed: bool,
    controller_error: BaseException | None,
) -> None:
    p1.STAGE_CODE = str(backend.identity_plan["stage_code"])
    p1.write_final_outputs(
        base=base,
        campaign_root=campaign_root,
        packet=packet,
        backend=backend,
        vm_registry=vm_registry,
        census=census,
    )
    descriptor_path = campaign_root / "campaign" / "aggregation-descriptor.json"
    descriptor = load_json(descriptor_path)
    descriptor["runtime_authorization"] = str(packet / "runtime-authorization.json")
    # Authorization is granted only after the create-only campaign seal has
    # been reproduced and, for checkpoint stages, matched byte-for-byte to the
    # pre-hidden registry.  A failed controller can therefore never leave a
    # promotion-capable descriptor behind.
    descriptor["aggregation_authorized"] = False
    write_json_atomic(descriptor_path, descriptor)
    result_path = campaign_root / "campaign" / "controller-result.json"
    result = load_json(result_path)
    result.update(
        {
            "status": (
                "EXECUTION_AND_EXACT_ID_TEARDOWN_COMPLETE"
                if completed
                else "EXECUTION_ABORTED_AND_EXACT_ID_TEARDOWN_COMPLETE"
            ),
            "execution_complete": completed,
            "aggregation_authorized": False,
            "controller_error_type": (
                None if controller_error is None else type(controller_error).__name__
            ),
            "controller_error_detail": (
                None if controller_error is None else str(controller_error)
            ),
        }
    )
    write_json_atomic(result_path, result)


def _seal_and_authorize_aggregation(
    *,
    pexec,
    campaign_root: Path,
    checkpoint_preseal_path: Path | None,
) -> dict[str, Any]:
    descriptor_path = campaign_root / "campaign" / "aggregation-descriptor.json"
    seal = pexec.aggregate_from_descriptor(descriptor_path, write_seal=True)
    manifest_path = campaign_root / "campaign" / "campaign-manifest.json"
    seal_path = campaign_root / "campaign" / "campaign-seal.json"
    manifest = load_json(manifest_path)
    if checkpoint_preseal_path is not None:
        preseal = load_json(checkpoint_preseal_path)
        if (
            manifest.get("attempts") != preseal.get("attempts")
            or manifest.get("audit_checkpoint_registry")
            != preseal.get("audit_checkpoint_registry")
        ):
            raise ControllerError(
                "final campaign attempts/checkpoint registry differ from the preseal"
            )
    descriptor = load_json(descriptor_path)
    descriptor["aggregation_authorized"] = True
    descriptor["campaign_manifest"] = str(manifest_path)
    descriptor["campaign_seal"] = str(seal_path)
    result_path = campaign_root / "campaign" / "controller-result.json"
    result = load_json(result_path)
    result.update(
        {
            "aggregation_authorized": True,
            "campaign_manifest": str(manifest_path),
            "campaign_seal": str(seal_path),
            "campaign_manifest_canonical_sha256": pexec.canonical_sha256(manifest),
            "campaign_seal_raw_sha256": pexec.sha256_file(seal_path),
            "checkpoint_registry_exactly_reproduced_after_teardown": (
                checkpoint_preseal_path is not None
            ),
        }
    )
    write_json_atomic(result_path, result)
    # This is the final mutable authorization write.  Any earlier failure leaves
    # the descriptor at aggregation_authorized=false.
    write_json_atomic(descriptor_path, descriptor)
    return seal


def _finalize_all(executor, registry, base, lifecycle_lock: threading.RLock) -> None:
    for identity in list(executor.active.values()):
        operator_killed = (
            identity.slot,
            identity.generation,
        ) in getattr(executor.backend, "audit_operator_kills", {})
        provider_absent = (
            executor.backend.describe_instance(identity.run_id, check=False) is None
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=(operator_killed or provider_absent),
            lifecycle_lock=lifecycle_lock,
        )
    for identity in registry.identities:
        key = (identity.slot, identity.generation)
        state = base.load_json(identity.state_path)
        if state.get("status") == "vm_lifecycle_final":
            continue
        if key in executor.providers and key in executor.partials:
            operator_killed = key in getattr(
                executor.backend, "audit_operator_kills", {}
            )
            provider_absent = (
                executor.backend.describe_instance(identity.run_id, check=False) is None
            )
            _finalize_identity(
                executor=executor,
                identity=identity,
                preempted=(operator_killed or provider_absent),
                lifecycle_lock=lifecycle_lock,
            )


def _provision_required_replacement(
    *,
    base,
    p1,
    pexec,
    executor,
    registry,
    backend,
    slot: str,
    selected_shape: str,
    preferred_zone: str,
    lock: threading.RLock,
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    planned_index: int,
    spend_ledger: Mapping[str, Any],
    campaign_root: Path,
    ceiling: float,
    lifecycle_lock: threading.RLock,
) -> None:
    science_started = threading.Event()
    science_started.set()
    stop_event = threading.Event()

    def pre_launch_guard() -> None:
        _guard_provider_mutation_cost(
            executor=executor,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            stage_code=stage_code,
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
            ceiling=ceiling,
            phase="required_replacement_prelaunch",
        )

    for attempt in range(len(ZONE_ROTATION) * 4):
        current_spend, _rows = _current_campaign_cost(executor, campaign_root)
        eligible_zones = _cost_eligible_zone_rotation(
            stage_code=stage_code,
            machine_type=selected_shape,
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
            prior_stage_spend=float(spend_ledger["estimated_spend_usd"]),
            current_campaign_spend=current_spend,
            ceiling=ceiling,
        )
        if not eligible_zones:
            raise HardCeilingStop(
                "no registered zone for the required replacement can finish "
                "inside the remaining stage ceiling"
            )
        ordered_zones = (
            [preferred_zone, *[zone for zone in eligible_zones if zone != preferred_zone]]
            if preferred_zone in eligible_zones
            else list(eligible_zones)
        )
        offset = attempt % len(ordered_zones)
        p1.ZONE_ROTATION = tuple(ordered_zones[offset:] + ordered_zones[:offset])
        errors: list[BaseException] = []
        backend.note(
            f"REPLACEMENT CAPACITY attempt {attempt + 1} for {slot}: fresh physical "
            f"generation, preferred surviving shape {selected_shape}, first zone "
            f"{p1.ZONE_ROTATION[0]}; no prior scientific state is reusable."
        )
        _audit_provision_loop(
            base=base,
            p1=p1,
            pexec=pexec,
            executor=executor,
            registry=registry,
            backend=backend,
            slot=slot,
            stop_event=stop_event,
            science_started=science_started,
            lock=lock,
            errors=errors,
            required=True,
            pre_launch_guard=pre_launch_guard,
        )
        if errors:
            if isinstance(errors[0], HardCeilingStop):
                raise errors[0]
            raise ControllerError("required replacement provisioning failed") from errors[0]
        identity = executor.active.get(slot)
        if identity is None:
            raise ControllerError("required replacement did not reach READY")
        provider = executor.providers[(slot, identity.generation)]
        current_spend, _rows = _current_campaign_cost(executor, campaign_root)
        forecast = _shape_stage_forecast(
            stage_code=stage_code,
            machine_type=str(provider["machine_type"]),
            region=str(provider["region"]),
            plan=plan,
            scientific=scientific,
            planned_index=planned_index,
        )
        eligible = (
            provider["machine_type"] == selected_shape
            and float(spend_ledger["estimated_spend_usd"])
            + current_spend
            + forecast
            < ceiling
        )
        if eligible:
            backend.note(
                f"REPLACEMENT READY {slot}/g{identity.generation}: exact instance "
                f"{provider['instance_numeric_id']}, shape {selected_shape}, zone "
                f"{provider['zone']}; cost guard PASS and fresh retry may continue."
            )
            p1.ZONE_ROTATION = ZONE_ROTATION
            return
        backend.note(
            f"REPLACEMENT SHAPE/COST FILTER {slot}/g{identity.generation}: landed "
            f"{provider['machine_type']} in {provider['zone']} with remaining forecast "
            f"${forecast:.2f}; exact-ID zero-attempt teardown starts."
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=False,
            lifecycle_lock=lifecycle_lock,
        )
    p1.ZONE_ROTATION = ZONE_ROTATION
    raise ControllerError("replacement search exhausted the bounded zone/shape ladder")


def _provision_cost_eligible_initial_generation(
    *,
    base,
    p1,
    pexec,
    executor,
    registry,
    backend,
    slot: str,
    lock: threading.RLock,
    stage_code: str,
    plan: Mapping[str, Any],
    scientific: Mapping[str, Any],
    spend_ledger: Mapping[str, Any],
    campaign_root: Path,
    ceiling: float,
    lifecycle_lock: threading.RLock,
) -> tuple[str, float]:
    science_started = threading.Event()
    science_started.set()
    stop_event = threading.Event()

    def pre_launch_guard() -> None:
        _guard_provider_mutation_cost(
            executor=executor,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            stage_code=stage_code,
            plan=plan,
            scientific=scientific,
            planned_index=0,
            ceiling=ceiling,
            phase="pre_scientific_cost_capacity_ladder",
        )

    for attempt in range(len(ZONE_ROTATION) * 4):
        current_spend, _rows = _current_campaign_cost(executor, campaign_root)
        cheapest_first = _cost_eligible_zone_rotation(
            stage_code=stage_code,
            machine_type="a2-highgpu-1g",
            plan=plan,
            scientific=scientific,
            planned_index=0,
            prior_stage_spend=float(spend_ledger["estimated_spend_usd"]),
            current_campaign_spend=current_spend,
            ceiling=ceiling,
        )
        if not cheapest_first:
            pre_launch_guard()
            raise HardCeilingStop(
                "no registered 1g zone can finish inside the remaining stage ceiling"
            )
        offset = attempt % len(cheapest_first)
        p1.ZONE_ROTATION = tuple(
            cheapest_first[offset:] + cheapest_first[:offset]
        )
        errors: list[BaseException] = []
        backend.note(
            f"COST-ELIGIBLE CAPACITY search {attempt + 1} for {slot}: first zone "
            f"{p1.ZONE_ROTATION[0]}, Spot 1g preferred under the ceiling amendment; "
            "provider stockout rotates the registered zones."
        )
        _audit_provision_loop(
            base=base,
            p1=p1,
            pexec=pexec,
            executor=executor,
            registry=registry,
            backend=backend,
            slot=slot,
            stop_event=stop_event,
            science_started=science_started,
            lock=lock,
            errors=errors,
            required=True,
            pre_launch_guard=pre_launch_guard,
        )
        if errors:
            p1.ZONE_ROTATION = ZONE_ROTATION
            if isinstance(errors[0], HardCeilingStop):
                raise errors[0]
            raise ControllerError("cost-eligible capacity search failed") from errors[0]
        identity = executor.active.get(slot)
        if identity is None:
            p1.ZONE_ROTATION = ZONE_ROTATION
            raise ControllerError("cost-eligible capacity search did not reach READY")
        provider = executor.providers[(slot, identity.generation)]
        current_spend, _rows = _current_campaign_cost(executor, campaign_root)
        forecast = _shape_stage_forecast(
            stage_code=stage_code,
            machine_type=str(provider["machine_type"]),
            region=str(provider["region"]),
            plan=plan,
            scientific=scientific,
            planned_index=0,
        )
        if (
            float(spend_ledger["estimated_spend_usd"])
            + current_spend
            + forecast
            < ceiling
        ):
            p1.ZONE_ROTATION = ZONE_ROTATION
            backend.note(
                f"COST-ELIGIBLE READY {slot}/g{identity.generation}: "
                f"{provider['machine_type']} in {provider['zone']}, remaining stage "
                f"forecast ${forecast:.2f}. Science begins immediately."
            )
            return str(provider["machine_type"]), forecast
        backend.note(
            f"COST-INELIGIBLE READY {slot}/g{identity.generation}: "
            f"{provider['machine_type']} in {provider['zone']} would put the stage "
            f"at/above ${ceiling:.2f}; exact-ID zero-attempt teardown starts without "
            "scientific dispatch."
        )
        _finalize_identity(
            executor=executor,
            identity=identity,
            preempted=False,
            lifecycle_lock=lifecycle_lock,
        )
    p1.ZONE_ROTATION = ZONE_ROTATION
    raise ControllerError("cost-eligible capacity search exhausted the bounded ladder")


def _deferred_evaluation_role(scientific: Mapping[str, Any]) -> str | None:
    modes = {
        str(cell.get("evaluation_mode", "development_endpoint"))
        for cell in scientific.get("cells", [])
        if isinstance(cell, Mapping)
    }
    deferred = modes & {
        "confirmation_audit_pending",
        "development_prediction_pending",
    }
    if len(deferred) > 1:
        raise ControllerError("one campaign cannot mix confirmation and prediction endpoints")
    if not deferred:
        return None
    mode = next(iter(deferred))
    return (
        "confirmation_audit"
        if mode == "confirmation_audit_pending"
        else "development_prediction_endpoint"
    )


def _checkpoint_preseal(
    *,
    pexec,
    stage_code: str,
    packet: Path,
    campaign_root: Path,
    registry,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    plan: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
    runtime_authorization: Mapping[str, Any],
    campaign_attempt: int,
) -> Path:
    value = pexec.build_audit_checkpoint_preseal(
        stage_code=stage_code,
        parent_manifest=parent,
        bound_manifest=bound,
        scientific_plan=scientific,
        roster=roster,
        parallel_plan=plan,
        vm_registry=registry.snapshot(),
        evaluation_registry=evaluation_registry,
        campaign_attempt=campaign_attempt,
        campaign_root=campaign_root,
        runtime_authorization=runtime_authorization,
    )
    path = campaign_root / "campaign" / "checkpoint-preseal.json"
    write_json_create_only(path, value)
    return path


def _a3_prediction_freeze(
    *,
    args: argparse.Namespace,
    checkpoint_preseal: Path,
    bound_manifest: Path,
    campaign_root: Path,
) -> Path:
    required = {
        "a3_mechanical_gate": args.a3_mechanical_gate,
        "a3_historical_phase_manifest": args.a3_historical_phase_manifest,
        "a3_recapture_campaign_manifest": args.a3_recapture_campaign_manifest,
        "a3_recapture_bound_manifest": args.a3_recapture_bound_manifest,
        "a3_recapture_campaign_root": args.a3_recapture_campaign_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ControllerError(f"A3 prediction freeze lacks inputs: {missing}")
    from scripts import audit_135m_kernel_law as kernel

    output = campaign_root / "campaign" / "a3-prediction-freeze.json"
    kernel.freeze(
        argparse.Namespace(
            mechanical_gate=args.a3_mechanical_gate,
            historical_phase_manifest=args.a3_historical_phase_manifest,
            recapture_campaign_manifest=args.a3_recapture_campaign_manifest,
            recapture_bound_manifest=args.a3_recapture_bound_manifest,
            recapture_campaign_root=args.a3_recapture_campaign_root,
            current_checkpoint_preseal=checkpoint_preseal,
            current_bound_manifest=bound_manifest,
            current_campaign_root=campaign_root,
            output=output,
            sealed_at_utc=None,
        )
    )
    return output


def _hidden_authorization(
    *,
    checkpoint_preseal: Path,
    bound_manifest: Path,
    campaign_root: Path,
    evaluation_role: str,
    prediction_freeze: Path | None,
) -> Path:
    from scripts import audit_135m_hidden_evaluator as hidden

    output = campaign_root / "campaign" / "hidden-authorization.json"
    hidden.authorize(
        argparse.Namespace(
            checkpoint_preseal=checkpoint_preseal,
            bound_manifest=bound_manifest,
            campaign_root=campaign_root,
            evaluation_role=evaluation_role,
            prediction_freeze=prediction_freeze,
            output=output,
        )
    )
    return output


def _latest_scientific_end(executor, identity) -> datetime:
    partial = load_json(
        executor._local_vm_root(identity) / "manifests" / "vm-partial-manifest.json"
    )
    ends = [
        parse_time(str(row["scientific_ended_at"]))
        for row in partial.get("attempts", [])
        if isinstance(row, Mapping) and isinstance(row.get("scientific_ended_at"), str)
    ]
    return max(ends, default=datetime.min.replace(tzinfo=timezone.utc))


def _select_hidden_survivor(executor):
    if not executor.active:
        raise ControllerError("deferred evaluation has no surviving READY generation")
    return max(
        executor.active.values(),
        key=lambda identity: (
            _latest_scientific_end(executor, identity),
            identity.slot,
        ),
    )


def _remote_hidden_command(
    *,
    identity,
    backend,
    science_root: str,
    evaluation_role: str,
    first_seed: int,
    remote_authorization: str,
    remote_preseal: str,
    remote_bound: str,
    remote_output: str,
) -> str:
    repo = "/tmp/yeto-best-paper"
    python = "/home/shou/venv/bin/python"
    command = [
        python,
        repo + "/scripts/audit_135m_hidden_evaluator.py",
        "evaluate",
        "--authorization",
        remote_authorization,
        "--checkpoint-preseal",
        remote_preseal,
        "--bound-manifest",
        remote_bound,
        "--model",
        science_root.rstrip("/") + "/inputs/model",
        "--python-executable",
        python,
        "--compare-script",
        repo + "/scripts/compare_diloco.py",
        "--device",
        "cuda",
        "--output-dir",
        remote_output,
    ]
    if evaluation_role == "confirmation_audit":
        command.extend(
            [
                "--source-data",
                science_root.rstrip("/") + "/inputs/train.parquet",
            ]
        )
    else:
        seed_root = (
            science_root.rstrip("/") + f"/phase-map/frozen-eval/seed-{first_seed}"
        )
        command.extend(
            [
                "--development-eval",
                seed_root + "/materialized/eval.jsonl",
                "--development-eval-freeze",
                seed_root + "/parallel-eval-freeze.json",
            ]
        )
    quoted = " ".join(shlex.quote(value) for value in command)
    return (
        "set -eu; export PYTHONPATH=/tmp/yeto-best-paper; "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
        + quoted
    )


def _sync_hidden_output(
    *, backend, identity, remote_output: str, local_parent: Path
) -> Path:
    local_parent.mkdir(parents=True, exist_ok=True)
    backend.gcloud(
        "compute",
        "scp",
        "--recurse",
        f"{identity.run_id}:{remote_output}",
        str(local_parent),
        "--project=model-training-497007",
        f"--zone={backend.zone_for_name(identity.run_id)}",
        "--quiet",
        timeout=3600,
    )
    local = local_parent / Path(remote_output).name
    if not all(
        (local / name).is_file()
        for name in ("audit-bundle.json", "audit-seal.json", "shared-unblind.json")
    ):
        raise ControllerError("hidden output sync lacks its complete seal/unblind trio")
    return local


def _execute_hidden_on_survivor(
    *,
    executor,
    backend,
    identity,
    campaign_root: Path,
    science_root: str,
    evaluation_role: str,
    first_seed: int,
    authorization: Path,
    checkpoint_preseal: Path,
    bound_manifest: Path,
    batch_attempt: int,
    prediction_freeze: Path | None,
) -> Path:
    stem = f"{identity.run_id}-hidden"
    remote_authorization = f"/home/shou/{stem}-authorization.json"
    remote_preseal = f"/home/shou/{stem}-preseal.json"
    remote_bound = f"/home/shou/{stem}-bound.json"
    remote_output = (
        f"/tmp/audit-135m-hidden/{identity.run_id}/batch-attempt-{batch_attempt}"
    )
    backend._stream_file(identity.run_id, authorization, remote_authorization)
    backend._stream_file(identity.run_id, checkpoint_preseal, remote_preseal)
    backend._stream_file(identity.run_id, bound_manifest, remote_bound)
    command = _remote_hidden_command(
        identity=identity,
        backend=backend,
        science_root=science_root,
        evaluation_role=evaluation_role,
        first_seed=first_seed,
        remote_authorization=remote_authorization,
        remote_preseal=remote_preseal,
        remote_bound=remote_bound,
        remote_output=remote_output,
    )
    try:
        result = backend.remote(identity.run_id, command, check=False, timeout=21600)
    except subprocess.TimeoutExpired as exc:
        if backend.describe_instance(identity.run_id, check=False) is None:
            raise HiddenSurvivorPreempted(
                "hidden evaluator survivor disappeared during the remote batch"
            ) from exc
        raise ControllerError("hidden evaluator remote command timed out") from exc
    if result.returncode:
        if backend.describe_instance(identity.run_id, check=False) is None:
            raise HiddenSurvivorPreempted(
                "hidden evaluator survivor was Spot-preempted before shared unblind"
            )
        raise ControllerError(
            "hidden evaluation failed before complete sealing; private output remains unopened"
        )
    try:
        local = _sync_hidden_output(
            backend=backend,
            identity=identity,
            remote_output=remote_output,
            local_parent=campaign_root / "campaign" / "hidden",
        )
    except BaseException as exc:
        if backend.describe_instance(identity.run_id, check=False) is None:
            raise HiddenSurvivorPreempted(
                "hidden evaluator survivor disappeared before sealed-output sync"
            ) from exc
        raise
    from scripts import audit_135m_phase_manifest as phase_promotion

    phase_promotion._verify_hidden(
        hidden_root=local,
        authorization_path=authorization,
        preseal=load_json(checkpoint_preseal),
        bound=load_json(bound_manifest),
        prediction_freeze_path=prediction_freeze,
    )
    return local


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet = args.packet_root.resolve()
    campaign_root = args.campaign_root.resolve()
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise ControllerError(f"campaign root is not empty: {campaign_root}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    identity_plan = load_json(packet / "identity-plan.json")
    review = load_json(packet / "review-packet.json")
    stage_code = str(identity_plan["stage_code"])
    target_width = int(identity_plan["target_width"])
    logical_slots = tuple(identity_plan.get("logical_slots", ()))
    target_1g_slot_count = int(identity_plan.get("target_1g_slot_count", 0))
    maximum_concurrent_blocks = int(
        identity_plan.get("maximum_concurrent_blocks", 0)
    )
    assembly_seconds = int(identity_plan["assembly_max_seconds"])
    if (
        review.get("status") != "SEALED_LAUNCH_AUTHORIZED"
        or review.get("stage_code") != stage_code
        or target_width != AUDIT_BLOCK_WIDTH
        or not logical_slots
        or len(logical_slots) != target_1g_slot_count
        or len(logical_slots) > len(SLOTS)
        or len(logical_slots) % AUDIT_BLOCK_WIDTH
        or any(slot not in SLOTS for slot in logical_slots)
        or review.get("logical_slots") != list(logical_slots)
        or int(review.get("target_1g_slot_count", 0)) != target_1g_slot_count
        or int(review.get("maximum_concurrent_blocks", 0))
        != maximum_concurrent_blocks
        or maximum_concurrent_blocks
        != min(MAX_CONCURRENT_BLOCKS, int(identity_plan["wave_count"]))
        or assembly_seconds > 480
        or tuple(identity_plan["zone_rotation"]) != ZONE_ROTATION
        or int(review.get("global_attached_a100_equivalent_at_preflight", 10**9))
        + PREFERRED_PROBE_A100S
        > GLOBAL_A100_CEILING
    ):
        raise ControllerError("packet launch identity/width/zone contract differs")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(P1_SESSION))
    sys.path.insert(0, "/private/tmp/yeto-p1r0-launcher")
    from scripts import run_parallel_phase_map as pexec
    import build_audit_135m_launch_packet as packet_builder

    packet_builder.configure_from_identity_plan(identity_plan)
    sys.modules["build_launch_packet"] = packet_builder
    p1 = load_module("audit_135m_p1_capacity_controller", P1_CONTROLLER)
    p1.STAGE_CODE = stage_code
    p1.SLOTS = logical_slots
    p1.ZONE_ROTATION = ZONE_ROTATION
    p1.NOTE_PATH = NOTE_PATH
    p1.RETRY_SECONDS = 600
    p1.ASSEMBLY_MAX_SECONDS = assembly_seconds
    base = load_module("audit_135m_reviewed_gcp_backend", R0_CONTROLLER)
    p1.patch_base(base, identity_plan)
    _bind_reviewed_backend_workspace(base)
    base.NOTE_PATH = NOTE_PATH
    base.GCLOUD_CONFIG = GCLOUD_CONFIG
    base.ALLOWED_ZONES = ZONE_ROTATION
    os.environ["P1AD_SCIENCE_ROOT"] = str(identity_plan["science_root"])

    roster = base.load_json(packet / "binding" / "parallel-roster.json")
    plan = base.load_json(packet / "binding" / "parallel-plan.json")
    scientific = base.load_json(
        packet / "materialized" / "scientific-randomization-plan.json"
    )
    bound = base.load_json(packet / "materialized" / "bound-manifest.json")
    parent = base.load_json(packet / "parent" / "parent-manifest.json")
    runtime_authorization = base.load_json(packet / "runtime-authorization.json")
    seed_registry = base.load_json(packet / "inputs" / "seed-bundle-registry.json")
    if (
        roster.get("stage_code") != stage_code
        or len(roster.get("launch_cells", [])) != identity_plan["launch_cell_count"]
        or len(plan.get("waves", [])) != identity_plan["wave_count"]
        or len(scientific.get("cells", [])) != identity_plan["launch_cell_count"]
    ):
        raise ControllerError("packet roster/plan dimensions differ")
    concurrent_binding = plan.get("audit_concurrent_block_binding")
    if (
        plan.get("schema") != "yeto_parallel_plan_v4"
        or not isinstance(concurrent_binding, Mapping)
        or concurrent_binding.get("amendment_raw_sha256")
        != pexec.sha256_file(
            REPO_ROOT / "docs" / "AMENDMENT-audit-135m-concurrent-blocks.md"
        )
        or int(concurrent_binding.get("block_width", 0)) != AUDIT_BLOCK_WIDTH
        or int(concurrent_binding.get("maximum_concurrent_blocks", 0))
        != MAX_CONCURRENT_BLOCKS
    ):
        raise ControllerError("packet concurrent-block amendment binding differs")
    evaluation_registry = _evaluation_registry(
        packet=packet,
        campaign_root=campaign_root,
        seed_registry=seed_registry,
        bound=bound,
    )
    registry = base.PreboundRegistry(
        pexec,
        prebound_nonces={
            row["slot"]: row["ownership_nonce"] for row in identity_plan["generations"]
        },
        stage_code=stage_code,
        study_id=bound["study_id"],
        roster_digest=identity_plan["roster_hash"],
        campaign_attempt=int(identity_plan["campaign_attempt"]),
        campaign_state_root=Path(identity_plan["campaign_state_root"]),
        campaign_artifact_root=identity_plan["campaign_artifact_root"],
    )
    backend = base.GcpBackend(
        pexec=pexec,
        packet_root=packet,
        campaign_root=campaign_root,
        evaluation_registry=evaluation_registry,
    )
    backend.private = (
        Path(identity_plan["controller_private_root"])
        / identity_plan["roster_tag"]
        / "c1"
    )
    backend.private.mkdir(parents=True, exist_ok=True)
    _install_operator_kill_lifecycle_patch(backend)
    _install_transient_provider_generation_monitor(
        backend=backend, campaign_root=campaign_root
    )
    _install_launch_census_and_direct_fallback_patch(
        base=base, backend=backend, campaign_root=campaign_root
    )
    executor = pexec.ParallelWaveExecutor(
        roster=roster,
        parallel_plan=plan,
        scientific_plan=scientific,
        bound_manifest=bound,
        registry=registry,
        campaign_root=campaign_root,
        backend=backend,
        available_slots=logical_slots,
        runtime_authorization=runtime_authorization,
    )
    ceiling = float(identity_plan["hard_ceiling_usd"])
    audit_stage = str(identity_plan["audit_stage"])
    authority = load_json(
        REPO_ROOT / "experiment-specs" / "tuned-baseline-audit-prereg.json"
    )
    stage_controls = authority.get("experiments", {}).get(audit_stage, {})
    if (
        int(stage_controls.get("width_cap", 0)) != MAX_CONCURRENT_BLOCKS
        or float(stage_controls.get("abort_burn_kill_usd", math.nan))
        != ABORT_BURN_KILL_USD
    ):
        raise ControllerError("registered width/abort-burn controls differ")
    spend_ledger = _load_spend_ledger(args.stage_spend_ledger, audit_stage, ceiling)
    census = executor._assert_capacity_census()
    if census["campaign_owned_vm_count"] or census["campaign_owned_attached_a100s"]:
        raise ControllerError("campaign tag already owns a cloud resource")
    global_census = _global_a100_census(backend)
    global_existing = int(global_census["total_attached_a100_equivalent"])
    global_headroom = GLOBAL_A100_CEILING - global_existing
    if global_headroom < PREFERRED_PROBE_A100S:
        raise ControllerError(
            f"global A100 headroom is {global_headroom}; a reviewed 1g Spot "
            "probe could exceed the total ceiling"
        )
    prelaunch_cost_lock = threading.RLock()

    def initial_pre_launch_guard() -> None:
        with prelaunch_cost_lock:
            _guard_abort_burn(
                executor=executor,
                campaign_root=campaign_root,
                stage_ledger=spend_ledger,
                phase="initial_pre_scientific_prelaunch",
            )
            _guard_provider_mutation_cost(
                executor=executor,
                campaign_root=campaign_root,
                stage_ledger=spend_ledger,
                stage_code=stage_code,
                plan=plan,
                scientific=scientific,
                planned_index=0,
                ceiling=ceiling,
                phase="initial_pre_scientific_prelaunch",
            )

    try:
        initial_pre_launch_guard()
    except HardCeilingStop as exc:
        backend.note(
            "HARD CEILING PREFLIGHT STOP before any provider mutation: "
            f"{exc}"
        )
        raise
    current_prelaunch_spend, _prelaunch_rows = _current_campaign_cost(
        executor, campaign_root
    )
    initial_cost_eligible_zones = _cost_eligible_zone_rotation(
        stage_code=stage_code,
        machine_type="a2-highgpu-1g",
        plan=plan,
        scientific=scientific,
        planned_index=0,
        prior_stage_spend=float(spend_ledger["estimated_spend_usd"]),
        current_campaign_spend=current_prelaunch_spend,
        ceiling=ceiling,
    )
    if not initial_cost_eligible_zones:
        raise HardCeilingStop(
            "no registered initial 1g zone can finish inside the remaining stage ceiling"
        )
    p1.ZONE_ROTATION = initial_cost_eligible_zones
    backend.note(
        "COST-ELIGIBLE INITIAL ZONE ROTATION bound before provider mutation: "
        f"{list(initial_cost_eligible_zones)}; every omitted region fails the "
        "registered remaining-stage forecast under the durable ledger."
    )
    initial_probe_width = min(
        MAX_PREFERRED_PROBE_WIDTH,
        len(logical_slots),
        global_headroom // PREFERRED_PROBE_A100S,
    )
    initial_slots = logical_slots[:initial_probe_width]
    backend.note(
        f"{stage_code.upper()} LAUNCH AUTHORIZED: {len(roster['launch_cells'])} cells, "
        f"registered target width {target_width}, global-headroom probe width "
        f"{initial_probe_width} ({global_existing}/{GLOBAL_A100_CEILING} A100s already "
        f"attached outside this campaign), hard ceiling ${ceiling:.2f}, current stage "
        f"spend ${float(spend_ledger['estimated_spend_usd']):.2f}, Spot only, "
        "losses SEALED/BLINDED, exact zero-resource census PASS."
    )

    stop_event = threading.Event()
    science_started = threading.Event()
    lock = threading.RLock()
    lifecycle_lock = threading.RLock()
    errors: list[BaseException] = []
    threads = []
    completed = False
    controller_error: BaseException | None = None
    watchdog: CostWatchdog | None = None
    checkpoint_preseal_path: Path | None = None
    prediction_freeze_path: Path | None = None
    hidden_authorization_path: Path | None = None
    hidden_root: Path | None = None
    campaign_seal: dict[str, Any] | None = None
    try:
        for slot in initial_slots:
            thread = threading.Thread(
                target=_audit_provision_loop,
                kwargs={
                    "base": base,
                    "p1": p1,
                    "pexec": pexec,
                    "executor": executor,
                    "registry": registry,
                    "backend": backend,
                    "slot": slot,
                    "stop_event": stop_event,
                    "science_started": science_started,
                    "lock": lock,
                    "errors": errors,
                    "required": False,
                    "pre_launch_guard": initial_pre_launch_guard,
                },
                name=f"audit-{stage_code}-initial-{slot}",
                daemon=False,
            )
            thread.start()
            threads.append(thread)
        first_ready_epoch: float | None = None
        while True:
            if errors:
                if isinstance(errors[0], HardCeilingStop):
                    raise errors[0]
                raise ControllerError("capacity thread failed") from errors[0]
            with lock:
                width = len(executor.active)
                ready_values = list(executor.ready_at.values())
            if width and first_ready_epoch is None:
                first_ready_epoch = min(parse_time(value).timestamp() for value in ready_values)
                backend.note(
                    f"INITIAL ASSEMBLY first READY width {width}; at most "
                    f"{assembly_seconds}s remain before immediate science/cleanup."
                )
            if width >= initial_probe_width:
                break
            if first_ready_epoch is not None and time.time() - first_ready_epoch >= assembly_seconds:
                break
            time.sleep(5)
        selection_error: ControllerError | None = None
        with lock:
            if not executor.active:
                raise ControllerError("assembly closed without a READY generation")
            current_spend, _generation_rows = _current_campaign_cost(
                executor, campaign_root
            )
            try:
                selected_shape, selected_slots, stage_forecast = (
                    _select_cost_eligible_shape(
                        executor=executor,
                        stage_code=stage_code,
                        ceiling=ceiling,
                        prior_spend=float(spend_ledger["estimated_spend_usd"]),
                        current_spend=current_spend,
                        plan=plan,
                        scientific=scientific,
                        planned_index=0,
                        target_width=initial_probe_width,
                    )
                )
            except ControllerError as exc:
                if "no landed machine shape" not in str(exc):
                    raise
                selection_error = exc
                selected_shape = ""
                selected_slots = ()
                stage_forecast = 0.0
            _guard_abort_burn(
                executor=executor,
                campaign_root=campaign_root,
                stage_ledger=spend_ledger,
                phase="initial_assembly_before_science",
            )
            science_started.set()
            stop_event.set()
        for thread in threads:
            thread.join()
        if errors:
            if isinstance(errors[0], HardCeilingStop):
                raise errors[0]
            raise ControllerError("capacity cleanup thread failed") from errors[0]
        if selection_error is not None:
            backend.note(
                "INITIAL LANDED FLEET has no shape/region that can finish inside the "
                "stage ceiling; every landed generation is torn down before science, "
                "then the bounded cheapest-region ladder continues."
            )
            _finalize_nonselected(
                executor=executor,
                selected_slots=set(),
                lifecycle_lock=lifecycle_lock,
            )
            selected_slot = initial_slots[0]
            selected_shape, stage_forecast = (
                _provision_cost_eligible_initial_generation(
                    base=base,
                    p1=p1,
                    pexec=pexec,
                    executor=executor,
                    registry=registry,
                    backend=backend,
                    slot=selected_slot,
                    lock=lock,
                    stage_code=stage_code,
                    plan=plan,
                    scientific=scientific,
                    spend_ledger=spend_ledger,
                    campaign_root=campaign_root,
                    ceiling=ceiling,
                    lifecycle_lock=lifecycle_lock,
                )
            )
            selected_slots = (selected_slot,)
        else:
            _finalize_nonselected(
                executor=executor,
                selected_slots=set(selected_slots),
                lifecycle_lock=lifecycle_lock,
            )
        preferred_replacement_zone = str(
            executor.providers[
                (
                    selected_slots[0],
                    executor.active[selected_slots[0]].generation,
                )
            ]["zone"]
        )
        remaining_assembly_seconds = max(
            0,
            assembly_seconds
            - int(
                0
                if first_ready_epoch is None
                else max(0.0, time.time() - first_ready_epoch)
            ),
        )
        selected_slots = _expand_homogeneous_initial_capacity(
            base=base,
            p1=p1,
            pexec=pexec,
            executor=executor,
            registry=registry,
            backend=backend,
            logical_slots=logical_slots,
            selected_shape=selected_shape,
            preferred_zone=preferred_replacement_zone,
            assembly_seconds=remaining_assembly_seconds,
            lock=lock,
            lifecycle_lock=lifecycle_lock,
            pre_launch_guard=initial_pre_launch_guard,
        )
        if set(executor.active) != set(selected_slots):
            raise ControllerError("post-assembly active slots differ from the selected set")
        if (
            int(_global_a100_census(backend)["total_attached_a100_equivalent"])
            > GLOBAL_A100_CEILING
        ):
            raise ControllerError("post-assembly global A100 census exceeds 16")
        backend.note(
            f"INITIAL ASSEMBLY CLOSED on shape {selected_shape}, slots "
            f"{list(selected_slots)}, concurrent block width "
            f"{len(selected_slots) // AUDIT_BLOCK_WIDTH}, stage-wide forecast "
            f"${stage_forecast:.2f}; batch 0 dispatch begins immediately."
        )
        _guard_abort_burn(
            executor=executor,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            phase="final_capacity_before_first_scientific_dispatch",
        )
        _cost_guard(
            executor=executor,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            stage_code=stage_code,
            plan=plan,
            scientific=scientific,
            planned_index=0,
            ceiling=ceiling,
        )
        watchdog = CostWatchdog(
            executor=executor,
            backend=backend,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            ceiling=ceiling,
            lifecycle_lock=lifecycle_lock,
        )
        watchdog.start()

        _execute_concurrent_block_batches(
            base=base,
            p1=p1,
            pexec=pexec,
            executor=executor,
            registry=registry,
            backend=backend,
            plan=plan,
            scientific=scientific,
            roster=roster,
            selected_shape=selected_shape,
            selected_slots=selected_slots,
            preferred_replacement_zone=preferred_replacement_zone,
            lock=lock,
            lifecycle_lock=lifecycle_lock,
            watchdog=watchdog,
            campaign_root=campaign_root,
            spend_ledger=spend_ledger,
            stage_code=stage_code,
            ceiling=ceiling,
        )
        watchdog.raise_if_triggered()
        evaluation_role = _deferred_evaluation_role(scientific)
        if evaluation_role is not None:
            checkpoint_preseal_path = _checkpoint_preseal(
                pexec=pexec,
                stage_code=stage_code,
                packet=packet,
                campaign_root=campaign_root,
                registry=registry,
                parent=parent,
                bound=bound,
                scientific=scientific,
                roster=roster,
                plan=plan,
                evaluation_registry=evaluation_registry,
                runtime_authorization=runtime_authorization,
                campaign_attempt=int(identity_plan["campaign_attempt"]),
            )
            if evaluation_role == "development_prediction_endpoint":
                if stage_code != "a3r0":
                    raise ControllerError(
                        "prediction-first hidden evaluation is restricted to A3-R0"
                    )
                prediction_freeze_path = _a3_prediction_freeze(
                    args=args,
                    checkpoint_preseal=checkpoint_preseal_path,
                    bound_manifest=packet
                    / "materialized"
                    / "bound-manifest.json",
                    campaign_root=campaign_root,
                )
            hidden_authorization_path = _hidden_authorization(
                checkpoint_preseal=checkpoint_preseal_path,
                bound_manifest=packet / "materialized" / "bound-manifest.json",
                campaign_root=campaign_root,
                evaluation_role=evaluation_role,
                prediction_freeze=prediction_freeze_path,
            )
            survivor = _select_hidden_survivor(executor)
            idle_seconds = (
                datetime.now(timezone.utc)
                - _latest_scientific_end(executor, survivor)
            ).total_seconds()
            if idle_seconds >= 600:
                raise ControllerError(
                    "selected hidden-evaluation survivor exceeded the 600s idle rail"
                )
            backend.note(
                f"CHECKPOINT PRESEAL PASS and hidden authorization SEALED; survivor "
                f"{survivor.slot}/g{survivor.generation} begins the complete "
                f"{evaluation_role} batch after {idle_seconds:.1f}s idle while every peer "
                "starts exact-ID teardown. Losses remain SEALED/BLINDED."
            )
            peers = [
                identity
                for identity in list(executor.active.values())
                if identity != survivor
            ]
            first_seed = min(int(seed) for seed in seed_registry["seeds"])
            batch_attempt = 1
            pending_peer_teardown = peers
            while True:
                hidden_failure: HiddenSurvivorPreempted | None = None
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(2, len(pending_peer_teardown) + 1)
                ) as pool:
                    hidden_future = pool.submit(
                        _execute_hidden_on_survivor,
                        executor=executor,
                        backend=backend,
                        identity=survivor,
                        campaign_root=campaign_root,
                        science_root=str(identity_plan["science_root"]),
                        evaluation_role=evaluation_role,
                        first_seed=first_seed,
                        authorization=hidden_authorization_path,
                        checkpoint_preseal=checkpoint_preseal_path,
                        bound_manifest=packet
                        / "materialized"
                        / "bound-manifest.json",
                        batch_attempt=batch_attempt,
                        prediction_freeze=prediction_freeze_path,
                    )
                    peer_futures = [
                        pool.submit(
                            _finalize_identity,
                            executor=executor,
                            identity=identity,
                            preempted=False,
                            lifecycle_lock=lifecycle_lock,
                        )
                        for identity in pending_peer_teardown
                    ]
                    while not hidden_future.done():
                        watchdog.raise_if_triggered()
                        for future in peer_futures:
                            if future.done():
                                future.result()
                        time.sleep(5)
                    try:
                        candidate_hidden_root = hidden_future.result()
                    except HiddenSurvivorPreempted as exc:
                        hidden_failure = exc
                        candidate_hidden_root = None
                    for future in peer_futures:
                        future.result()
                    if hidden_failure is None:
                        watchdog.raise_if_triggered()
                        assert candidate_hidden_root is not None
                        hidden_root = candidate_hidden_root
                        pool.submit(
                            _finalize_identity,
                            executor=executor,
                            identity=survivor,
                            preempted=False,
                            lifecycle_lock=lifecycle_lock,
                        ).result()
                if hidden_failure is None:
                    break
                backend.note(
                    f"HIDDEN BATCH attempt {batch_attempt} lost exact survivor "
                    f"{survivor.slot}/g{survivor.generation} to Spot before shared "
                    "unblind; the complete batch will restart from the same sealed "
                    "checkpoint registry on a fresh physical generation."
                )
                if executor.active.get(survivor.slot) == survivor:
                    _finalize_identity(
                        executor=executor,
                        identity=survivor,
                        preempted=True,
                        lifecycle_lock=lifecycle_lock,
                    )
                watchdog.raise_if_triggered()
                _provision_required_replacement(
                    base=base,
                    p1=p1,
                    pexec=pexec,
                    executor=executor,
                    registry=registry,
                    backend=backend,
                    slot=survivor.slot,
                    selected_shape=selected_shape,
                    preferred_zone=preferred_replacement_zone,
                    lock=lock,
                    stage_code=stage_code,
                    plan=plan,
                    scientific=scientific,
                    planned_index=len(plan["waves"]),
                    spend_ledger=spend_ledger,
                    campaign_root=campaign_root,
                    ceiling=ceiling,
                    lifecycle_lock=lifecycle_lock,
                )
                survivor = executor.active[survivor.slot]
                pending_peer_teardown = []
                batch_attempt += 1
            backend.note(
                "HIDDEN BATCH SEALED/SHARED-UNBLINDED and synchronized only after its "
                "complete bundle; survivor exact-ID teardown PASS."
            )
        completed = True
    except BaseException as exc:
        controller_error = exc
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()
        try:
            _finalize_all(executor, registry, base, lifecycle_lock)
        except BaseException as teardown_error:
            if controller_error is None:
                controller_error = teardown_error
            else:
                backend.note(f"SECONDARY TEARDOWN ERROR: {teardown_error}")
        if watchdog is not None:
            watchdog.stop()
            try:
                watchdog.raise_if_triggered()
            except BaseException as watchdog_error:
                if controller_error is None:
                    controller_error = watchdog_error
                else:
                    backend.note(f"SECONDARY COST WATCHDOG ERROR: {watchdog_error}")
        census = executor._assert_capacity_census()
        if census["campaign_owned_vm_count"] or census["campaign_owned_attached_a100s"]:
            zero_error = ControllerError("final exact-ID teardown did not reach zero census")
            if controller_error is None:
                controller_error = zero_error
        current_cost, generation_costs = _current_campaign_cost(executor, campaign_root)
        final_stage_spend = float(spend_ledger["estimated_spend_usd"]) + current_cost
        if final_stage_spend >= ceiling:
            ceiling_error = HardCeilingStop(
                f"final estimated stage spend ${final_stage_spend:.6f} reaches/exceeds "
                f"the registered ${ceiling:.2f} ceiling"
            )
            if controller_error is None:
                controller_error = ceiling_error
        if controller_error is not None:
            completed = False
        transient_registry = _transient_provider_lifecycle_registry(campaign_root)
        transient_registry_path = (
            campaign_root / "campaign" / "transient-provider-registry.json"
        )
        write_json_atomic(transient_registry_path, transient_registry)
        scientific_attempt_started = _campaign_scientific_attempt_started(campaign_root)
        prior_abort_burn = float(
            spend_ledger["pre_science_aborted_launch_spend_usd"]
        )
        updated_abort_burn = (
            prior_abort_burn
            if scientific_attempt_started or completed
            else prior_abort_burn + current_cost
        )
        campaign_cost = {
            "schema": "audit_135m_campaign_cost_v1",
            "stage_code": stage_code,
            "roster_hash": identity_plan["roster_hash"],
            "completed": completed,
            "scientific_attempt_started": scientific_attempt_started,
            "counts_toward_pre_science_abort_burn": (
                not scientific_attempt_started and not completed
            ),
            "estimated_cost_usd": round(current_cost, 6),
            "generations": generation_costs,
            "transient_provider_registry": {
                "path": transient_registry_path.relative_to(campaign_root).as_posix(),
                "raw_sha256": _sha256_file(transient_registry_path),
                "count": int(transient_registry["count"]),
            },
            "final_zero_census": census,
        }
        campaigns = list(spend_ledger["campaigns"])
        campaigns.append(campaign_cost)
        updated_ledger = {
            **spend_ledger,
            "abort_burn_kill_usd": ABORT_BURN_KILL_USD,
            "pre_science_aborted_launch_spend_usd": round(
                updated_abort_burn, 6
            ),
            "abort_burn_kill_exceeded": updated_abort_burn > ABORT_BURN_KILL_USD,
            "estimated_spend_usd": round(
                float(spend_ledger["estimated_spend_usd"]) + current_cost, 6
            ),
            "campaigns": campaigns,
            "updated_at_utc": utc_now(),
        }
        write_json_atomic(args.stage_spend_ledger, updated_ledger)
        write_json_atomic(campaign_root / "campaign" / "campaign-cost-final.json", campaign_cost)
        if registry.identities:
            vm_registry = registry.snapshot()
            try:
                _final_outputs(
                    base=base,
                    p1=p1,
                    campaign_root=campaign_root,
                    packet=packet,
                    backend=backend,
                    vm_registry=vm_registry,
                    census=census,
                    completed=completed,
                    controller_error=controller_error,
                )
                if completed and controller_error is None:
                    campaign_seal = _seal_and_authorize_aggregation(
                        pexec=pexec,
                        campaign_root=campaign_root,
                        checkpoint_preseal_path=checkpoint_preseal_path,
                    )
            except BaseException as finalization_error:
                completed = False
                if controller_error is None:
                    controller_error = finalization_error
                else:
                    backend.note(
                        f"SECONDARY FINAL SEAL ERROR: {finalization_error}"
                    )
                result_path = campaign_root / "campaign" / "controller-result.json"
                if result_path.is_file():
                    result = load_json(result_path)
                    result.update(
                        {
                            "status": "EXECUTION_ABORTED_AND_EXACT_ID_TEARDOWN_COMPLETE",
                            "execution_complete": False,
                            "aggregation_authorized": False,
                            "controller_error_type": type(controller_error).__name__,
                            "controller_error_detail": str(controller_error),
                        }
                    )
                    write_json_atomic(result_path, result)
        if campaign_cost["completed"] != completed:
            campaign_cost["completed"] = completed
            write_json_atomic(
                campaign_root / "campaign" / "campaign-cost-final.json", campaign_cost
            )
            updated_ledger["campaigns"][-1]["completed"] = completed
            write_json_atomic(args.stage_spend_ledger, updated_ledger)
    if controller_error is not None:
        raise controller_error
    return {
        "status": "SEALED_EXECUTION_AND_EXACT_ID_TEARDOWN_COMPLETE",
        "stage_code": stage_code,
        "campaign_root": str(campaign_root),
        "estimated_campaign_cost_usd": campaign_cost["estimated_cost_usd"],
        "estimated_stage_spend_usd": updated_ledger["estimated_spend_usd"],
        "hard_ceiling_usd": ceiling,
        "pre_science_aborted_launch_spend_usd": updated_ledger[
            "pre_science_aborted_launch_spend_usd"
        ],
        "abort_burn_kill_usd": ABORT_BURN_KILL_USD,
        "aggregation_descriptor": str(
            campaign_root / "campaign" / "aggregation-descriptor.json"
        ),
        "campaign_manifest": str(
            campaign_root / "campaign" / "campaign-manifest.json"
        ),
        "campaign_seal": str(campaign_root / "campaign" / "campaign-seal.json"),
        "checkpoint_preseal": (
            None if checkpoint_preseal_path is None else str(checkpoint_preseal_path)
        ),
        "prediction_freeze": (
            None if prediction_freeze_path is None else str(prediction_freeze_path)
        ),
        "hidden_authorization": (
            None
            if hidden_authorization_path is None
            else str(hidden_authorization_path)
        ),
        "hidden_root": None if hidden_root is None else str(hidden_root),
        "campaign_seal_status": None if campaign_seal is None else campaign_seal["status"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stage-spend-ledger", type=Path, required=True)
    parser.add_argument("--a3-mechanical-gate", type=Path)
    parser.add_argument("--a3-historical-phase-manifest", type=Path)
    parser.add_argument("--a3-recapture-campaign-manifest", type=Path)
    parser.add_argument("--a3-recapture-bound-manifest", type=Path)
    parser.add_argument("--a3-recapture-campaign-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        print(f"audit campaign controller error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
