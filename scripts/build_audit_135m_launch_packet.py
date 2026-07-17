#!/usr/bin/env python3
"""Freeze one file-streamed audit-135M launch packet.

The input directory must already contain the non-authorizing audit
materialization and deterministic parallel binding.  This builder binds the
pushed scientific commit, current-stage seed bundle, exact runtime
authorization, up to six generation-1 identities, Spot-only GCP specs, cost ceiling,
and the reviewed P1 worker/backend substrate.  It performs no provider mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
R0_ROOT = Path("/private/tmp/yeto-p1r0-launcher/p1r0-session")
R0_BUILDER = R0_ROOT / "build_launch_packet.py"
R0_WORKER = R0_ROOT / "p1r0_vm_worker.py"
BUCKET = "gs://yeto-exp2-52-model-training-497007"
PREREG_SHA256 = "5198d62090ea307a5b8c7151f66088ddf8c57782b00591da93b1465f1c146eb7"
PREREG_PATH = "experiment-specs/tuned-baseline-audit-prereg.json"
MODEL_URI = (
    "gs://yeto-exp2-52-model-training-497007/prelaunch/"
    "bp-p0a-0af7f4a-20260714a/model-93efa2f.tar#1784089423172165"
)
DATA_URI = (
    "gs://yeto-exp2-52-model-training-497007/prelaunch/"
    "bp-p0a-0af7f4a-20260714a/train.parquet#1784090284099303"
)
MODEL_ARCHIVE_SHA256 = (
    "53d15a96a333e33c6a7a9224dbe6392a2480420bd40a327588797d03b625e4c3"
)
DATA_SHA256 = "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
AUDIT_BLOCK_WIDTH = 3
MAX_CONCURRENT_BLOCKS = 2
SLOTS = tuple(
    f"v{index}" for index in range(AUDIT_BLOCK_WIDTH * MAX_CONCURRENT_BLOCKS)
)
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
PROTECTED_INSTANCE_ID = "3908640733128066700"
GCLOUD_CONFIG = "/private/tmp/yeto-gcloud-admin-codex"
GLOBAL_A100_CEILING = 16
PREFERRED_PROBE_A100S = 1


def _source_commit() -> str:
    override = os.environ.get("AUDIT_135M_SOURCE_COMMIT")
    if override is not None:
        if len(override) != 40 or any(character not in "0123456789abcdef" for character in override):
            raise RuntimeError("AUDIT_135M_SOURCE_COMMIT is not a full lowercase Git SHA")
        return override
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


SOURCE_COMMIT = _source_commit()

_ACTIVE: dict[str, Any] = {}


class PacketError(RuntimeError):
    """The packet inputs or prelaunch gates are not exact."""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PacketError(f"cannot load reviewed helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = load_module("audit_135m_r0_packet_helpers", R0_BUILDER)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return _base.sha256_file(path)


def write_json_create_only(path: Path, value: object) -> None:
    _base.write_json_create_only(path, value)


def write_text_create_only(
    path: Path, value: str, *, executable: bool = False
) -> None:
    _base.write_text_create_only(path, value, executable=executable)


def deterministic_tar(paths: list[tuple[Path, str]], output: Path) -> None:
    _base.deterministic_tar(paths, output)


def _configure_base() -> None:
    if not _ACTIVE:
        raise PacketError("launch packet runtime configuration is absent")
    _base.CONTROLLER_COMMIT = SOURCE_COMMIT
    _base.SOURCE_COMMIT = SOURCE_COMMIT
    _base.SOURCE_ROOT = REPO_ROOT
    _base.SCIENCE_ROOT = Path(_ACTIVE["science_root"])
    _base.STATE_ROOT = Path(_ACTIVE["campaign_state_root_base"])
    _base.HARNESS_STATE_ROOT = Path(_ACTIVE["harness_state_root"])
    _base.CAMPAIGN_ATTEMPT = int(_ACTIVE["campaign_attempt"])
    _base.ZONE_BY_SLOT = {
        slot: str(_ACTIVE["initial_zone"]) for slot in _ACTIVE["logical_slots"]
    }
    _base.MODEL_URI = MODEL_URI
    _base.DATA_URI = DATA_URI
    _base.MODEL_ARCHIVE_SHA256 = MODEL_ARCHIVE_SHA256
    _base.DATA_SHA256 = DATA_SHA256
    _base.PREREG_COMMIT = SOURCE_COMMIT
    _base.PREREG_PATH = PREREG_PATH
    _base.PREREG_SHA256 = PREREG_SHA256


def configure_from_identity_plan(identity_plan: Mapping[str, Any]) -> None:
    if identity_plan.get("source_commit") != SOURCE_COMMIT:
        raise PacketError("identity plan source commit differs from the executing checkout")
    _ACTIVE.clear()
    _ACTIVE.update(
        {
            "stage_code": identity_plan["stage_code"],
            "science_root": identity_plan["science_root"],
            "campaign_attempt": identity_plan["campaign_attempt"],
            "campaign_state_root_base": identity_plan[
                "campaign_state_root_base"
            ],
            "harness_state_root": identity_plan["harness_state_root"],
            "initial_zone": identity_plan["initial_zone"],
            "logical_slots": tuple(identity_plan["logical_slots"]),
        }
    )
    _configure_base()


def bootstrap_text(**kwargs: Any) -> str:
    _configure_base()
    return _base.bootstrap_text(**kwargs)


def spec_value(**kwargs: Any) -> dict[str, object]:
    _configure_base()
    value = _base.spec_value(**kwargs)
    cloud = value["cloud"]
    execution = value["execution"]
    assert isinstance(cloud, dict) and isinstance(execution, dict)
    if (
        cloud.get("machine_type") != "a2-highgpu-4g"
        or cloud.get("provisioning_model") != "SPOT"
    ):
        raise PacketError("reviewed base spec is not the expected Spot A2 shape")
    cloud["machine_type"] = "a2-highgpu-1g"
    cloud["accelerator_count"] = 1
    labels = cloud["labels"]
    assert isinstance(labels, dict)
    labels["stage"] = str(_ACTIVE["stage_code"])
    labels["campaign"] = "audit-135m"
    execution["source_authority"] = {
        "ref": "refs/heads/experiment/best-paper-phase-map",
        "ancestor_commit": SOURCE_COMMIT,
        "ancestor_path": PREREG_PATH,
        "ancestor_sha256": PREREG_SHA256,
    }
    return value


def _copy_create_only(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PacketError(f"refusing to overwrite packet artifact: {destination}")
    shutil.copyfile(source, destination)


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PacketError(f"{label} must be a JSON object")
    return value


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


def _cloud_preflight(roster_tag: str, artifact_root: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["CLOUDSDK_CONFIG"] = GCLOUD_CONFIG

    def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command), env=env, text=True, capture_output=True, check=False
        )

    instances = run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project=model-training-497007",
            "--format=json",
        ]
    )
    if instances.returncode:
        raise PacketError(f"cannot list GCP instances: {instances.stderr[-2000:]}")
    rows = json.loads(instances.stdout or "[]")
    owned = [
        row
        for row in rows
        if (row.get("labels") or {}).get("campaign-tag") == roster_tag
    ]
    if owned:
        raise PacketError("campaign tag already owns a GCP instance")
    live_a100_rows = [
        {
            "name": row.get("name"),
            "instance_numeric_id": str(row.get("id")),
            "a100_count": _instance_a100_count(row),
            "campaign_tag": (row.get("labels") or {}).get("campaign-tag"),
        }
        for row in rows
        if row.get("status") != "TERMINATED" and _instance_a100_count(row) > 0
    ]
    live_a100_count = sum(row["a100_count"] for row in live_a100_rows)
    if live_a100_count + PREFERRED_PROBE_A100S > GLOBAL_A100_CEILING:
        raise PacketError(
            "global A100 census leaves insufficient room for one reviewed 4g-first probe"
        )
    protected = [row for row in rows if str(row.get("id")) == PROTECTED_INSTANCE_ID]
    if len(protected) != 1:
        raise PacketError("protected instance census differs")
    prefix = run(
        [
            "gcloud",
            "storage",
            "ls",
            "--all-versions",
            artifact_root.rstrip("/") + "/**",
        ]
    )
    if prefix.returncode == 0 and prefix.stdout.strip():
        raise PacketError("new campaign artifact root is not empty")
    prefix_text = (prefix.stdout + prefix.stderr).casefold()
    if prefix.returncode and not any(
        token in prefix_text for token in ("matched no", "not found", "no urls matched")
    ):
        raise PacketError("cannot prove campaign artifact root empty")
    return {
        "schema": "audit_135m_cloud_readonly_preflight_v1",
        "status": "PASS",
        "cloudsdk_config": GCLOUD_CONFIG,
        "campaign_tag": roster_tag,
        "campaign_owned_instances": [],
        "protected_instance_numeric_id": PROTECTED_INSTANCE_ID,
        "protected_instance_untouched": True,
        "visible_instance_count": len(rows),
        "global_attached_a100_equivalent": live_a100_count,
        "global_a100_ceiling": GLOBAL_A100_CEILING,
        "preferred_probe_a100s": PREFERRED_PROBE_A100S,
        "global_a100_inventory": live_a100_rows,
        "global_ceiling_after_one_preferred_probe_pass": True,
        "artifact_root": artifact_root,
        "artifact_root_empty": True,
    }


def _registered_cost_forecast(
    *, stage_code: str, hard_ceiling: float
) -> dict[str, Any]:
    audit_stage = {"a1": "A1", "a3": "A3", "a4": "A4"}[stage_code[:2]]
    authority_path = REPO_ROOT / PREREG_PATH
    if sha256_file(authority_path) != PREREG_SHA256:
        raise PacketError("registered cost authority bytes differ")
    authority = _load(authority_path, "audit preregistration")
    block = authority.get("costs", {}).get("blocks", {}).get(audit_stage, {})
    cost_range = block.get("range_usd")
    if (
        not isinstance(cost_range, list)
        or len(cost_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in cost_range
        )
        or float(block.get("hard_ceiling_usd", -1.0)) != hard_ceiling
    ):
        raise PacketError("registered stage cost range/ceiling differs")
    low, high = (float(cost_range[0]), float(cost_range[1]))
    if high >= hard_ceiling:
        raise PacketError("registered stage forecast does not leave ceiling headroom")
    return {
        "schema": "audit_135m_prelaunch_cost_forecast_v1",
        "status": "PASS",
        "audit_stage": audit_stage,
        "stage_code": stage_code,
        "registered_range_usd": [low, high],
        "registered_range_includes_spot_preemption_reserve": True,
        "registered_spot_preemption_reserve_fraction": 0.25,
        "registered_hard_ceiling_reserve_fraction": 0.30,
        "hard_ceiling_usd": hard_ceiling,
        "forecast_upper_usd": high,
        "ceiling_headroom_usd": hard_ceiling - high,
        "hard_stop_before_exceeding_ceiling": True,
        "spot_only": True,
    }


def _cost_efficient_target_width(parallel_plan: Mapping[str, Any]) -> int:
    if parallel_plan.get("schema") == "yeto_parallel_plan_v4":
        return AUDIT_BLOCK_WIDTH
    group_sizes = [
        len(wave.get("assigned_cells_in_dispatch_order", []))
        for wave in parallel_plan.get("waves", [])
    ]
    if not group_sizes or any(size <= 0 for size in group_sizes):
        raise PacketError("parallel plan has an empty wave")
    billed_slot_batches = {
        width: sum(width * ((size + width - 1) // width) for size in group_sizes)
        for width in range(1, len(SLOTS) + 1)
    }
    minimum = min(billed_slot_batches.values())
    return max(
        width for width, billed in billed_slot_batches.items() if billed == minimum
    )


def _command_flag(command: Sequence[Any], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise PacketError(f"scientific command has malformed {flag}")
    value = command[positions[0] + 1]
    if not isinstance(value, str):
        raise PacketError(f"scientific command {flag} value is not a string")
    return value


def _validate_guest_scientific_paths(
    *, scientific: Mapping[str, Any], science_root: str
) -> None:
    expected_repo = "/tmp/yeto-best-paper"
    expected_python = "/home/shou/venv/bin/python"
    expected_compare = expected_repo + "/scripts/compare_diloco.py"
    expected_model = science_root.rstrip("/") + "/inputs/model"
    frozen_prefix = science_root.rstrip("/") + "/phase-map/frozen-eval/seed-"
    cells = scientific.get("cells")
    if not isinstance(cells, list) or not cells:
        raise PacketError("scientific plan has no cells")
    for raw in cells:
        if not isinstance(raw, Mapping):
            raise PacketError("scientific plan contains a malformed cell")
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or command[0] != expected_python
            or command[1] != expected_compare
            or _command_flag(command, "--model") != expected_model
        ):
            raise PacketError("scientific command guest Python/repo/model path differs")
        data = _command_flag(command, "--data")
        development = _command_flag(command, "--prebound-development-eval")
        seed = str(raw.get("seed"))
        expected_seed_root = frozen_prefix + seed
        if data != expected_seed_root + "/materialized/train.jsonl" or development != (
            expected_seed_root + "/materialized/eval.jsonl"
        ):
            raise PacketError("scientific command frozen train/evaluation path differs")


def build(args: argparse.Namespace) -> dict[str, Any]:
    packet = args.packet_root.resolve()
    if not packet.is_dir():
        raise PacketError("packet root with materialized/binding inputs must exist")
    materialized = packet / "materialized"
    binding_dir = packet / "binding"
    binding = _load(binding_dir / "parallel-binding.json", "parallel binding")
    roster = _load(binding_dir / "parallel-roster.json", "parallel roster")
    parallel_plan = _load(binding_dir / "parallel-plan.json", "parallel plan")
    bound = _load(materialized / "bound-manifest.json", "bound manifest")
    scientific = _load(
        materialized / "scientific-randomization-plan.json", "scientific plan"
    )
    stage_code = str(binding.get("stage_code"))
    if (
        stage_code not in {
            "a1d",
            "a1x",
            "a1c",
            "a3k",
            "a3r0",
            "a3x",
            "a4d",
            "a4b",
            "a4c",
            "a4x",
        }
        or roster.get("stage_code") != stage_code
        or bound.get("launch_authorized") is not False
        or binding.get("launch_authorized") is not False
        or bound.get("frozen", {}).get("git_commit") != SOURCE_COMMIT
    ):
        raise PacketError("audit binding identity/authorization differs")
    concurrent_binding = parallel_plan.get("audit_concurrent_block_binding")
    amendment_path = REPO_ROOT / "docs" / "AMENDMENT-audit-135m-concurrent-blocks.md"
    if (
        parallel_plan.get("schema") != "yeto_parallel_plan_v4"
        or not isinstance(concurrent_binding, Mapping)
        or concurrent_binding.get("amendment_raw_sha256")
        != sha256_file(amendment_path)
    ):
        raise PacketError("audit concurrent-block amendment binding differs")
    if subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", SOURCE_COMMIT, "--"]
    ).returncode:
        raise PacketError("working source differs from the pushed scientific commit")
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/experiment/best-paper-phase-map"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()[0]
    if remote != SOURCE_COMMIT:
        raise PacketError("remote scientific branch no longer matches SOURCE_COMMIT")

    authorization = dict(binding["runtime_authorization_required_fields"])
    if set(authorization) != {
        "schema",
        "launch_authorized",
        "stage_code",
        "audit_135m_design_contract_hash",
        "roster_hash",
        "parallel_plan_hash",
        "bound_manifest_canonical_sha256",
        "scientific_randomization_plan_hash",
        "hard_ceiling_usd",
        "spot_only",
        "maximum_attached_a100_equivalent",
        "max_idle_before_science_seconds",
    }:
        raise PacketError("runtime authorization field set differs")
    runtime_authorization_path = packet / "runtime-authorization.json"
    write_json_create_only(runtime_authorization_path, authorization)
    runtime_authorization_hash = canonical_sha256(authorization)

    seed_registry = _load(args.seed_bundle_registry, "seed-bundle registry")
    if (
        seed_registry.get("stage_code") != stage_code
        or seed_registry.get("audit_objects_included") is not False
        or seed_registry.get("audit_model_evaluation_accesses") != []
    ):
        raise PacketError("seed bundle is not the exact current-stage blind suffix")
    seed_bundle = args.seed_bundle_registry.parent / seed_registry["bundle_path"]
    if sha256_file(seed_bundle) != seed_registry["bundle_sha256"]:
        raise PacketError("seed bundle SHA-256 differs from its registry")
    _copy_create_only(seed_bundle, packet / "inputs" / "frozen-seed347.tar.gz")
    _copy_create_only(
        args.seed_bundle_registry, packet / "inputs" / "seed-bundle-registry.json"
    )
    _copy_create_only(args.parent_manifest, packet / "parent" / "parent-manifest.json")
    if canonical_sha256(_load(args.parent_manifest, "parent manifest")) != binding[
        "parent_manifest_canonical_sha256"
    ]:
        raise PacketError("packet parent canonical hash differs from binding")

    initial_zone = args.initial_zone
    if initial_zone not in ZONE_ROTATION:
        raise PacketError("initial zone is outside the survival-weighted registry")
    campaign_attempt = 1
    science_root = str(args.science_root)
    if not science_root.startswith("/tmp/audit-135m-science/"):
        raise PacketError("science root must be a fresh /tmp/audit-135m-science suffix")
    _validate_guest_scientific_paths(
        scientific=scientific, science_root=science_root
    )
    roster_tag = str(binding["roster_tag"])
    logical_slots = tuple(binding.get("logical_slots", ()))
    if (
        not logical_slots
        or len(logical_slots) > len(SLOTS)
        or len(logical_slots) % AUDIT_BLOCK_WIDTH
        or any(slot not in SLOTS for slot in logical_slots)
    ):
        raise PacketError("audit binding has an invalid concurrent logical-slot set")
    run_ids = {
        slot: f"bp-{stage_code}-{roster_tag}-c{campaign_attempt}-{slot}-g1"
        for slot in logical_slots
    }
    joined_run_ids = "--".join(run_ids[slot] for slot in logical_slots)
    artifact_root = (
        f"{BUCKET}/audit-135m-{stage_code}-{roster_tag}-c1/{joined_run_ids}"
    )
    state_root_base = f"/tmp/audit-135m-{stage_code}-pexec-state"
    state_root = Path(state_root_base) / roster_tag / "c1"
    harness_state_root = f"/tmp/yeto-audit-135m-{stage_code}-state"
    controller_private_root = f"/tmp/audit-135m-{stage_code}-controller"
    _ACTIVE.clear()
    _ACTIVE.update(
        {
            "stage_code": stage_code,
            "science_root": science_root,
            "campaign_attempt": campaign_attempt,
            "campaign_state_root_base": state_root_base,
            "harness_state_root": harness_state_root,
            "initial_zone": initial_zone,
            "logical_slots": logical_slots,
        }
    )
    _configure_base()

    identity_rows = []
    configs: list[tuple[Path, str]] = []
    for slot in logical_slots:
        run_id = run_ids[slot]
        nonce = secrets.token_hex(16)
        artifact_prefix = f"{artifact_root}/vms/{slot}/g1/"
        labels = {
            "campaign-tag": roster_tag,
            "logical-slot": slot,
            "physical-generation": "1",
            "run-id": run_id,
            "ownership-nonce": nonce,
        }
        identity = {
            "slot": slot,
            "generation": 1,
            "run_id": run_id,
            "ownership_nonce": nonce,
            "state_path": str(state_root / f"{run_id}.json"),
            "artifact_prefix": artifact_prefix,
            "provider_record_path": artifact_prefix + "provider/provider-evidence.json",
            "partial_manifest_path": artifact_prefix
            + "manifests/vm-partial-manifest.json",
            "lifecycle_record_path": artifact_prefix
            + "manifests/vm-lifecycle-final.json",
            "labels": labels,
            "harness_state_path": f"{harness_state_root}/{run_id}.json",
            "region": initial_zone.rsplit("-", 1)[0],
            "zone": initial_zone,
            "planned_zone": initial_zone,
        }
        identity_rows.append(identity)
        remote_run = f"/tmp/runs/{run_id}/vms/{slot}/g1"
        config = {
            "schema": "audit_135m_vm_worker_config_v1",
            "stage_code": stage_code,
            "source_commit": SOURCE_COMMIT,
            "controller_commit": SOURCE_COMMIT,
            "repo_root": "/tmp/yeto-best-paper",
            "packet_root": remote_run + "/packet",
            "science_root": science_root,
            "remote_run_dir": remote_run,
            "run_id": run_id,
            "slot": slot,
            "generation": 1,
            "ownership_nonce": nonce,
            "identity": identity,
            "artifact_prefix": artifact_prefix,
            "generation_campaign_relative": f"vms/{slot}/g1",
            "roster_hash": binding["roster_hash"],
            "parallel_plan_hash": binding["parallel_plan_hash"],
            "supersedes_parallel_plan_hash": binding[
                "supersedes_parallel_plan_hash"
            ],
            "bound_manifest_canonical_sha256": binding[
                "bound_manifest_canonical_sha256"
            ],
            "scientific_randomization_plan_hash": binding[
                "scientific_randomization_plan_hash"
            ],
            "runtime_authorization_hash": runtime_authorization_hash,
            "data_sha256": DATA_SHA256,
        }
        config_path = packet / "runtime" / "configs" / f"{slot}.json"
        write_json_create_only(config_path, config)
        configs.append((config_path, f"configs/{slot}.json"))

    identity_plan = {
        "schema": "audit_135m_generation_identity_plan_v1",
        "stage_code": stage_code,
        "source_commit": SOURCE_COMMIT,
        "audit_stage": roster["audit_stage"],
        "study_id": bound["study_id"],
        "campaign_attempt": campaign_attempt,
        "roster_hash": binding["roster_hash"],
        "roster_tag": roster_tag,
        "parallel_plan_hash": binding["parallel_plan_hash"],
        "campaign_artifact_root": artifact_root,
        "campaign_state_root": str(state_root),
        "campaign_state_root_base": state_root_base,
        "harness_state_root": harness_state_root,
        "controller_private_root": controller_private_root,
        "science_root": science_root,
        "initial_zone": initial_zone,
        "zone_rotation": list(ZONE_ROTATION),
        "logical_slots": list(logical_slots),
        "target_width": _cost_efficient_target_width(parallel_plan),
        "maximum_concurrent_blocks": min(
            MAX_CONCURRENT_BLOCKS, len(parallel_plan["waves"])
        ),
        "target_1g_slot_count": len(logical_slots),
        "assembly_max_seconds": 480,
        "hard_ceiling_usd": roster["hard_ceiling_usd"],
        "runtime_authorization_hash": runtime_authorization_hash,
        "launch_cell_count": len(roster["launch_cells"]),
        "wave_count": len(parallel_plan["waves"]),
        "generations": identity_rows,
    }
    write_json_create_only(packet / "identity-plan.json", identity_plan)

    revision = {
        "schema": "audit_135m_controller_revision_binding_v1",
        "status": "BOUND",
        "controller_commit": SOURCE_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "science_commands_unchanged": True,
        "audit_prereg_sha256": PREREG_SHA256,
        "roster_hash": binding["roster_hash"],
        "parallel_plan_hash": binding["parallel_plan_hash"],
        "runtime_authorization_hash": runtime_authorization_hash,
        "allowed_zones": list(ZONE_ROTATION),
        "shape_fallback_order": ["a2-highgpu-1g"],
        "shape_fallback_trigger": "operator_ceiling_amendment_cheapest_survivable_shape",
    }
    write_json_create_only(
        materialized / "controller-amendment-revision.json", revision
    )
    forecast = _registered_cost_forecast(
        stage_code=stage_code, hard_ceiling=float(roster["hard_ceiling_usd"])
    )
    write_json_create_only(packet / "gates" / "prelaunch-cost-forecast.json", forecast)

    worker_copy = packet / "runtime" / "p1r0_vm_worker.py"
    _copy_create_only(args.worker_wrapper, worker_copy)
    worker_copy.chmod(0o755)
    _copy_create_only(R0_WORKER, packet / "runtime" / "p1r0_vm_worker_base.py")
    runtime_paths = [
        (worker_copy, "p1r0_vm_worker.py"),
        (packet / "runtime" / "p1r0_vm_worker_base.py", "p1r0_vm_worker_base.py"),
        (REPO_ROOT / "scripts" / "run_parallel_phase_map.py", "run_parallel_phase_map.py"),
        (materialized / "bound-manifest.json", "bound-manifest.json"),
        (materialized / "scientific-randomization-plan.json", "scientific-randomization-plan.json"),
        (args.parent_manifest, "parent-manifest.json"),
        (binding_dir / "parallel-roster.json", "parallel-roster.json"),
        (binding_dir / "parallel-plan.json", "parallel-plan.json"),
        (binding_dir / "parallel-binding.json", "parallel-binding.json"),
        (materialized / "controller-amendment-revision.json", "controller-amendment-revision.json"),
        (runtime_authorization_path, "runtime-authorization.json"),
        (args.seed_bundle_registry, "seed-bundle-registry.json"),
        *configs,
    ]
    runtime_tar = packet / "runtime" / "p1r0-runtime.tar.gz"
    deterministic_tar(runtime_paths, runtime_tar)
    runtime_hash = sha256_file(runtime_tar)
    frozen_path = packet / "inputs" / "frozen-seed347.tar.gz"
    frozen_hash = sha256_file(frozen_path)

    vm_rows = []
    for identity in identity_rows:
        slot = str(identity["slot"])
        run_id = str(identity["run_id"])
        remote_run = f"/tmp/runs/{run_id}/vms/{slot}/g1"
        bootstrap_remote = f"/home/shou/{run_id}-bootstrap.sh"
        runtime_remote = f"/home/shou/{run_id}-runtime.tar.gz"
        frozen_remote = f"/home/shou/{run_id}-frozen-seed347.tar.gz"
        provider_remote = f"/home/shou/{run_id}-parallel-provider.json"
        bootstrap_path = packet / "vms" / slot / "bootstrap.sh"
        write_text_create_only(
            bootstrap_path,
            bootstrap_text(
                slot=slot,
                run_id=run_id,
                remote_run_dir=remote_run,
                runtime_sha256=runtime_hash,
                frozen_sha256=frozen_hash,
                config_relative=f"configs/{slot}.json",
            ),
            executable=True,
        )
        spec_path = packet / "vms" / slot / "optimizer-harness.json"
        write_json_create_only(
            spec_path,
            spec_value(
                identity=identity,
                zone=initial_zone,
                remote_run_dir=remote_run,
                artifact_prefix=str(identity["artifact_prefix"]),
                bootstrap_remote=bootstrap_remote,
                runtime_remote=runtime_remote,
                frozen_remote=frozen_remote,
                provider_remote=provider_remote,
            ),
        )
        validation = subprocess.run(
            [sys.executable, "-m", "yeto.optimizer_harness", "validate", str(spec_path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        syntax = subprocess.run(
            ["bash", "-n", str(bootstrap_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if validation.returncode or syntax.returncode:
            raise PacketError(
                "harness/bootstrap validation failed: "
                + (validation.stdout + validation.stderr + syntax.stdout + syntax.stderr)[-4000:]
            )
        vm_rows.append(
            {
                "slot": slot,
                "run_id": run_id,
                "ownership_nonce": identity["ownership_nonce"],
                "planned_zone": initial_zone,
                "artifact_prefix": identity["artifact_prefix"],
                "harness_state_path": identity["harness_state_path"],
                "pexec_state_path": identity["state_path"],
                "bootstrap_path": str(bootstrap_path),
                "bootstrap_sha256": sha256_file(bootstrap_path),
                "spec_path": str(spec_path),
                "spec_sha256": sha256_file(spec_path),
                "config_sha256": sha256_file(
                    packet / "runtime" / "configs" / f"{slot}.json"
                ),
                "remote_bootstrap_path": bootstrap_remote,
                "remote_runtime_path": runtime_remote,
                "remote_frozen_bundle_path": frozen_remote,
                "remote_provider_path": provider_remote,
            }
        )

    source_bundle = packet / "source" / f"yeto-{SOURCE_COMMIT[:7]}.bundle"
    source_bundle.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "bundle",
            "create",
            str(source_bundle),
            "refs/heads/experiment/best-paper-phase-map",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(["git", "bundle", "verify", str(source_bundle)], check=True)

    preflight = _cloud_preflight(roster_tag, artifact_root)
    write_json_create_only(packet / "gates" / "cloud-readonly-preflight.json", preflight)
    review = {
        "schema": "audit_135m_launch_review_packet_v1",
        "status": "SEALED_LAUNCH_AUTHORIZED",
        "stage_code": stage_code,
        "study_id": bound["study_id"],
        "source_commit": SOURCE_COMMIT,
        "remote_branch_sha": remote,
        "audit_prereg_sha256": PREREG_SHA256,
        "parent_manifest_canonical_sha256": binding[
            "parent_manifest_canonical_sha256"
        ],
        "bound_manifest_canonical_sha256": binding[
            "bound_manifest_canonical_sha256"
        ],
        "scientific_randomization_plan_hash": binding[
            "scientific_randomization_plan_hash"
        ],
        "roster_hash": binding["roster_hash"],
        "parallel_plan_hash": binding["parallel_plan_hash"],
        "runtime_authorization_hash": runtime_authorization_hash,
        "runtime_tar_sha256": runtime_hash,
        "seed_bundle_sha256": frozen_hash,
        "source_bundle_sha256": sha256_file(source_bundle),
        "hard_ceiling_usd": roster["hard_ceiling_usd"],
        "cost_forecast": forecast,
        "spot_only": True,
        "maximum_attached_a100_equivalent": 16,
        "audit_concurrent_block_binding": parallel_plan.get(
            "audit_concurrent_block_binding"
        ),
        "maximum_concurrent_blocks": identity_plan[
            "maximum_concurrent_blocks"
        ],
        "target_1g_slot_count": identity_plan["target_1g_slot_count"],
        "logical_slots": identity_plan["logical_slots"],
        "global_attached_a100_equivalent_at_preflight": preflight[
            "global_attached_a100_equivalent"
        ],
        "maximum_initial_preferred_probe_width_under_global_ceiling": (
            GLOBAL_A100_CEILING - preflight["global_attached_a100_equivalent"]
        )
        // PREFERRED_PROBE_A100S,
        "max_idle_before_science_seconds": 600,
        "zone_rotation": list(ZONE_ROTATION),
        "shape_fallback_order": ["a2-highgpu-1g"],
        "audit_objects_included": False,
        "audit_model_evaluation_accesses": [],
        "vms": vm_rows,
    }
    write_json_create_only(packet / "review-packet.json", review)
    digest = sha256_file(packet / "review-packet.json")
    write_text_create_only(
        packet / "review-packet.json.sha256", f"{digest}  review-packet.json\n"
    )
    return {
        "status": "SEALED_LAUNCH_AUTHORIZED",
        "stage_code": stage_code,
        "review_packet": str(packet / "review-packet.json"),
        "review_packet_sha256": digest,
        "roster_hash": binding["roster_hash"],
        "runtime_authorization_hash": runtime_authorization_hash,
        "hard_ceiling_usd": roster["hard_ceiling_usd"],
        "cloud_mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--seed-bundle-registry", type=Path, required=True)
    parser.add_argument(
        "--worker-wrapper",
        type=Path,
        default=REPO_ROOT / "scripts" / "audit_135m_vm_worker.py",
    )
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--initial-zone", choices=ZONE_ROTATION, default="us-east1-b")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(args)
    except (OSError, KeyError, ValueError, PacketError) as exc:
        print(f"audit launch-packet error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
