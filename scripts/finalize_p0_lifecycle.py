#!/usr/bin/env python3
"""Finalize a sealed P0 acquisition after exact Spot VM and disk teardown.

The pre-delete acquisition manifest is never modified.  This program verifies
its immutable GCS object seal, copies exact deletion proofs into the artifact
tree, and writes a distinct final ``sealed_results`` manifest plus an external
lifecycle envelope binding both manifest generations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class FinalizationError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object")
    return value


def write_object(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise FinalizationError(f"{label} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise FinalizationError(f"{label} must be timezone-aware")
    return parsed


def require_numeric(value: Any, label: str) -> str:
    rendered = str(value or "")
    if not re.fullmatch(r"[0-9]+", rendered):
        raise FinalizationError(f"{label} must be an exact numeric provider ID")
    return rendered


def _artifact_object(
    deletion: dict[str, Any], role: str
) -> dict[str, Any]:
    seal = deletion.get("artifact_object_seal")
    objects = seal.get("objects") if isinstance(seal, dict) else None
    matches = [
        item
        for item in objects or []
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matches) != 1:
        raise FinalizationError(f"deletion evidence lacks one {role} object seal")
    return matches[0]


def finalize(
    root: Path,
    deletion_evidence: Path,
    *,
    expected_instance_id: str,
) -> dict[str, Any]:
    root = root.resolve()
    deletion = read_object(deletion_evidence, "deletion evidence")
    if (
        deletion.get("status") != "DELETED"
        or deletion.get("verified_instance_absent") is not True
        or deletion.get("verified_boot_disk_absent") is not True
    ):
        raise FinalizationError("finalization requires verified VM and disk deletion")
    instance_id = require_numeric(deletion.get("deleted_instance_id"), "instance ID")
    if instance_id != require_numeric(expected_instance_id, "expected instance ID"):
        raise FinalizationError("deleted instance ID differs from explicit authority")
    boot_disk_id = require_numeric(
        deletion.get("deleted_boot_disk_id"), "boot disk ID"
    )

    acquisition_manifest_path = root / "phase-map-acquisition-manifest.json"
    live_manifest_path = root / "phase-map-manifest.json"
    acquisition_checksum_path = root / "acquisition.sha256"
    acquisition_seal_path = root / "acquisition-seal.json"
    acquisition_manifest = read_object(
        acquisition_manifest_path, "P0 acquisition manifest"
    )
    if acquisition_manifest.get("status") != "sealed_acquisition_pending_teardown":
        raise FinalizationError("input is not a pending sealed P0 acquisition")
    kind = (acquisition_manifest.get("lineage") or {}).get("descendant_kind")
    if kind not in ("p0a_single_gpu_bound", "p0b_four_gpu_bound"):
        raise FinalizationError("legacy or non-canary descendant kind is forbidden")
    if live_manifest_path.read_bytes() != acquisition_manifest_path.read_bytes():
        raise FinalizationError("pre-delete live manifest differs from immutable copy")

    acquisition_raw_hash = sha256_file(acquisition_manifest_path)
    acquisition_canonical_hash = sha256_bytes(canonical_json(acquisition_manifest))
    phase_object = _artifact_object(deletion, "phase_map_manifest")
    checksum_object = _artifact_object(deletion, "acquisition_checksum")
    artifact_uri = str(deletion.get("artifact_uri", "")).rstrip("/")
    if (
        phase_object.get("uri") != f"{artifact_uri}/phase-map-manifest.json"
        or phase_object.get("sha256") != acquisition_raw_hash
        or checksum_object.get("uri") != f"{artifact_uri}/acquisition.sha256"
        or checksum_object.get("sha256") != sha256_file(acquisition_checksum_path)
    ):
        raise FinalizationError("immutable GCS object seal does not bind acquisition")
    seal = read_object(acquisition_seal_path, "acquisition seal")
    if (
        seal.get("schema") != "yeto_phase_map_acquisition_seal_v1"
        or seal.get("phase_map_manifest_sha256") != acquisition_raw_hash
        or seal.get("phase_map_manifest_canonical_sha256")
        != acquisition_canonical_hash
        or seal.get("loss_blind_mechanical_seal") is not True
    ):
        raise FinalizationError("acquisition seal does not bind immutable manifest")

    not_found = deletion.get("provider_not_found_verification")
    accelerator = deletion.get("post_delete_accelerator_proof")
    if not isinstance(not_found, dict) or not isinstance(accelerator, dict):
        raise FinalizationError("deletion evidence lacks absence/accelerator proofs")
    instance_proof = not_found.get("instance")
    disk_proof = not_found.get("boot_disk")
    if (
        not isinstance(instance_proof, dict)
        or instance_proof.get("result") != "NOT_FOUND"
        or str(instance_proof.get("provider_id")) != instance_id
        or not isinstance(disk_proof, dict)
        or disk_proof.get("result") != "NOT_FOUND"
        or str(disk_proof.get("provider_id")) != boot_disk_id
        or accelerator.get("campaign_owned_accelerators") != 0
    ):
        raise FinalizationError("exact resource absence proof is invalid")

    artifact_sealed_at = str(deletion["artifact_object_seal"]["sealed_at_utc"])
    deletion_requested_at = str(deletion.get("deletion_requested_at_utc"))
    deletion_completed_at = str(deletion.get("deleted_at_utc"))
    finalized_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ordered = [
        parse_time(artifact_sealed_at, "artifact seal"),
        parse_time(deletion_requested_at, "deletion request"),
        parse_time(deletion_completed_at, "deletion completion"),
        parse_time(finalized_at, "finalization"),
    ]
    if not (ordered[0] <= ordered[1] < ordered[2] < ordered[3]):
        raise FinalizationError(
            "lifecycle must order seal <= request < deletion < final"
        )

    lifecycle_dir = root / "lifecycle"
    instance_path = lifecycle_dir / "instance-not-found.json"
    disk_path = lifecycle_dir / "disk-not-found.json"
    accelerator_path = lifecycle_dir / "zero-campaign-accelerators.json"
    deletion_copy = lifecycle_dir / "deletion-evidence.json"
    write_object(instance_path, instance_proof)
    write_object(disk_path, disk_proof)
    write_object(accelerator_path, accelerator)
    if deletion_copy.resolve() != deletion_evidence.resolve():
        deletion_copy.parent.mkdir(parents=True, exist_ok=True)
        temporary = deletion_copy.with_name(deletion_copy.name + ".tmp")
        temporary.write_bytes(deletion_evidence.read_bytes())
        temporary.replace(deletion_copy)
    if sha256_file(deletion_copy) != sha256_file(deletion_evidence):
        raise FinalizationError("deletion evidence copy is not byte-identical")

    final_manifest = deepcopy(acquisition_manifest)
    transition = {
        "acquisition_status": "sealed_acquisition_pending_teardown",
        "acquisition_manifest_sha256": acquisition_raw_hash,
        "acquisition_manifest_canonical_sha256": acquisition_canonical_hash,
        "acquisition_checksum_sha256": sha256_file(acquisition_checksum_path),
        "acquisition_seal_sha256": sha256_file(acquisition_seal_path),
        "final_manifest_status": "sealed_results",
        "deletion_evidence_sha256": sha256_file(deletion_copy),
        "finalized_at": finalized_at,
    }
    for index, result in enumerate(final_manifest.get("results", [])):
        if not isinstance(result, dict) or not isinstance(result.get("hardware"), dict):
            raise FinalizationError(f"results[{index}] lacks hardware evidence")
        hardware = result["hardware"]
        if kind == "p0b_four_gpu_bound":
            required_acquisition = (
                "provisioning_started_at",
                "provisioning_completed_at",
                "nvidia_smi_inventory_uri",
                "nvidia_smi_inventory_sha256",
                "learner_gpu_map_uri",
                "learner_gpu_map_sha256",
                "barrier_version_trace_uri",
                "barrier_version_trace_sha256",
                "distinct_a100_gpu_uuid_count",
                "learner_gpu_uuid_bijection",
            )
            missing = [field for field in required_acquisition if field not in hardware]
            if missing:
                raise FinalizationError(
                    f"P0b results[{index}] lacks attempt evidence: {missing}"
                )
        hardware.update(transition)
        hardware.update(
            {
                "artifact_sealed_at": artifact_sealed_at,
                "deletion_requested_at": deletion_requested_at,
                "deletion_completed_at": deletion_completed_at,
                "instance_not_found_evidence_uri": (
                    f"lifecycle/{instance_path.name}"
                ),
                "instance_not_found_evidence_sha256": sha256_file(instance_path),
                "disk_not_found_evidence_uri": f"lifecycle/{disk_path.name}",
                "disk_not_found_evidence_sha256": sha256_file(disk_path),
                "zero_accelerator_evidence_uri": (
                    f"lifecycle/{accelerator_path.name}"
                ),
                "zero_accelerator_evidence_sha256": sha256_file(accelerator_path),
            }
        )
    final_manifest["status"] = "sealed_results"
    write_object(live_manifest_path, final_manifest)
    final_raw_hash = sha256_file(live_manifest_path)
    final_canonical_hash = sha256_bytes(canonical_json(final_manifest))
    envelope = {
        "schema": "yeto_p0_lifecycle_finalization_v1",
        "status": "SEALED",
        "descendant_kind": kind,
        "acquisition_manifest_sha256": acquisition_raw_hash,
        "acquisition_manifest_canonical_sha256": acquisition_canonical_hash,
        "acquisition_checksum_sha256": sha256_file(acquisition_checksum_path),
        "acquisition_seal_sha256": sha256_file(acquisition_seal_path),
        "artifact_object_seal_sha256": sha256_bytes(
            canonical_json(deletion["artifact_object_seal"])
        ),
        "deletion_evidence_sha256": sha256_file(deletion_copy),
        "final_manifest_sha256": final_raw_hash,
        "final_manifest_canonical_sha256": final_canonical_hash,
        "artifact_sealed_at": artifact_sealed_at,
        "deletion_requested_at": deletion_requested_at,
        "deletion_completed_at": deletion_completed_at,
        "finalized_at": finalized_at,
        "instance_numeric_id": instance_id,
        "boot_disk_numeric_id": boot_disk_id,
        "campaign_owned_accelerators_after_deletion": 0,
    }
    envelope_path = root / "phase-map-lifecycle-seal.json"
    write_object(envelope_path, envelope)
    (root / "phase-map.sha256").write_text(
        f"{final_raw_hash}  phase-map-manifest.json\n"
    )
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--deletion-evidence", type=Path, required=True)
    parser.add_argument("--expected-instance-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        envelope = finalize(
            args.run_root,
            args.deletion_evidence,
            expected_instance_id=args.expected_instance_id,
        )
    except (OSError, ValueError, FinalizationError) as exc:
        print(f"P0 finalization error: {exc}")
        return 2
    print(json.dumps(envelope, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
