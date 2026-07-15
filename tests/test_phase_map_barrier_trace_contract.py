from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase_map_barrier_contract_validator",
    ROOT / "scripts" / "validate_phase_map.py",
)
validator = importlib.util.module_from_spec(SPEC)
sys.modules["phase_map_barrier_contract_validator"] = validator
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)

HEX = "a" * 64


def _p0b_row() -> tuple[dict, dict]:
    hardware = {
        "provider": "gcp",
        "zone": "us-central1-a",
        "region": "us-central1",
        "market": "spot",
        "instance_type": "a2-highgpu-4g",
        "instance_name": "p0b-unit",
        "instance_id": "9001",
        "instance_numeric_id": "9001",
        "boot_disk_name": "p0b-unit-disk",
        "boot_disk_numeric_id": "8001",
        "image_id": "7290368630472593484",
        "source_image_numeric_id": "7290368630472593484",
        "provisioning_evidence_uri": "gs://bucket/p0b/provider.json",
        "provisioning_evidence_sha256": HEX,
        "provisioning_started_at": "2026-07-14T11:00:00Z",
        "provisioning_completed_at": "2026-07-14T11:05:00Z",
        "nvidia_smi_inventory_uri": "gs://bucket/p0b/nvidia-smi.json",
        "nvidia_smi_inventory_sha256": HEX,
        "learner_gpu_map_uri": "gs://bucket/p0b/gpu-map.json",
        "learner_gpu_map_sha256": HEX,
        "barrier_version_trace_uri": "gs://bucket/p0b/barrier-version-trace.json",
        "barrier_version_trace_sha256": HEX,
        "barrier_trace_validated": True,
        "base_versions_match": True,
        "no_inner_step_while_blocked": True,
        "barrier_trace_learner_count": 4,
        "barrier_trace_commit_count": 32,
        "barrier_trace_inner_steps_per_learner": 128,
        "distinct_a100_gpu_uuid_count": 4,
        "learner_gpu_uuid_bijection": {
            "0": "GPU-a",
            "1": "GPU-b",
            "2": "GPU-c",
            "3": "GPU-d",
        },
        "artifact_sealed_at": "2026-07-14T13:05:00Z",
        "deletion_requested_at": "2026-07-14T13:10:00Z",
        "deletion_completed_at": "2026-07-14T13:20:00Z",
        "instance_not_found_evidence_uri": "lifecycle/instance-not-found.json",
        "instance_not_found_evidence_sha256": HEX,
        "disk_not_found_evidence_uri": "lifecycle/disk-not-found.json",
        "disk_not_found_evidence_sha256": HEX,
        "zero_accelerator_evidence_uri": "lifecycle/zero-accelerators.json",
        "zero_accelerator_evidence_sha256": HEX,
    }
    row = {
        "cell_id": "p0b-h16-mu0-eta0p0875-s337",
        "attempt_id": "p0b-h16-mu0-eta0p0875-s337-attempt-1",
        "work": {"outer_steps": 32, "fixed_window_microsteps": 16},
        "observed_work": {
            "per_fragment_outer_steps": {
                str(fragment): 8 for fragment in range(4)
            }
        },
        "hardware": hardware,
    }
    manifest = {
        "frozen": {"image_id": "7290368630472593484"},
        "protocol": {"fragments": 4},
        "canary_hardware_evidence_policy": {
            "p0b_result_hardware_required_fields": [
                "nvidia_smi_inventory_uri",
                "nvidia_smi_inventory_sha256",
                "learner_gpu_map_uri",
                "learner_gpu_map_sha256",
                "barrier_version_trace_uri",
                "barrier_version_trace_sha256",
            ]
        },
    }
    return manifest, row


def _errors(manifest: dict, row: dict) -> list[str]:
    errors: list[str] = []
    validator._validate_p0b_hardware(manifest, [row], errors)
    return errors


def test_valid_p0b_barrier_summary_is_bound_to_raw_registry() -> None:
    manifest, row = _p0b_row()
    assert _errors(manifest, row) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("barrier_trace_validated", False, "barrier_trace_validated must be true"),
        ("base_versions_match", False, "base_versions_match must be true"),
        (
            "no_inner_step_while_blocked",
            False,
            "no_inner_step_while_blocked must be true",
        ),
        (
            "barrier_trace_learner_count",
            3,
            "barrier_trace_learner_count must be exactly 4",
        ),
        (
            "barrier_trace_commit_count",
            31,
            "barrier_trace_commit_count must equal",
        ),
        (
            "barrier_trace_inner_steps_per_learner",
            512,
            "barrier_trace_inner_steps_per_learner must equal",
        ),
    ],
)
def test_p0b_barrier_summary_rejects_missing_or_failed_causal_attestation(
    field: str, value: object, message: str
) -> None:
    manifest, row = _p0b_row()
    row["hardware"][field] = value
    assert message in "\n".join(_errors(manifest, row))

    missing_manifest, missing_row = _p0b_row()
    del missing_row["hardware"][field]
    assert message in "\n".join(_errors(missing_manifest, missing_row))


def test_p0b_barrier_summary_rejects_unbound_registry_or_update_count() -> None:
    manifest, row = _p0b_row()
    del row["hardware"]["barrier_version_trace_sha256"]
    row["observed_work"]["per_fragment_outer_steps"]["3"] = 7
    errors = "\n".join(_errors(manifest, row))
    assert "missing P0b evidence fields" in errors
    assert "barrier_version_trace_sha256 must be" in errors
    assert "exactly 8 applied updates per fragment" in errors


def test_p0b_barrier_summary_keeps_uuid_bijection_as_separate_required_proof() -> None:
    manifest, row = _p0b_row()
    row = copy.deepcopy(row)
    row["hardware"]["learner_gpu_uuid_bijection"]["3"] = "GPU-a"
    assert "four distinct GPU UUIDs" in "\n".join(_errors(manifest, row))
