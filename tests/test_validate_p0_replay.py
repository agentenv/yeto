from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_p0_replay", ROOT / "scripts" / "validate_p0_replay.py"
)
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)

FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_p0_lifecycle", ROOT / "scripts" / "finalize_p0_lifecycle.py"
)
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = finalizer
assert FINALIZER_SPEC.loader is not None
FINALIZER_SPEC.loader.exec_module(finalizer)


def _write_checkpoint(path: Path, versions, params, momentum, global_step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray(struct.pack("<IQI", replay.CKPT_MAGIC, global_step, 4))
    for fragment in range(4):
        values = np.asarray(params[fragment], dtype="<f4")
        buffer = np.asarray(momentum[fragment], dtype="<f4")
        payload.extend(struct.pack("<QQ", versions[fragment], values.size))
        payload.extend(values.tobytes())
        payload.extend(buffer.tobytes())
    payload.extend(struct.pack("<I", 0))
    path.write_bytes(bytes(payload))


def _fixture(tmp_path: Path):
    root = tmp_path / "p0"
    cell_id = "p0-h2-mu0p5-eta0p1-s337"
    attempt = root / "cells" / cell_id / "attempt-1"
    frozen = root / "frozen-eval" / "seed-337" / "materialized"
    frozen.mkdir(parents=True)
    train_file = frozen / "train.jsonl"
    development_file = frozen / "eval.jsonl"
    train_file.write_text('{"messages":[]}\n')
    development_file.write_text('{"messages":[]}\n')
    capture = attempt / "work" / "m4" / "syncer_probe"
    (capture / "states").mkdir(parents=True)
    (capture / "candidates").mkdir()
    command = [
        "python3",
        "compare_diloco.py",
        "--data",
        str(train_file),
        "--prebound-development-eval",
        str(development_file),
        "--outer-lr",
        "0.1",
        "--outer-momentum",
        "0.5",
        "--fixed-window-microsteps",
        "2",
        "--seq-len",
        "1",
        "--shuffle-rows-seed",
        "337",
        "--training-seed",
        "337337",
        "--syncer-probe-capture",
        "--syncer-probe-capture-every",
        "1",
        "--strict-quorum",
        "--barrier-sync",
        "--version-matched-anchor",
    ]
    (attempt / "command.json").parent.mkdir(parents=True, exist_ok=True)
    (attempt / "command.json").write_text(json.dumps(command))
    command_hash = replay.sha256_bytes(replay.canonical_json(command))
    layout = {
        "matrix_merge": "rda",
        "fragments": [
            {
                "id": fragment,
                "merge_mode": "avg" if fragment == 0 else "rda",
                "tensors": [{"name": f"p{fragment}", "numel": 2}],
            }
            for fragment in range(4)
        ],
    }
    for learner in range(4):
        layout_path = (
            attempt
            / "work"
            / "m4"
            / f"learner-{learner}"
            / "resolved-layout.json"
        )
        layout_path.parent.mkdir(parents=True)
        layout_path.write_text(json.dumps(layout))

    versions = [0, 0, 0, 0]
    params = [np.asarray([1.0 + i, 2.0 + i], dtype="<f4") for i in range(4)]
    momentum = [np.zeros(2, dtype="<f4") for _ in range(4)]
    tape_rows = []
    index_rows = []
    for step in range(1, 5):
        fragment = step - 1
        base_version = versions[fragment]
        state_name = f"state_before_step_{step:08d}.ckpt"
        _write_checkpoint(
            capture / "states" / state_name,
            versions,
            params,
            momentum,
            global_step=step - 1,
        )
        deltas = []
        for learner in range(4):
            delta = np.asarray(
                [0.01 * (learner + 1), 0.02 * (learner + 1)], dtype="<f4"
            )
            deltas.append(delta)
            candidate_name = (
                f"candidate_step_{step:08d}_fragment_{fragment:04d}_"
                f"learner_{learner:04d}.f32"
            )
            (capture / "candidates" / candidate_name).write_bytes(
                (params[fragment] - delta).astype("<f4").tobytes()
            )
            index_rows.append(
                {
                    "step": step,
                    "syncer_global_step": step - 1,
                    "fragment": fragment,
                    "current_fragment_version": versions[fragment],
                    "learner_id": learner,
                    "base_version": base_version,
                    "c_steps": 2,
                    "c_tokens": 2,
                    "weight": 2.0,
                    "state_checkpoint": f"states/{state_name}",
                    "candidate_f32": f"candidates/{candidate_name}",
                }
            )
        merged = sum(deltas, np.zeros(2, dtype="<f4")) * np.float32(0.25)
        momentum[fragment] = np.float32(0.5) * momentum[fragment] + merged
        direction = merged + np.float32(0.5) * momentum[fragment]
        params[fragment] = params[fragment] - np.float32(0.1) * direction
        versions[fragment] = step
        tape_rows.append(
            {
                "step": step,
                "fragment": fragment,
                "gnorm": replay.l2(merged),
                "outer_step_norm": replay.l2(np.float32(0.1) * direction),
                "responders": [
                    {
                        "id": learner,
                        "base_version": base_version,
                        "c_steps": 2,
                        "c_tokens": 2,
                        "weight": 2.0,
                        "anchor_base_resolved": True,
                    }
                    for learner in range(4)
                ],
            }
        )
    (capture / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    tape_path = attempt / "work" / "m4" / "tape.jsonl"
    tape_path.write_text("".join(json.dumps(row) + "\n" for row in tape_rows))
    _write_checkpoint(
        attempt / "work" / "m4" / "state.ckpt",
        versions,
        params,
        momentum,
        global_step=4,
    )
    provider = {
        "provider": "gcp",
        "market": "spot",
        "project": "test-project",
        "zone": "us-central1-b",
        "instance_name": "p0-vm",
        "instance_id": "123",
        "boot_disk_id": "456",
        "source_image_id": "7290368630472593484",
    }
    provider_path = root / "provider-evidence" / "instance-123.json"
    provider_path.parent.mkdir(parents=True)
    provider_path.write_text(json.dumps(provider, sort_keys=True) + "\n")
    provider_sha = replay.sha256_file(provider_path)
    manifest = {
        "status": "sealed_acquisition_pending_teardown",
        "lineage": {"descendant_kind": "p0a_single_gpu_bound"},
        "frozen": {
            "git_commit": "a" * 40,
            "image_id": "7290368630472593484",
            "cell_command_hashes": {cell_id: command_hash},
        },
        "expected_cells": [
            {
                "cell_id": cell_id,
                "h": 2,
                "mu": 0.5,
                "eta": 0.1,
                "seed": 337,
                "training_seed": 337337,
            }
        ],
        "results": [
            {
                "cell_id": cell_id,
                "attempt": 1,
                "status": "COMPLETED",
                "h": 2,
                "mu": 0.5,
                "eta": 0.1,
                "seed": 337,
                "training_seed": 337337,
                "command_hash": command_hash,
                "hardware": {
                    **provider,
                    "provisioning_evidence_sha256": provider_sha,
                },
            }
        ],
    }
    manifest_path = root / "phase-map-manifest.json"
    acquisition_manifest_path = root / "phase-map-acquisition-manifest.json"
    finalizer.write_object(acquisition_manifest_path, manifest)
    manifest_path.write_bytes(acquisition_manifest_path.read_bytes())
    now = datetime.now(timezone.utc)
    sealed_at = now - timedelta(minutes=3)
    (root / "acquisition-seal.json").write_text(
        json.dumps(
            {
                "schema": "yeto_phase_map_acquisition_seal_v1",
                "sealed_at_utc": sealed_at.isoformat(),
                "phase_map_manifest_sha256": replay.sha256_file(
                    acquisition_manifest_path
                ),
                "phase_map_manifest_canonical_sha256": replay.sha256_bytes(
                    replay.canonical_json(manifest)
                ),
                "loss_blind_mechanical_seal": True,
            }
        )
    )
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    ]
    lines = [
        f"{replay.sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths)
    ]
    acquisition = root / "acquisition.sha256"
    acquisition.write_text("\n".join(lines) + "\n")
    deleted = tmp_path / "deleted.json"
    requested_at = now - timedelta(minutes=2)
    deleted_at = now - timedelta(minutes=1)
    artifact_uri = "gs://bucket/p0"
    deleted.write_text(
        json.dumps(
            {
                "status": "DELETED",
                "project": "test-project",
                "zone": "us-central1-b",
                "instance_name": "p0-vm",
                "artifact_uri": artifact_uri,
                "repo_commit": "a" * 40,
                "source_image_id": "7290368630472593484",
                "verified_instance_absent": True,
                "verified_boot_disk_absent": True,
                "deletion_requested_at_utc": requested_at.isoformat(),
                "deleted_at_utc": deleted_at.isoformat(),
                "deleted_instance_id": "123",
                "deleted_instance_name": "p0-vm",
                "deleted_boot_disk_id": "456",
                "deleted_boot_disk_name": "p0-vm",
                "provider_not_found_verification": {
                    "instance": {
                        "name": "p0-vm",
                        "provider_id": "123",
                        "result": "NOT_FOUND",
                        "verified_at_utc": deleted_at.isoformat(),
                    },
                    "boot_disk": {
                        "name": "p0-vm",
                        "provider_id": "456",
                        "result": "NOT_FOUND",
                        "verified_at_utc": deleted_at.isoformat(),
                    },
                },
                "post_delete_accelerator_proof": {
                    "project": "test-project",
                    "campaign_owned_accelerators": 0,
                    "total_active_accelerators": 4,
                    "inventory_sha256": "f" * 64,
                    "queried_at_utc": deleted_at.isoformat(),
                },
                "artifact_object_seal": {
                    "schema": "optimizer_harness_artifact_object_seal_v1",
                    "sealed_at_utc": (
                        sealed_at + timedelta(seconds=30)
                    ).isoformat(),
                    "objects": [
                        {
                            "role": "phase_map_manifest",
                            "uri": f"{artifact_uri}/phase-map-manifest.json",
                            "generation": "11",
                            "metageneration": "1",
                            "size": acquisition_manifest_path.stat().st_size,
                            "sha256": replay.sha256_file(
                                acquisition_manifest_path
                            ),
                        },
                        {
                            "role": "acquisition_checksum",
                            "uri": f"{artifact_uri}/acquisition.sha256",
                            "generation": "12",
                            "metageneration": "1",
                            "size": acquisition.stat().st_size,
                            "sha256": replay.sha256_file(acquisition),
                        },
                    ],
                },
            }
        )
    )
    finalizer.finalize(root, deleted, expected_instance_id="123")
    return root, root / "lifecycle" / "deletion-evidence.json"


def _source_attestation(_manifest):
    return {
        "replay_validator_git_commit": "a" * 40,
        "replay_validator_script_sha256": "b" * 64,
        "replay_validator_git_blob_sha256": "b" * 64,
        "authoritative_prereg_source_commit": replay.AUTHORITATIVE_PREREG_COMMIT,
        "authoritative_prereg_template_sha256": replay.AUTHORITATIVE_PREREG_SHA256,
    }


def _manifest_attestation(_manifest):
    return {
        "phase_map_integrity_status": "VALIDATED",
        "phase_map_validator_report_sha256": "c" * 64,
    }


def _refresh_acquisition(root: Path, deleted: Path):
    acquisition = root / "acquisition.sha256"
    excluded = {
        acquisition,
        root / "phase-map-manifest.json",
        root / "phase-map.sha256",
        root / "phase-map-lifecycle-seal.json",
    }
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path not in excluded
        and (root / "lifecycle") not in path.parents
    ]
    acquisition.write_text(
        "\n".join(
            f"{replay.sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in sorted(paths)
        )
        + "\n"
    )
    state = json.loads(deleted.read_text())
    item = next(
        row
        for row in state["artifact_object_seal"]["objects"]
        if row["role"] == "acquisition_checksum"
    )
    item["sha256"] = replay.sha256_file(acquisition)
    item["size"] = acquisition.stat().st_size
    finalizer.write_object(deleted, state)
    _refinalize(root, deleted)


def _refinalize(root: Path, deleted: Path):
    (root / "phase-map-manifest.json").write_bytes(
        (root / "phase-map-acquisition-manifest.json").read_bytes()
    )
    finalizer.finalize(root, deleted, expected_instance_id="123")


def test_replay_all_steps_only_after_verified_deletion(tmp_path):
    root, deleted = _fixture(tmp_path)
    registry = replay.verify_acquisition(root, root / "acquisition.sha256")
    assert "phase-map-acquisition-manifest.json" in registry
    assert "phase-map-manifest.json" not in registry
    finalized = json.loads((root / "phase-map-manifest.json").read_text())
    assert finalized["status"] == "sealed_results"
    assert finalized["results"][0]["hardware"]["acquisition_status"] == (
        "sealed_acquisition_pending_teardown"
    )
    output = tmp_path / "replay.json"
    report = replay.validate(
        root,
        deleted,
        output,
        atol=2e-6,
        rtol=2e-6,
        tape_rtol=2e-4,
        source_verifier=_source_attestation,
        manifest_validator=_manifest_attestation,
    )
    assert report["status"] == "PASS"
    assert report["gpu_deleted_before_replay"] is True
    assert report["cells"][0]["replayed_attempts"][0]["commit_count"] == 4
    assert output.with_suffix(".json.sha256").is_file()


def test_lifecycle_rejects_post_seal_scientific_mutation_even_if_rehashed(tmp_path):
    root, deleted = _fixture(tmp_path)
    manifest_path = root / "phase-map-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["results"][0]["mu"] = 0.9
    finalizer.write_object(manifest_path, manifest)
    envelope_path = root / "phase-map-lifecycle-seal.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["final_manifest_sha256"] = replay.sha256_file(manifest_path)
    envelope["final_manifest_canonical_sha256"] = replay.sha256_bytes(
        replay.canonical_json(manifest)
    )
    finalizer.write_object(envelope_path, envelope)
    with pytest.raises(replay.ReplayError, match="mutates pre-delete scientific"):
        replay.validate_lifecycle_finalization(root, manifest, deleted)


def test_finalizer_rejects_legacy_canary_kind(tmp_path):
    root, deleted = _fixture(tmp_path)
    acquisition_path = root / "phase-map-acquisition-manifest.json"
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["lineage"]["descendant_kind"] = "p0_canary_bound"
    finalizer.write_object(acquisition_path, acquisition)
    (root / "phase-map-manifest.json").write_bytes(acquisition_path.read_bytes())
    with pytest.raises(finalizer.FinalizationError, match="legacy"):
        finalizer.finalize(root, deleted, expected_instance_id="123")


def test_p0b_launched_attempt_without_uuid_barrier_evidence_cannot_finalize(
    tmp_path,
):
    root, deleted = _fixture(tmp_path)
    acquisition_path = root / "phase-map-acquisition-manifest.json"
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["lineage"]["descendant_kind"] = "p0b_four_gpu_bound"
    finalizer.write_object(acquisition_path, acquisition)
    seal_path = root / "acquisition-seal.json"
    seal = json.loads(seal_path.read_text())
    seal["phase_map_manifest_sha256"] = replay.sha256_file(acquisition_path)
    seal["phase_map_manifest_canonical_sha256"] = replay.sha256_bytes(
        replay.canonical_json(acquisition)
    )
    finalizer.write_object(seal_path, seal)
    state = json.loads(deleted.read_text())
    phase_object = next(
        item
        for item in state["artifact_object_seal"]["objects"]
        if item["role"] == "phase_map_manifest"
    )
    phase_object["sha256"] = replay.sha256_file(acquisition_path)
    phase_object["size"] = acquisition_path.stat().st_size
    finalizer.write_object(deleted, state)
    with pytest.raises(finalizer.FinalizationError, match="lacks attempt evidence"):
        _refresh_acquisition(root, deleted)


def test_replay_rejects_capture_not_in_sealed_acquisition(tmp_path):
    root, deleted = _fixture(tmp_path)
    acquisition = root / "acquisition.sha256"
    lines = [
        line
        for line in acquisition.read_text().splitlines()
        if "candidate_step_00000001" not in line
    ]
    acquisition.write_text("\n".join(lines) + "\n")
    deletion_state = json.loads(deleted.read_text())
    acquisition_object = next(
        item
        for item in deletion_state["artifact_object_seal"]["objects"]
        if item["role"] == "acquisition_checksum"
    )
    acquisition_object["sha256"] = replay.sha256_file(acquisition)
    acquisition_object["size"] = acquisition.stat().st_size
    finalizer.write_object(deleted, deletion_state)
    _refinalize(root, deleted)
    with pytest.raises(replay.ReplayError, match="capture is not sealed"):
        replay.validate(
            root,
            deleted,
            tmp_path / "replay.json",
            atol=2e-6,
            rtol=2e-6,
            tape_rtol=2e-4,
            source_verifier=_source_attestation,
            manifest_validator=_manifest_attestation,
        )


def test_replay_rejects_live_gpu_state(tmp_path):
    root, deleted = _fixture(tmp_path)
    state = json.loads(deleted.read_text())
    state["status"] = "RUNNING_EXPERIMENT"
    deleted.write_text(json.dumps(state))
    with pytest.raises(replay.ReplayError, match="requires verified exact VM"):
        replay.validate(
            root,
            deleted,
            tmp_path / "replay.json",
            atol=2e-6,
            rtol=2e-6,
            tape_rtol=2e-4,
            source_verifier=_source_attestation,
            manifest_validator=_manifest_attestation,
        )


def test_replay_rejects_inflated_tolerance(tmp_path):
    root, deleted = _fixture(tmp_path)
    with pytest.raises(replay.ReplayError, match="tolerances are frozen"):
        replay.validate(
            root,
            deleted,
            tmp_path / "replay.json",
            atol=1.0,
            rtol=replay.PARAM_RTOL,
            tape_rtol=replay.TAPE_NORM_RTOL,
            source_verifier=_source_attestation,
            manifest_validator=_manifest_attestation,
        )


def test_replay_rejects_sealed_nonfinite_candidate(tmp_path):
    root, deleted = _fixture(tmp_path)
    candidate = next(root.rglob("candidate_step_00000001*.f32"))
    values = np.fromfile(candidate, dtype="<f4")
    values[0] = np.float32(np.nan)
    candidate.write_bytes(values.tobytes())
    _refresh_acquisition(root, deleted)
    with pytest.raises(replay.ReplayError, match="nonfinite"):
        replay.validate(
            root,
            deleted,
            tmp_path / "replay.json",
            atol=replay.PARAM_ATOL,
            rtol=replay.PARAM_RTOL,
            tape_rtol=replay.TAPE_NORM_RTOL,
            source_verifier=_source_attestation,
            manifest_validator=_manifest_attestation,
        )


def test_replay_rejects_command_hash_or_coordinate_drift(tmp_path):
    root, deleted = _fixture(tmp_path)
    command_path = next(root.rglob("attempt-1/command.json"))
    command = json.loads(command_path.read_text())
    command[command.index("--outer-lr") + 1] = "0.2"
    command_path.write_text(json.dumps(command))
    _refresh_acquisition(root, deleted)
    with pytest.raises(replay.ReplayError, match="command hash"):
        replay.validate(
            root,
            deleted,
            tmp_path / "replay.json",
            atol=replay.PARAM_ATOL,
            rtol=replay.PARAM_RTOL,
            tape_rtol=replay.TAPE_NORM_RTOL,
            source_verifier=_source_attestation,
            manifest_validator=_manifest_attestation,
        )


def test_replay_accepts_authenticated_same_vm_whole_block_retry_history(tmp_path):
    root, deleted = _fixture(tmp_path)
    manifest_path = root / "phase-map-acquisition-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first = manifest["results"][0]
    infra = {
        **first,
        "attempt": 2,
        "status": "INFRA_FAILURE",
        "failure_reason": "provider_spot_preemption",
    }
    final = {**first, "attempt": 3}
    source = root / "cells" / first["cell_id"] / "attempt-1"
    destination = root / "cells" / first["cell_id"] / "attempt-3"
    shutil.copytree(source, destination)
    manifest["results"] = [first, infra, final]
    finalizer.write_object(manifest_path, manifest)
    seal_path = root / "acquisition-seal.json"
    seal = json.loads(seal_path.read_text())
    seal["phase_map_manifest_sha256"] = replay.sha256_file(manifest_path)
    seal["phase_map_manifest_canonical_sha256"] = replay.sha256_bytes(
        replay.canonical_json(manifest)
    )
    seal_path.write_text(json.dumps(seal))
    state = json.loads(deleted.read_text())
    phase_object = next(
        row
        for row in state["artifact_object_seal"]["objects"]
        if row["role"] == "phase_map_manifest"
    )
    phase_object["sha256"] = replay.sha256_file(manifest_path)
    phase_object["size"] = manifest_path.stat().st_size
    deleted.write_text(json.dumps(state))
    _refresh_acquisition(root, deleted)

    report = replay.validate(
        root,
        deleted,
        tmp_path / "replay.json",
        atol=replay.PARAM_ATOL,
        rtol=replay.PARAM_RTOL,
        tape_rtol=replay.TAPE_NORM_RTOL,
        source_verifier=_source_attestation,
        manifest_validator=_manifest_attestation,
    )

    assert report["cell_count"] == 1
    assert report["replayed_scientific_attempt_count"] == 2
    assert report["cells"][0]["final_attempt"] == 3
