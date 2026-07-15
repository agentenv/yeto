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
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--learner-max-steps",
        "2",
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
    trace_entries = []
    for learner in range(4):
        trace_path = (
            attempt
            / "work"
            / "m4"
            / f"learner-{learner}"
            / "barrier-version-trace.jsonl"
        )
        events = []

        def emit(event, local_step, **fields):
            events.append(
                {
                    "schema": "yeto_barrier_trace_v1",
                    "event_seq": len(events) + 1,
                    "learner_id": learner,
                    "local_step": local_step,
                    "event": event,
                    **fields,
                }
            )

        for fragment in range(4):
            emit(
                "initial_broadcast_applied",
                0,
                fragment=fragment,
                broadcast_version=0,
                awaiting_fragments=[],
            )
        emit("inner_step_started", 1, awaiting_fragments=[])
        emit("inner_step_started", 2, awaiting_fragments=[])
        awaiting = []
        for step in range(1, 5):
            fragment = step - 1
            awaiting.append(fragment)
            emit(
                "push_sent",
                2,
                fragment=fragment,
                pull_step=step,
                base_version=0,
                c_steps=2,
                c_tokens=2,
                awaiting_fragments=list(awaiting),
            )
        for step in range(1, 5):
            fragment = step - 1
            awaiting.remove(fragment)
            emit(
                "broadcast_applied",
                2,
                fragment=fragment,
                pushed_base_version=0,
                broadcast_version=step,
                awaiting_fragments=list(awaiting),
            )
        trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))
        trace_entries.append(
            {
                "learner_id": learner,
                "path": trace_path.relative_to(attempt).as_posix(),
                "sha256": replay.sha256_file(trace_path),
                "size_bytes": trace_path.stat().st_size,
            }
        )
    barrier_registry_path = attempt / "report" / "barrier-version-trace.json"
    barrier_registry_path.parent.mkdir(parents=True, exist_ok=True)
    barrier_registry = {
        "schema": "yeto_barrier_version_trace_v1",
        "learner_count": 4,
        "syncer_tape": {
            "path": tape_path.relative_to(attempt).as_posix(),
            "sha256": replay.sha256_file(tape_path),
            "size_bytes": tape_path.stat().st_size,
        },
        "learner_traces": trace_entries,
    }
    barrier_registry_path.write_text(
        json.dumps(barrier_registry, sort_keys=True, separators=(",", ":")) + "\n"
    )
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
                    "barrier_version_trace_uri": (
                        "gs://bucket/p0/"
                        + barrier_registry_path.relative_to(root).as_posix()
                    ),
                    "barrier_version_trace_sha256": replay.sha256_file(
                        barrier_registry_path
                    ),
                    "barrier_version_trace_canonical_sha256": replay.sha256_bytes(
                        replay.canonical_json(barrier_registry)
                    ),
                    "barrier_trace_validated": True,
                    "base_versions_match": True,
                    "no_inner_step_while_blocked": True,
                    "barrier_trace_learner_count": 4,
                    "barrier_trace_commit_count": 4,
                    "barrier_trace_inner_steps_per_learner": 2,
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


def test_p0b_replay_source_rebind_is_bound_to_exact_erratum(monkeypatch):
    head = "b" * 40
    script_bytes = (ROOT / replay.SCRIPT_RELATIVE_PATH).read_bytes()
    prereg_bytes = (ROOT / replay.AUTHORITATIVE_PREREG_RELATIVE_PATH).read_bytes()
    erratum_bytes = (ROOT / replay.P0B_REPLAY_ERRATUM_PATH).read_bytes()

    def fake_git_output(*args: str) -> bytes:
        if args == ("rev-parse", "HEAD"):
            return f"{head}\n".encode()
        if args[0] in {"cat-file", "merge-base"}:
            return b""
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if args == (
            "show",
            f"{head}:{replay.P0B_REPLAY_ERRATUM_PATH.as_posix()}",
        ):
            return erratum_bytes
        if args == ("show", f"{head}:{replay.SCRIPT_RELATIVE_PATH.as_posix()}"):
            return script_bytes
        if args == (
            "show",
            f"{replay.AUTHORITATIVE_PREREG_COMMIT}:"
            f"{replay.AUTHORITATIVE_PREREG_RELATIVE_PATH.as_posix()}",
        ):
            return prereg_bytes
        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(replay, "git_output", fake_git_output)
    manifest = {
        "frozen": {"git_commit": replay.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT},
        "lineage": {
            "authoritative_prereg_path": (
                replay.AUTHORITATIVE_PREREG_RELATIVE_PATH.as_posix()
            ),
            "authoritative_prereg_source_commit": replay.AUTHORITATIVE_PREREG_COMMIT,
            "authoritative_prereg_template_sha256": replay.AUTHORITATIVE_PREREG_SHA256,
        },
    }

    attestation = replay.verify_replay_source(manifest)

    assert attestation["replay_validator_git_commit"] == head
    assert attestation["replay_source_rebind_from_git_commit"] == (
        replay.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT
    )
    assert attestation["replay_source_rebind_erratum_path"] == (
        replay.P0B_REPLAY_ERRATUM_PATH.as_posix()
    )
    assert attestation["replay_source_rebind_erratum_sha256"] == (
        replay.P0B_REPLAY_ERRATUM_SHA256
    )


def test_p0b_replay_source_rebind_rejects_wrong_erratum(monkeypatch):
    monkeypatch.setattr(
        replay,
        "git_output",
        lambda *args: (
            f"{'b' * 40}\n".encode()
            if args == ("rev-parse", "HEAD")
            else b"tampered"
            if args[0] == "show"
            else b""
        ),
    )
    manifest = {
        "frozen": {"git_commit": replay.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT}
    }

    with pytest.raises(replay.ReplayError, match="exact authority document"):
        replay.verify_replay_source(manifest)


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


def test_same_second_object_seal_and_deletion_request_remain_ordered(tmp_path):
    root, deleted = _fixture(tmp_path)
    state = json.loads(deleted.read_text())
    state["artifact_object_seal"]["sealed_at_utc"] = state[
        "deletion_requested_at_utc"
    ]
    finalizer.write_object(deleted, state)
    _refinalize(root, deleted)
    output = tmp_path / "same-second-replay.json"

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


def _barrier_replay_inputs(root: Path):
    manifest = json.loads((root / "phase-map-manifest.json").read_text())
    result = manifest["results"][0]
    attempt = root / "cells" / result["cell_id"] / "attempt-1"
    tape = replay.read_jsonl(attempt / "work/m4/tape.jsonl")
    registry = replay.verify_acquisition(root, root / "acquisition.sha256")
    return result, attempt, tape, registry


def _reseal_barrier_trace_for_unit(
    root: Path,
    result: dict,
    attempt: Path,
    acquisition_registry: dict[str, str],
    learner: int,
):
    registry_path = attempt / "report/barrier-version-trace.json"
    trace_path = attempt / f"work/m4/learner-{learner}/barrier-version-trace.jsonl"
    barrier_registry = json.loads(registry_path.read_text())
    entry = next(
        row
        for row in barrier_registry["learner_traces"]
        if row["learner_id"] == learner
    )
    entry["sha256"] = replay.sha256_file(trace_path)
    entry["size_bytes"] = trace_path.stat().st_size
    registry_path.write_text(
        json.dumps(barrier_registry, sort_keys=True, separators=(",", ":")) + "\n"
    )
    trace_relative = trace_path.relative_to(root).as_posix()
    registry_relative = registry_path.relative_to(root).as_posix()
    acquisition_registry[trace_relative] = replay.sha256_file(trace_path)
    acquisition_registry[registry_relative] = replay.sha256_file(registry_path)
    result["hardware"]["barrier_version_trace_sha256"] = replay.sha256_file(
        registry_path
    )
    result["hardware"]["barrier_version_trace_canonical_sha256"] = (
        replay.sha256_bytes(replay.canonical_json(barrier_registry))
    )


def test_cpu_replay_rejects_inner_step_between_push_and_broadcast(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    trace = attempt / "work/m4/learner-1/barrier-version-trace.jsonl"
    events = replay.read_jsonl(trace)
    push_index = next(
        index for index, event in enumerate(events) if event["event"] == "push_sent"
    )
    events.insert(
        push_index + 1,
        {
            "schema": "yeto_barrier_trace_v1",
            "event_seq": 0,
            "learner_id": 1,
            "local_step": events[push_index]["local_step"] + 1,
            "event": "inner_step_started",
            "awaiting_fragments": [],
        },
    )
    for sequence, event in enumerate(events, 1):
        event["event_seq"] = sequence
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _reseal_barrier_trace_for_unit(root, result, attempt, registry, 1)

    with pytest.raises(replay.ReplayError, match="while blocked"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


def test_cpu_replay_rejects_post_final_broadcast_extra_inner_step(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    trace = attempt / "work/m4/learner-1/barrier-version-trace.jsonl"
    events = replay.read_jsonl(trace)
    events.append(
        {
            "schema": "yeto_barrier_trace_v1",
            "event_seq": len(events) + 1,
            "learner_id": 1,
            "local_step": 3,
            "event": "inner_step_started",
            "awaiting_fragments": [],
        }
    )
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _reseal_barrier_trace_for_unit(root, result, attempt, registry, 1)

    with pytest.raises(replay.ReplayError, match="inner-step count"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


@pytest.mark.parametrize(
    "command",
    [
        ["--learner-max-steps", "3"],
        ["--learner-max-steps", "2", "--learner-max-steps", "2"],
        [],
    ],
)
def test_cpu_replay_rejects_nonexact_learner_step_cap(command):
    with pytest.raises(replay.ReplayError, match="learner step cap"):
        replay.validate_exact_learner_max_steps(command, [{}, {}, {}, {}], h=2)


def test_cpu_replay_rejects_rehashed_late_initial_broadcast(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    trace = attempt / "work/m4/learner-3/barrier-version-trace.jsonl"
    events = replay.read_jsonl(trace)
    events[1]["local_step"] = 1
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _reseal_barrier_trace_for_unit(root, result, attempt, registry, 3)

    with pytest.raises(replay.ReplayError, match="initial broadcast prefix"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


def test_cpu_replay_rejects_missing_broadcast_even_when_rehashed(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    trace = attempt / "work/m4/learner-2/barrier-version-trace.jsonl"
    events = replay.read_jsonl(trace)
    removed = False
    retained = []
    for event in events:
        if not removed and event["event"] == "broadcast_applied":
            removed = True
            continue
        retained.append(event)
    for sequence, event in enumerate(retained, 1):
        event["event_seq"] = sequence
    trace.write_text("".join(json.dumps(event) + "\n" for event in retained))
    _reseal_barrier_trace_for_unit(root, result, attempt, registry, 2)

    with pytest.raises(replay.ReplayError, match="awaiting state|coverage"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


def test_cpu_replay_rejects_barrier_trace_omitted_from_acquisition(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    omitted = attempt / "work/m4/learner-3/barrier-version-trace.jsonl"
    registry.pop(omitted.relative_to(root).as_posix())

    with pytest.raises(replay.ReplayError, match="acquisition-bound"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


def test_cpu_replay_rejects_rehashed_late_fragment_window(tmp_path):
    root, _deleted = _fixture(tmp_path)
    result, attempt, tape, registry = _barrier_replay_inputs(root)
    trace = attempt / "work/m4/learner-0/barrier-version-trace.jsonl"
    events = replay.read_jsonl(trace)
    for event in events:
        if event["event"] in ("push_sent", "broadcast_applied"):
            event["local_step"] = 3
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _reseal_barrier_trace_for_unit(root, result, attempt, registry, 0)

    with pytest.raises(replay.ReplayError, match="push does not biject"):
        replay.validate_barrier_version_trace(
            root, attempt, result, tape, registry, h=2, seq_len=1
        )


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
    final = json.loads(json.dumps(first))
    final["attempt"] = 3
    final["hardware"]["barrier_version_trace_uri"] = final["hardware"][
        "barrier_version_trace_uri"
    ].replace("/attempt-1/", "/attempt-3/")
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
