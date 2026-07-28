#!/usr/bin/env python3
"""Preflight or drain one full-node v7 smoke, pilot, or main-grid queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


MIN_FREE_BYTES = 1_000_000_000_000
MODEL_INVENTORY_SHA256 = (
    "32c8f34fa11f07ffde3eedb32435b39a78590ea102b7923bbc1d9b4df7b51c4c"
)
TELEMETRY_SCHEMA = "yeto_rho_telemetry_v1"
BARRIER_TRACE_SCHEMA = "yeto_barrier_trace_v1"
FRAGMENT_COUNT = 4
STAGES = ("SMOKE", "PILOT", "MAIN")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def gpu_inventory() -> dict[int, dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    inventory = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise RuntimeError(f"malformed nvidia-smi row: {line!r}")
        index = int(parts[0])
        inventory[index] = {
            "index": index,
            "uuid": parts[1],
            "name": parts[2],
            "memory_total_mib": int(parts[3]),
            "memory_used_mib": int(parts[4]),
            "utilization_percent": int(parts[5]),
        }
    return inventory


def gpu_sample() -> dict:
    try:
        return {str(index): record for index, record in gpu_inventory().items()}
    except Exception as exc:
        return {"error": str(exc)}


def model_inventory() -> dict:
    rows = []
    for path in sorted(common.MODEL.iterdir()):
        resolved = path.resolve(strict=True)
        rows.append(
            {
                "name": path.name,
                "blob": resolved.name,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "entries": len(rows),
        "bytes": sum(item["bytes"] for item in rows),
        "canonical_inventory_sha256": common.canonical_sha256(rows),
    }


def storage_proof() -> tuple[dict, list[str]]:
    errors = []
    proof = {
        "result_link": str(common.RESULT_LINK),
        "result_target_expected": str(common.RESULT_TARGET),
        "is_symlink": common.RESULT_LINK.is_symlink(),
    }
    try:
        resolved = common.RESULT_LINK.resolve(strict=True)
        proof["result_target_resolved"] = str(resolved)
        proof["same_device_as_data"] = (
            resolved.stat().st_dev == Path("/data").stat().st_dev
        )
        usage = shutil.disk_usage(resolved)
        proof["filesystem_total_bytes"] = usage.total
        proof["filesystem_used_bytes"] = usage.used
        proof["filesystem_free_bytes"] = usage.free
        findmnt = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "--target", str(resolved)],
            capture_output=True,
            text=True,
        )
        proof["findmnt"] = findmnt.stdout.strip()
        if not common.RESULT_LINK.is_symlink():
            errors.append("v7 results path is not a symlink")
        if resolved != common.RESULT_TARGET:
            errors.append(
                f"v7 results resolve to {resolved}, not {common.RESULT_TARGET}"
            )
        if not proof["same_device_as_data"]:
            errors.append("v7 results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v7 results filesystem has only {usage.free} bytes free")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def manifest_kind(manifest: dict) -> str:
    schema = manifest.get("schema")
    if schema == "yeto_outer_mup_v7_27b_lora_prep_manifest_v1":
        return "PREP"
    if schema == "yeto_outer_mup_v7_27b_lora_launch_manifest_v1":
        return "MAIN"
    raise ValueError("launch manifest schema mismatch")


def validate_command(cell: dict, errors: list[str]) -> None:
    cell_id = cell.get("cell_id", "<missing>")
    command = cell.get("command", [])
    if common.canonical_sha256(command) != cell.get("command_hash"):
        errors.append(f"initial command hash mismatch: {cell_id}")
    retries = cell.get("registered_retry_commands", [])
    if len(retries) != 1 or common.canonical_sha256(
        retries[0].get("command", [])
    ) != retries[0].get("command_hash"):
        errors.append(f"registered retry command mismatch: {cell_id}")
    required_values = {
        "--settings": "m2",
        "--tuning": "lora",
        "--shard": "fsdp",
        "--learner-gpus": "4",
        "--gpu-offset": "0",
        "--seq-len": "128",
        "--micro-batch-size": "1",
        "--inner-lr": "0.0003",
        "--lora-r": "16",
        "--lora-alpha": "32",
        "--eval-rows": "1024",
        "--fixed-window-microsteps": str(cell.get("h")),
        "--fixed-window-tokens": str(cell.get("fixed_window_tokens")),
        "--syncer-total-steps": str(4 * int(cell.get("t", 0))),
        "--learner-max-steps": str(cell.get("s")),
        "--outer-momentum": str(cell.get("mu")),
        "--outer-lr": repr(float(cell.get("eta", math.nan))),
        "--pipeline-depth": "4",
        "--wan-streams": "0",
    }
    for flag, expected in required_values.items():
        if command_value(command, flag) != expected:
            errors.append(f"{cell_id}: {flag} binding mismatch")
    for flag in (
        "--strict-quorum",
        "--barrier-sync",
        "--version-matched-anchor",
        "--rho-telemetry",
        "--skip-baseline",
        "--skip-untrained-eval",
    ):
        if flag not in command:
            errors.append(f"{cell_id}: required flag missing: {flag}")
    for flag in ("--work-dir", "--report-dir"):
        value = command_value(command, flag)
        if value is None or not value.startswith(str(common.RESULT_LINK) + "/"):
            errors.append(f"{cell_id}: {flag} escapes v7 result root")


def verify_preflight(manifest_path: Path, node_label: str, proof_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    try:
        kind = manifest_kind(manifest)
    except ValueError as exc:
        kind = "INVALID"
        errors.append(str(exc))
    if manifest.get("status") != "REGISTERED":
        errors.append("launch manifest status is not REGISTERED")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(
        manifest_path
    ):
        errors.append("launch manifest sidecar is absent or mismatched")

    repo = Path("/root/yeto")
    try:
        head = git(repo, "rev-parse", "HEAD")
        if head != manifest.get("source", {}).get("git_commit"):
            errors.append("node Git commit mismatch")
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        if manifest.get("registration", {}).get("git_commit") != head:
            errors.append("v7 must execute at its manifest registration commit")
    except Exception as exc:
        head = None
        errors.append(f"Git preflight failed: {exc}")

    contract = manifest.get("contract", {})
    expected_contract = common.contract_record()
    for field in (
        "json_sha256",
        "md_sha256",
        "analyzer_sha256",
        "scientific_registration_git_commit",
    ):
        if contract.get(field) != expected_contract[field]:
            errors.append(f"manifest contract {field} mismatch")
    for path, digest in (
        (
            repo / common.CONTRACT_JSON.relative_to(common.REPO),
            common.CONTRACT_JSON_SHA256,
        ),
        (repo / common.CONTRACT_MD.relative_to(common.REPO), common.CONTRACT_MD_SHA256),
        (repo / common.ANALYZER.relative_to(common.REPO), common.ANALYZER_SHA256),
    ):
        if not path.is_file() or common.sha256_file(path) != digest:
            errors.append(f"registered file hash mismatch: {path}")

    for label in ("train", "eval"):
        record = manifest.get("inputs", {}).get(label, {})
        path = Path(record.get("path", ""))
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            errors.append(f"{label} input size mismatch: {path}")
        elif common.sha256_file(path) != record.get("sha256"):
            errors.append(f"{label} input hash mismatch: {path}")
    try:
        model = model_inventory()
        if model["entries"] != 29 or model["bytes"] != 55586107940:
            errors.append("27B snapshot entry/byte inventory mismatch")
        if model["canonical_inventory_sha256"] != MODEL_INVENTORY_SHA256:
            errors.append("27B snapshot canonical inventory hash mismatch")
    except Exception as exc:
        model = {"error": str(exc)}
        errors.append(f"model inventory failed: {exc}")

    cells = manifest.get("cells", [])
    ids = set()
    for cell in cells:
        if cell.get("cell_id") in ids:
            errors.append(f"duplicate cell id: {cell.get('cell_id')}")
        ids.add(cell.get("cell_id"))
        validate_command(cell, errors)
    if kind == "PREP" and (
        len(cells) != 4
        or len([cell for cell in cells if cell.get("stage") == "SMOKE"]) != 1
        or len([cell for cell in cells if cell.get("stage") == "PILOT"]) != 3
    ):
        errors.append("prep stage cell counts are incorrect")
    if kind == "MAIN":
        variant = manifest.get("grid", {}).get("variant")
        expected_cells = 48 if variant == "FULL_48" else 45
        if len(cells) != expected_cells or manifest.get("stage") != "V7_27B_LORA_GRID":
            errors.append("main grid cell count/stage is incorrect")

    try:
        inventory = gpu_inventory()
        if set(inventory) != set(range(8)):
            errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
        if any("H200" not in record["name"].upper() for record in inventory.values()):
            errors.append("non-H200 device in node inventory")
        occupied = [
            index
            for index, record in inventory.items()
            if record["memory_used_mib"] != 0 or record["utilization_percent"] != 0
        ]
        if occupied:
            errors.append(f"GPU preflight is not idle: {occupied}")
    except Exception as exc:
        inventory = {}
        errors.append(f"GPU inventory failed: {exc}")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    proof = {
        "schema": "yeto_outer_mup_v7_node_authority_v1",
        "node": node_label,
        "checked_at_utc": utc_now(),
        "manifest_kind": kind,
        "manifest_path": str(manifest_path),
        "manifest_sha256": common.sha256_file(manifest_path),
        "git_commit": head,
        "model": model,
        "gpu_inventory": inventory,
        "storage": storage,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    common.write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_authority(
    path: Path, manifest_path: Path, stage: str, proof_path: Path
) -> dict:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v7_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("launch authority is not authorized")
    if authority.get("manifest_sha256") != common.sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    if authority.get("stage") != stage:
        errors.append("launch authority stage mismatch")
    proofs = authority.get("node_proof_sha256", {})
    node = json.loads(proof_path.read_text()).get("node")
    if proofs.get(node) != common.sha256_file(proof_path):
        errors.append("launch authority binds another node proof")
    started = authority.get("wall_clock_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("launch authority lacks numeric wall times")
    elif deadline <= started:
        errors.append("launch authority deadline is not after start")
    if stage == "MAIN" and deadline - started != 108000:
        errors.append(
            "main launch authority does not encode the registered 30h ceiling"
        )
    if not authority.get("v6_drain_proof_sha256"):
        errors.append("launch authority lacks a v6 drain-proof binding")
    if stage == "PILOT" and not authority.get("smoke_evidence_sha256"):
        errors.append("pilot authority lacks the smoke-evidence binding")
    if stage == "MAIN" and not authority.get("pilot_readout_sha256"):
        errors.append("main authority lacks the pilot-readout binding")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_groups(
    path: Path | None, manifest_path: Path, manifest: dict
) -> set[str]:
    if path is None:
        return set()
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v7_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != common.sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        errors.append("retry groups are empty, duplicated, or malformed")
        groups = []
    known = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    if set(groups) - known:
        errors.append("retry authority contains an unknown group")
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if authority.get("reason") not in allowed:
        errors.append("retry authority reason is not registered")
    if errors:
        raise SystemExit("; ".join(errors))
    return set(groups)


def require_file(failures: list[str], artifacts: dict, label: str, path: Path) -> bool:
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return False
    artifacts[label] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }
    return True


def validate_barrier_registry(
    attempt_root: Path, tape_rows: list[dict], expected: dict
) -> dict:
    """Verify trace hashes and replay each logical learner's barrier state."""

    registry_path = attempt_root / "report" / "barrier-version-trace.json"
    registry = json.loads(registry_path.read_text())
    learner_count = int(expected["learner_count"])
    if (
        registry.get("schema") != "yeto_barrier_version_trace_v1"
        or registry.get("learner_count") != learner_count
    ):
        raise ValueError("barrier registry schema/learner count mismatch")

    def verify_entry(entry: object, relative: str, label: str) -> Path:
        if not isinstance(entry, dict) or entry.get("path") != relative:
            raise ValueError(f"{label} registry path mismatch")
        path = attempt_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} registry artifact is missing or unsafe")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"{label} registry size is malformed")
        if size != path.stat().st_size:
            raise ValueError(f"{label} registry size mismatch")
        if entry.get("sha256") != common.sha256_file(path):
            raise ValueError(f"{label} registry hash mismatch")
        return path

    tape_path = verify_entry(
        registry.get("syncer_tape"), "work/m2/tape.jsonl", "syncer tape"
    )
    if read_jsonl(tape_path) != tape_rows:
        raise ValueError("barrier registry tape differs from validated event tape")

    entries = registry.get("learner_traces")
    if not isinstance(entries, list) or len(entries) != learner_count:
        raise ValueError("barrier registry learner trace count mismatch")
    trace_paths = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("barrier registry learner entry is malformed")
        learner_id = entry.get("learner_id")
        if (
            isinstance(learner_id, bool)
            or not isinstance(learner_id, int)
            or learner_id not in range(learner_count)
            or learner_id in trace_paths
        ):
            raise ValueError("barrier registry learner IDs are not exact")
        trace_paths[learner_id] = verify_entry(
            entry,
            f"work/m2/learner-{learner_id}/barrier-version-trace.jsonl",
            f"learner {learner_id} trace",
        )
    if set(trace_paths) != set(range(learner_count)):
        raise ValueError("barrier registry learner coverage is incomplete")

    expected_pushes = {}
    for tape_row in tape_rows:
        step = tape_row.get("step")
        fragment = tape_row.get("fragment")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or isinstance(fragment, bool)
            or not isinstance(fragment, int)
            or fragment not in range(FRAGMENT_COUNT)
        ):
            raise ValueError("event tape cannot define barrier push identities")
        for responder in tape_row.get("responders", []):
            learner_id = responder.get("id")
            key = (learner_id, step)
            if (
                isinstance(learner_id, bool)
                or not isinstance(learner_id, int)
                or learner_id not in range(learner_count)
                or key in expected_pushes
            ):
                raise ValueError("event tape barrier responder identity is malformed")
            expected_pushes[key] = {
                "fragment": fragment,
                "base_version": responder.get("base_version"),
                "c_steps": responder.get("c_steps"),
                "c_tokens": responder.get("c_tokens"),
            }

    full_fragments = set(range(FRAGMENT_COUNT))
    push_counts = {}
    broadcast_counts = {}
    inner_counts = {}
    for learner_id in range(learner_count):
        awaiting = {}
        reset_step = {fragment: 0 for fragment in full_fragments}
        initial_fragments = set()
        seen_pushes = set()
        seen_broadcasts = set()
        next_inner_step = 1
        previous_local_step = 0
        for sequence, row in enumerate(read_jsonl(trace_paths[learner_id]), 1):
            if (
                row.get("schema") != BARRIER_TRACE_SCHEMA
                or row.get("event_seq") != sequence
                or row.get("learner_id") != learner_id
            ):
                raise ValueError(f"learner {learner_id} trace identity/sequence mismatch")
            local_step = row.get("local_step")
            if (
                isinstance(local_step, bool)
                or not isinstance(local_step, int)
                or local_step < previous_local_step
            ):
                raise ValueError(f"learner {learner_id} trace local step is not monotone")
            previous_local_step = local_step
            event = row.get("event")
            declared_awaiting = row.get("awaiting_fragments")
            if event == "initial_broadcast_applied":
                fragment = row.get("fragment")
                if (
                    local_step != 0
                    or isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment not in full_fragments
                    or fragment in initial_fragments
                    or row.get("broadcast_version") != 0
                    or declared_awaiting != []
                    or awaiting
                    or next_inner_step != 1
                ):
                    raise ValueError(f"learner {learner_id} initial barrier prefix is invalid")
                initial_fragments.add(fragment)
                continue
            if event == "inner_step_started":
                if (
                    initial_fragments != full_fragments
                    or awaiting
                    or declared_awaiting != []
                    or local_step != next_inner_step
                ):
                    raise ValueError(f"learner {learner_id} stepped while barrier-blocked")
                next_inner_step += 1
                continue
            if event == "push_sent":
                fragment = row.get("fragment")
                pull_step = row.get("pull_step")
                if (
                    isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment not in full_fragments
                    or isinstance(pull_step, bool)
                    or not isinstance(pull_step, int)
                ):
                    raise ValueError(f"learner {learner_id} push identity is malformed")
                expected_push = expected_pushes.get((learner_id, pull_step))
                observed_push = {
                    "fragment": fragment,
                    "base_version": row.get("base_version"),
                    "c_steps": row.get("c_steps"),
                    "c_tokens": row.get("c_tokens"),
                }
                if (
                    initial_fragments != full_fragments
                    or expected_push != observed_push
                    or pull_step in seen_pushes
                    or fragment in awaiting
                    or local_step - reset_step[fragment] != expected["c_steps"]
                    or row.get("c_steps") != expected["c_steps"]
                    or row.get("c_tokens") != expected["c_tokens"]
                ):
                    raise ValueError(f"learner {learner_id} push does not match tape/window")
                awaiting[fragment] = (
                    pull_step,
                    row.get("base_version"),
                    local_step,
                )
                seen_pushes.add(pull_step)
            elif event == "broadcast_applied":
                fragment = row.get("fragment")
                pending = awaiting.get(fragment)
                if pending is None:
                    raise ValueError(f"learner {learner_id} broadcast has no pending push")
                pull_step, base_version, push_local_step = pending
                if (
                    row.get("pushed_base_version") != base_version
                    or row.get("broadcast_version") != pull_step
                    or isinstance(base_version, bool)
                    or not isinstance(base_version, int)
                    or pull_step <= base_version
                    or local_step != push_local_step
                    or pull_step in seen_broadcasts
                ):
                    raise ValueError(f"learner {learner_id} broadcast release is invalid")
                del awaiting[fragment]
                reset_step[fragment] = local_step
                seen_broadcasts.add(pull_step)
            else:
                raise ValueError(f"learner {learner_id} trace event is unknown: {event!r}")
            if declared_awaiting != sorted(awaiting):
                raise ValueError(f"learner {learner_id} declared barrier state is false")

        expected_steps = {
            step
            for expected_learner, step in expected_pushes
            if expected_learner == learner_id
        }
        if (
            awaiting
            or seen_pushes != expected_steps
            or seen_broadcasts != expected_steps
            or initial_fragments != full_fragments
            or next_inner_step - 1 != expected["learner_steps_per_learner"]
            or set(reset_step.values()) != {expected["learner_steps_per_learner"]}
        ):
            raise ValueError(f"learner {learner_id} barrier trace coverage is incomplete")
        push_counts[str(learner_id)] = len(seen_pushes)
        broadcast_counts[str(learner_id)] = len(seen_broadcasts)
        inner_counts[str(learner_id)] = next_inner_step - 1

    return {
        "validated": True,
        "registry_sha256": common.sha256_file(registry_path),
        "learner_count": learner_count,
        "commit_count": len(tape_rows),
        "push_counts": push_counts,
        "broadcast_counts": broadcast_counts,
        "inner_step_counts": inner_counts,
    }


def validate_layouts(
    work: Path, expected: dict, failures: list[str], artifacts: dict
) -> None:
    layout_hashes = []
    for learner in range(expected["learner_count"]):
        path = work / f"learner-{learner}" / "resolved-layout.json"
        if not require_file(failures, artifacts, f"learner_{learner}_layout", path):
            continue
        try:
            layout = json.loads(path.read_text())
            fragments = layout.get("fragments", [])
            tensors = [
                tensor
                for fragment in fragments
                for tensor in fragment.get("tensors", [])
            ]
            if len(fragments) != 4:
                failures.append(f"learner {learner} fragment count mismatch")
            if len(tensors) != expected["lora_trainable_tensors"]:
                failures.append(f"learner {learner} trainable tensor count mismatch")
            if (
                sum(int(tensor["numel"]) for tensor in tensors)
                != expected["lora_trainable_parameters"]
            ):
                failures.append(f"learner {learner} trainable parameter count mismatch")
            if any(tensor.get("dtype") != "float32" for tensor in tensors):
                failures.append(f"learner {learner} LoRA dtype mismatch")
            if layout.get("tuning") != "lora" or layout.get("wire_dtype") != "bf16":
                failures.append(f"learner {learner} layout tuning/wire mismatch")
            layout_hashes.append(layout.get("layout_hash"))
        except Exception as exc:
            failures.append(f"learner {learner} layout validation error: {exc}")
    if len(layout_hashes) == expected["learner_count"] and len(set(layout_hashes)) != 1:
        failures.append("logical learner layout hashes differ")


def validate_cell(cell: dict, attempt_root: Path, command: list[str]) -> dict:
    expected = cell["expected"]
    work = attempt_root / "work" / "m2"
    report = attempt_root / "report"
    failures = []
    artifacts = {}
    tape_rows = []
    barrier_validation = None

    telemetry_path = work / "rho-telemetry.jsonl"
    if require_file(failures, artifacts, "rho_telemetry", telemetry_path):
        try:
            rows = read_jsonl(telemetry_path)
            if len(rows) != expected["telemetry_rows"]:
                failures.append("rho telemetry row count mismatch")
            for index, row in enumerate(rows, 1):
                if (
                    row.get("schema") != TELEMETRY_SCHEMA
                    or row.get("outer_step") != index
                ):
                    failures.append(f"rho telemetry schema/step mismatch at {index}")
                    break
                cross = row.get("cross_worker", {})
                if cross.get("worker_count") != 2:
                    failures.append(f"rho worker count mismatch at {index}")
                    break
                norm = row.get("pseudo_gradient", {}).get("l2_norm")
                if not isinstance(norm, (int, float)) or not math.isfinite(norm):
                    failures.append(f"rho norm is nonfinite at {index}")
                    break
        except Exception as exc:
            failures.append(f"rho telemetry validation error: {exc}")

    tape_path = work / "tape.jsonl"
    if require_file(failures, artifacts, "event_tape", tape_path):
        try:
            rows = read_jsonl(tape_path)
            tape_rows = rows
            if len(rows) != expected["outer_steps"]:
                failures.append("event tape row count mismatch")
            for index, row in enumerate(rows, 1):
                if row.get("step") != index:
                    failures.append(f"event tape step mismatch at {index}")
                    break
                responders = row.get("responders")
                if not isinstance(responders, list) or sorted(
                    responder.get("id") for responder in responders
                ) != [0, 1]:
                    failures.append(f"strict quorum responder mismatch at {index}")
                    break
                if any(
                    responder.get("c_steps") != expected["c_steps"]
                    or responder.get("c_tokens") != expected["c_tokens"]
                    for responder in responders
                ):
                    failures.append(f"fixed-window counter mismatch at {index}")
                    break
        except Exception as exc:
            failures.append(f"event tape validation error: {exc}")

    for label, path in (
        ("checkpoint", work / "state.ckpt"),
        ("syncer_log", work / "syncer.log"),
        ("export_log", work / "export.log"),
        ("export_adapter", work / "export" / "adapter_model.safetensors"),
        ("export_config", work / "export" / "adapter_config.json"),
        ("results", report / "results.jsonl"),
        ("per_example_loss", report / "per-example-loss" / "m2.jsonl"),
        ("eval_rows", report / "eval-provenance" / "eval_rows.jsonl"),
        ("eval_provenance", report / "eval-provenance" / "eval_provenance.json"),
        ("acquisition_state", report / "acquisition-state.json"),
        ("barrier_registry", report / "barrier-version-trace.json"),
        ("recorded_command", attempt_root / "command.sh"),
        ("recorded_git_commit", attempt_root / "git_commit.txt"),
        ("recorded_git_diff", attempt_root / "git_diff.patch"),
    ):
        require_file(failures, artifacts, label, path)
    for learner in range(2):
        for suffix, filename in (
            ("log", f"learner-{learner}.log"),
            ("adapter", f"learner-{learner}/adapter_model.safetensors"),
            ("adapter_config", f"learner-{learner}/adapter_config.json"),
        ):
            require_file(
                failures,
                artifacts,
                f"learner_{learner}_{suffix}",
                work / filename,
            )
    validate_layouts(work, expected, failures, artifacts)
    registry_path = report / "barrier-version-trace.json"
    if tape_path.is_file() and registry_path.is_file():
        try:
            barrier_validation = validate_barrier_registry(
                attempt_root, tape_rows, expected
            )
        except Exception as exc:
            failures.append(f"barrier trace validation error: {exc}")

    command_path = attempt_root / "command.sh"
    if (
        command_path.is_file()
        and command_path.read_text() != shlex.join(command[1:]) + "\n"
    ):
        failures.append("recorded command differs from registered argv")
    commit_path = attempt_root / "git_commit.txt"
    if (
        commit_path.is_file()
        and commit_path.read_text().strip() != cell["source_git_commit"]
    ):
        failures.append("recorded Git commit mismatch")
    diff_path = attempt_root / "git_diff.patch"
    if diff_path.is_file() and diff_path.read_text():
        failures.append("recorded Git diff is nonempty")

    scientific_divergence = False
    results_path = report / "results.jsonl"
    if results_path.is_file():
        try:
            rows = read_jsonl(results_path)
            if len(rows) != 1 or rows[0].get("arm") != "m2":
                failures.append("results do not contain exactly one m2 row")
            else:
                result = rows[0]
                loss = result.get("eval_loss")
                if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                    scientific_divergence = True
                if result.get("eval_rows") != expected["eval_rows"]:
                    failures.append("evaluation row count mismatch")
                if result.get("learner_exit_codes") != [0, 0]:
                    failures.append("logical learner exit code mismatch")
                if result.get("syncer_exit_code") != 0:
                    failures.append("syncer exit code mismatch")
        except Exception as exc:
            failures.append(f"results validation error: {exc}")
    eval_rows_path = report / "eval-provenance" / "eval_rows.jsonl"
    if eval_rows_path.is_file():
        try:
            if len(read_jsonl(eval_rows_path)) != expected["eval_rows"]:
                failures.append("evaluation provenance row count mismatch")
        except Exception as exc:
            failures.append(f"evaluation provenance validation error: {exc}")

    for learner in range(2):
        log_path = work / f"learner-{learner}.log"
        if log_path.is_file():
            text = log_path.read_text(errors="replace")
            matches = re.findall(
                r"inner loop done at local_step=(\d+) global_step=(\d+)", text
            )
            if not matches:
                failures.append(f"learner {learner} lacks completion line")
            elif int(matches[-1][0]) != expected["learner_steps_per_learner"]:
                failures.append(f"learner {learner} local-step count mismatch")
            elif int(matches[-1][1]) != expected["outer_steps"]:
                failures.append(f"learner {learner} outer-step count mismatch")

    state_path = report / "acquisition-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            expected_status = "DIVERGED" if scientific_divergence else "COMPLETED"
            if (
                state.get("phase") != "endpoint_recorded"
                or state.get("scientific_status") != expected_status
            ):
                failures.append("acquisition state does not match endpoint status")
        except Exception as exc:
            failures.append(f"acquisition-state validation error: {exc}")

    if failures:
        status = "INVALID_WORK"
    elif scientific_divergence:
        status = "SCIENTIFIC_DIVERGENCE"
    else:
        status = "COMPLETED"
    return {
        "schema": "yeto_outer_mup_cell_evidence_v1",
        "cell_id": cell["cell_id"],
        "validated_at_utc": utc_now(),
        "status": status,
        "failures": failures,
        "expected": expected,
        "observed_artifacts": artifacts,
        "barrier_trace_validation": barrier_validation,
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": common.canonical_sha256(command),
    }


def kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def mark_not_run(cell: dict, attempt_root: Path, attempt: int, reason: str) -> None:
    attempt_root.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(
        attempt_root / "evidence.json",
        {
            "schema": "yeto_outer_mup_cell_evidence_v1",
            "cell_id": cell["cell_id"],
            "validated_at_utc": utc_now(),
            "status": "NOT_RUN_WALL_CEILING",
            "failures": [reason],
            "seed": cell["seed"],
            "training_seed": cell["training_seed"],
            "attempt_number": attempt,
        },
    )


def run_queue(
    manifest_path: Path,
    node_label: str,
    stage: str,
    proof_path: Path,
    authority_path: Path,
    attempt: int,
    retry_authority_path: Path | None,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("node") != node_label:
        raise SystemExit("node authority proof is missing or invalid")
    if proof.get("manifest_sha256") != common.sha256_file(manifest_path):
        raise SystemExit("node proof binds another manifest")
    authority = load_authority(authority_path, manifest_path, stage, proof_path)
    deadline = float(authority["hard_deadline_unix_s"])
    retry_groups = load_retry_groups(retry_authority_path, manifest_path, manifest)
    if attempt == 2 and not retry_groups:
        raise SystemExit("attempt 2 requires a retry authority")
    if attempt == 1 and retry_authority_path is not None:
        raise SystemExit("attempt 1 forbids a retry authority")
    queue = [
        cell
        for cell in manifest["cells"]
        if cell.get("assignment", {}).get("node") == node_label
        and (stage == "MAIN" or cell.get("stage") == stage)
        and (attempt == 1 or cell.get("retry_group_id") in retry_groups)
    ]
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    status_path = (
        common.RESULT_LINK
        / "_controller"
        / "slots-v7"
        / f"{stage.lower()}-{node_label}.json"
    )
    lock_path = (
        common.RESULT_LINK
        / "_controller"
        / "locks-v7"
        / f"{stage.lower()}-{node_label}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"v7 full-node controller already active: {lock_path}"
            ) from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = common.RESULT_LINK / cell["cell_id"] / f"attempt-{attempt}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = json.loads(evidence_path.read_text())
                if evidence.get("status") in ("COMPLETED", "SCIENTIFIC_DIVERGENCE"):
                    completed += 1
                    continue
                raise SystemExit(
                    f"refusing existing nonterminal attempt: {cell['cell_id']}"
                )
            if attempt_root.exists():
                raise SystemExit(
                    f"refusing to overwrite existing attempt: {attempt_root}"
                )
            if time.time() >= deadline:
                mark_not_run(
                    cell, attempt_root, attempt, "registered wall ceiling reached"
                )
                failures += 1
                continue
            inventory = gpu_inventory()
            occupied = [
                index
                for index, record in inventory.items()
                if record["memory_used_mib"] != 0 or record["utilization_percent"] != 0
            ]
            if occupied:
                raise SystemExit(
                    f"refusing v7 cell on occupied full node: GPUs {occupied}"
                )
            command = (
                cell["command"]
                if attempt == 1
                else cell["registered_retry_commands"][0]["command"]
            )
            expected_hash = (
                cell["command_hash"]
                if attempt == 1
                else cell["registered_retry_commands"][0]["command_hash"]
            )
            if common.canonical_sha256(command) != expected_hash:
                raise SystemExit(f"command hash changed: {cell['cell_id']}")
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt}-{uuid.uuid4()}"
            common.write_json_atomic(
                attempt_root / "attempt-start.json",
                {
                    "schema": "yeto_outer_mup_attempt_start_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "node": node_label,
                    "gpu_inventory": inventory,
                    "start_utc": utc_now(),
                    "command": command,
                    "command_hash": expected_hash,
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "git_commit": manifest["source"]["git_commit"],
                    "launch_authority_sha256": common.sha256_file(authority_path),
                },
            )
            common.write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v7_slot_status_v1",
                    "state": "RUNNING",
                    "stage": stage,
                    "node": node_label,
                    "cell_id": cell["cell_id"],
                    "queue_index": queue_index,
                    "queue_total": len(queue),
                    "updated_at_utc": utc_now(),
                },
            )
            log_path = attempt_root / "controller.stdout.log"
            started = time.monotonic()
            env = dict(os.environ)
            env.update(
                {
                    "HF_HOME": "/root/yeto-hf-cache",
                    "HF_DATASETS_CACHE": "/data/yeto-v7-datasets-cache",
                    "PYTHONPATH": "/root/yeto",
                }
            )
            with log_path.open("wb") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd="/root/yeto",
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                termination = None
                cell_deadline = min(
                    deadline, time.time() + float(cell["timeout_minutes"]) * 60.0
                )
                while process.poll() is None:
                    if time.time() >= cell_deadline:
                        termination = (
                            "wall_ceiling"
                            if time.time() >= deadline
                            else "cell_watchdog"
                        )
                        kill_process_group(process)
                        break
                    common.write_json_atomic(
                        status_path,
                        {
                            "schema": "yeto_outer_mup_v7_slot_status_v1",
                            "state": "RUNNING",
                            "stage": stage,
                            "node": node_label,
                            "cell_id": cell["cell_id"],
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "elapsed_seconds": time.monotonic() - started,
                            "gpu_sample": gpu_sample(),
                            "updated_at_utc": utc_now(),
                        },
                    )
                    time.sleep(30)
                return_code = process.wait()
            common.write_json_atomic(
                attempt_root / "attempt-end.json",
                {
                    "schema": "yeto_outer_mup_attempt_end_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "node": node_label,
                    "end_utc": utc_now(),
                    "wall_seconds": time.monotonic() - started,
                    "process_return_code": return_code,
                    "termination": termination,
                },
            )
            if termination == "wall_ceiling":
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "NOT_RUN_WALL_CEILING",
                    "failures": ["registered wall ceiling terminated the cell"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            elif return_code == 0 and termination is None:
                evidence = validate_cell(cell, attempt_root, command)
            else:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "INFRA_FAILURE",
                    "failures": [
                        f"process exited {return_code}; termination={termination}"
                    ],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            evidence["attempt_id"] = attempt_id
            evidence["attempt_number"] = attempt
            evidence["attempt_start_sha256"] = common.sha256_file(
                attempt_root / "attempt-start.json"
            )
            evidence["attempt_end_sha256"] = common.sha256_file(
                attempt_root / "attempt-end.json"
            )
            common.write_json_atomic(evidence_path, evidence)
            if evidence["status"] in ("COMPLETED", "SCIENTIFIC_DIVERGENCE"):
                completed += 1
            else:
                failures += 1
            common.write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v7_slot_status_v1",
                    "state": "BETWEEN_CELLS",
                    "stage": stage,
                    "node": node_label,
                    "last_cell_id": cell["cell_id"],
                    "last_status": evidence["status"],
                    "completed": completed,
                    "failures": failures,
                    "queue_index": queue_index + 1,
                    "queue_total": len(queue),
                    "updated_at_utc": utc_now(),
                },
            )
        common.write_json_atomic(
            status_path,
            {
                "schema": "yeto_outer_mup_v7_slot_status_v1",
                "state": "DRAINED",
                "stage": stage,
                "node": node_label,
                "completed": completed,
                "failures": failures,
                "queue_total": len(queue),
                "updated_at_utc": utc_now(),
            },
        )
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", choices=common.NODES, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        verify_preflight(args.manifest, args.node_label, args.proof)
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.stage is None or args.launch_authority is None:
        parser.error("--stage and --launch-authority are required for execution")
    manifest = json.loads(args.manifest.read_text())
    kind = manifest_kind(manifest)
    if args.stage == "MAIN" and kind != "MAIN":
        parser.error("MAIN execution requires the v7 main manifest")
    if args.stage in ("SMOKE", "PILOT") and kind != "PREP":
        parser.error("SMOKE/PILOT execution requires the v7 prep manifest")
    return run_queue(
        args.manifest,
        args.node_label,
        args.stage,
        args.proof,
        args.launch_authority,
        args.attempt,
        args.retry_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
