#!/usr/bin/env python3
"""Preflight or drain one hash-bound v4 scale GPU queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

try:
    from run_slot_v3 import (
        canonical_sha256,
        git,
        gpu_inventory,
        sha256_file,
        utc_now,
        validate_cell,
        write_json_atomic,
    )
except ModuleNotFoundError:  # package import in tests; direct script uses branch above
    from scripts.run_slot_v3 import (
        canonical_sha256,
        git,
        gpu_inventory,
        sha256_file,
        utc_now,
        validate_cell,
        write_json_atomic,
    )


CONTRACT_JSON_SHA256 = "3ab2c66b849078deb3d7ca786faf9a053ce3d2405b1344e9bfd4cc6a47080bc6"
CONTRACT_MD_SHA256 = "682d20c79a4718c9e9b61d557fee0df6b1b163a05b8aca4164d388256bc097cb"
ANALYZER_SHA256 = "654eb63d5a830b275dfb64d2c9e178ff83c2355a7ec88e02c8c46e2cf1300471"
RESULT_LINK = Path("/root/yeto-results-v4")
RESULT_TARGET = Path("/data/yeto-results-v4")
MIN_FREE_BYTES = 1_000_000_000_000


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def storage_proof() -> tuple[dict, list[str]]:
    errors = []
    proof = {
        "result_link": str(RESULT_LINK),
        "result_target_expected": str(RESULT_TARGET),
        "is_symlink": RESULT_LINK.is_symlink(),
    }
    try:
        resolved = RESULT_LINK.resolve(strict=True)
        proof["result_target_resolved"] = str(resolved)
        proof["same_device_as_data"] = resolved.stat().st_dev == Path("/data").stat().st_dev
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
        if not RESULT_LINK.is_symlink():
            errors.append("v4 results path is not a symlink")
        if resolved != RESULT_TARGET:
            errors.append(f"v4 results resolve to {resolved}, not {RESULT_TARGET}")
        if not proof["same_device_as_data"]:
            errors.append("v4 results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v4 results filesystem has only {usage.free} bytes free")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def verify_preflight(manifest_path: Path, node_label: str, proof_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v4_scale_launch_manifest_v1":
        errors.append("launch manifest schema mismatch")
    if manifest.get("manifest_variant") != "v4_scale_raw_tscan":
        errors.append("launch manifest variant mismatch")
    if manifest.get("stage") != "V4_SCALE" or len(manifest.get("cells", [])) != 48:
        errors.append("launch manifest is not the complete 48-cell V4_SCALE stage")
    if manifest.get("status") != "AUTHORIZED":
        errors.append("launch manifest is not authorized")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(manifest_path):
        errors.append("launch manifest sidecar is absent or mismatched")

    repo = Path("/root/yeto")
    try:
        head = git(repo, "rev-parse", "HEAD")
        if head != manifest["source"]["git_commit"]:
            errors.append("node Git commit mismatch")
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        if manifest["registration"]["git_commit"] != head:
            errors.append("v4 must execute at its registration commit")
    except Exception as exc:
        errors.append(f"Git preflight failed: {exc}")

    constants = {
        "json_sha256": CONTRACT_JSON_SHA256,
        "md_sha256": CONTRACT_MD_SHA256,
        "analyzer_sha256": ANALYZER_SHA256,
    }
    contract = manifest.get("contract", {})
    for field, expected in constants.items():
        if contract.get(field) != expected:
            errors.append(f"manifest contract {field} differs from controller constant")
    contract_paths = {
        "json_sha256": repo / "experiment-specs/outer-mup-v4-scale-prereg.json",
        "md_sha256": repo / "experiment-specs/outer-mup-v4-scale-prereg.md",
        "analyzer_sha256": repo / "scripts/analyze_v4.py",
    }
    for field, path in contract_paths.items():
        if not path.is_file() or sha256_file(path) != constants[field]:
            errors.append(f"registered file hash mismatch: {path}")

    for record in manifest.get("inputs", {}).get("files", []):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            errors.append(f"input size mismatch: {path}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"input hash mismatch: {path}")

    cells = manifest.get("cells", [])
    seen = set()
    for cell in cells:
        cell_id = cell.get("cell_id", "<missing>")
        if cell_id in seen:
            errors.append(f"duplicate cell id: {cell_id}")
        seen.add(cell_id)
        command = cell.get("command", [])
        if canonical_sha256(command) != cell.get("command_hash"):
            errors.append(f"initial command hash mismatch: {cell_id}")
        retries = cell.get("registered_retry_commands", [])
        if len(retries) != 1 or canonical_sha256(retries[0].get("command", [])) != retries[0].get("command_hash"):
            errors.append(f"registered retry command mismatch: {cell_id}")
        if "--rho-telemetry" not in command:
            errors.append(f"rho telemetry missing: {cell_id}")
        if "--outer-bias-correction" in command:
            errors.append(f"v4 raw arm unexpectedly enables bias correction: {cell_id}")
        if command_value(command, "--gpu-offset") != str(cell.get("assignment", {}).get("gpu")):
            errors.append(f"GPU binding mismatch: {cell_id}")
        if command_value(command, "--outer-momentum") != str(cell.get("mu")):
            errors.append(f"momentum binding mismatch: {cell_id}")
        for flag in ("--work-dir", "--report-dir"):
            value = command_value(command, flag)
            if value is None or not value.startswith(str(RESULT_LINK) + "/"):
                errors.append(f"{flag} escapes roomy v4 results: {cell_id}")

    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in node inventory")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    proof = {
        "schema": "yeto_outer_mup_v4_node_authority_v1",
        "node": node_label,
        "checked_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest.get("source", {}).get("git_commit"),
        "gpu_inventory": inventory,
        "storage": storage,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_launch_authority(path: Path, manifest_path: Path) -> dict:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v4_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("launch authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    started = authority.get("wall_clock_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("launch authority lacks numeric wall times")
    elif deadline - started != 43_200:
        errors.append("launch authority does not encode the registered 12h ceiling")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_authority(path: Path, manifest_path: Path, manifest: dict) -> set[str]:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v4_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    allowed_reasons = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if authority.get("reason") not in allowed_reasons:
        errors.append("retry authority reason is not registered")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        errors.append("retry authority groups are empty, duplicated, or malformed")
        groups = []
    known_groups = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    if set(groups) - known_groups:
        errors.append("retry authority contains an unknown group")
    if errors:
        raise SystemExit("; ".join(errors))
    return set(groups)


def kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def gpu_sample(gpu: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {"raw": result.stdout.strip(), "return_code": result.returncode}


def mark_not_run(cell: dict, attempt_root: Path, attempt_number: int, reason: str) -> None:
    attempt_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        attempt_root / "evidence.json",
        {
            "schema": "yeto_outer_mup_cell_evidence_v1",
            "cell_id": cell["cell_id"],
            "validated_at_utc": utc_now(),
            "status": "NOT_RUN_WALL_CEILING",
            "failures": [reason],
            "seed": cell["seed"],
            "training_seed": cell["training_seed"],
            "attempt_number": attempt_number,
        },
    )


def run_queue(
    manifest_path: Path,
    node_label: str,
    gpu: int,
    proof_path: Path,
    launch_authority_path: Path,
    attempt_number: int,
    retry_authority_path: Path | None,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("node") != node_label:
        raise SystemExit("node authority proof is missing or invalid")
    if proof.get("manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("node authority proof binds another manifest")
    authority = load_launch_authority(launch_authority_path, manifest_path)
    deadline = float(authority["hard_deadline_unix_s"])
    queue = [
        cell
        for cell in manifest["cells"]
        if cell["assignment"] == {"node": node_label, "gpu": gpu}
    ]
    if attempt_number == 2:
        if retry_authority_path is None:
            raise SystemExit("attempt 2 requires a registered retry authority")
        retry_groups = load_retry_authority(retry_authority_path, manifest_path, manifest)
        queue = [cell for cell in queue if cell["retry_group_id"] in retry_groups]
    elif retry_authority_path is not None:
        raise SystemExit("retry authority is forbidden for attempt 1")
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    status_path = RESULT_LINK / "_controller" / "slots-v4" / f"{node_label}-gpu{gpu}.json"
    lock_path = RESULT_LINK / "_controller" / "locks-v4" / f"{node_label}-gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU slot controller already active: {lock_path}") from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = RESULT_LINK / cell["cell_id"] / f"attempt-{attempt_number}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = json.loads(evidence_path.read_text())
                if evidence.get("status") == "COMPLETED":
                    completed += 1
                    continue
                raise SystemExit(f"refusing existing noncompleted attempt: {cell['cell_id']}")
            if attempt_root.exists():
                raise SystemExit(f"refusing to overwrite existing attempt: {attempt_root}")
            if time.time() >= deadline:
                mark_not_run(cell, attempt_root, attempt_number, "registered 12h wall ceiling reached before launch")
                failures += 1
                continue
            command = (
                cell["command"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command"]
            )
            expected_hash = (
                cell["command_hash"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command_hash"]
            )
            if canonical_sha256(command) != expected_hash:
                raise SystemExit(f"command hash changed: {cell['cell_id']}")
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt_number}-{uuid.uuid4()}"
            start = {
                "schema": "yeto_outer_mup_attempt_start_v1",
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "cell_id": cell["cell_id"],
                "node": node_label,
                "gpu_index": gpu,
                "gpu_uuid": inventory[gpu]["uuid"],
                "gpu_name": inventory[gpu]["name"],
                "start_utc": utc_now(),
                "command": command,
                "command_hash": expected_hash,
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "git_commit": manifest["source"]["git_commit"],
                "launch_authority_sha256": sha256_file(launch_authority_path),
            }
            write_json_atomic(attempt_root / "attempt-start.json", start)
            write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v4_slot_status_v1",
                    "state": "RUNNING",
                    "node": node_label,
                    "gpu": gpu,
                    "cell_id": cell["cell_id"],
                    "queue_index": queue_index,
                    "queue_total": len(queue),
                    "updated_at_utc": utc_now(),
                },
            )
            log_path = attempt_root / "controller.stdout.log"
            started_monotonic = time.monotonic()
            with log_path.open("wb") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd="/root/yeto",
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ),
                    start_new_session=True,
                )
                wall_stopped = False
                while process.poll() is None:
                    if time.time() >= deadline:
                        wall_stopped = True
                        kill_process_group(process)
                        break
                    write_json_atomic(
                        status_path,
                        {
                            "schema": "yeto_outer_mup_v4_slot_status_v1",
                            "state": "RUNNING",
                            "node": node_label,
                            "gpu": gpu,
                            "cell_id": cell["cell_id"],
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "elapsed_seconds": time.monotonic() - started_monotonic,
                            "gpu_sample": gpu_sample(gpu),
                            "updated_at_utc": utc_now(),
                        },
                    )
                    time.sleep(30)
                return_code = process.wait()
            end = {
                "schema": "yeto_outer_mup_attempt_end_v1",
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "cell_id": cell["cell_id"],
                "node": node_label,
                "gpu_index": gpu,
                "end_utc": utc_now(),
                "wall_seconds": time.monotonic() - started_monotonic,
                "process_return_code": return_code,
                "wall_ceiling_terminated": wall_stopped,
            }
            write_json_atomic(attempt_root / "attempt-end.json", end)
            if wall_stopped:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "NOT_RUN_WALL_CEILING",
                    "failures": ["registered 12h wall ceiling terminated the cell"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            elif return_code == 0:
                evidence = validate_cell(cell, attempt_root, command)
            else:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "INFRA_FAILURE",
                    "failures": [f"scientific process exited {return_code} before valid completion"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            evidence["attempt_id"] = attempt_id
            evidence["attempt_number"] = attempt_number
            evidence["attempt_start_sha256"] = sha256_file(attempt_root / "attempt-start.json")
            evidence["attempt_end_sha256"] = sha256_file(attempt_root / "attempt-end.json")
            write_json_atomic(evidence_path, evidence)
            if evidence["status"] == "COMPLETED":
                completed += 1
            else:
                failures += 1
            write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v4_slot_status_v1",
                    "state": "BETWEEN_CELLS",
                    "node": node_label,
                    "gpu": gpu,
                    "last_cell_id": cell["cell_id"],
                    "last_status": evidence["status"],
                    "completed": completed,
                    "failures": failures,
                    "queue_index": queue_index + 1,
                    "queue_total": len(queue),
                    "updated_at_utc": utc_now(),
                },
            )
        write_json_atomic(
            status_path,
            {
                "schema": "yeto_outer_mup_v4_slot_status_v1",
                "state": "DRAINED",
                "node": node_label,
                "gpu": gpu,
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
    parser.add_argument("--node-label", choices=("h200-n1", "h200-n2"), required=True)
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        verify_preflight(args.manifest, args.node_label, args.proof)
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.gpu is None or args.launch_authority is None:
        parser.error("--gpu and --launch-authority are required for queue execution")
    return run_queue(
        args.manifest,
        args.node_label,
        args.gpu,
        args.proof,
        args.launch_authority,
        args.attempt,
        args.retry_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
