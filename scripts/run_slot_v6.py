#!/usr/bin/env python3
"""Preflight or drain one hash-bound v6 factorial GPU queue."""

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
except ModuleNotFoundError:
    from scripts.run_slot_v3 import (
        canonical_sha256,
        git,
        gpu_inventory,
        sha256_file,
        utc_now,
        validate_cell,
        write_json_atomic,
    )


CONTRACT_JSON_SHA256 = "00e04e8544443aa4bec1ad34b1b032920fb9e14cb01d6adba61e8332c8ede6b4"
CONTRACT_MD_SHA256 = "c8ac606bc13403b50f09f358b574f285e5dda1d2612dfb735d7031286957211b"
ANALYZER_SHA256 = "4862681e6d7dbbf55e95ae45e15e0a0170e5a35a17a73f0df01240951b576f1a"
RESULT_LINK = Path("/root/yeto-results-v6")
RESULT_TARGET = Path("/data/yeto-results-v6")
MIN_FREE_BYTES = 1_000_000_000_000
EXPECTED_CELLS = 900


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
        if not RESULT_LINK.is_symlink():
            errors.append("v6 results path is not a symlink")
        if resolved != RESULT_TARGET:
            errors.append(f"v6 results resolve to {resolved}, not {RESULT_TARGET}")
        if not proof["same_device_as_data"]:
            errors.append("v6 results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v6 results filesystem has only {usage.free} bytes free")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def verify_file_record(path: Path, record: dict, errors: list[str], label: str) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        errors.append(f"size mismatch for {label}: {path}")
        return
    if sha256_file(path) != record["sha256"]:
        errors.append(f"hash mismatch for {label}: {path}")


def verify_inputs(manifest: dict, repo: Path, errors: list[str]) -> dict:
    inputs = manifest.get("inputs", {})
    input_manifest_record = inputs.get("input_manifest", {})
    input_manifest_path = Path(input_manifest_record.get("path", ""))
    verify_file_record(input_manifest_path, input_manifest_record, errors, "v6 input manifest")
    combined = {}
    if input_manifest_path.is_file():
        try:
            combined = json.loads(input_manifest_path.read_text())
            if combined.get("schema") != "yeto_outer_mup_v6_inputs_v1":
                errors.append("v6 input manifest schema mismatch")
            if sorted(map(int, combined.get("seeds", {}))) != [601, 607, 613, 617, 619]:
                errors.append("v6 input manifest seed set mismatch")
            for seed, record in combined.get("seeds", {}).items():
                for label, item in record.get("files", {}).items():
                    verify_file_record(
                        Path(item["path"]), item, errors, f"seed {seed} {label}"
                    )
        except Exception as exc:
            errors.append(f"cannot validate v6 input manifest: {exc}")

    capacity = inputs.get("token_capacity", {})
    capacity_path = Path(capacity.get("report_path", ""))
    verify_file_record(
        capacity_path,
        {"sha256": capacity.get("report_sha256")},
        errors,
        "v6 token-capacity report",
    )
    if capacity_path.is_file():
        try:
            report = json.loads(capacity_path.read_text())
            if report.get("status") != "PASS":
                errors.append("v6 token-capacity report is not PASS")
            minimum = report.get("minimum_across_all_seeds_and_learners", {}).get(
                "blocks"
            )
            required = report.get("pipeline", {}).get("required_blocks_per_learner")
            if not isinstance(minimum, int) or minimum < 10_240:
                errors.append(f"v6 token capacity is only {minimum}")
            if required != 10_240:
                errors.append(f"v6 capacity report required-block count is {required}")
        except Exception as exc:
            errors.append(f"cannot validate v6 capacity report: {exc}")

    for key in ("input_builder",):
        record = inputs.get(key, {})
        verify_file_record(
            repo / record.get("path", ""), record, errors, f"registered {key}"
        )
    verifier = capacity.get("verifier", {})
    verify_file_record(
        repo / verifier.get("path", ""), verifier, errors, "registered input verifier"
    )

    model = inputs.get("model", {})
    model_root = Path(model.get("path", ""))
    for name, record in model.get("files", {}).items():
        verify_file_record(model_root / name, record, errors, f"model {name}")
    return {
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": (
            sha256_file(input_manifest_path) if input_manifest_path.is_file() else None
        ),
        "token_capacity_path": str(capacity_path),
        "token_capacity_sha256": (
            sha256_file(capacity_path) if capacity_path.is_file() else None
        ),
        "seed_count": len(combined.get("seeds", {})),
    }


def verify_preflight(manifest_path: Path, node_label: str, proof_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        errors.append("launch manifest schema mismatch")
    if manifest.get("manifest_variant") != "v6_full_T_by_S_factorial":
        errors.append("launch manifest variant mismatch")
    if manifest.get("stage") != "V6_FACTORIAL" or len(
        manifest.get("cells", [])
    ) != EXPECTED_CELLS:
        errors.append("manifest is not the complete 900-cell V6_FACTORIAL stage")
    if manifest.get("status") != "REGISTERED":
        errors.append("launch manifest status is not REGISTERED")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(
        manifest_path
    ):
        errors.append("launch manifest sidecar is absent or mismatched")

    repo = Path("/root/yeto")
    try:
        head = git(repo, "rev-parse", "HEAD")
        if head != manifest["source"]["git_commit"]:
            errors.append("node Git commit mismatch")
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        if manifest["registration"]["git_commit"] != head:
            errors.append("v6 must execute at its registration commit")
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
        "json_sha256": repo / "experiment-specs/outer-mup-v6-factorial-prereg.json",
        "md_sha256": repo / "experiment-specs/outer-mup-v6-factorial-prereg.md",
        "analyzer_sha256": repo / "scripts/analyze_v6.py",
    }
    for field, path in contract_paths.items():
        if not path.is_file() or sha256_file(path) != constants[field]:
            errors.append(f"registered file hash mismatch: {path}")

    input_proof = verify_inputs(manifest, repo, errors)
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
        if len(retries) != 1 or canonical_sha256(
            retries[0].get("command", [])
        ) != retries[0].get("command_hash"):
            errors.append(f"registered retry command mismatch: {cell_id}")
        if "--rho-telemetry" not in command:
            errors.append(f"rho telemetry missing: {cell_id}")
        correction = "--outer-bias-correction" in command
        if correction != (cell.get("arm") == "corrected"):
            errors.append(f"bias-correction binding mismatch: {cell_id}")
        expected_values = {
            "--gpu-offset": cell.get("assignment", {}).get("gpu"),
            "--outer-momentum": cell.get("mu"),
            "--fixed-window-microsteps": cell.get("h"),
            "--fixed-window-tokens": cell.get("h", 0) * 128,
            "--syncer-total-steps": cell.get("t", 0) * 4,
            "--learner-max-steps": cell.get("s"),
            "--training-seed": cell.get("training_seed"),
        }
        for flag, expected in expected_values.items():
            if command_value(command, flag) != str(expected):
                errors.append(f"{flag} binding mismatch: {cell_id}")
        if cell.get("s", 0) % cell.get("h", 1) or cell.get("s", 0) // cell.get(
            "h", 1
        ) != cell.get("t"):
            errors.append(f"T/S/H closure mismatch: {cell_id}")
        for flag in ("--work-dir", "--report-dir"):
            value = command_value(command, flag)
            if value is None or not value.startswith(str(RESULT_LINK) + "/"):
                errors.append(f"{flag} escapes roomy v6 results: {cell_id}")

    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in node inventory")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    proof = {
        "schema": "yeto_outer_mup_v6_node_authority_v1",
        "node": node_label,
        "checked_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest.get("source", {}).get("git_commit"),
        "gpu_inventory": inventory,
        "storage": storage,
        "inputs": input_proof,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_gate_proof(path: Path) -> dict:
    proof = json.loads(path.read_text())
    errors = []
    if proof.get("schema") != "yeto_outer_mup_v6_gate_proof_v1":
        errors.append("gate-proof schema mismatch")
    if proof.get("status") != "PASS":
        errors.append("gate proof is not PASS")
    if proof.get("v4", {}).get("unique_completed_cells") != 48:
        errors.append("gate proof does not contain v4 48/48 completion")
    for node in ("h200-n1", "h200-n2"):
        if proof.get("v4", {}).get("run_slot_processes", {}).get(node) != []:
            errors.append(f"gate proof has active v4 processes on {node}")
    if not str(proof.get("v5", {}).get("verdict_line", "")).startswith(
        "G5 VERDICT"
    ):
        errors.append("gate proof lacks a G5 VERDICT line")
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_launch_authority(
    path: Path, manifest_path: Path, gate_proof_path: Path
) -> dict:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v6_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("launch authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    if authority.get("gate_proof_sha256") != sha256_file(gate_proof_path):
        errors.append("launch authority binds another gate proof")
    load_gate_proof(gate_proof_path)
    started = authority.get("wall_clock_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(
        deadline, (int, float)
    ):
        errors.append("launch authority lacks numeric wall times")
    elif deadline - started != 108_000:
        errors.append("launch authority does not encode the registered 30h ceiling")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_authority(path: Path, manifest_path: Path, manifest: dict) -> set[str]:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v6_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    if authority.get("reason") not in set(
        manifest.get("retry_contract", {}).get("allowed_reasons", [])
    ):
        errors.append("retry authority reason is not registered")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        errors.append("retry authority groups are empty, duplicated, or malformed")
        groups = []
    known = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    if set(groups) - known:
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


def mark_not_run(cell: dict, attempt_root: Path, attempt_number: int) -> None:
    attempt_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        attempt_root / "evidence.json",
        {
            "schema": "yeto_outer_mup_cell_evidence_v1",
            "cell_id": cell["cell_id"],
            "validated_at_utc": utc_now(),
            "status": "NOT_RUN_WALL_CEILING",
            "failures": ["registered 30h wall ceiling reached before launch"],
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
    gate_proof_path: Path,
    attempt_number: int,
    retry_authority_path: Path | None,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("node") != node_label:
        raise SystemExit("node authority proof is missing or invalid")
    if proof.get("manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("node authority proof binds another manifest")
    authority = load_launch_authority(
        launch_authority_path, manifest_path, gate_proof_path
    )
    deadline = float(authority["hard_deadline_unix_s"])
    queue = [
        cell
        for cell in manifest["cells"]
        if cell["assignment"] == {"node": node_label, "gpu": gpu}
    ]
    if attempt_number == 2:
        if retry_authority_path is None:
            raise SystemExit("attempt 2 requires a registered retry authority")
        retry_groups = load_retry_authority(
            retry_authority_path, manifest_path, manifest
        )
        queue = [cell for cell in queue if cell["retry_group_id"] in retry_groups]
    elif retry_authority_path is not None:
        raise SystemExit("retry authority is forbidden for attempt 1")
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    status_path = (
        RESULT_LINK / "_controller" / "slots-v6" / f"{node_label}-gpu{gpu}.json"
    )
    lock_path = (
        RESULT_LINK / "_controller" / "locks-v6" / f"{node_label}-gpu{gpu}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    wall_reached = False
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
                raise SystemExit(
                    f"refusing existing noncompleted attempt: {cell['cell_id']}"
                )
            if attempt_root.exists():
                raise SystemExit(f"refusing to overwrite existing attempt: {attempt_root}")
            if time.time() >= deadline:
                mark_not_run(cell, attempt_root, attempt_number)
                failures += 1
                wall_reached = True
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
                "gate_proof_sha256": sha256_file(gate_proof_path),
            }
            write_json_atomic(attempt_root / "attempt-start.json", start)
            write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v6_slot_status_v1",
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
                        wall_reached = True
                        kill_process_group(process)
                        break
                    write_json_atomic(
                        status_path,
                        {
                            "schema": "yeto_outer_mup_v6_slot_status_v1",
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
                return_code = process.returncode
            try:
                evidence = validate_cell(cell, attempt_root, command)
            except Exception as exc:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "INVALID_WORK",
                    "failures": [f"validator exception: {exc}"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            if wall_stopped:
                evidence["status"] = "NOT_RUN_WALL_CEILING"
                evidence.setdefault("failures", []).append(
                    "registered 30h wall ceiling reached during cell"
                )
            elif return_code != 0 and evidence.get("status") == "COMPLETED":
                evidence["status"] = "INVALID_WORK"
                evidence.setdefault("failures", []).append(
                    f"controller observed process exit {return_code}"
                )
            evidence.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "node": node_label,
                    "gpu_index": gpu,
                    "return_code": return_code,
                    "command_hash": expected_hash,
                    "start_utc": start["start_utc"],
                    "end_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                }
            )
            write_json_atomic(evidence_path, evidence)
            write_json_atomic(
                attempt_root / "attempt-end.json",
                {
                    "schema": "yeto_outer_mup_attempt_end_v1",
                    "attempt_id": attempt_id,
                    "cell_id": cell["cell_id"],
                    "attempt_number": attempt_number,
                    "return_code": return_code,
                    "status": evidence["status"],
                    "end_utc": evidence["end_utc"],
                    "evidence_sha256": sha256_file(evidence_path),
                },
            )
            if evidence["status"] == "COMPLETED":
                completed += 1
            else:
                failures += 1
        write_json_atomic(
            status_path,
            {
                "schema": "yeto_outer_mup_v6_slot_status_v1",
                "state": "WALL_CEILING" if wall_reached else "DRAINED",
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
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--gate-proof", type=Path)
    parser.add_argument("--attempt-number", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    args = parser.parse_args()
    if args.preflight == args.run:
        raise SystemExit("choose exactly one of --preflight or --run")
    if args.preflight:
        verify_preflight(
            args.manifest.resolve(), args.node_label, args.proof.resolve()
        )
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.gpu is None or args.gpu not in range(8):
        raise SystemExit("--run requires --gpu in 0..7")
    if args.launch_authority is None or args.gate_proof is None:
        raise SystemExit("--run requires --launch-authority and --gate-proof")
    return run_queue(
        args.manifest.resolve(),
        args.node_label,
        args.gpu,
        args.proof.resolve(),
        args.launch_authority.resolve(),
        args.gate_proof.resolve(),
        args.attempt_number,
        args.retry_authority.resolve() if args.retry_authority else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
