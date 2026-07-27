#!/usr/bin/env python3
"""Preflight or drain one hash-bound v9 one-GPU/four-GPU queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402
from run_slot_v3 import (  # noqa: E402
    canonical_sha256,
    git,
    gpu_inventory,
    validate_cell,
)


CONTRACT_JSON_SHA256 = "46686fd91e301555bf9f337bf58398d4ad81bdbbd5c249ac5bc081a56670d22c"
CONTRACT_MD_SHA256 = "07032889dc02c4d76e0c894e803a63fe232c1c3ddbc1e9ff69fc073487c29fe8"
ANALYZER_SHA256 = "efc18c8105a85072e1de3b58fa353488da15ff32a3406878265b8cb10035aba8"
PREDICTIONS_SHA256 = "97e02dcad63782978ac51b320621e5a681236518cb0d5db19454b8981549ca9c"
RESULT_LINK = Path("/root/yeto-results-v9")
RESULT_TARGET = Path("/data/yeto-results-v9")
MIN_FREE_BYTES = 1_000_000_000_000
STAGE_COUNTS = {"stage_1p7b": 16, "stage_7b": 12}


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
        usage = shutil.disk_usage(resolved)
        proof.update(
            {
                "result_target_resolved": str(resolved),
                "same_device_as_data": resolved.stat().st_dev
                == Path("/data").stat().st_dev,
                "filesystem_total_bytes": usage.total,
                "filesystem_used_bytes": usage.used,
                "filesystem_free_bytes": usage.free,
            }
        )
        mount = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "--target", str(resolved)],
            capture_output=True,
            text=True,
        )
        proof["findmnt"] = mount.stdout.strip()
        if not RESULT_LINK.is_symlink():
            errors.append("v9 result path is not a symlink")
        if resolved != RESULT_TARGET:
            errors.append(f"v9 results resolve to {resolved}, not {RESULT_TARGET}")
        if not proof["same_device_as_data"]:
            errors.append("v9 results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v9 filesystem has only {usage.free} free bytes")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def gpu_compute_processes() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return [{"query_error": result.stderr.strip() or result.stdout.strip()}]
    records = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            records.append({"raw": line})
        else:
            records.append(
                {
                    "pid": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mib": float(parts[3]),
                }
            )
    return records


def verify_file_record(record: dict, errors: list[str], label: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
    elif path.stat().st_size != int(record["bytes"]):
        errors.append(f"size mismatch for {label}: {path}")
    elif sha256_file(path) != record["sha256"]:
        errors.append(f"hash mismatch for {label}: {path}")


def verify_preflight(
    manifest_path: Path, node_label: str, stage: str, proof_path: Path
) -> dict:
    manifest = read_json(manifest_path)
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v9_launch_manifest_v1":
        errors.append("launch manifest schema mismatch")
    if (
        manifest.get("stage") != "V9_SEALED_SCALE"
        or len(manifest.get("cells", [])) != 28
    ):
        errors.append("manifest is not the complete 28-cell v9 stage")
    if manifest.get("status") != "REGISTERED":
        errors.append("manifest is not REGISTERED")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(
        manifest_path
    ):
        errors.append("manifest SHA-256 sidecar is missing or mismatched")
    stage_cells = [
        cell for cell in manifest.get("cells", []) if cell.get("stage") == stage
    ]
    if len(stage_cells) != STAGE_COUNTS[stage]:
        errors.append(f"manifest stage {stage} has {len(stage_cells)} cells")

    repo = Path("/root/yeto")
    try:
        head = git(repo, "rev-parse", "HEAD")
        if head != manifest.get("source", {}).get("git_commit"):
            errors.append("node Git commit differs from manifest")
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        if manifest.get("registration", {}).get("git_commit") != head:
            errors.append("v9 does not execute at its registration commit")
    except Exception as exc:
        errors.append(f"Git preflight failed: {exc}")

    constants = {
        "contract_json": (
            repo / "experiment-specs/outer-mup-v9-sealed-scale-prereg.json",
            CONTRACT_JSON_SHA256,
        ),
        "contract_md": (
            repo / "experiment-specs/outer-mup-v9-sealed-scale-prereg.md",
            CONTRACT_MD_SHA256,
        ),
        "analyzer": (repo / "scripts/analyze_v9.py", ANALYZER_SHA256),
        "predictions": (
            repo / "experiment-specs/outer-mup-v9-sealed-predictions.json",
            PREDICTIONS_SHA256,
        ),
    }
    for label, (path, expected) in constants.items():
        if expected.startswith("__V9_"):
            errors.append(f"controller has unresolved frozen constant {label}")
        elif not path.is_file() or sha256_file(path) != expected:
            errors.append(f"registered file hash mismatch: {path}")
    if manifest.get("contract", {}).get("sha256") != CONTRACT_JSON_SHA256:
        errors.append("manifest binds another v9 contract")
    if manifest.get("predictions", {}).get("sha256") != PREDICTIONS_SHA256:
        errors.append("manifest binds another prediction seal")

    inputs = manifest.get("inputs", {})
    for label in ("training_jsonl", "development_jsonl", "scale_manifest"):
        try:
            verify_file_record(inputs[label], errors, label)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed input record {label}: {exc}")
    model_key = "smollm2_1p7b" if stage == "stage_1p7b" else "qwen2p5_7b"
    try:
        model = manifest["models"][model_key]
        for name, record in model["files"].items():
            path = Path(model["path"]) / name
            bound = {**record, "path": str(path)}
            verify_file_record(bound, errors, f"{model_key}/{name}")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed model record {model_key}: {exc}")

    seen = set()
    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id", "<missing>")
        if cell_id in seen:
            errors.append(f"duplicate cell id {cell_id}")
        seen.add(cell_id)
        command = cell.get("command", [])
        if canonical_sha256(command) != cell.get("command_hash"):
            errors.append(f"command hash mismatch: {cell_id}")
        retries = cell.get("registered_retry_commands", [])
        if len(retries) != 1 or canonical_sha256(
            retries[0].get("command", [])
        ) != retries[0].get("command_hash"):
            errors.append(f"retry command hash mismatch: {cell_id}")
        if "--rho-telemetry" not in command or "--barrier-sync" not in command:
            errors.append(f"scientific evidence flags missing: {cell_id}")
        expected_slots = len(cell.get("assignment", {}).get("gpus", []))
        if command_value(command, "--gpu-slots") != str(expected_slots):
            errors.append(f"GPU width mismatch: {cell_id}")
        if command_value(command, "--gpu-offset") != str(
            min(cell.get("assignment", {}).get("gpus", [-1]))
        ):
            errors.append(f"GPU offset mismatch: {cell_id}")
        correction = "--outer-bias-correction" in command
        if correction != (cell.get("arm") == "corrected"):
            errors.append(f"bias-correction mismatch: {cell_id}")
        for flag in ("--work-dir", "--report-dir"):
            value = command_value(command, flag)
            if value is None or not value.startswith(str(RESULT_LINK) + "/"):
                errors.append(f"{flag} escapes v9 result root: {cell_id}")

    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in inventory")
    compute = gpu_compute_processes()
    if compute:
        errors.append(f"node has {len(compute)} active/unknown compute processes")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    proof = {
        "schema": "yeto_outer_mup_v9_node_preflight_v1",
        "node": node_label,
        "stage": stage,
        "checked_at_utc": utc_now(),
        "checked_at_unix_s": time.time(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest.get("source", {}).get("git_commit"),
        "gpu_inventory": inventory,
        "compute_processes": compute,
        "storage": storage,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_launch_authority(
    path: Path, manifest_path: Path, stage: str, node_label: str, proof_path: Path
) -> dict:
    authority = read_json(path)
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v9_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED" or authority.get("stage") != stage:
        errors.append("launch authority status/stage mismatch")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    node_record = authority.get("node_preflights", {}).get(node_label, {})
    if node_record.get("sha256") != sha256_file(proof_path):
        errors.append("authority does not bind this node preflight")
    started = authority.get("program_wall_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("authority lacks numeric program wall times")
    elif deadline <= started or time.time() >= deadline:
        errors.append("authority deadline is invalid or already elapsed")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_authority(
    path: Path, manifest_path: Path, manifest: dict, stage: str
) -> set[str]:
    authority = read_json(path)
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v9_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED" or authority.get("stage") != stage:
        errors.append("retry authority status/stage mismatch")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if authority.get("reason") not in allowed:
        errors.append("retry reason is not registered")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        errors.append("retry groups are missing, empty, or duplicated")
        groups = []
    known = {
        cell["retry_group_id"]
        for cell in manifest.get("cells", [])
        if cell.get("stage") == stage
    }
    if set(groups) - known:
        errors.append("retry authority includes an unknown stage group")
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


def gpu_sample(gpus: list[int]) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={','.join(map(str, gpus))}",
            "--query-gpu=index,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {"raw": result.stdout.strip(), "return_code": result.returncode}


def mark_not_run(
    cell: dict, attempt_root: Path, attempt_number: int, reason: str
) -> None:
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
    *,
    manifest_path: Path,
    node_label: str,
    stage: str,
    slot_id: str,
    proof_path: Path,
    launch_authority_path: Path,
    attempt_number: int,
    retry_authority_path: Path | None,
) -> int:
    manifest = read_json(manifest_path)
    repo = Path("/root/yeto")
    registered_git_commit = manifest["source"]["git_commit"]
    execution_git_commit = git(repo, "rev-parse", "HEAD")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SystemExit("v9 execution worktree is dirty")
    if attempt_number == 1 and execution_git_commit != registered_git_commit:
        raise SystemExit("attempt 1 must execute at the registered Git commit")
    if attempt_number == 2 and subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            registered_git_commit,
            execution_git_commit,
        ],
        capture_output=True,
    ).returncode:
        raise SystemExit(
            "attempt 2 execution commit is not a descendant of the registration"
        )
    proof = read_json(proof_path)
    if (
        proof.get("status") != "PASS"
        or proof.get("node") != node_label
        or proof.get("stage") != stage
        or proof.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise SystemExit("node preflight is missing, stale, or invalid")
    authority = load_launch_authority(
        launch_authority_path, manifest_path, stage, node_label, proof_path
    )
    deadline = float(authority["hard_deadline_unix_s"])
    queue = [
        cell
        for cell in manifest["cells"]
        if cell["stage"] == stage
        and cell["slot_id"] == slot_id
        and cell["assignment"]["node"] == node_label
    ]
    if not queue:
        raise SystemExit(f"manifest has no {stage} queue {slot_id} on {node_label}")
    if attempt_number == 2:
        if retry_authority_path is None:
            raise SystemExit("attempt 2 requires retry authority")
        retry_groups = load_retry_authority(
            retry_authority_path, manifest_path, manifest, stage
        )
        queue = [cell for cell in queue if cell["retry_group_id"] in retry_groups]
    elif retry_authority_path is not None:
        raise SystemExit("retry authority is forbidden for attempt 1")
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    status_path = RESULT_LINK / "_controller" / "slots-v9" / stage / f"{slot_id}.json"
    lock_path = RESULT_LINK / "_controller" / "locks-v9" / stage / f"{slot_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"v9 slot controller already active: {lock_path}") from exc
        for queue_index, cell in enumerate(queue):
            gpus = [int(value) for value in cell["assignment"]["gpus"]]
            attempt_root = RESULT_LINK / cell["cell_id"] / f"attempt-{attempt_number}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = read_json(evidence_path)
                if evidence.get("status") == "COMPLETED":
                    completed += 1
                    continue
                raise SystemExit(
                    f"refusing existing noncompleted attempt: {cell['cell_id']}"
                )
            if attempt_root.exists():
                raise SystemExit(
                    f"refusing to overwrite existing attempt: {attempt_root}"
                )
            if time.time() >= deadline:
                mark_not_run(
                    cell,
                    attempt_root,
                    attempt_number,
                    "registered v9 program wall ceiling reached before launch",
                )
                failures += 1
                continue
            command_record = (
                {"command": cell["command"], "command_hash": cell["command_hash"]}
                if attempt_number == 1
                else cell["registered_retry_commands"][0]
            )
            command = command_record["command"]
            expected_hash = command_record["command_hash"]
            if canonical_sha256(command) != expected_hash:
                raise SystemExit(f"command hash changed: {cell['cell_id']}")
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt_number}-{uuid.uuid4()}"
            start = {
                "schema": "yeto_outer_mup_attempt_start_v1",
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "cell_id": cell["cell_id"],
                "stage": stage,
                "node": node_label,
                "gpu_indices": gpus,
                "gpu_uuids": [inventory[gpu]["uuid"] for gpu in gpus],
                "gpu_names": [inventory[gpu]["name"] for gpu in gpus],
                "start_utc": utc_now(),
                "command": command,
                "command_hash": expected_hash,
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "git_commit": execution_git_commit,
                "registered_source_git_commit": registered_git_commit,
                "launch_authority_sha256": sha256_file(launch_authority_path),
            }
            write_json_atomic(attempt_root / "attempt-start.json", start)
            write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v9_slot_status_v1",
                    "state": "RUNNING",
                    "stage": stage,
                    "node": node_label,
                    "slot_id": slot_id,
                    "gpus": gpus,
                    "cell_id": cell["cell_id"],
                    "queue_index": queue_index,
                    "queue_total": len(queue),
                    "completed": completed,
                    "failures": failures,
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
                            "schema": "yeto_outer_mup_v9_slot_status_v1",
                            "state": "RUNNING",
                            "stage": stage,
                            "node": node_label,
                            "slot_id": slot_id,
                            "gpus": gpus,
                            "cell_id": cell["cell_id"],
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "completed": completed,
                            "failures": failures,
                            "elapsed_seconds": time.monotonic() - started_monotonic,
                            "gpu_sample": gpu_sample(gpus),
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
                "stage": stage,
                "node": node_label,
                "gpu_indices": gpus,
                "end_utc": utc_now(),
                "wall_seconds": time.monotonic() - started_monotonic,
                "process_return_code": return_code,
                "wall_ceiling_terminated": wall_stopped,
                "git_commit": execution_git_commit,
                "registered_source_git_commit": registered_git_commit,
            }
            write_json_atomic(attempt_root / "attempt-end.json", end)
            if wall_stopped:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "NOT_RUN_WALL_CEILING",
                    "failures": ["registered v9 wall ceiling terminated the cell"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            elif return_code == 0:
                validation_cell = {
                    **cell,
                    "source_git_commit": execution_git_commit,
                }
                evidence = validate_cell(validation_cell, attempt_root, command)
            else:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "INFRA_FAILURE",
                    "failures": [f"scientific process exited {return_code}"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": expected_hash,
                }
            evidence.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "attempt_start_sha256": sha256_file(
                        attempt_root / "attempt-start.json"
                    ),
                    "attempt_end_sha256": sha256_file(
                        attempt_root / "attempt-end.json"
                    ),
                    "git_commit": execution_git_commit,
                    "registered_source_git_commit": registered_git_commit,
                }
            )
            write_json_atomic(evidence_path, evidence)
            if evidence["status"] == "COMPLETED":
                completed += 1
            else:
                failures += 1
            write_json_atomic(
                status_path,
                {
                    "schema": "yeto_outer_mup_v9_slot_status_v1",
                    "state": "BETWEEN_CELLS",
                    "stage": stage,
                    "node": node_label,
                    "slot_id": slot_id,
                    "gpus": gpus,
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
                "schema": "yeto_outer_mup_v9_slot_status_v1",
                "state": "DRAINED",
                "stage": stage,
                "node": node_label,
                "slot_id": slot_id,
                "gpus": sorted(
                    {gpu for cell in queue for gpu in cell["assignment"]["gpus"]}
                ),
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
    parser.add_argument("--stage", choices=tuple(STAGE_COUNTS), required=True)
    parser.add_argument("--slot-id")
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        proof = verify_preflight(args.manifest, args.node_label, args.stage, args.proof)
        print(
            json.dumps(
                {
                    "node": args.node_label,
                    "stage": args.stage,
                    "status": proof["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.slot_id is None or args.launch_authority is None:
        parser.error("--slot-id and --launch-authority are required to run a queue")
    return run_queue(
        manifest_path=args.manifest,
        node_label=args.node_label,
        stage=args.stage,
        slot_id=args.slot_id,
        proof_path=args.proof,
        launch_authority_path=args.launch_authority,
        attempt_number=args.attempt,
        retry_authority_path=args.retry_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
