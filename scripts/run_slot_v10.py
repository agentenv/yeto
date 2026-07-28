#!/usr/bin/env python3
"""Wait for reduced-v9 islands, then drain hash-bound one-GPU V10 queues."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_slot_v3 import gpu_inventory, validate_cell  # noqa: E402
from v10_common import (  # noqa: E402
    V10Error,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


MANIFEST_SCHEMA = "yeto_outer_mup_v10_freshtransfer_launch_manifest_v1"
RESULT_LINK = Path("/root/yeto-results-v10")
RESULT_TARGET = Path("/data/yeto-results-v10")
REPO = Path("/root/yeto-v10")
MIN_FREE_BYTES = 500_000_000_000
POLL_SECONDS = 15
PRIORITY_PATTERN = (
    r"[r]un_slot_v9.py|[l]aunch_v9_stage.py|[7][bB]-wave|[r]un_slot_tonight85.py"
)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise V10Error(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def verify_file(record: dict, label: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise V10Error(f"missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise V10Error(f"size mismatch for {label}: {path}")
    if sha256_file(path) != record["sha256"]:
        raise V10Error(f"hash mismatch for {label}: {path}")


def verify_storage() -> dict:
    if not RESULT_LINK.is_symlink():
        raise V10Error("V10 result root is not a symlink")
    resolved = RESULT_LINK.resolve(strict=True)
    if resolved != RESULT_TARGET:
        raise V10Error(f"V10 result link resolves to {resolved}")
    if resolved.stat().st_dev != Path("/data").stat().st_dev:
        raise V10Error("V10 result root is not on /data")
    usage = shutil.disk_usage(resolved)
    if usage.free < MIN_FREE_BYTES:
        raise V10Error(f"V10 filesystem has only {usage.free} free bytes")
    return {
        "link": str(RESULT_LINK),
        "target": str(resolved),
        "free_bytes": usage.free,
    }


def verify_manifest(path: Path, node_label: str) -> dict:
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise V10Error("manifest schema mismatch")
    if manifest.get("status") != "REGISTERED" or len(manifest.get("cells", [])) != 18:
        raise V10Error("manifest is not the complete registered 18-cell design")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(path):
        raise V10Error("manifest SHA-256 sidecar is missing or mismatched")
    head = git("rev-parse", "HEAD")
    if head != manifest.get("source", {}).get("git_commit"):
        raise V10Error("V10 checkout differs from manifest source commit")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise V10Error("V10 checkout is dirty")
    if manifest.get("registration", {}).get("git_commit") != head:
        raise V10Error("manifest registration commit differs from execution")
    for label, artifact in manifest.get("frozen_artifacts", {}).items():
        target = REPO / artifact["path"]
        bound = {**artifact, "path": str(target)}
        verify_file(bound, f"frozen artifact {label}")
    for label in ("training_jsonl", "heldout_audit_jsonl", "scale_manifest"):
        verify_file(manifest["inputs"][label], label)
    model = manifest["model"]
    for name, record in model["files"].items():
        verify_file({**record, "path": str(Path(model["path"]) / name)}, f"model/{name}")
    seen = set()
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        if cell_id in seen:
            raise V10Error(f"duplicate cell id {cell_id}")
        seen.add(cell_id)
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise V10Error(f"command hash mismatch {cell_id}")
        retries = cell.get("registered_retry_commands", [])
        if len(retries) != 1 or canonical_sha256(retries[0]["command"]) != retries[0]["command_hash"]:
            raise V10Error(f"retry command hash mismatch {cell_id}")
        if cell["assignment"]["node"] not in ("h200-n1", "h200-n2"):
            raise V10Error(f"invalid node assignment {cell_id}")
        command = cell["command"]
        if "--outer-bias-correction" in command:
            raise V10Error(f"bias-corrected flag in raw V10 cell {cell_id}")
        if command_value(command, "--outer-momentum") != "0.9":
            raise V10Error(f"momentum mismatch {cell_id}")
        if command_value(command, "--gpu-offset") != str(cell["assignment"]["gpu"]):
            raise V10Error(f"GPU binding mismatch {cell_id}")
        if command_value(command, "--prebound-development-eval") != manifest["inputs"]["heldout_audit_jsonl"]["path"]:
            raise V10Error(f"held-out stream mismatch {cell_id}")
    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        raise V10Error(f"expected GPUs 0..7, got {sorted(inventory)}")
    if any("H200" not in record["name"].upper() for record in inventory.values()):
        raise V10Error("non-H200 device in inventory")
    proof = {
        "schema": "yeto_outer_mup_v10_node_proof_v1",
        "node": node_label,
        "checked_at_utc": utc_now(),
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "git_commit": head,
        "gpu_inventory": inventory,
        "storage": verify_storage(),
        "status": "PASS",
    }
    proof_path = RESULT_LINK / "_controller" / f"preflight-{node_label}.json"
    write_json_atomic(proof_path, proof)
    return manifest


def pgrep_priority() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", PRIORITY_PATTERN], capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        raise V10Error(f"priority pgrep failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def compute_by_uuid() -> dict[str, list[dict]]:
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
        raise V10Error(result.stderr.strip() or result.stdout.strip())
    records: dict[str, list[dict]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise V10Error(f"unparseable compute row: {line}")
        records.setdefault(parts[1], []).append(
            {
                "pid": int(parts[0]),
                "process_name": parts[2],
                "used_memory_mib": float(parts[3]),
            }
        )
    return records


def v9_island_id(node_label: str, gpu: int) -> str:
    return f"{node_label}-gpu0-3" if gpu < 4 else f"{node_label}-gpu4-7"


def slot_available(node_label: str, gpu: int, inventory: dict) -> tuple[bool, dict]:
    lines = pgrep_priority()
    island = v9_island_id(node_label, gpu)
    # A live v9 island controller can start its next registered cell even if a
    # GPU is momentarily empty, so absence of compute alone never authorizes V10.
    island_controllers = [line for line in lines if "run_slot_v9.py" in line and island in line]
    tonight = [line for line in lines if "run_slot_tonight85.py" in line]
    compute = compute_by_uuid().get(inventory[gpu]["uuid"], [])
    available = not island_controllers and not tonight and not compute
    return available, {
        "v9_island": island,
        "v9_controllers": island_controllers,
        "tonight_controllers": tonight,
        "compute": compute,
    }


def gpu_sample(gpu: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {"return_code": result.returncode, "raw": result.stdout.strip()}


def kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def load_retry_groups(path: Path, manifest_path: Path, manifest: dict) -> set[str]:
    authority = read_json(path)
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v10_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    if authority.get("reason") not in manifest["retry_contract"]["allowed_reasons"]:
        errors.append("retry reason is not registered")
    groups = authority.get("retry_group_ids")
    known = {cell["retry_group_id"] for cell in manifest["cells"]}
    if not isinstance(groups, list) or not groups or set(groups) - known:
        errors.append("retry group list is empty or unknown")
        groups = []
    if errors:
        raise V10Error("; ".join(errors))
    return set(groups)


def run_queue(
    manifest_path: Path,
    node_label: str,
    slot_id: str,
    attempt_number: int,
    retry_authority: Path | None,
) -> int:
    manifest = verify_manifest(manifest_path, node_label)
    queue = [
        cell
        for cell in manifest["cells"]
        if cell["assignment"]["node"] == node_label and cell["slot_id"] == slot_id
    ]
    if attempt_number == 2:
        if retry_authority is None:
            raise V10Error("attempt 2 requires --retry-authority")
        groups = load_retry_groups(retry_authority, manifest_path, manifest)
        queue = [cell for cell in queue if cell["retry_group_id"] in groups]
    elif retry_authority is not None:
        raise V10Error("retry authority is forbidden for attempt 1")
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    if not queue:
        raise V10Error(f"no registered queue for {slot_id}/attempt-{attempt_number}")
    gpu = int(queue[0]["assignment"]["gpu"])
    lock_path = RESULT_LINK / "_controller" / "locks" / f"{slot_id}.lock"
    status_path = RESULT_LINK / "_controller" / "slots" / f"{slot_id}.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V10Error(f"V10 slot controller already active: {slot_id}") from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = RESULT_LINK / cell["cell_id"] / f"attempt-{attempt_number}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = read_json(evidence_path)
                if evidence.get("status") == "COMPLETED":
                    completed += 1
                    continue
                raise V10Error(f"existing noncompleted attempt: {cell['cell_id']}")
            if attempt_root.exists():
                raise V10Error(f"refusing to overwrite {attempt_root}")
            record = (
                {"command": cell["command"], "command_hash": cell["command_hash"]}
                if attempt_number == 1
                else cell["registered_retry_commands"][0]
            )
            command = record["command"]
            if canonical_sha256(command) != record["command_hash"]:
                raise V10Error(f"command hash changed: {cell['cell_id']}")
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
                "command_hash": record["command_hash"],
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "git_commit": git("rev-parse", "HEAD"),
                "registered_source_git_commit": manifest["source"]["git_commit"],
                "manifest_sha256": sha256_file(manifest_path),
            }
            write_json_atomic(attempt_root / "attempt-start.json", start)
            log_path = attempt_root / "controller.stdout.log"
            started = time.monotonic()
            environment = {
                **os.environ,
                **{key: str(value) for key, value in manifest["environment"].items()},
                "PYTHONUNBUFFERED": "1",
            }
            with log_path.open("wb") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                while process.poll() is None:
                    write_json_atomic(
                        status_path,
                        {
                            "schema": "yeto_outer_mup_v10_slot_status_v1",
                            "state": "RUNNING",
                            "node": node_label,
                            "slot_id": slot_id,
                            "gpu": gpu,
                            "cell_id": cell["cell_id"],
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "completed": completed,
                            "failures": failures,
                            "controller_pid": os.getpid(),
                            "scientific_process_pid": process.pid,
                            "scientific_process_group_id": os.getpgid(process.pid),
                            "elapsed_seconds": time.monotonic() - started,
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
                "wall_seconds": time.monotonic() - started,
                "process_return_code": return_code,
                "git_commit": git("rev-parse", "HEAD"),
            }
            write_json_atomic(attempt_root / "attempt-end.json", end)
            if return_code == 0:
                evidence = validate_cell(cell, attempt_root, command)
            else:
                evidence = {
                    "schema": "yeto_outer_mup_cell_evidence_v1",
                    "cell_id": cell["cell_id"],
                    "validated_at_utc": utc_now(),
                    "status": "INFRA_FAILURE",
                    "failures": [f"scientific process exited {return_code}"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "command_hash": record["command_hash"],
                }
            evidence.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "attempt_start_sha256": sha256_file(attempt_root / "attempt-start.json"),
                    "attempt_end_sha256": sha256_file(attempt_root / "attempt-end.json"),
                    "manifest_sha256": sha256_file(manifest_path),
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
                    "schema": "yeto_outer_mup_v10_slot_status_v1",
                    "state": "BETWEEN_CELLS",
                    "node": node_label,
                    "slot_id": slot_id,
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
                "schema": "yeto_outer_mup_v10_slot_status_v1",
                "state": "DRAINED",
                "node": node_label,
                "slot_id": slot_id,
                "gpu": gpu,
                "completed": completed,
                "failures": failures,
                "queue_total": len(queue),
                "updated_at_utc": utc_now(),
            },
        )
    return 0 if failures == 0 else 2


def wait_and_run(manifest_path: Path, node_label: str) -> int:
    manifest = verify_manifest(manifest_path, node_label)
    inventory = gpu_inventory()
    local_slots = sorted(
        {
            cell["slot_id"]
            for cell in manifest["cells"]
            if cell["assignment"]["node"] == node_label
        },
        key=lambda value: int(value.rsplit("gpu", 1)[1]),
    )
    pending = set(local_slots)
    children: dict[str, subprocess.Popen] = {}
    parent_status = RESULT_LINK / "_controller" / f"wait-controller-{node_label}.json"
    log_root = RESULT_LINK / "_controller" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    while pending or children:
        for slot_id in sorted(pending):
            gpu = int(slot_id.rsplit("gpu", 1)[1])
            available, proof = slot_available(node_label, gpu, inventory)
            if not available:
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(manifest_path),
                "--node-label",
                node_label,
                "--slot-id",
                slot_id,
                "--attempt",
                "1",
            ]
            log_handle = (log_root / f"{slot_id}.log").open("ab")
            child = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
                    "TMPDIR": "/data/tmp",
                    "PYTHONUNBUFFERED": "1",
                },
                start_new_session=True,
            )
            log_handle.close()
            children[slot_id] = child
            pending.remove(slot_id)
            write_json_atomic(
                RESULT_LINK / "_controller" / "claims" / f"{slot_id}.json",
                {
                    "schema": "yeto_outer_mup_v10_gpu_claim_v1",
                    "claimed_at_utc": utc_now(),
                    "node": node_label,
                    "gpu": gpu,
                    "slot_id": slot_id,
                    "loss_blind_free_proof": proof,
                    "child_pid": child.pid,
                },
            )
        finished = {}
        for slot_id, child in list(children.items()):
            code = child.poll()
            if code is not None:
                finished[slot_id] = code
                del children[slot_id]
        write_json_atomic(
            parent_status,
            {
                "schema": "yeto_outer_mup_v10_wait_controller_v1",
                "state": "WAITING_OR_RUNNING" if pending or children else "DRAINED",
                "node": node_label,
                "controller_pid": os.getpid(),
                "pending_slots": sorted(pending),
                "active_slots": {slot: child.pid for slot, child in children.items()},
                "just_finished": finished,
                "priority_pattern_registration": "run_slot_v10.py",
                "updated_at_utc": utc_now(),
            },
        )
        if pending or children:
            time.sleep(POLL_SECONDS)
    statuses = []
    for slot_id in local_slots:
        path = RESULT_LINK / "_controller" / "slots" / f"{slot_id}.json"
        statuses.append(read_json(path))
    failures = sum(int(record.get("failures", 0)) for record in statuses)
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", required=True, choices=("h200-n1", "h200-n2"))
    parser.add_argument("--wait-and-run", action="store_true")
    parser.add_argument("--slot-id")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    args = parser.parse_args()
    try:
        if args.wait_and_run:
            if args.slot_id is not None or args.attempt != 1 or args.retry_authority:
                parser.error("--wait-and-run cannot be combined with slot/retry options")
            return wait_and_run(args.manifest.resolve(), args.node_label)
        if not args.slot_id:
            parser.error("--slot-id is required without --wait-and-run")
        return run_queue(
            args.manifest.resolve(),
            args.node_label,
            args.slot_id,
            args.attempt,
            args.retry_authority.resolve() if args.retry_authority else None,
        )
    except (KeyError, OSError, TypeError, ValueError, V10Error) as exc:
        print(f"V10 controller error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
