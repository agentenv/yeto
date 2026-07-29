#!/usr/bin/env python3
"""Drain one hash-bound day-3 GPU queue with append-only attempt evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import day3_common as common  # noqa: E402
from run_slot_v3 import validate_cell as validate_standard_cell  # noqa: E402


TERMINAL_STATUSES = {
    "COMPLETED",
    "SCIENTIFIC_DIVERGENCE",
    "INFRA_FAILURE",
    "INVALID_WORK",
}
SUCCESS_TERMINAL_STATUSES = {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}
ORPHAN_QUIET_SECONDS = 120


def kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    inventory = subprocess.run(
        ["ps", "-e", "-o", "pid=", "-o", "pgid="],
        capture_output=True,
        text=True,
        check=True,
    )
    owned = []
    for line in inventory.stdout.splitlines():
        pid_text, pgid_text = line.split()
        if int(pgid_text) == process.pid:
            owned.append(int(pid_text))
    if process.pid not in owned:
        owned.append(process.pid)
    # Every signal target is an exact PID resolved from our unique session;
    # no pattern-based or process-group signal can touch a foreign workload.
    for pid in sorted(set(owned), key=lambda value: value == process.pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    survivors = set(owned)
    while survivors and time.monotonic() < deadline:
        for pid in tuple(survivors):
            if pid == process.pid and process.poll() is not None:
                survivors.remove(pid)
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                survivors.remove(pid)
        if survivors:
            time.sleep(0.5)
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)


def gpu_sample(gpu: int) -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    sample: dict[str, object] = {
        "return_code": result.returncode,
        "raw": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode:
        return sample
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    try:
        sample.update(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    except (IndexError, ValueError) as exc:
        sample["parse_error"] = str(exc)
    processes = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    sample["compute_query_return_code"] = processes.returncode
    sample["compute_processes"] = [
        line.strip() for line in processes.stdout.splitlines() if line.strip()
    ]
    if processes.returncode:
        sample["compute_query_stderr"] = processes.stderr.strip()
    return sample


def capacity_ready(sample: dict[str, object], capacity: dict) -> tuple[bool, str]:
    if (
        sample.get("return_code") != 0
        or sample.get("compute_query_return_code") != 0
        or "parse_error" in sample
    ):
        return False, "GPU inventory query/parse failure"
    if sample.get("compute_processes"):
        return False, f"foreign compute process present: {sample['compute_processes']}"
    required_free = int(capacity["minimum_free_before_launch_mib"])
    maximum_used = int(capacity["exclusive_max_prelaunch_used_mib"])
    if int(sample["memory_free_mib"]) < required_free:
        return (
            False,
            f"free={sample['memory_free_mib']} MiB < required={required_free} MiB",
        )
    if int(sample["memory_used_mib"]) > maximum_used:
        return (
            False,
            f"used={sample['memory_used_mib']} MiB > exclusive ceiling={maximum_used} MiB",
        )
    return True, "capacity clear"


def write_slot_state(
    path: Path,
    *,
    state: str,
    program: str,
    queue_id: str,
    node: str,
    gpu: int,
    attempt: int,
    queue_index: int,
    queue_total: int,
    completed: int,
    failures: int,
    cell_id: str | None = None,
    **extra: object,
) -> None:
    payload: dict[str, object] = {
        "schema": "yeto_day3_slot_status_v1",
        "state": state,
        "program": program,
        "queue_id": queue_id,
        "node": node,
        "gpu": gpu,
        "queue_index": queue_index,
        "queue_total": queue_total,
        "completed": completed,
        "failures": failures,
        "attempt": attempt,
        "updated_at_utc": common.utc_now(),
    }
    if cell_id is not None:
        payload["cell_id"] = cell_id
    payload.update(extra)
    common.write_json_atomic(path, payload)


def wait_for_cell_capacity(
    *,
    gpu: int,
    capacity: dict,
    status_path: Path,
    program: str,
    queue_id: str,
    node: str,
    attempt: int,
    queue_index: int,
    queue_total: int,
    completed: int,
    failures: int,
    cell_id: str,
) -> dict[str, object]:
    required_samples = int(capacity.get("consecutive_clear_samples", 2))
    interval = int(capacity.get("clear_sample_interval_seconds", 5))
    consecutive = 0
    while consecutive < required_samples:
        sample = gpu_sample(gpu)
        ready, reason = capacity_ready(sample, capacity)
        consecutive = consecutive + 1 if ready else 0
        write_slot_state(
            status_path,
            state="CAPACITY_CHECK" if ready else "WAITING_FOR_CAPACITY",
            program=program,
            queue_id=queue_id,
            node=node,
            gpu=gpu,
            attempt=attempt,
            queue_index=queue_index,
            queue_total=queue_total,
            completed=completed,
            failures=failures,
            cell_id=cell_id,
            gpu_sample=sample,
            capacity_reason=reason,
            consecutive_clear_samples=consecutive,
            required_clear_samples=required_samples,
        )
        if consecutive < required_samples:
            time.sleep(interval if ready else 30)
    return sample


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for root, directories, files in os.walk(path):
        for name in directories + files:
            try:
                newest = max(newest, (Path(root) / name).stat().st_mtime)
            except FileNotFoundError:
                continue
    return newest


def attempt_processes(attempt_root: Path) -> str:
    # Bracketing the first character prevents pgrep from matching itself.
    escaped = re.escape(str(attempt_root)).replace("yeto", "[y]eto", 1)
    result = subprocess.run(
        ["pgrep", "-af", escaped], capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        raise SystemExit(f"orphan process inventory failed: {result.stderr.strip()}")
    return result.stdout.strip()


def wait_for_orphan_quiet(attempt_root: Path) -> float:
    last_mtime = None
    while True:
        active = attempt_processes(attempt_root)
        current_mtime = newest_mtime(attempt_root)
        age = time.time() - current_mtime
        if not active and age >= ORPHAN_QUIET_SECONDS and current_mtime == last_mtime:
            return current_mtime
        last_mtime = current_mtime
        time.sleep(30)


def has_recorded_endpoint(attempt_root: Path) -> bool:
    path = attempt_root / "report" / "results.jsonl"
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return False
    loss = rows[0].get("eval_loss")
    return isinstance(loss, (int, float)) and not isinstance(loss, bool)


def park_incomplete_attempt(
    attempt_root: Path,
    *,
    program: str,
    queue_id: str,
    cell_id: str,
    attempt: int,
    quiet_mtime: float,
) -> None:
    parked_root = (
        common.RESULT_ROOT
        / "_controller"
        / "parked"
        / program
        / cell_id
        / f"attempt-{attempt}-orphan-{uuid.uuid4()}"
    )
    parked_root.parent.mkdir(parents=True, exist_ok=True)
    attempt_root.replace(parked_root)
    common.write_json_atomic(
        parked_root.parent / f"{parked_root.name}.json",
        {
            "schema": "yeto_day3_parked_orphan_v1",
            "parked_at_utc": common.utc_now(),
            "program": program,
            "queue_id": queue_id,
            "cell_id": cell_id,
            "attempt": attempt,
            "source_path": str(attempt_root),
            "parked_path": str(parked_root),
            "mtime_guard_seconds": ORPHAN_QUIET_SECONDS,
            "quiet_tree_mtime": quiet_mtime,
            "active_process_inventory": "CLEAR",
        },
    )


def verify_file_record(repo: Path, record: dict) -> None:
    path = repo / record["path"]
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or common.sha256_file(path) != record["sha256"]
    ):
        raise SystemExit(f"execution file mismatch: {path}")


def verify_manifest(
    path: Path,
    manifest: dict,
    node: str,
    queue_id: str,
    *,
    attempt: int,
    execution_gpu: int,
) -> dict:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise SystemExit("day-3 manifest SHA-256 sidecar is missing or mismatched")
    if manifest.get("schema") != "yeto_day3_launch_manifest_v1":
        raise SystemExit("day-3 manifest schema mismatch")
    program = manifest.get("program")
    if manifest.get("status") != "AUTHORIZED" or program not in {"v19", "v18", "v16", "v14"}:
        raise SystemExit("manifest does not authorize a supported day-3 program")
    queues = [queue for queue in manifest["queues"] if queue["queue_id"] == queue_id]
    if len(queues) != 1 or queues[0]["node"] != node:
        raise SystemExit(f"no unique {queue_id} authority for {node}")
    queue = queues[0]
    repo = common.REMOTE_REPO
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != manifest["source"]["git_commit"]:
        raise SystemExit(f"node commit {head} differs from the day-3 manifest")
    tracked = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if tracked:
        raise SystemExit("tracked node Git worktree changes forbid day-3 execution")
    registration = manifest["registration"]
    verify_file_record(repo, registration)
    for record in manifest["execution_files"].values():
        verify_file_record(repo, record)
    syncer = Path(manifest["syncer"]["path"])
    if not syncer.is_file() or common.sha256_file(syncer) != manifest["syncer"]["sha256"]:
        raise SystemExit("release syncer binary hash mismatch")

    root = common.RESULT_ROOT
    if not root.is_symlink() or root.resolve() != common.RESULT_TARGET:
        raise SystemExit("day-3 result root is not the registered /data LVM symlink")
    findmnt = subprocess.run(
        ["findmnt", "-T", str(common.RESULT_TARGET), "-n", "-o", "SOURCE"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not findmnt.startswith("/dev/mapper/"):
        raise SystemExit(f"day-3 result target is not on an LVM root: {findmnt}")

    inputs = manifest["inputs"].get("content", manifest["inputs"])
    input_record = manifest["inputs"].get("manifest")
    if input_record is not None:
        input_path = Path(input_record["path"])
        if not input_path.is_file() or common.sha256_file(input_path) != input_record["sha256"]:
            raise SystemExit("combined input manifest hash mismatch")
    model_proof = manifest["inputs"].get("model_proof")
    if model_proof is not None:
        proof_path = Path(model_proof["path"])
        if not proof_path.is_file() or common.sha256_file(proof_path) != model_proof["sha256"]:
            raise SystemExit("model proof hash mismatch")
    model = inputs["model"]
    for name, record in model["files"].items():
        model_file = Path(model["path"]) / name
        if not model_file.is_file() or common.sha256_file(model_file) != record["sha256"]:
            raise SystemExit(f"V19 model file mismatch: {model_file}")
    queue_cells = [cell for cell in manifest["cells"] if cell["queue_id"] == queue_id]
    if len(queue_cells) != int(queue["scientific_cells"]):
        raise SystemExit("day-3 queue cardinality differs from its authority")
    if program == "v19":
        seeds = {str(cell["seed"]) for cell in queue_cells}
        if len(seeds) != 1:
            raise SystemExit("a V19 queue must contain one paired seed")
        seed = next(iter(seeds))
        for label in ("train", "audit", "split_provenance"):
            record = inputs["seeds"][seed]["files"][label]
            input_file = Path(record["path"])
            if not input_file.is_file() or common.sha256_file(input_file) != record["sha256"]:
                raise SystemExit(f"V19 seed input mismatch: {input_file}")
        endpoint_paths = {cell["eval_path"] for cell in queue_cells}
        if endpoint_paths != {inputs["seeds"][seed]["files"]["audit"]["path"]}:
            raise SystemExit("V19 queue endpoint is not the frozen audit stream")
    else:
        for label in ("train", "eval"):
            record = inputs["files"][label]
            input_file = Path(record["path"])
            if not input_file.is_file() or common.sha256_file(input_file) != record["sha256"]:
                raise SystemExit(f"{program.upper()} frozen input mismatch: {input_file}")
            rows = sum(1 for line in input_file.open(encoding="utf-8") if line.strip())
            if rows != int(record["rows"]):
                raise SystemExit(f"{program.upper()} frozen input row count mismatch: {input_file}")
        if {cell["train_path"] for cell in queue_cells} != {inputs["files"]["train"]["path"]}:
            raise SystemExit(f"{program.upper()} queue train path mismatch")
        if {cell["eval_path"] for cell in queue_cells} != {inputs["files"]["eval"]["path"]}:
            raise SystemExit(f"{program.upper()} queue endpoint path mismatch")
    for cell in queue_cells:
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise SystemExit(f"initial command hash mismatch: {cell['cell_id']}")
        for retry in cell["registered_retry_commands"]:
            if common.canonical_sha256(retry["command"]) != retry["command_hash"]:
                raise SystemExit(f"retry command hash mismatch: {cell['cell_id']}")
    if attempt == 1 and execution_gpu != int(queue["gpu"]):
        raise SystemExit("attempt-1 GPU differs from the queue authority")
    if attempt == 2:
        for cell in queue_cells:
            matching = [
                record
                for record in cell["registered_retry_commands"]
                if int(record["attempt_number"]) == 2
                and record.get("node") == node
                and int(record.get("gpu", -1)) == execution_gpu
            ]
            if len(matching) != 1:
                raise SystemExit(
                    f"{cell['cell_id']}: retry GPU lacks a unique registered command"
                )
    capacity = manifest.get("capacity")
    if not isinstance(capacity, dict) or not capacity.get("recheck_before_every_cell"):
        raise SystemExit("manifest lacks the shared-host per-cell capacity contract")
    protected = {
        (record.get("node"), int(record.get("gpu", -1)))
        for record in capacity.get("protected_slots", [])
    }
    if (node, execution_gpu) in protected:
        raise SystemExit(f"refusing protected shared-host GPU {node}:{execution_gpu}")
    sample = gpu_sample(execution_gpu)
    if sample["return_code"] != 0:
        raise SystemExit("assigned GPU inventory query failed")
    return queue


def command_record(cell: dict, attempt: int, *, node: str, gpu: int) -> dict:
    if attempt == 1:
        return {"command": cell["command"], "command_hash": cell["command_hash"]}
    records = [
        record
        for record in cell["registered_retry_commands"]
        if int(record["attempt_number"]) == attempt
        and record.get("node") == node
        and int(record.get("gpu", -1)) == gpu
    ]
    if len(records) != 1:
        raise SystemExit(f"{cell['cell_id']}: no unique attempt-{attempt} command")
    return records[0]


def failure_evidence(cell: dict, status: str, failure: str, command_hash: str) -> dict:
    return {
        "schema": "yeto_outer_mup_cell_evidence_v1",
        "cell_id": cell["cell_id"],
        "validated_at_utc": common.utc_now(),
        "status": status,
        "failures": [failure],
        "expected": cell["expected"],
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": command_hash,
    }


def recover_recorded_endpoint(
    *,
    cell: dict,
    attempt_root: Path,
    command: list[str],
    command_hash: str,
    attempt: int,
    manifest: dict,
    manifest_path: Path,
    retry_authority_sha256: str | None,
    node: str,
    gpu: int,
    quiet_mtime: float,
) -> dict:
    start_path = attempt_root / "attempt-start.json"
    if not start_path.is_file():
        evidence = failure_evidence(
            cell,
            "INVALID_WORK",
            "orphaned finite endpoint lacks attempt-start authority",
            command_hash,
        )
        common.write_json_atomic(
            start_path,
            {
                "schema": "yeto_day3_recovered_missing_attempt_start_v1",
                "cell_id": cell["cell_id"],
                "recovered_at_utc": common.utc_now(),
            },
        )
    else:
        evidence = validate_standard_cell(
            {**cell, "source_git_commit": manifest["source"]["git_commit"]},
            attempt_root,
            command,
        )
    end_path = attempt_root / "attempt-end.json"
    if not end_path.is_file():
        common.write_json_atomic(
            end_path,
            {
                "schema": "yeto_day3_attempt_end_v1",
                "attempt_number": attempt,
                "cell_id": cell["cell_id"],
                "end_utc": common.utc_now(),
                "process_return_code": None,
                "timed_out": False,
                "recovered_after_controller_loss": True,
                "quiet_tree_mtime": quiet_mtime,
            },
        )
    post_sample = gpu_sample(gpu)
    evidence.update(
        {
            "attempt_number": attempt,
            "attempt_start_sha256": common.sha256_file(start_path),
            "attempt_end_sha256": common.sha256_file(end_path),
            "git_commit": manifest["source"]["git_commit"],
            "manifest_sha256": common.sha256_file(manifest_path),
            "retry_group_id": cell["retry_group_id"],
            "retry_authority_sha256": retry_authority_sha256,
            "execution_node": node,
            "execution_gpu": gpu,
            "postlaunch_gpu_sample": post_sample,
            "recovered_after_controller_loss": True,
            "finite_endpoint_preserved_without_relaunch": True,
        }
    )
    return evidence


def run_queue(
    manifest_path: Path,
    node: str,
    queue_id: str,
    attempt: int,
    retry_authority: Path | None,
) -> int:
    manifest = common.read_json(manifest_path)
    queue = sorted(
        [cell for cell in manifest["cells"] if cell["queue_id"] == queue_id],
        key=lambda cell: cell["slot_queue_index"],
    )
    queues = [record for record in manifest["queues"] if record["queue_id"] == queue_id]
    if len(queues) != 1:
        raise SystemExit("queue authority is absent or duplicated")
    execution_gpu = int(queues[0]["gpu"])
    retry_authority_sha256 = None
    if attempt == 2:
        if retry_authority is None or not retry_authority.is_file():
            raise SystemExit("attempt 2 requires a hash-bound retry authority")
        authority = common.read_json(retry_authority)
        if (
            authority.get("schema") != "yeto_day3_retry_authority_v1"
            or authority.get("manifest_sha256") != common.sha256_file(manifest_path)
            or authority.get("program") != manifest.get("program")
            or authority.get("queue_id") != queue_id
            or authority.get("node") != node
            or authority.get("retry_group_id")
            not in {cell["retry_group_id"] for cell in queue}
            or authority.get("attempt") != 2
            or authority.get("finite_endpoint_seen") is not False
        ):
            raise SystemExit("attempt-2 retry authority mismatch")
        execution_gpu = int(authority.get("gpu", -1))
        retry_authority_sha256 = common.sha256_file(retry_authority)
        queue = [cell for cell in queue if cell["retry_group_id"] == authority["retry_group_id"]]
        if not queue:
            raise SystemExit("retry authority selects no cells from this queue")
    verify_manifest(
        manifest_path,
        manifest,
        node,
        queue_id,
        attempt=attempt,
        execution_gpu=execution_gpu,
    )
    program = manifest["program"]
    capacity = manifest["capacity"]
    status_path = (
        common.RESULT_ROOT / "_controller" / "slots" / program / f"{queue_id}-a{attempt}.json"
    )
    lock_path = (
        common.RESULT_ROOT / "_controller" / "locks" / program / f"{queue_id}-a{attempt}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"queue controller already active: {lock_path}") from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = common.RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = common.read_json(evidence_path)
                if evidence.get("status") in TERMINAL_STATUSES:
                    if evidence.get("status") in SUCCESS_TERMINAL_STATUSES:
                        completed += 1
                    else:
                        failures += 1
                    continue
                raise SystemExit(f"refusing nonterminal existing evidence: {evidence_path}")
            record = command_record(cell, attempt, node=node, gpu=execution_gpu)
            command = record["command"]
            if common.canonical_sha256(command) != record["command_hash"]:
                raise SystemExit(f"{cell['cell_id']}: command hash changed")
            if attempt_root.exists():
                quiet_mtime = wait_for_orphan_quiet(attempt_root)
                if has_recorded_endpoint(attempt_root):
                    evidence = recover_recorded_endpoint(
                        cell=cell,
                        attempt_root=attempt_root,
                        command=command,
                        command_hash=record["command_hash"],
                        attempt=attempt,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        retry_authority_sha256=retry_authority_sha256,
                        node=node,
                        gpu=execution_gpu,
                        quiet_mtime=quiet_mtime,
                    )
                    common.write_json_atomic(evidence_path, evidence)
                    if evidence["status"] in SUCCESS_TERMINAL_STATUSES:
                        completed += 1
                    else:
                        failures += 1
                    continue
                park_incomplete_attempt(
                    attempt_root,
                    program=program,
                    queue_id=queue_id,
                    cell_id=cell["cell_id"],
                    attempt=attempt,
                    quiet_mtime=quiet_mtime,
                )
            prelaunch_sample = wait_for_cell_capacity(
                gpu=execution_gpu,
                capacity=capacity,
                status_path=status_path,
                program=program,
                queue_id=queue_id,
                node=node,
                attempt=attempt,
                queue_index=queue_index,
                queue_total=len(queue),
                completed=completed,
                failures=failures,
                cell_id=cell["cell_id"],
            )
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt}-{uuid.uuid4()}"
            common.write_json_atomic(
                attempt_root / "attempt-start.json",
                {
                    "schema": "yeto_day3_attempt_start_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "program": program,
                    "queue_id": queue_id,
                    "node": node,
                    "gpus": [execution_gpu],
                    "registered_primary_gpus": cell["assignment"]["gpus"],
                    "prelaunch_gpu_sample": prelaunch_sample,
                    "capacity_contract": capacity,
                    "start_utc": common.utc_now(),
                    "command": command,
                    "command_hash": record["command_hash"],
                    "manifest_sha256": common.sha256_file(manifest_path),
                    "git_commit": manifest["source"]["git_commit"],
                    "retry_authority_sha256": retry_authority_sha256,
                },
            )
            write_slot_state(
                status_path,
                state="RUNNING",
                program=program,
                queue_id=queue_id,
                node=node,
                gpu=execution_gpu,
                attempt=attempt,
                queue_index=queue_index,
                queue_total=len(queue),
                completed=completed,
                failures=failures,
                cell_id=cell["cell_id"],
            )
            log_path = attempt_root / "controller.stdout.log"
            started = time.monotonic()
            environment = dict(os.environ)
            environment.update(manifest["environment"])
            environment.update(
                {"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
            )
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(common.REMOTE_REPO),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                timed_out = False
                timeout_seconds = float(cell["timeout_minutes"]) * 60.0
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= timeout_seconds:
                        timed_out = True
                        kill_group(process)
                        break
                    write_slot_state(
                        status_path,
                        state="RUNNING",
                        program=program,
                        queue_id=queue_id,
                        node=node,
                        gpu=execution_gpu,
                        attempt=attempt,
                        queue_index=queue_index,
                        queue_total=len(queue),
                        completed=completed,
                        failures=failures,
                        cell_id=cell["cell_id"],
                        elapsed_seconds=elapsed,
                        gpu_sample=gpu_sample(execution_gpu),
                    )
                    time.sleep(30)
                return_code = process.wait()
            time.sleep(2)
            postlaunch_sample = gpu_sample(execution_gpu)
            try:
                log_tail = log_path.read_text(errors="replace")[-131072:]
            except OSError:
                log_tail = ""
            oom_observed = "out of memory" in log_tail.lower()
            common.write_json_atomic(
                attempt_root / "attempt-end.json",
                {
                    "schema": "yeto_day3_attempt_end_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "end_utc": common.utc_now(),
                    "wall_seconds": time.monotonic() - started,
                    "process_return_code": return_code,
                    "timed_out": timed_out,
                    "gpu": execution_gpu,
                    "postlaunch_gpu_sample": postlaunch_sample,
                    "oom_observed_in_controller_log": oom_observed,
                },
            )
            if timed_out:
                evidence = failure_evidence(
                    cell, "INFRA_FAILURE", "registered cell timeout", record["command_hash"]
                )
            elif return_code:
                evidence = failure_evidence(
                    cell,
                    "INFRA_FAILURE",
                    f"scientific process exited {return_code} before a valid endpoint",
                    record["command_hash"],
                )
            else:
                evidence = validate_standard_cell(
                    {**cell, "source_git_commit": manifest["source"]["git_commit"]},
                    attempt_root,
                    command,
                )
            evidence.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "attempt_start_sha256": common.sha256_file(attempt_root / "attempt-start.json"),
                    "attempt_end_sha256": common.sha256_file(attempt_root / "attempt-end.json"),
                    "git_commit": manifest["source"]["git_commit"],
                    "manifest_sha256": common.sha256_file(manifest_path),
                    "retry_group_id": cell["retry_group_id"],
                    "retry_authority_sha256": retry_authority_sha256,
                    "execution_node": node,
                    "execution_gpu": execution_gpu,
                    "postlaunch_gpu_sample": postlaunch_sample,
                    "oom_observed_in_controller_log": oom_observed,
                    "attempt2_relocation_required": evidence.get("status")
                    == "INFRA_FAILURE",
                }
            )
            common.write_json_atomic(evidence_path, evidence)
            if evidence["status"] in SUCCESS_TERMINAL_STATUSES:
                completed += 1
            else:
                failures += 1
        write_slot_state(
            status_path,
            state="DRAINED",
            program=program,
            queue_id=queue_id,
            node=node,
            gpu=execution_gpu,
            attempt=attempt,
            queue_index=len(queue),
            queue_total=len(queue),
            completed=completed,
            failures=failures,
        )
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", choices=common.NODES, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    args = parser.parse_args()
    if args.attempt == 1 and args.retry_authority is not None:
        parser.error("--retry-authority is only valid for attempt 2")
    return run_queue(
        args.manifest,
        args.node_label,
        args.queue_id,
        args.attempt,
        args.retry_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
