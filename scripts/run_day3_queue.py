#!/usr/bin/env python3
"""Drain one hash-bound day-3 GPU queue with append-only attempt evidence."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import day3_common as common  # noqa: E402
from run_slot_v3 import validate_cell as validate_standard_cell  # noqa: E402


TERMINAL_STATUSES = {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}


def kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def gpu_sample(gpu: int) -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {"return_code": result.returncode, "raw": result.stdout.strip()}


def verify_file_record(repo: Path, record: dict) -> None:
    path = repo / record["path"]
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or common.sha256_file(path) != record["sha256"]
    ):
        raise SystemExit(f"execution file mismatch: {path}")


def verify_manifest(path: Path, manifest: dict, node: str, queue_id: str) -> dict:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise SystemExit("day-3 manifest SHA-256 sidecar is missing or mismatched")
    if manifest.get("schema") != "yeto_day3_launch_manifest_v1":
        raise SystemExit("day-3 manifest schema mismatch")
    if manifest.get("status") != "AUTHORIZED" or manifest.get("program") != "v19":
        raise SystemExit("manifest does not authorize V19")
    queues = [queue for queue in manifest["queues"] if queue["queue_id"] == queue_id]
    if len(queues) != 1 or queues[0]["node"] != node:
        raise SystemExit(f"no unique {queue_id} authority for {node}")
    queue = queues[0]
    repo = Path("/root/yeto")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != manifest["source"]["git_commit"]:
        raise SystemExit(f"node commit {head} differs from the V19 manifest")
    tracked = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if tracked:
        raise SystemExit("tracked node Git worktree changes forbid V19")
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

    input_record = manifest["inputs"]["manifest"]
    input_path = Path(input_record["path"])
    if not input_path.is_file() or common.sha256_file(input_path) != input_record["sha256"]:
        raise SystemExit("V19 combined input manifest hash mismatch")
    inputs = manifest["inputs"]["content"]
    model = inputs["model"]
    for name, record in model["files"].items():
        model_file = Path(model["path"]) / name
        if not model_file.is_file() or common.sha256_file(model_file) != record["sha256"]:
            raise SystemExit(f"V19 model file mismatch: {model_file}")
    queue_cells = [cell for cell in manifest["cells"] if cell["queue_id"] == queue_id]
    if len(queue_cells) != 6:
        raise SystemExit("V19 queue must contain exactly six rungs")
    seed = str(queue_cells[0]["seed"])
    for label in ("train", "audit", "split_provenance"):
        record = inputs["seeds"][seed]["files"][label]
        input_file = Path(record["path"])
        if not input_file.is_file() or common.sha256_file(input_file) != record["sha256"]:
            raise SystemExit(f"V19 seed input mismatch: {input_file}")
    audit_hashes = {cell["eval_path"] for cell in queue_cells}
    if audit_hashes != {inputs["seeds"][seed]["files"]["audit"]["path"]}:
        raise SystemExit("V19 queue endpoint is not the frozen audit stream")
    for cell in queue_cells:
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise SystemExit(f"initial command hash mismatch: {cell['cell_id']}")
        for retry in cell["registered_retry_commands"]:
            if common.canonical_sha256(retry["command"]) != retry["command_hash"]:
                raise SystemExit(f"retry command hash mismatch: {cell['cell_id']}")
    sample = gpu_sample(int(queue["gpu"]))
    if sample["return_code"] != 0:
        raise SystemExit("assigned GPU inventory query failed")
    return queue


def command_record(cell: dict, attempt: int) -> dict:
    if attempt == 1:
        return {"command": cell["command"], "command_hash": cell["command_hash"]}
    records = [
        record
        for record in cell["registered_retry_commands"]
        if int(record["attempt_number"]) == attempt
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


def run_queue(
    manifest_path: Path,
    node: str,
    queue_id: str,
    attempt: int,
    retry_authority: Path | None,
) -> int:
    manifest = common.read_json(manifest_path)
    queue_authority = verify_manifest(manifest_path, manifest, node, queue_id)
    queue = sorted(
        [cell for cell in manifest["cells"] if cell["queue_id"] == queue_id],
        key=lambda cell: cell["slot_queue_index"],
    )
    retry_authority_sha256 = None
    if attempt == 2:
        if retry_authority is None or not retry_authority.is_file():
            raise SystemExit("attempt 2 requires a hash-bound retry authority")
        authority = common.read_json(retry_authority)
        if (
            authority.get("schema") != "yeto_day3_retry_authority_v1"
            or authority.get("manifest_sha256") != common.sha256_file(manifest_path)
            or authority.get("retry_group_id") != queue_id
            or authority.get("attempt") != 2
        ):
            raise SystemExit("attempt-2 retry authority mismatch")
        retry_authority_sha256 = common.sha256_file(retry_authority)
    status_path = (
        common.RESULT_ROOT / "_controller" / "slots" / "v19" / f"{queue_id}-a{attempt}.json"
    )
    lock_path = (
        common.RESULT_ROOT / "_controller" / "locks" / "v19" / f"{queue_id}-a{attempt}.lock"
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
                    completed += 1
                    continue
                raise SystemExit(f"refusing existing failed attempt: {evidence_path}")
            if attempt_root.exists():
                raise SystemExit(f"refusing to overwrite existing attempt: {attempt_root}")
            record = command_record(cell, attempt)
            command = record["command"]
            if common.canonical_sha256(command) != record["command_hash"]:
                raise SystemExit(f"{cell['cell_id']}: command hash changed")
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt}-{uuid.uuid4()}"
            common.write_json_atomic(
                attempt_root / "attempt-start.json",
                {
                    "schema": "yeto_day3_attempt_start_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "program": "v19",
                    "queue_id": queue_id,
                    "node": node,
                    "gpus": cell["assignment"]["gpus"],
                    "start_utc": common.utc_now(),
                    "command": command,
                    "command_hash": record["command_hash"],
                    "manifest_sha256": common.sha256_file(manifest_path),
                    "git_commit": manifest["source"]["git_commit"],
                    "retry_authority_sha256": retry_authority_sha256,
                },
            )
            common.write_json_atomic(
                status_path,
                {
                    "schema": "yeto_day3_slot_status_v1",
                    "state": "RUNNING",
                    "program": "v19",
                    "queue_id": queue_id,
                    "node": node,
                    "gpu": queue_authority["gpu"],
                    "cell_id": cell["cell_id"],
                    "queue_index": queue_index,
                    "queue_total": len(queue),
                    "completed": completed,
                    "failures": failures,
                    "attempt": attempt,
                    "updated_at_utc": common.utc_now(),
                },
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
                    cwd="/root/yeto",
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
                    common.write_json_atomic(
                        status_path,
                        {
                            "schema": "yeto_day3_slot_status_v1",
                            "state": "RUNNING",
                            "program": "v19",
                            "queue_id": queue_id,
                            "node": node,
                            "gpu": queue_authority["gpu"],
                            "cell_id": cell["cell_id"],
                            "elapsed_seconds": elapsed,
                            "gpu_sample": gpu_sample(int(queue_authority["gpu"])),
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "completed": completed,
                            "failures": failures,
                            "attempt": attempt,
                            "updated_at_utc": common.utc_now(),
                        },
                    )
                    time.sleep(30)
                return_code = process.wait()
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
                }
            )
            common.write_json_atomic(evidence_path, evidence)
            if evidence["status"] in TERMINAL_STATUSES:
                completed += 1
            else:
                failures += 1
        common.write_json_atomic(
            status_path,
            {
                "schema": "yeto_day3_slot_status_v1",
                "state": "DRAINED",
                "program": "v19",
                "queue_id": queue_id,
                "node": node,
                "gpu": queue_authority["gpu"],
                "attempt": attempt,
                "completed": completed,
                "failures": failures,
                "queue_total": len(queue),
                "updated_at_utc": common.utc_now(),
            },
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
