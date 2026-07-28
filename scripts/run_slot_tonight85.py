#!/usr/bin/env python3
"""Drain one hash-bound tonight-8.5 GPU or full-node island queue."""

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

import tonight85_common as common  # noqa: E402
from run_node_v7 import validate_cell as validate_v7_cell  # noqa: E402
from run_slot_v3 import validate_cell as validate_standard_cell  # noqa: E402


def kill_group(process: subprocess.Popen) -> None:
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
    return {"return_code": result.returncode, "raw": result.stdout.strip()}


def verify_manifest(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(
        path
    ):
        raise SystemExit("manifest SHA-256 sidecar is missing or mismatched")
    if manifest.get("schema") != "yeto_tonight85_launch_manifest_v1":
        raise SystemExit("manifest schema mismatch")
    repo = Path("/root/yeto")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != manifest.get("source", {}).get("git_commit"):
        raise SystemExit(f"node commit {head} differs from manifest source")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty:
        raise SystemExit("node Git worktree is dirty")
    for record in manifest.get("registration", {}).values():
        registered = repo / record["path"]
        if (
            not registered.is_file()
            or common.sha256_file(registered) != record["sha256"]
        ):
            raise SystemExit(f"registered contract mismatch: {registered}")
    dynamic = manifest.get("dynamic_artifact")
    if dynamic:
        prediction = repo / dynamic["path"]
        if (
            not prediction.is_file()
            or common.sha256_file(prediction) != dynamic["sha256"]
        ):
            raise SystemExit(f"dynamic prediction artifact mismatch: {prediction}")


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


def evidence_for_failure(
    cell: dict, status: str, failure: str, command_hash: str
) -> dict:
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
    stage: str,
    slot_id: str,
    attempt: int,
    retry_groups: set[str] | None,
) -> int:
    manifest = common.read_json(manifest_path)
    verify_manifest(manifest_path, manifest)
    queue = [
        cell
        for cell in manifest["cells"]
        if cell["stage"] == stage
        and cell["slot_id"] == slot_id
        and cell["assignment"]["node"] == node
    ]
    if retry_groups is not None:
        queue = [cell for cell in queue if cell["retry_group_id"] in retry_groups]
    if not queue:
        raise SystemExit(f"no {stage}/{slot_id} queue for {node}")
    queue.sort(key=lambda cell: cell["slot_queue_index"])
    status_path = (
        common.RESULT_ROOT
        / "_controller"
        / "slots"
        / stage
        / f"{slot_id}-a{attempt}.json"
    )
    lock_path = (
        common.RESULT_ROOT
        / "_controller"
        / "locks"
        / stage
        / f"{slot_id}-a{attempt}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"slot controller already active: {lock_path}") from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = common.RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = common.read_json(evidence_path)
                if evidence.get("status") in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}:
                    completed += 1
                    continue
                raise SystemExit(f"refusing existing failed attempt: {evidence_path}")
            if attempt_root.exists():
                raise SystemExit(
                    f"refusing to overwrite existing attempt: {attempt_root}"
                )
            record = command_record(cell, attempt)
            command = record["command"]
            if common.canonical_sha256(command) != record["command_hash"]:
                raise SystemExit(f"{cell['cell_id']}: command hash changed")
            for required in (cell["model_path"], cell["train_path"], cell["eval_path"]):
                if not Path(required).exists():
                    raise SystemExit(
                        f"{cell['cell_id']}: missing staged input {required}"
                    )
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt}-{uuid.uuid4()}"
            common.write_json_atomic(
                attempt_root / "attempt-start.json",
                {
                    "schema": "yeto_tonight85_attempt_start_v1",
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "cell_id": cell["cell_id"],
                    "program": cell["program"],
                    "stage": stage,
                    "node": node,
                    "gpus": cell["assignment"]["gpus"],
                    "start_utc": common.utc_now(),
                    "command": command,
                    "command_hash": record["command_hash"],
                    "manifest_sha256": common.sha256_file(manifest_path),
                    "git_commit": manifest["source"]["git_commit"],
                },
            )
            common.write_json_atomic(
                status_path,
                {
                    "schema": "yeto_tonight85_slot_status_v1",
                    "state": "RUNNING",
                    "stage": stage,
                    "slot_id": slot_id,
                    "node": node,
                    "cell_id": cell["cell_id"],
                    "queue_index": queue_index,
                    "queue_total": len(queue),
                    "completed": completed,
                    "failures": failures,
                    "updated_at_utc": common.utc_now(),
                },
            )
            log_path = attempt_root / "controller.stdout.log"
            started = time.monotonic()
            env = dict(os.environ)
            env.update(
                {
                    "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
                    "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    command,
                    cwd="/root/yeto",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
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
                            "schema": "yeto_tonight85_slot_status_v1",
                            "state": "RUNNING",
                            "stage": stage,
                            "slot_id": slot_id,
                            "node": node,
                            "cell_id": cell["cell_id"],
                            "elapsed_seconds": elapsed,
                            "gpu_sample": gpu_sample(cell["assignment"]["gpus"]),
                            "queue_index": queue_index,
                            "queue_total": len(queue),
                            "completed": completed,
                            "failures": failures,
                            "updated_at_utc": common.utc_now(),
                        },
                    )
                    time.sleep(30)
                return_code = process.wait()
            common.write_json_atomic(
                attempt_root / "attempt-end.json",
                {
                    "schema": "yeto_tonight85_attempt_end_v1",
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
                evidence = evidence_for_failure(
                    cell,
                    "INFRA_FAILURE",
                    "registered cell timeout",
                    record["command_hash"],
                )
            elif return_code:
                evidence = evidence_for_failure(
                    cell,
                    "INFRA_FAILURE",
                    f"scientific process exited {return_code} before a valid endpoint",
                    record["command_hash"],
                )
            else:
                validation_cell = {
                    **cell,
                    "source_git_commit": manifest["source"]["git_commit"],
                }
                evidence = (
                    validate_v7_cell(validation_cell, attempt_root, command)
                    if cell.get("island")
                    else validate_standard_cell(validation_cell, attempt_root, command)
                )
            evidence.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt,
                    "attempt_start_sha256": common.sha256_file(
                        attempt_root / "attempt-start.json"
                    ),
                    "attempt_end_sha256": common.sha256_file(
                        attempt_root / "attempt-end.json"
                    ),
                    "git_commit": manifest["source"]["git_commit"],
                    "retry_group_id": cell["retry_group_id"],
                }
            )
            common.write_json_atomic(evidence_path, evidence)
            if evidence["status"] in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}:
                completed += 1
            else:
                failures += 1
        common.write_json_atomic(
            status_path,
            {
                "schema": "yeto_tonight85_slot_status_v1",
                "state": "DRAINED",
                "stage": stage,
                "slot_id": slot_id,
                "node": node,
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
    parser.add_argument("--stage", required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--retry-groups",
        help="comma-separated registered retry groups; required for attempt 2",
    )
    args = parser.parse_args()
    if args.attempt == 2 and not args.retry_groups:
        parser.error("attempt 2 requires --retry-groups")
    if args.attempt == 1 and args.retry_groups:
        parser.error("--retry-groups is only valid for attempt 2")
    groups = set(args.retry_groups.split(",")) if args.retry_groups else None
    return run_queue(
        args.manifest,
        args.node_label,
        args.stage,
        args.slot_id,
        args.attempt,
        groups,
    )


if __name__ == "__main__":
    raise SystemExit(main())
